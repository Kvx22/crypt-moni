"""Real-account brokers. Both only ever act on orders a human approved.

CoinbaseBroker   Coinbase Advanced Trade API through Coinbase's official Python
                 SDK (`pip install coinbase-advanced-py`). Runs inside the
                 scanner, so an order you approve at 3 a.m. goes out at 3 a.m.
                 Use a CDP API key created inside a dedicated portfolio with
                 View + Trade permission only (never Transfer).

RobinhoodHandoff Robinhood's agentic account is reachable only through Claude's
                 Robinhood connector, not from a program on your machine. This
                 broker reads the balances Claude last synced into
                 data/robinhood_snapshot.json; approved orders wait in the
                 outbox until a Claude session places them (`export-approved`,
                 then `ack`). See EXECUTION.md.
"""
from __future__ import annotations

import json
import math
import os
import time

from .models import Account, Fill


def _d(x):
    """SDK responses are typed objects in newer versions and dicts in older ones."""
    if isinstance(x, dict):
        return x
    if hasattr(x, "to_dict"):
        return x.to_dict()
    return json.loads(json.dumps(x, default=lambda o: o.__dict__))


def _floor(x, inc):
    inc = float(inc)
    if inc <= 0:
        return x
    n = math.floor(x / inc + 1e-9) * inc
    dec = max(0, -int(math.floor(math.log10(inc)))) if inc < 1 else 0
    return round(n, dec)


def _fmt(x, inc):
    s = f"{_floor(x, inc):.10f}".rstrip("0").rstrip(".")
    return s or "0"


class CoinbaseBroker:
    name = "coinbase"
    queues = True          # orders go to the outbox for approval
    can_execute = True     # approved orders are placed by this process
    fees_paid = 0.0

    def __init__(self, key_file=None, quote="USD", client=None, cache_s=20.0):
        if client is None:
            try:
                from coinbase.rest import RESTClient
            except ImportError as e:
                raise RuntimeError("Coinbase SDK missing: pip install coinbase-advanced-py") from e
            if key_file and os.path.exists(key_file):
                client = RESTClient(key_file=key_file)
            elif os.environ.get("COINBASE_API_KEY") and os.environ.get("COINBASE_API_SECRET"):
                client = RESTClient(api_key=os.environ["COINBASE_API_KEY"], api_secret=os.environ["COINBASE_API_SECRET"])
            else:
                raise RuntimeError(f"No Coinbase API key: put the CDP key JSON at {key_file} "
                                   "or set COINBASE_API_KEY and COINBASE_API_SECRET")
        self.c, self.quote, self.cache_s = client, quote, cache_s
        self._acct, self._acct_ts, self._products = None, 0.0, {}

    def balances(self):
        r = _d(self.c.get_accounts(limit=250))
        bal, avail = {}, {}
        for a in r.get("accounts", []):
            cur = a.get("currency")
            av = float((a.get("available_balance") or {}).get("value", 0) or 0)
            hold = float((a.get("hold") or {}).get("value", 0) or 0)
            bal[cur] = bal.get(cur, 0.0) + av + hold
            avail[cur] = avail.get(cur, 0.0) + av
        return bal, avail

    def account(self, quotes):
        if self._acct is not None and time.time() - self._acct_ts < self.cache_s:
            bal, avail = self._acct
        else:
            bal, avail = self.balances()
            self._acct, self._acct_ts = (bal, avail), time.time()
        pos = {}
        for sym, q in quotes.items():
            base = sym.split("-")[0]
            qty = bal.get(base, 0.0)
            if qty > 0:
                pos[sym] = dict(qty=qty, price=q[0], value=qty * q[0], available=avail.get(base, 0.0))
        cash = avail.get(self.quote, 0.0)
        return Account(cash + sum(p["value"] for p in pos.values()), cash, pos)

    def product(self, pid):
        if pid not in self._products:
            self._products[pid] = _d(self.c.get_product(pid))
        return self._products[pid]

    def place(self, o, quotes, ts, client_order_id):
        pid = f"{o.symbol.split('-')[0]}-{self.quote}"
        price = quotes[o.symbol][0]
        p = self.product(pid)
        if o.side == "buy":
            size = _floor(o.notional_usd, p.get("quote_increment", "0.01"))
            if size < float(p.get("quote_min_size") or 1):
                return Fill(o.symbol, o.side, o.notional_usd, 0, price, ts, "rejected", f"below {pid} minimum order")
            r = _d(self.c.market_order_buy(client_order_id=client_order_id, product_id=pid,
                                           quote_size=_fmt(size, p.get("quote_increment", "0.01"))))
            qty = size / price
        else:
            held = self.account(quotes).positions.get(o.symbol, {}).get("available", 0.0)
            qty = _floor(min(o.notional_usd / price, held), p.get("base_increment", "0.00000001"))
            if qty <= 0 or qty < float(p.get("base_min_size") or 0):
                return Fill(o.symbol, o.side, o.notional_usd, 0, price, ts, "rejected", f"below {pid} minimum size or nothing to sell")
            r = _d(self.c.market_order_sell(client_order_id=client_order_id, product_id=pid,
                                            base_size=_fmt(qty, p.get("base_increment", "0.00000001"))))
            size = qty * price
        self._acct = None  # balances changed
        if r.get("success"):
            oid = (r.get("success_response") or {}).get("order_id") or r.get("order_id", "")
            return Fill(o.symbol, o.side, round(size, 2), qty, price, ts, "sent", oid)
        err = r.get("error_response") or {}
        return Fill(o.symbol, o.side, o.notional_usd, 0, price, ts, "rejected",
                    str(err.get("message") or err.get("error") or r.get("failure_reason") or r)[:300])


class RobinhoodHandoff:
    """Robinhood agentic account: balances from Claude's last sync, orders placed by Claude."""
    name = "robinhood"
    queues = True
    can_execute = False    # this process can't reach the agentic account; a Claude session places approved orders
    fees_paid = 0.0

    def __init__(self, snapshot="data/robinhood_snapshot.json", max_age_h=24.0):
        self.path, self.max_age = snapshot, max_age_h * 3600

    def read(self):
        if not os.path.exists(self.path):
            return None
        with open(self.path) as f:
            return json.load(f)

    def age(self):
        s = self.read()
        return None if not s else time.time() - float(s.get("ts", 0))

    def account(self, quotes):
        s = self.read()
        if not s:
            return Account(0.0, 0.0, {})
        cash = float(s.get("cash", 0.0))
        pos = {}
        for sym, qty in (s.get("positions") or {}).items():
            if sym in quotes and float(qty) > 0:
                pos[sym] = dict(qty=float(qty), price=quotes[sym][0], value=float(qty) * quotes[sym][0], available=float(qty))
        # holdings outside the board's universe (other stocks you own) count toward equity but are never traded
        other = float(s.get("other_value", 0.0))
        return Account(cash + other + sum(p["value"] for p in pos.values()), cash, pos)

    def stale(self):
        a = self.age()
        return a is None or a > self.max_age


def make_broker(cfg):
    ex = cfg.get("execution", {})
    kind = ex.get("broker", "paper")
    if kind == "coinbase":
        return CoinbaseBroker(ex.get("key_file", "secrets/cdp_api_key.json"), ex.get("quote_currency", "USD"))
    if kind == "robinhood":
        return RobinhoodHandoff(ex.get("snapshot", "data/robinhood_snapshot.json"), ex.get("snapshot_max_age_h", 24))
    return None
