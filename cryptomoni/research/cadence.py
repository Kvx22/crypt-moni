"""The 2-hour cadence paper books: a side-by-side test of trading more often, but pickier.

The board rebalances every `rebalance_every_hours` (8 by default). The question is whether
looking every 2 hours, while only buying on real conviction, waiting out a cooldown after
each trade on a symbol and capping trades per day, does better after fees. Two paper books
run beside the main book on the same target weights and the same prices:

  base     rebalances on the board's own schedule, no extra rules (what the board does today)
  cadence  looks every `every_hours`, and its orders pass through `cadence_filter`

Both are paper books: nothing here can reach a broker. Their state lives in the board
store (kv "cadence_books") so they survive restarts; `stats()` compares them for the
stage switch that may one day hand the cadence to the real book.
"""
from __future__ import annotations

import dataclasses
import math

from ..feeds import DAY, HOUR
from ..paper import PaperBroker

RULES = dict(every_hours=2, min_conviction=0.35, cooldown_hours=6, max_trades_per_day=6)
KV = "cadence_books"
HIST_MAX = 24 * 120


def _day(ts):
    return str(int(ts // DAY))


def cadence_filter(orders, conviction, last_trade, trades_today, now, rules=None):
    """Keep the orders the cadence rules allow. Returns (keep, dropped), dropped = [(order, reason)].
    Buys need conviction >= min_conviction; nothing trades a symbol within cooldown_hours of its
    last trade; then sells first, buys by size, until max_trades_per_day is used up."""
    r = dict(RULES, **(rules or {}))
    keep, dropped = [], []
    for o in orders:
        c = (conviction or {}).get(o.symbol)
        last = (last_trade or {}).get(o.symbol)
        if o.side == "buy" and (c is None or not c >= r["min_conviction"]):
            have = "no conviction yet" if c is None else f"conviction {c:+.2f}"
            dropped.append((o, f"{have} is below the {r['min_conviction']:.2f} needed to buy"))
        elif last is not None and now - last < r["cooldown_hours"] * HOUR:
            dropped.append((o, f"traded {(now - last) / HOUR:.1f}h ago; waiting out the {r['cooldown_hours']:g}h cooldown"))
        else:
            keep.append(o)
    keep.sort(key=lambda o: (o.side != "sell", -o.notional_usd))
    room = max(0, int(r["max_trades_per_day"]) - int(trades_today or 0))
    for o in keep[room:]:
        dropped.append((o, f"the {int(r['max_trades_per_day'])} trades a day are used up"))
    return keep[:room], dropped


def max_drawdown(equity):
    """Deepest fall from a running high, as a negative fraction (0 when it never fell)."""
    hi, dd = -math.inf, 0.0
    for e in equity:
        if e is None or not e > 0:
            continue
        hi = max(hi, e)
        dd = min(dd, e / hi - 1)
    return dd


class CadenceBooks:
    """The base and cadence paper books. Engine thread only (it writes the board store)."""

    def __init__(self, cfg, store):
        self.s = store
        self.paper = dict(cfg.get("paper") or {})
        self.start_cash = float(self.paper.get("start_cash", 10000.0))
        self.rules = dict(RULES, **(cfg.get("cadence") or {}))
        self.rules["every_hours"] = max(1, int(self.rules["every_hours"]))
        self.base_every = max(1, int(cfg.get("rebalance_every_hours", 8)))
        self.min_notional = 1.0
        self._fresh()
        try:
            self._load(store.kv_get(KV) or {})
        except Exception:  # noqa: BLE001  (a damaged saved state starts the books over, never stops the engine)
            self._fresh()

    def _fresh(self):
        self.books = {"base": PaperBroker(**self.paper), "cad": PaperBroker(**self.paper)}
        self.last_trade, self.per_day, self.hist = {}, {}, []      # per_day: UTC day number -> cadence trades
        self.trades, self.px = {"base": 0, "cad": 0}, {}           # px: last price seen per symbol (for equity)
        self.start_ts = self.last_bar = None

    def _load(self, st):
        """Strict: every number is coerced, anything malformed raises and the caller starts over."""
        for k, b in self.books.items():
            if st.get(k):
                b.load(st[k])
                b.cash, b.fees_paid = float(b.cash), float(b.fees_paid)
                b.qty = {str(s): float(q) for s, q in b.qty.items()}
        self.last_trade = {str(s): float(t) for s, t in (st.get("last_trade") or {}).items()}
        self.per_day = {str(int(d)): int(n) for d, n in (st.get("per_day") or {}).items()}
        self.hist = [[float(t), float(b), float(c)] for t, b, c in (st.get("hist") or [])]
        self.trades = {k: int((st.get("trades") or {}).get(k, 0)) for k in ("base", "cad")}
        self.px = {str(s): float(p) for s, p in (st.get("px") or {}).items()}
        self.start_ts = None if st.get("start_ts") is None else float(st["start_ts"])
        self.last_bar = st.get("last_bar")

    # ------------------------------------------------------------------ the hourly step
    def step(self, now, bar_ts, weights, conviction, quotes, plan_fn):
        """One closed hour. Rebalances whichever books are due, logs equity, saves.
        Returns dict(base=[fills], cad=[fills], dropped=[(symbol, side, reason)]) or None
        when this bar was already handled."""
        if bar_ts is None or bar_ts == self.last_bar:
            return None
        self.last_bar = bar_ts
        if self.start_ts is None:
            self.start_ts = now
        fresh = {}
        for s, x in (quotes or {}).items():
            try:
                p = float(x[0])
            except (TypeError, ValueError, IndexError):
                continue
            if p > 0 and math.isfinite(p):
                self.px[s] = p
                fresh[s] = (p, now)
        q = {s: (p, now) for s, p in self.px.items()}   # last known prices: for valuing the books only
        hour = int(bar_ts // HOUR)
        out = dict(base=[], cad=[], dropped=[])
        if hour % self.base_every == 0:
            out["base"] = self._rebalance("base", now, weights, q, fresh, plan_fn)
        if hour % self.rules["every_hours"] == 0:
            out["cad"], out["dropped"] = self._rebalance("cad", now, weights, q, fresh, plan_fn, conviction)
        eq = [self._equity(k, q) for k in ("base", "cad")]
        self.hist = (self.hist + [[bar_ts, round(eq[0], 2), round(eq[1], 2)]])[-HIST_MAX:]
        today = int(_day(now))
        self.per_day = {d: n for d, n in self.per_day.items() if int(d) >= today - 2}
        self.save()
        return out

    def _equity(self, k, q):
        try:
            return float(self.books[k].account(q).equity)
        except Exception:  # noqa: BLE001
            return float(self.books[k].cash)

    def _rebalance(self, k, now, weights, q, fresh, plan_fn, conviction=None):
        """Valued at last known prices; planned and filled only at this hour's quotes."""
        book = self.books[k]
        try:
            acct = book.account(q)
            if acct.equity <= 0:
                return ([], []) if k == "cad" else []
            orders = list(plan_fn(acct, dict(weights or {}), fresh) or [])
        except Exception:  # noqa: BLE001  (a planning failure skips this round for this book only)
            return ([], []) if k == "cad" else []
        dropped = []
        if k == "cad":
            orders, dr = cadence_filter(orders, conviction or {}, self.last_trade, self.per_day.get(_day(now), 0), now, self.rules)
            dropped = [(o.symbol, o.side, why) for o, why in dr]
        fills = []
        orders.sort(key=lambda o: (o.side != "sell", -o.notional_usd))   # sells raise the cash buys spend
        for o in orders:
            if o.symbol not in fresh or not (o.notional_usd > 0) or o.side not in ("buy", "sell"):
                continue
            if o.side == "buy" and o.notional_usd > book.cash:
                if book.cash < self.min_notional:
                    continue
                o = dataclasses.replace(o, notional_usd=book.cash)   # paper cash is the only limit
            try:
                f = book.place(o, fresh, now)
            except Exception:  # noqa: BLE001
                continue
            if f.status != "filled":
                continue
            fills.append(dict(symbol=f.symbol, side=f.side, notional_usd=f.notional_usd, price=f.price, fee=f.fee_usd))
            self.trades[k] = self.trades.get(k, 0) + 1
            if k == "cad":
                self.last_trade[o.symbol] = now
                self.per_day[_day(now)] = self.per_day.get(_day(now), 0) + 1
        return (fills, dropped) if k == "cad" else fills

    def save(self):
        self.s.kv_set(KV, dict(base=self.books["base"].state(), cad=self.books["cad"].state(), last_trade=self.last_trade,
                               per_day=self.per_day, hist=self.hist, trades=self.trades, px=self.px,
                               start_ts=self.start_ts, last_bar=self.last_bar, start_cash=self.start_cash))

    # ------------------------------------------------------------------ reading
    def stats(self, now):
        """dict(days, base_ret, cad_ret, base_dd, cad_dd, base_fees, cad_fees, base_trades, cad_trades).
        Returns are on the starting cash; drawdowns (negative) over the kept history."""
        base_eq = self.hist[-1][1] if self.hist else self.start_cash
        cad_eq = self.hist[-1][2] if self.hist else self.start_cash
        sc = self.start_cash if self.start_cash > 0 else 1.0
        return dict(days=max(0.0, (now - self.start_ts) / DAY) if self.start_ts is not None else 0.0,
                    base_ret=base_eq / sc - 1, cad_ret=cad_eq / sc - 1,
                    base_dd=max_drawdown(h[1] for h in self.hist), cad_dd=max_drawdown(h[2] for h in self.hist),
                    base_fees=float(self.books["base"].fees_paid), cad_fees=float(self.books["cad"].fees_paid),
                    base_trades=int(self.trades.get("base", 0)), cad_trades=int(self.trades.get("cad", 0)))

    def view(self, now=None):
        """For the dashboard: stats, the rules, and the last 200 equity points."""
        if now is None:
            now = (self.hist[-1][0] + HOUR) if self.hist else (self.start_ts or 0.0)
        return dict(self.stats(now), rules=dict(self.rules), base_every_hours=self.base_every, start_cash=self.start_cash,
                    history=self.hist[-200:])
