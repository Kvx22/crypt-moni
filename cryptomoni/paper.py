"""Paper broker: simulated fills with fees and slippage. It is the only broker in
this project, so nothing here can reach a real exchange or move real money."""
from __future__ import annotations

from .models import Account, Fill


class PaperBroker:
    name = "paper"

    def __init__(self, start_cash=10000.0, fee_bps=40.0, slippage_bps=5.0):
        self.cash = float(start_cash)
        self.qty = {}
        self.fee, self.slip = fee_bps / 1e4, slippage_bps / 1e4
        self.fees_paid = 0.0

    def account(self, quotes):
        pos = {}
        for s, q in self.qty.items():
            if q > 0 and s in quotes:
                p = quotes[s][0]
                pos[s] = dict(qty=q, price=p, value=q * p)
        return Account(self.cash + sum(p["value"] for p in pos.values()), self.cash, pos)

    def place(self, o, quotes, ts):
        price = quotes[o.symbol][0]
        if o.side == "buy":
            px = price * (1 + self.slip)
            fee = o.notional_usd * self.fee
            spend = o.notional_usd
            if spend > self.cash + 1e-9:
                return Fill(o.symbol, o.side, o.notional_usd, 0, price, ts, "rejected", "insufficient paper cash")
            qty = (spend - fee) / px
            self.cash -= spend
            self.qty[o.symbol] = self.qty.get(o.symbol, 0.0) + qty
        else:
            px = price * (1 - self.slip)
            held = self.qty.get(o.symbol, 0.0)
            qty = min(o.notional_usd / px, held)
            if qty <= 0:
                return Fill(o.symbol, o.side, o.notional_usd, 0, price, ts, "rejected", "no paper position")
            gross = qty * px
            fee = gross * self.fee
            self.cash += gross - fee
            self.qty[o.symbol] = held - qty
            if self.qty[o.symbol] * price < 0.01:
                self.qty.pop(o.symbol)
        self.fees_paid += fee
        return Fill(o.symbol, o.side, round(o.notional_usd, 2), qty, px, ts, "filled", "paper", round(fee, 4))

    def state(self):
        return dict(cash=self.cash, qty=self.qty, fees=self.fees_paid)

    def load(self, s):
        self.cash, self.qty, self.fees_paid = s["cash"], dict(s["qty"]), s.get("fees", 0.0)
