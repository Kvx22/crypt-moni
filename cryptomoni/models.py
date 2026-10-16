"""Plain data types shared by engine, paper broker and safety. No logic, no ML."""
from __future__ import annotations

from dataclasses import dataclass, asdict, field


@dataclass
class Order:
    symbol: str
    side: str                   # "buy" | "sell"
    notional_usd: float         # dollars (fractional crypto)
    order_type: str = "market"  # "market" | "limit"
    limit_price: float | None = None
    reason: str = ""

    def d(self):
        return asdict(self)


@dataclass
class Fill:
    symbol: str
    side: str
    notional_usd: float
    qty: float
    price: float
    ts: float
    status: str = "filled"      # filled | rejected | dry_run | error
    detail: str = ""
    fee_usd: float = 0.0


@dataclass
class Account:
    equity: float
    cash: float
    positions: dict = field(default_factory=dict)   # symbol -> dict(qty, price, value)


@dataclass
class Bar:
    ts: int          # bar start, unix seconds (UTC)
    open: float
    high: float
    low: float
    close: float
    volume: float    # base-currency volume
