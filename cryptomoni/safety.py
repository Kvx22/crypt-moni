"""Independent safety layer.

HARD RULE: this module imports nothing from the brain (regime, advisors,
scanner, allocation). It is deterministic, has no learned parameters, and sits
between every order intent and the (paper) broker. If the model is wrong or
the process is confused, these checks still hold.

Paper by default. "live" mode is refused unless it is armed twice (config
`live_armed: true` AND environment variable CM_LIVE=1). Even then the engine
only queues orders: nothing is sent until a human approves it, and approved
orders come back through this gate with fresh prices before they go out.

Pieces:
  KillSwitch      - a file on disk. If it exists, no order passes. The dashboard
                    button, the watchdog, or `python run.py kill` create it; only
                    `python run.py reset-kill` clears it.
  CircuitBreaker  - drawdown / day-loss state machine -> exposure cap, buy block,
                    latched HALT that only a human can reset.
  SafetyGate      - per-order checks: allowlist, size caps, no shorting, position
                    caps, cash reserve, price band, stale quotes, cooldown,
                    turnover and order-count limits.
"""
from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field

from .models import Order


@dataclass
class SafetyConfig:
    mode: str = "paper"                      # "paper" | "live" (human-approved orders only)
    live_armed: bool = False                 # must ALSO be true (plus env CM_LIVE=1) for live orders
    max_per_trade_usd: float | None = None   # UNSET => every order rejected
    on_cap_exceeded: str = "clamp"           # "clamp" to the cap, or "reject"
    max_orders_per_day: int = 60
    max_daily_turnover_pct: float = 0.60     # of equity, buys+sells, per UTC day
    min_cash_pct: float = 0.02               # buys must leave this much cash
    default_max_position_pct: float = 0.15
    max_position_pct: dict = field(default_factory=lambda: {"BTC-USD": 0.50, "ETH-USD": 0.35})
    allowlist: list = field(default_factory=list)       # empty => nothing allowed
    price_band_pct: float = 0.02             # limit price within 2% of the quote
    max_quote_age_s: float = 300             # crypto trades 24/7: quotes go stale fast
    order_cooldown_s: float = 3 * 3600       # same symbol+side
    kill_file: str = "KILL"
    # circuit breaker (crypto-sized: it swings 2-4x harder than stocks)
    dd_yellow: float = 0.10
    dd_orange: float = 0.15
    dd_red_halt: float = 0.25
    cap_yellow: float = 0.60
    cap_orange: float = 0.30
    release_gap: float = 0.03
    day_loss_block: float = 0.05
    hwm_decay_per_day: float = 0.0005        # high-water mark forgets ~17%/yr so one drawdown can't pin the cap forever

    @staticmethod
    def from_dict(d):
        base = SafetyConfig()
        for k, v in (d or {}).items():
            if hasattr(base, k):
                setattr(base, k, v)
        return base


def live_armed(c: "SafetyConfig") -> bool:
    return c.mode == "live" and c.live_armed is True and os.environ.get("CM_LIVE") == "1"


# ----------------------------------------------------------------------------
class KillSwitch:
    def __init__(self, path):
        self.path = path

    def engaged(self):
        return os.path.exists(self.path)

    def trip(self, why="manual"):
        with open(self.path, "w") as f:
            f.write(f"{time.time():.0f} {why}\n")

    def reason(self):
        try:
            with open(self.path) as f:
                return f.read().strip()
        except OSError:
            return ""

    def reset(self):
        if os.path.exists(self.path):
            os.remove(self.path)


@dataclass
class BreakerState:
    level: int = 0                 # 0 ok, 1 yellow, 2 orange, 3 halt
    name: str = "OK"
    exposure_cap: float = 1.0
    allow_buys: bool = True
    halted: bool = False
    drawdown: float = 0.0
    day_pnl_pct: float = 0.0
    hwm: float = 0.0
    reasons: list = field(default_factory=list)


class CircuitBreaker:
    NAMES = ["OK", "YELLOW", "ORANGE", "HALT"]

    def __init__(self, cfg: SafetyConfig, saved: dict | None = None):
        self.cfg = cfg
        s = saved or {}
        self.hwm = s.get("hwm", 0.0)
        self.level = s.get("level", 0)
        self.halted = s.get("halted", False)

    def save(self):
        return dict(hwm=self.hwm, level=self.level, halted=self.halted)

    def reset(self, equity):
        """Human-only. Re-bases the high-water mark and clears the halt."""
        self.hwm, self.level, self.halted = equity, 0, False

    def update(self, equity, day_open_equity, new_day=True) -> BreakerState:
        c = self.cfg
        if new_day:
            self.hwm *= 1 - c.hwm_decay_per_day
        self.hwm = max(self.hwm, equity)
        dd = 0.0 if self.hwm <= 0 else 1 - equity / self.hwm
        thr = [0.0, c.dd_yellow, c.dd_orange, c.dd_red_halt]
        lvl = self.level
        while lvl < 3 and dd >= thr[lvl + 1]:
            lvl += 1
        while 0 < lvl < 3 and dd < thr[lvl] - c.release_gap:
            lvl -= 1
        if lvl == 3:
            self.halted = True
        self.level = lvl if not self.halted else 3
        day = 0.0 if not day_open_equity else equity / day_open_equity - 1
        reasons = []
        cap = {0: 1.0, 1: c.cap_yellow, 2: c.cap_orange, 3: 0.0}[self.level]
        if self.level:
            reasons.append(f"drawdown {dd:.1%} from high-water mark -> {self.NAMES[self.level]}")
        allow = self.level < 3
        if day <= -c.day_loss_block:
            allow = False
            reasons.append(f"day loss {day:.1%} <= -{c.day_loss_block:.0%}: no new buys today (UTC)")
        if self.halted:
            reasons.append("HALTED: sells only, until a human runs `python run.py reset-breaker`")
        return BreakerState(self.level, self.NAMES[self.level], cap, allow, self.halted, dd, day, self.hwm, reasons)


# ----------------------------------------------------------------------------
@dataclass
class GateContext:
    now: float
    equity: float
    cash: float
    positions: dict                 # sym -> market value USD
    quotes: dict                    # sym -> (price, ts)
    breaker: BreakerState
    orders_today: int = 0
    turnover_today: float = 0.0
    last_order_ts: dict = field(default_factory=dict)   # (sym, side) -> ts
    pending_buys: float = 0.0       # buys already approved this batch
    pending_pos: dict = field(default_factory=dict)     # sym -> delta value approved this batch


@dataclass
class Verdict:
    ok: bool
    order: Order | None
    reasons: list
    clamped: bool = False


class SafetyGate:
    def __init__(self, cfg: SafetyConfig, kill: KillSwitch | None = None):
        self.cfg = cfg
        self.kill = kill or KillSwitch(cfg.kill_file)

    def _no(self, why, *more):
        return Verdict(False, None, [why, *more])

    def check(self, o: Order, x: GateContext) -> Verdict:
        c = self.cfg
        if self.kill.engaged():
            return self._no(f"KILL SWITCH engaged ({self.kill.reason()})")
        if c.mode not in ("paper", "live"):
            return self._no(f"unknown mode {c.mode!r}")
        if c.mode == "live" and not live_armed(c):
            return self._no("live mode not armed (needs live_armed=true in config AND env CM_LIVE=1)")
        if c.max_per_trade_usd is None or not (c.max_per_trade_usd > 0):
            return self._no("MAX_PER_TRADE_USD is unset: refusing all orders")
        if o.side not in ("buy", "sell"):
            return self._no(f"bad side {o.side!r}")
        if not (isinstance(o.notional_usd, (int, float)) and math.isfinite(o.notional_usd) and o.notional_usd > 0):
            return self._no("notional must be a positive finite number")
        sym = o.symbol.upper()
        if sym not in {s.upper() for s in c.allowlist}:
            return self._no(f"{sym} not on allowlist")
        q = x.quotes.get(sym)
        if q is None:
            return self._no(f"no quote for {sym}")
        price, qts = q
        if not (price and price > 0 and math.isfinite(price)):
            return self._no(f"bad quote for {sym}")
        if x.now - qts > c.max_quote_age_s:
            return self._no(f"stale quote for {sym} ({(x.now - qts) / 60:.0f} min old)")
        last = x.last_order_ts.get((sym, o.side))
        if last is not None and x.now - last < c.order_cooldown_s:
            return self._no(f"cooldown: {sym} {o.side} placed {(x.now - last) / 3600:.1f}h ago")
        if o.order_type == "limit":
            if not o.limit_price or o.limit_price <= 0:
                return self._no("limit order without price")
            if abs(o.limit_price / price - 1) > c.price_band_pct:
                return self._no(f"limit {o.limit_price:.4g} is >{c.price_band_pct:.0%} from quote {price:.4g}")
        reasons, clamped = [], False
        notional = o.notional_usd
        held = x.positions.get(sym, 0.0) + x.pending_pos.get(sym, 0.0)

        if o.side == "sell":
            if held <= 0.01:
                return self._no(f"no {sym} position to sell (shorting is disabled)")
            if notional > held:
                notional, clamped = held, True
                reasons.append(f"clamped to position value ${held:,.2f}")
        else:
            if not x.breaker.allow_buys:
                return self._no("breaker: buys blocked", *x.breaker.reasons)
            room_cash = x.cash - x.pending_buys - c.min_cash_pct * x.equity
            if room_cash <= 0:
                return self._no("no cash above the reserve")
            if notional > room_cash:
                notional, clamped = room_cash, True
                reasons.append(f"clamped to cash above reserve ${room_cash:,.2f}")
            cap_pct = c.max_position_pct.get(sym, c.default_max_position_pct)
            room_pos = cap_pct * x.equity - held
            if room_pos <= 0:
                return self._no(f"{sym} already at position cap {cap_pct:.0%}")
            if notional > room_pos:
                notional, clamped = room_pos, True
                reasons.append(f"clamped to {sym} cap {cap_pct:.0%} of equity")
        if notional > c.max_per_trade_usd:
            if c.on_cap_exceeded == "reject":
                return self._no(f"${notional:,.2f} exceeds MAX_PER_TRADE_USD ${c.max_per_trade_usd:,.2f}")
            notional, clamped = c.max_per_trade_usd, True
            reasons.append(f"clamped to MAX_PER_TRADE_USD ${c.max_per_trade_usd:,.2f}")
        if x.orders_today >= c.max_orders_per_day:
            return self._no(f"daily order limit {c.max_orders_per_day} reached")
        if x.turnover_today + notional > c.max_daily_turnover_pct * x.equity:
            room = c.max_daily_turnover_pct * x.equity - x.turnover_today
            if room < 1.0:
                return self._no(f"daily turnover cap {c.max_daily_turnover_pct:.0%} reached")
            notional, clamped = room, True
            reasons.append(f"clamped to remaining turnover ${room:,.2f}")
        if notional < 1.0:
            return self._no("order under $1 after limits")
        out = Order(sym, o.side, round(notional, 2), o.order_type, o.limit_price, o.reason)
        return Verdict(True, out, reasons, clamped)

    @staticmethod
    def commit(v: Verdict, x: GateContext):
        """Book an approved order into the context so later orders in the batch see it."""
        o = v.order
        x.orders_today += 1
        x.turnover_today += o.notional_usd
        x.last_order_ts[(o.symbol, o.side)] = x.now
        if o.side == "buy":
            x.pending_buys += o.notional_usd
            x.pending_pos[o.symbol] = x.pending_pos.get(o.symbol, 0.0) + o.notional_usd
        else:
            x.pending_pos[o.symbol] = x.pending_pos.get(o.symbol, 0.0) - o.notional_usd
            x.cash += o.notional_usd  # sale proceeds are usable by later buys in the batch
