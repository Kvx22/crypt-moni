"""What every source adapter looks like.

An adapter is one source (one site or API). It does exactly one thing: fetch, parse,
return Items. It never writes to the database, never sleeps (the HTTP client paces it),
and raises SourceError (or a subclass) when the source fails; the worker slot that runs
it handles retries, standbys and health.

    class CoinDeskRSS(Adapter):
        name, team, kind = "coindesk_rss", "catalysts", "news"
        assets = ("crypto",)
        interval_s = 900
        domains = ("www.coindesk.com",)
        def fetch(self, ctx):
            xml = ctx.http.text("https://www.coindesk.com/arc/outboundfeeds/rss/")
            return [Item(...) for e in parse_feed(xml)]

Per-symbol adapters (a news search per ticker, a filings lookup per company) set
`per_symbol = True`; `ctx.symbols` then holds only their shard of the subscribed
symbols, requested symbols first.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..http import SourceError  # noqa: F401  (re-exported for adapters)
from ..types import Item  # noqa: F401


@dataclass
class FetchContext:
    http: object                     # research.http.Http
    now: float                       # time.time() (or the simulated clock)
    symbols: dict                    # {symbol: asset}: what to look at (per_symbol: this shard; else everything subscribed)
    requested: list = field(default_factory=list)   # symbols advisors asked about (highest priority first)
    state: dict = field(default_factory=dict)       # persisted per adapter between runs (seen ids, last product list, ...)
    cfg: dict = field(default_factory=dict)         # the "research" config block
    names: dict = field(default_factory=dict)       # {symbol: [display names]} from aliases, for building search queries
    world: object = None             # simulator handle (sim adapters only)

    def log(self, msg):
        self.state.setdefault("_log", [])
        self.state["_log"] = (self.state["_log"] + [str(msg)[:200]])[-5:]


class Adapter:
    name = "base"
    team = "catalysts"               # one of types.TEAMS
    kind = "news"                    # default evidence kind of its items
    assets = ("crypto", "stock")     # which assets it can cover
    interval_s = 900.0               # base polling interval (the scheduler stretches/shrinks it by usefulness)
    per_symbol = False               # True: ctx.symbols is this slot's shard
    max_symbols_per_run = 8          # per_symbol adapters look at no more than this many per run
    domains = ()                     # hosts it talks to (shown on the health panel)
    official = False                 # its items come straight from an exchange/regulator
    sim = False                      # a simulator adapter (only used with the sim feed)
    role = "primary"                 # "primary" | "standby" (takes over when its primary fails) | "dormant" (auto-growth reserve)
    standby_for = None               # standby adapters: the primary's name
    description = ""

    def fetch(self, ctx: FetchContext) -> list:
        raise NotImplementedError

    def covers(self, asset):
        return asset in self.assets


def crypto_base(symbol):
    """BTC-USD -> BTC"""
    return symbol.split("-")[0]


def sym_assets(ctx, asset):
    """The context's symbols of one asset class, requested ones first."""
    req = [s for s in ctx.requested if ctx.symbols.get(s) == asset]
    rest = [s for s, a in ctx.symbols.items() if a == asset and s not in req]
    return req + rest
