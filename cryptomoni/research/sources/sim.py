"""Simulator sources: invented news, risk alerts, crowd mood and listings for the sim feed.

The simulated market (feeds.SimFeed) plants a signal in every coin: the next hour's
idiosyncratic return leans with world.phase() x the coin's last-24-hour idiosyncratic mean.
SimNews reads that planted lean and writes a headline that agrees with it 70% of the time
(and contradicts it otherwise), so the whole research loop (labelling, aggregation, outcome
logging, the research desk's learned weights, the stage gates) has something real but
noisy to find, end to end, without the internet.

Two rules keep the simulation honest:
  * deterministic: every adapter draws from its own numpy Generator seeded from its name and
    a run counter kept in ctx.state, so the same run gives the same items;
  * hands off the market: nothing here calls world.rng (or anything that uses it, like
    world.ticker), so turning the research sources on never changes the simulated prices.

All of it is fictional: publishers are "sim-wire-N", URLs are sim://, and no headline names
a real outlet or person.
"""
from __future__ import annotations

import zlib

import numpy as np

from ..aliases import CRYPTO_NAMES
from ..http import SourceError
from ..types import Item
from .base import Adapter, crypto_base

# (catalyst, bullish template, bearish template). Plain English, fictional, and free of the
# risk words (hack, lawsuit, halt...) so text rules never invent a risk flag on a news item.
TEMPLATES = (
    ("partnership", "{name} signs payments integration with a large fintech app",
     "Major payments partner winds down its {name} integration"),
    ("product", "{name} developers ship a long-awaited wallet upgrade ahead of schedule",
     "{name} delays its flagship product launch to next quarter"),
    ("upgrade", "{name} network upgrade goes live smoothly, fees drop sharply",
     "{name} network upgrade postponed after testnet problems"),
    ("funding", "Asset manager discloses a growing {name} treasury position",
     "Large fund reports trimming its {name} holdings"),
    ("analyst", "Research desk upgrades {name} ({ticker}) to outperform",
     "Research desk downgrades {name} ({ticker}) to underperform"),
    ("unlock", "{name} foundation extends token lockups by a year",
     "Large {name} token unlock scheduled for this week"),
    ("listing", "{ticker} gains new trading pairs on a major exchange",
     "Major exchange trims {ticker} trading pairs"),
)

SOCIAL_WORDS = {1: "upbeat", -1: "gloomy", 0: "mixed"}


def _name(ctx, symbol):
    names = (ctx.names or {}).get(symbol) if isinstance(ctx.names, dict) else None
    if isinstance(names, (list, tuple)) and names and isinstance(names[0], str) and names[0]:
        return names[0]
    base = crypto_base(symbol)
    return (CRYPTO_NAMES.get(base) or [base])[0]


def planted(world, symbol):
    """The simulator's planted next-hour lean for one symbol (a small float; 0 for BTC, which
    has no idiosyncratic part), or None when the world does not know the symbol."""
    try:
        i = list(world.symbols).index(symbol)
        return float(world.phase()) * float(np.asarray(world.idio).mean(0)[i])
    except (ValueError, IndexError, TypeError, AttributeError):
        return None


class _Sim(Adapter):
    sim = True
    assets = ("crypto",)
    interval_s = 3600
    domains = ("simulator",)

    def _start(self, ctx):
        """-> (private rng, run number, world clock). Raises SourceError without a world."""
        w = ctx.world
        if w is None or not hasattr(w, "symbols") or not hasattr(w, "idio") or not hasattr(w, "phase"):
            raise SourceError("simulator: no simulated market attached")
        try:
            n = int(ctx.state.get("runs", 0)) + 1
        except (TypeError, ValueError):
            n = 1
        ctx.state["runs"] = n
        rng = np.random.default_rng([zlib.crc32(self.name.encode("utf-8")), n])
        clock = float(getattr(w, "clock", ctx.now) or ctx.now)
        return rng, n, clock

    def _coins(self, ctx):
        """Subscribed crypto symbols the world knows, in a fixed order (so picks are reproducible)."""
        known = set(getattr(ctx.world, "symbols", []) or [])
        return sorted(s for s, a in (ctx.symbols or {}).items() if a == "crypto" and s in known)


class SimNews(_Sim):
    name, team, kind = "sim_news", "catalysts", "news"
    role = "primary"
    description = "Simulated newswire: headlines that lean with the simulator's planted signal 70% of the time."
    publishers = tuple(f"sim-wire-{i}" for i in range(1, 7))
    url_prefix = "sim://news/"
    agree_p, confidence = 0.7, 0.6

    def fetch(self, ctx):
        rng, n, clock = self._start(ctx)
        cands = [(s, planted(ctx.world, s)) for s in self._coins(ctx)]
        cands = [(s, v) for s, v in cands if v]           # BTC (no planted lean) gets no news
        if not cands:
            return []
        k = min(len(cands), int(rng.integers(2, 5)))
        out = []
        for j, idx in enumerate(rng.choice(len(cands), size=k, replace=False)):
            sym, lean = cands[int(idx)]
            truth = 1 if lean > 0 else -1
            d = truth if rng.random() < self.agree_p else -truth
            cat, bull, bear = TEMPLATES[int(rng.integers(len(TEMPLATES)))]
            pub = self.publishers[int(rng.integers(len(self.publishers)))]
            name, ticker = _name(ctx, sym), crypto_base(sym)
            title = (bull if d > 0 else bear).format(name=name, ticker=ticker)
            text = (f"{title}. Simulated report from {pub} about {name} ({sym}); invented for the simulator, "
                    f"not real news.")
            out.append(Item(source=self.name, url=f"{self.url_prefix}{n}-{int(clock)}-{j}", title=title, text=text,
                            published_ts=clock, symbols=[sym], kind=self.kind, publisher=pub, facts=dict(sim=True),
                            catalyst=cat, direction=d, risk_flags=[], confidence=self.confidence))
        return out


class SimWire(SimNews):
    name = "sim_wire"
    role, standby_for = "standby", "sim_news"
    description = "Second simulated newswire (different outlets): stands in for sim_news."
    publishers = tuple(f"sim-desk-{i}" for i in range(1, 7))
    url_prefix = "sim://wire/"


class SimRisk(_Sim):
    name, team, kind = "sim_risk", "risk", "news"
    role = "primary"
    description = "Simulated risk desk: a rare exploit or trading-halt alert on a random coin."
    p = 0.05

    def fetch(self, ctx):
        rng, n, clock = self._start(ctx)
        coins = self._coins(ctx)
        if not coins or rng.random() >= self.p:
            return []
        sym = coins[int(rng.integers(len(coins)))]
        name = _name(ctx, sym)
        if rng.random() < 0.5:
            return [Item(source=self.name, url=f"sim://risk/exploit/{sym}/{n}", kind="news", publisher="sim-wire-security",
                         title=f"{name} bridge exploited, attacker drains funds (simulated)",
                         text=f"Simulated alert: an exploit on a {name} bridge contract. Invented for the simulator.",
                         published_ts=clock, symbols=[sym], facts=dict(sim=True), catalyst="security", direction=-1,
                         risk_flags=["exploit"], confidence=0.8)]
        return [Item(source=self.name, url=f"sim://risk/halt/{sym}/{n}", kind="official", publisher="sim-exchange",
                     title=f"Sim Exchange halts {sym} trading for maintenance (simulated)",
                     text=f"Simulated exchange notice: {sym} order books are paused. Invented for the simulator.",
                     published_ts=clock, symbols=[sym], facts=dict(sim=True, status="halted"), catalyst="outage",
                     direction=-1, risk_flags=["halt"], confidence=0.95, official=True)]


class SimSocial(_Sim):
    name, team, kind = "sim_social", "sentiment", "sentiment"
    role = "primary"
    description = "Simulated crowd mood per coin: agrees with the planted signal only 55% of the time."
    agree_p = 0.55

    def fetch(self, ctx):
        rng, n, clock = self._start(ctx)
        out = []
        for sym in self._coins(ctx):
            lean = planted(ctx.world, sym)
            if lean:
                truth = 1 if lean > 0 else -1
                d = truth if rng.random() < self.agree_p else -truth
            else:
                d = 0
            total = int(rng.integers(12, 40))
            share = {1: rng.uniform(0.7, 0.85), -1: rng.uniform(0.15, 0.35), 0: rng.uniform(0.4, 0.6)}[d]
            bulls = int(round(total * share))
            out.append(Item(source=self.name, url=f"sim://social/{sym}/{n}-{int(clock)}", kind=self.kind,
                            publisher="sim-social", published_ts=clock, symbols=[sym],
                            title=f"Simulated chatter on {_name(ctx, sym)} is {SOCIAL_WORDS[d]}: "
                                  f"{bulls} of {total} tagged posts bullish",
                            text="Aggregate of simulated social posts over the last 6 hours. Invented for the simulator.",
                            facts=dict(sim=True, bulls=bulls, bears=total - bulls, n=total), catalyst="sentiment",
                            direction=d, risk_flags=[], confidence=0.4))
        return out


class SimListings(_Sim):
    name, team, kind = "sim_listings", "discovery", "discovery"
    role = "primary"
    description = "Simulated exchange listings of coins the board does not follow yet."
    p = 0.15

    def fetch(self, ctx):
        rng, n, clock = self._start(ctx)
        listed = ctx.state.get("listed")
        listed = [s for s in listed if isinstance(s, str)] if isinstance(listed, list) else []
        subs = set(ctx.symbols or {})
        cands = sorted(s for s in (getattr(ctx.world, "symbols", []) or [])
                       if isinstance(s, str) and "-" in s and s not in subs and s not in listed)
        if not cands or rng.random() >= self.p:
            return []
        sym = cands[int(rng.integers(len(cands)))]
        ctx.state["listed"] = (listed + [sym])[-200:]
        return [Item(source=self.name, url=f"sim://listing/{sym}", kind=self.kind, publisher="sim-exchange",
                     title=f"Sim Exchange lists {sym} (simulated)",
                     text=f"Simulated listing notice: {_name(ctx, sym)} ({sym}) trading opens today. Invented for the simulator.",
                     published_ts=clock, symbols=[sym], facts=dict(sim=True, product_id=sym), catalyst="listing",
                     direction=1, risk_flags=[], confidence=0.8, official=True)]


SIM_ADAPTERS = [SimNews, SimWire, SimRisk, SimSocial, SimListings]
