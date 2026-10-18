"""Plain data types for the research pipeline. No logic beyond (de)serialising."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

# evidence kinds: what sort of thing a note is (the four-source rule wants at least two kinds)
EVIDENCE = ("news", "filing", "market", "sentiment", "discovery", "search", "official")

# teams: every adapter belongs to one
TEAMS = ("catalysts", "risk", "filings", "market", "sentiment", "discovery")


@dataclass
class Item:
    """One thing an adapter fetched: an article, a filing, a market reading, a post.

    `symbols` are the symbols the adapter KNOWS the item is about (a per-symbol query, a
    filing by that company, a funding rate for that coin). Leave it empty for general
    feeds; the labeller then finds mentions in the text.

    Structured adapters (market data, exchange status, insider filings) can pre-label the
    item with catalyst / direction / risk_flags / confidence; text items leave those None
    and the extractor labels them.
    """
    source: str                     # adapter name, e.g. "coindesk_rss"
    url: str                        # canonical link (or a stable pseudo-URL for data readings: "data://kraken_futures/BTC-USD/2026-10-18T05")
    title: str
    text: str = ""                  # summary or extracted body (<= ~4000 chars)
    published_ts: float | None = None
    symbols: list = field(default_factory=list)
    kind: str = "news"              # one of EVIDENCE
    publisher: str = ""             # outlet or domain ("coindesk.com", "sec.gov", "reddit:r/CryptoCurrency")
    facts: dict = field(default_factory=dict)      # structured numbers (funding_rate, shares, price, tvl_change_1d, ...)
    catalyst: str | None = None     # pre-label (labels.CATALYSTS)
    direction: int | None = None    # -1 / 0 / +1
    risk_flags: list = field(default_factory=list) # labels.RISK_FLAGS
    confidence: float | None = None # 0..1 for pre-labels
    official: bool = False          # straight from the exchange / regulator (fast-path for risk flags)
    quarantine: bool = False        # came from the forever-search sandbox: untrusted until corroborated

    def d(self):
        return asdict(self)

    @staticmethod
    def from_d(d):
        known = {k: v for k, v in d.items() if k in Item.__dataclass_fields__}
        return Item(**known)


@dataclass
class Note:
    """One labelled claim about one symbol, derived from one item."""
    symbol: str
    asset: str                      # "crypto" | "stock"
    ts: float                       # when it happened (published, else fetched)
    fetched_ts: float
    source: str
    team: str
    publisher: str
    evidence: str                   # one of EVIDENCE
    catalyst: str
    direction: int
    confidence: float
    risk_flags: list
    severity: float                 # 0..1, max severity of its risk flags (0 when none)
    label_by: str                   # "rules" | "model" | "human" | "seed" | "source"
    title: str
    url: str
    facts: dict = field(default_factory=dict)
    cluster: str = ""               # near-duplicate fingerprint: syndicated copies share it
    quarantine: bool = False
    official: bool = False
    expires_ts: float = 0.0
    item_id: int | None = None
    id: int | None = None

    def d(self):
        return asdict(self)


def dumps(x):
    return json.dumps(x, separators=(",", ":"), default=float)


def loads(s, default=None):
    if s is None or s == "":
        return default
    try:
        return json.loads(s)
    except (TypeError, ValueError):
        return default
