"""Crypto sources: news feeds, exchange status, DeFi fundamentals, market structure,
sentiment and new listings.

Every adapter here reads one public endpoint that needs no key. Text sources (news
feeds, status-page incidents) hand their items to the labeller with no symbols; the
labeller finds the coins they mention. Structured sources (exchange status, funding
rates, order books, TVL, the fear & greed index, listings) already know what they are
about, so they pre-label their items: catalyst, direction, risk flags and a confidence.

Structured sources only speak when something is worth saying: a product halts or comes
back, funding gets crowded, a book leans hard to one side, a new pair appears. They keep
what they need to notice a change (last states, trailing readings, known product ids) in
ctx.state, which the worker pool persists between runs.

Each primary has a standby on a different venue that the pool switches on when the
primary keeps failing (Coinbase <-> Kraken, CoinDesk <-> Cointelegraph, ...).
"""
from __future__ import annotations

import math
import re
import time

from ..http import RateLimited, SourceError
from ..text import clean, domain, parse_feed
from ..types import Item
from .base import Adapter, crypto_base, sym_assets

DAY = 86400.0
NEWS_MAX_AGE = 3 * DAY
FUTURE_SKEW = 900.0           # a timestamp up to 15 min ahead is clock skew; further ahead is junk

KRAKEN_ALIAS = {"XBT": "BTC", "XDG": "DOGE"}
KRAKEN_FROM = {v: k for k, v in KRAKEN_ALIAS.items()}
# Kraken's old four-letter asset codes ("XXBT", "ZUSD") carry an X/Z class prefix
_KRAKEN_LEGACY = {"XBT", "XDG", "ETH", "ETC", "LTC", "XRP", "XLM", "XMR", "ZEC", "MLN", "REP",
                  "USD", "EUR", "GBP", "CAD", "JPY", "AUD", "CHF"}

_TICKER = re.compile(r"^[A-Z0-9]{1,12}$")


# ---------------------------------------------------------------------------- small helpers

def _f(x):
    """A finite float from a number or numeric string, else None (bools are not numbers)."""
    if isinstance(x, bool) or x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _hour(now):
    return time.strftime("%Y-%m-%dT%H", time.gmtime(now))


def _day(now):
    return time.strftime("%Y-%m-%d", time.gmtime(now))


def _usd(v):
    """12_300_000_000 -> "$12.3B"."""
    a = abs(v)
    for div, unit in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "k")):
        if a >= div:
            return f"${v / div:.1f}{unit}"
    return f"${v:,.0f}"


def _pct(v):
    return f"{v:+.0f}%" if abs(v) >= 1 else f"{v:+.1f}%"


def _ticker(s):
    """A clean upper-case ticker or "" (CoinGecko and DefiLlama carry some odd symbols)."""
    s = (s if isinstance(s, str) else "").strip().upper()
    return s if _TICKER.match(s) else ""


def _crypto_syms(ctx):
    return sym_assets(ctx, "crypto")


def _kraken_code(code):
    """Kraken asset code -> board base: "XXBT" -> "BTC", "XDG" -> "DOGE", "ZUSD" -> "USD"."""
    c = (code if isinstance(code, str) else "").strip().upper()
    if len(c) == 4 and c[0] in "XZ" and c[1:] in _KRAKEN_LEGACY:
        c = c[1:]
    return KRAKEN_ALIAS.get(c, c)


def _kraken_result(payload, what):
    if not isinstance(payload, dict):
        raise SourceError(f"kraken {what}: expected a JSON object")
    errs = payload.get("error")
    if isinstance(errs, list) and errs:
        raise SourceError(f"kraken {what}: {str(errs[0])[:120]}")
    res = payload.get("result")
    if not isinstance(res, dict) or not res:
        raise SourceError(f"kraken {what}: no result")
    return res


def _kraken_pairs(payload):
    """AssetPairs -> {"BTC-USD": {"pair": "XXBTZUSD", "status": "online", ...}} for USD quotes only."""
    res = _kraken_result(payload, "AssetPairs")
    out = {}
    for key, p in res.items():
        if not isinstance(p, dict) or not isinstance(key, str) or key.endswith(".d"):
            continue   # ".d" were the old dark-pool books
        ws = p.get("wsname")
        if isinstance(ws, str) and ws.count("/") == 1:
            base, quote = (_kraken_code(x) for x in ws.split("/"))
        else:
            base, quote = _kraken_code(p.get("base")), _kraken_code(p.get("quote"))
        if quote != "USD" or not _ticker(base):
            continue
        sym = f"{base}-USD"
        status = p.get("status") if isinstance(p.get("status"), str) else "online"
        out.setdefault(sym, dict(pair=key, altname=str(p.get("altname") or key)[:30], status=status.lower()))
    if not out:
        raise SourceError("kraken AssetPairs: no USD pairs in the result")
    return out


# ---------------------------------------------------------------------------- news feeds

def rss_items(ctx, url, source, publisher, kind="news", max_items=60):
    """A news feed -> Items with no symbols (the labeller maps mentions). Entries older than
    three days, from the future, or without a title or a web link are dropped."""
    entries = parse_feed(ctx.http.text(url))
    out, seen = [], set()
    for e in entries:
        title, link = (e.get("title") or "").strip(), (e.get("link") or "").strip()
        if not title or not link.lower().startswith(("http://", "https://")) or link in seen:
            continue
        ts = e.get("published_ts")
        if ts is not None:
            if ts > ctx.now + FUTURE_SKEW or ts < ctx.now - NEWS_MAX_AGE:
                continue
            ts = min(ts, ctx.now)
        seen.add(link)
        out.append(Item(source=source, url=link, title=title[:300], text=e.get("summary") or "",
                        published_ts=ts, symbols=[], kind=kind, publisher=publisher))
        if len(out) >= max_items:
            break
    return out


class _Feed(Adapter):
    team, kind = "catalysts", "news"
    assets = ("crypto",)
    interval_s = 900.0
    feed_url = ""

    def fetch(self, ctx):
        return rss_items(ctx, self.feed_url, self.name, domain(self.feed_url), kind=self.kind)


class CoinDeskRSS(_Feed):
    name = "coindesk_rss"
    feed_url = "https://www.coindesk.com/arc/outboundfeeds/rss/"
    domains = ("www.coindesk.com",)
    description = "CoinDesk headlines (RSS)."


class CointelegraphRSS(_Feed):
    name = "cointelegraph_rss"
    feed_url = "https://cointelegraph.com/rss"
    domains = ("cointelegraph.com",)
    role, standby_for = "standby", "coindesk_rss"
    description = "Cointelegraph headlines (RSS); stands in for CoinDesk."


class DecryptRSS(_Feed):
    name = "decrypt_rss"
    feed_url = "https://decrypt.co/feed"
    domains = ("decrypt.co",)
    description = "Decrypt headlines (RSS)."


class CryptoSlateRSS(_Feed):
    name = "cryptoslate_rss"
    feed_url = "https://cryptoslate.com/feed/"
    domains = ("cryptoslate.com",)
    role, standby_for = "standby", "decrypt_rss"
    description = "CryptoSlate headlines (RSS); stands in for Decrypt."


# ---------------------------------------------------------------------------- exchange status

def status_items(ctx, source, venue, readings):
    """readings: {symbol: dict(state="ok"|"halted"|"delisted", status=str, why=str, facts=dict)}.
    One Item per symbol per state change; the last state lives in ctx.state["last"].
    A product seen halted on the very first look is reported (that is news); one seen fine
    is not. A product that goes missing from the listing keeps its last known state."""
    last = ctx.state.get("last")
    last = last if isinstance(last, dict) else {}
    out = []
    hour = _hour(ctx.now)
    for sym, r in readings.items():
        state, prev = r["state"], last.get(sym)
        last[sym] = state
        if state == prev or (prev is None and state == "ok"):
            continue
        facts = dict(r.get("facts") or {}, status=r.get("status", ""), state=state, previous=prev or "")
        note = r.get("why") or ""
        if state == "ok":
            title = f"{venue}: {sym} trading has resumed"
            text = f"{venue} lists {sym} as tradable again after it was {prev}."
            cat, d, flags, conf = "outage", 1, [], 0.9
        elif state == "delisted":
            title = f"{venue} has delisted {sym}"
            text = f"{venue} reports {sym} as delisted." + (f" {note}" if note else "")
            cat, d, flags, conf = "delisting", -1, ["delisting"], 0.95
        else:
            title = f"{venue}: {sym} trading is halted" + (f" ({r.get('status')})" if r.get("status") not in ("", "online") else "")
            text = f"{venue} reports {sym} as not tradable right now." + (f" {note}" if note else "")
            cat, d, flags, conf = "outage", -1, ["halt"], 0.95
        out.append(Item(source=source, url=f"data://{source}/{sym}/{state}/{hour}", title=title, text=text[:600],
                        published_ts=ctx.now, symbols=[sym], kind="official", publisher=venue.lower() + ".com",
                        facts=facts, catalyst=cat, direction=d, risk_flags=flags, confidence=conf, official=True))
    ctx.state["last"] = last
    return out


def _coinbase_products(ctx):
    data = ctx.http.json("https://api.exchange.coinbase.com/products")
    if not isinstance(data, list) or not data:
        raise SourceError("coinbase /products: expected a non-empty list")
    prods = [p for p in data if isinstance(p, dict) and isinstance(p.get("id"), str)]
    if not prods:
        raise SourceError("coinbase /products: no products with an id")
    return prods


def _coinbase_state(p):
    status = str(p.get("status") or "").lower()
    if status == "delisted":
        return "delisted"
    if p.get("trading_disabled") is True or p.get("cancel_only") is True or status != "online":
        return "halted"
    return "ok"


class CoinbaseStatus(Adapter):
    name, team, kind = "coinbase_status", "risk", "official"
    assets = ("crypto",)
    interval_s = 600.0
    official = True
    domains = ("api.exchange.coinbase.com",)
    description = "Coinbase product status: flags halted or delisted pairs you follow, and when they resume."

    def fetch(self, ctx):
        by_id = {p["id"].upper(): p for p in _coinbase_products(ctx)}
        readings = {}
        for sym in _crypto_syms(ctx):
            p = by_id.get(sym.upper())
            if p is None:
                continue
            flags = {k: bool(p.get(k)) for k in ("trading_disabled", "cancel_only", "limit_only", "post_only")
                     if p.get(k) is not None}
            readings[sym] = dict(state=_coinbase_state(p), status=str(p.get("status") or "")[:30],
                                 why=clean(p.get("status_message") or "")[:300], facts=flags)
        return status_items(ctx, self.name, "Coinbase", readings)


# Kraken pair statuses that still let you trade (reduce_only = margin positions can only shrink)
_KRAKEN_TRADABLE = {"online", "post_only", "limit_only", "reduce_only"}


class KrakenStatus(Adapter):
    name, team, kind = "kraken_status", "risk", "official"
    assets = ("crypto",)
    interval_s = 600.0
    official = True
    domains = ("api.kraken.com",)
    role, standby_for = "standby", "coinbase_status"
    description = "Kraken pair status; stands in for Coinbase product status."

    def fetch(self, ctx):
        pairs = _kraken_pairs(ctx.http.json("https://api.kraken.com/0/public/AssetPairs"))
        readings = {}
        for sym in _crypto_syms(ctx):
            p = pairs.get(sym.upper())
            if p is None:
                continue
            st = p["status"]
            state = "delisted" if st == "delisted" else ("ok" if st in _KRAKEN_TRADABLE else "halted")
            readings[sym] = dict(state=state, status=st[:30], facts={"pair": p["pair"]})
        return status_items(ctx, self.name, "Kraken", readings)


# ---------------------------------------------------------------------------- status-page incidents

# Statuspage writes each update as "<strong>Status</strong> - message", newest first
_UPDATE = re.compile(r"\b(Resolved|Completed|Postmortem|Monitoring|Identified|Investigating|Update|"
                     r"Scheduled|In progress|Verifying)\s+-\s")
_TRADING = re.compile(r"\b(trad(?:e|es|ing)|withdraw(?:s|al|als|ing)?|sends?|sending)\b", re.I)
_RESOLVED = re.compile(r"\b(resolved|completed)\b", re.I)
_ALIASES = None


def _mentions_coin(text):
    """Does the text name any coin at all (subscribed or not)?"""
    global _ALIASES
    if _ALIASES is None:
        try:
            from ..aliases import CRYPTO_NAMES, Aliases
            _ALIASES = Aliases({f"{t}-USD": "crypto" for t in CRYPTO_NAMES})
        except Exception:   # aliases missing or broken: treat every incident as market-wide
            _ALIASES = False
    if not _ALIASES:
        return False
    try:
        return bool(_ALIASES.find(text))
    except Exception:
        return False


def incident_items(ctx, url, source, venue, max_age=2 * DAY):
    entries = parse_feed(ctx.http.text(url))
    out, seen = [], set()
    for e in entries:
        title, link, ts = (e.get("title") or "").strip(), (e.get("link") or "").strip(), e.get("published_ts")
        if not title or not link.lower().startswith(("http://", "https://")) or link in seen:
            continue
        if ts is None or ts < ctx.now - max_age or ts > ctx.now + FUTURE_SKEW:
            continue   # the history feed goes back months: an undated entry can't be shown as recent
        seen.add(link)
        summary = e.get("summary") or ""
        m = _UPDATE.search(summary)
        latest = m.group(1).lower() if m else ""
        resolved = latest in ("resolved", "completed", "postmortem") or (not latest and bool(
            _RESOLVED.search(title + " " + summary)))
        scheduled = latest == "scheduled"
        names_trading = bool(_TRADING.search(title + " " + summary))
        facts = {"status": latest or ("resolved" if resolved else "open")}
        if not _mentions_coin(title + ". " + summary):
            facts["scope"] = "exchange"
        if resolved or scheduled:
            d, flags, conf = 0, [], 0.6
        else:
            d, flags, conf = -1, (["outage"] if names_trading else []), (0.8 if names_trading else 0.5)
        out.append(Item(source=source, url=link, title=f"{venue} status: {title}"[:300], text=summary,
                        published_ts=min(ts, ctx.now), symbols=[], kind="official", publisher=domain(url),
                        facts=facts, catalyst="outage", direction=d, risk_flags=flags, confidence=conf,
                        official=True))
    return out


class CoinbaseIncidents(Adapter):
    name, team, kind = "coinbase_incidents", "risk", "official"
    assets = ("crypto",)
    interval_s = 900.0
    official = True
    domains = ("status.coinbase.com",)
    description = "Coinbase status-page incidents from the last two days (delayed sends, trading issues)."

    def fetch(self, ctx):
        return incident_items(ctx, "https://status.coinbase.com/history.rss", self.name, "Coinbase")


class KrakenIncidents(Adapter):
    name, team, kind = "kraken_incidents", "risk", "official"
    assets = ("crypto",)
    interval_s = 900.0
    official = True
    domains = ("status.kraken.com",)
    role, standby_for = "standby", "coinbase_incidents"
    description = "Kraken status-page incidents; stands in for Coinbase's."

    def fetch(self, ctx):
        return incident_items(ctx, "https://status.kraken.com/history.rss", self.name, "Kraken")


# ---------------------------------------------------------------------------- DeFi TVL

MIN_TVL = 50e6
MAX_SANE_CHANGE = 300.0     # a +900% week is a data glitch on DefiLlama's side, not a signal


def tvl_item(ctx, source, sym, name, tvl, ch1, ch7, label="DefiLlama", up=10.0, down=-15.0):
    """The shared TVL Item. Direction from the 7-day change (else the 1-day one with half
    the thresholds); confidence 0.5..0.8 with the size of the move."""
    ch, window = (ch7, "on the week") if ch7 is not None else (ch1, "on the day")
    lo, hi = (down, up) if ch7 is not None else (down / 2, up / 2)
    d, conf = 0, 0.5
    if ch is not None and abs(ch) <= MAX_SANE_CHANGE:
        d = 1 if ch >= hi else (-1 if ch <= lo else 0)
        conf = round(0.5 + 0.3 * min(1.0, abs(ch) / (2 * hi)), 3)
    move = f", {_pct(ch)} {window}" if ch is not None else ""
    facts = {"tvl": round(tvl, 2), "change_1d": None if ch1 is None else round(ch1, 3),
             "change_7d": None if ch7 is None else round(ch7, 3), "name": name}
    return Item(source=source, url=f"data://defillama/{sym}/{_day(ctx.now)}",   # shared by primary and standby: no double count
                title=f"{name} TVL {_usd(tvl)}{move} ({label})",
                text=f"Total value locked in {name}: {_usd(tvl)}.", published_ts=ctx.now, symbols=[sym],
                kind="market", publisher="defillama.com", facts=facts, catalyst="other", direction=d,
                confidence=conf)


class DefiLlamaTVL(Adapter):
    name, team, kind = "defillama_tvl", "filings", "market"
    assets = ("crypto",)
    interval_s = 3600.0
    domains = ("api.llama.fi",)
    description = "DefiLlama protocol TVL for the coins you follow: big weekly inflows or outflows."

    def fetch(self, ctx):
        data = ctx.http.json("https://api.llama.fi/protocols")
        if not isinstance(data, list) or not data:
            raise SourceError("defillama /protocols: expected a non-empty list")
        want = {crypto_base(s).upper(): s for s in _crypto_syms(ctx)}
        best = {}
        for p in data:
            if not isinstance(p, dict) or str(p.get("category") or "").upper() == "CEX":
                continue   # exchange reserves are not a protocol's TVL
            base, tvl = _ticker(p.get("symbol")), _f(p.get("tvl"))
            if base not in want or tvl is None or tvl < MIN_TVL:
                continue
            if base not in best or tvl > best[base][1]:
                best[base] = (p, tvl)
        out = []
        for base, (p, tvl) in best.items():
            name = clean(p.get("name") or "")[:60] or base
            out.append(tvl_item(ctx, self.name, want[base], name, tvl, _f(p.get("change_1d")), _f(p.get("change_7d"))))
        return out


class DefiLlamaChains(Adapter):
    name, team, kind = "defillama_chains", "filings", "market"
    assets = ("crypto",)
    interval_s = 3600.0
    domains = ("api.llama.fi",)
    role, standby_for = "standby", "defillama_tvl"
    description = "DefiLlama chain TVL (day-over-day from its own history); stands in for protocol TVL."

    def fetch(self, ctx):
        data = ctx.http.json("https://api.llama.fi/v2/chains")
        if not isinstance(data, list) or not data:
            raise SourceError("defillama /v2/chains: expected a non-empty list")
        want = {crypto_base(s).upper(): s for s in _crypto_syms(ctx)}
        best = {}
        for c in data:
            if not isinstance(c, dict):
                continue
            base, tvl = _ticker(c.get("tokenSymbol")), _f(c.get("tvl"))
            if base in want and tvl is not None and tvl >= MIN_TVL and (base not in best or tvl > best[base][1]):
                best[base] = (c, tvl)
        hist = ctx.state.get("days")
        hist = hist if isinstance(hist, dict) else {}
        today = _day(ctx.now)
        ago = {k: _day(ctx.now - k * DAY) for k in (1, 7)}
        out = []
        for base, (c, tvl) in best.items():
            name = clean(c.get("name") or "")[:60] or base
            h = hist.get(name) if isinstance(hist.get(name), dict) else {}
            ch = {}
            for k, day in ago.items():
                old = _f(h.get(day))
                ch[k] = (tvl / old - 1) * 100 if old and old > 0 else None
            h[today] = round(tvl, 2)
            hist[name] = {d: v for d, v in sorted(h.items())[-9:]}
            out.append(tvl_item(ctx, self.name, want[base], name, tvl, ch[1], ch[7]))
        ctx.state["days"] = hist   # only chains that matched a followed coin ever get an entry
        return out


# ---------------------------------------------------------------------------- funding rates

FUND_HIST = 48            # trailing readings kept per coin (24 h at the 30-min interval)
FUND_MIN_HIST = 12
FUND_STD_FLOOR = 5e-5     # 0.005% per 8 h: below this a "spread" is just rounding
FUND_Z = 2.0
FUND_REPEAT_S = 6 * 3600  # a persisting extreme is re-reported at most this often


def _robust(xs):
    xs = sorted(xs)
    n = len(xs)
    med = xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2
    mad = sorted(abs(x - med) for x in xs)
    m = mad[n // 2] if n % 2 else (mad[n // 2 - 1] + mad[n // 2]) / 2
    return med, max(1.4826 * m, FUND_STD_FLOOR)


def funding_items(ctx, source, venue, rates, publisher, min_cross=8):
    """rates: {"SOL-USD": dict(funding=<relative rate per 8 h>, oi=..., instrument=...)} for EVERY perp
    the venue lists (the cross-section). Items only for subscribed coins at an extreme: crowded
    longs (z >= 2 against both the market and the coin's own trailing readings) -> -1, deeply
    negative -> +1. A persisting extreme is re-reported at most every 6 h."""
    hist = ctx.state.get("hist") if isinstance(ctx.state.get("hist"), dict) else {}
    ext = ctx.state.get("ext") if isinstance(ctx.state.get("ext"), dict) else {}
    cross = [r["funding"] for r in rates.values()]
    med, sd = _robust(cross) if len(cross) >= min_cross else (None, None)
    out = []
    for sym in _crypto_syms(ctx):
        r = rates.get(sym)
        if r is None:
            continue
        x = r["funding"]
        past = [v for v in (hist.get(sym) or []) if _f(v) is not None]
        hist[sym] = (past + [x])[-FUND_HIST:]
        zs = []
        if med is not None:
            zs.append((x - med) / sd)
        if len(past) >= FUND_MIN_HIST:
            mu = sum(past) / len(past)
            own_sd = max(math.sqrt(sum((v - mu) ** 2 for v in past) / len(past)), FUND_STD_FLOOR)
            zs.append((x - mu) / own_sd)
        if not zs:
            continue
        # both views must agree: take the one closer to zero
        z = min(zs) if all(v > 0 for v in zs) else (max(zs) if all(v < 0 for v in zs) else 0.0)
        sign = -1 if (z >= FUND_Z and x > 0) else (1 if (z <= -FUND_Z and x < 0) else 0)
        prev = ext.get(sym) if isinstance(ext.get(sym), dict) else {}
        if sign == 0:
            ext.pop(sym, None)
            continue
        if prev.get("sign") == sign and ctx.now - (_f(prev.get("ts")) or 0) < FUND_REPEAT_S:
            continue
        ext[sym] = {"sign": sign, "ts": ctx.now}
        base = crypto_base(sym)
        lvl = f"{x * 100:+.3f}% per 8h"
        if sign < 0:
            title = f"{base} perpetual funding at {lvl}, far above normal: longs crowded ({venue})"
            text = (f"Traders are paying a lot to stay long {base}. Crowded longs often unwind, "
                    f"so this reads as a contrarian warning.")
        else:
            title = f"{base} perpetual funding at {lvl}, deeply negative: shorts crowded ({venue})"
            text = f"Traders are paying to stay short {base}. Crowded shorts can be squeezed."
        out.append(Item(source=source, url=f"data://{source}/{sym}/{_hour(ctx.now)}", title=title, text=text,
                        published_ts=ctx.now, symbols=[sym], kind="market", publisher=publisher,
                        facts={"funding": round(x, 8), "z": round(z, 2), "oi": r.get("oi"),
                               "instrument": r.get("instrument", ""), "cross_median": None if med is None else round(med, 8)},
                        catalyst="market_structure", direction=sign, confidence=0.55))
    ctx.state["hist"] = {k: v for k, v in hist.items() if k in rates or k in ctx.symbols}
    ctx.state["ext"] = ext
    return out


class KrakenFuturesFunding(Adapter):
    name, team, kind = "kraken_funding", "market", "market"
    assets = ("crypto",)
    interval_s = 1800.0
    domains = ("futures.kraken.com",)
    description = "Kraken Futures perpetual funding: flags crowded longs or shorts in the coins you follow."

    def fetch(self, ctx):
        data = ctx.http.json("https://futures.kraken.com/derivatives/api/v3/tickers")
        tickers = data.get("tickers") if isinstance(data, dict) else None
        if not isinstance(tickers, list) or not tickers:
            raise SourceError("kraken futures tickers: no 'tickers' list")
        rates = {}
        for t in tickers:
            if not isinstance(t, dict) or t.get("suspended") is True:
                continue
            sym = t.get("symbol") if isinstance(t.get("symbol"), str) else ""
            if t.get("tag") not in (None, "perpetual") or not sym.upper().startswith("PF_") or not sym.upper().endswith("USD"):
                continue
            base = KRAKEN_ALIAS.get(sym[3:-3].upper(), sym[3:-3].upper())
            fr, px = _f(t.get("fundingRate")), _f(t.get("markPrice"))
            if not _ticker(base) or fr is None or not px or px <= 0:
                continue
            # Kraken quotes an ABSOLUTE hourly rate (USD per contract); relative per 8 h is comparable across coins
            rates.setdefault(f"{base}-USD", dict(funding=fr / px * 8, oi=_f(t.get("openInterest")), instrument=sym))
        if not rates:
            raise SourceError("kraken futures tickers: no usable perpetuals")
        return funding_items(ctx, self.name, "Kraken Futures", rates, "futures.kraken.com")


class DeribitFunding(Adapter):
    name, team, kind = "deribit_funding", "market", "market"
    assets = ("crypto",)
    interval_s = 1800.0
    domains = ("www.deribit.com",)
    role, standby_for = "standby", "kraken_funding"
    description = "Deribit perpetual funding; stands in for Kraken Futures."
    URL = "https://www.deribit.com/api/v2/public/get_book_summary_by_currency?currency={}&kind=future"

    def fetch(self, ctx):
        rates, errors, ok = {}, [], 0
        for cur in ("BTC", "ETH", "USDC"):   # inverse BTC/ETH first: they win over the USDC-margined twins
            try:
                data = ctx.http.json(self.URL.format(cur))
            except RateLimited:
                raise
            except SourceError as e:
                errors.append(str(e))
                continue
            res = data.get("result") if isinstance(data, dict) else None
            if not isinstance(res, list):
                errors.append(f"deribit {cur}: no result list")
                continue
            ok += 1
            for r in res:
                name = r.get("instrument_name") if isinstance(r, dict) else None
                if not isinstance(name, str) or not name.endswith("-PERPETUAL"):
                    continue
                base = name[: -len("-PERPETUAL")].split("_")[0].upper()
                f8 = _f(r.get("funding_8h"))
                if f8 is None:
                    f8 = _f(r.get("current_funding"))
                if not _ticker(base) or f8 is None:
                    continue
                rates.setdefault(f"{base}-USD", dict(funding=f8, oi=_f(r.get("open_interest")), instrument=name))
        if not ok:
            raise SourceError(errors[0] if errors else "deribit: no data")
        if not rates:
            raise SourceError("deribit: no perpetuals in the book summaries")
        return funding_items(ctx, self.name, "Deribit", rates, "deribit.com", min_cross=4)


# ---------------------------------------------------------------------------- order books

BOOK_BAND = 0.01
BOOK_MIN_USD = 20_000.0
BOOK_IMB = 0.35


def book_item(ctx, source, venue, sym, bids, asks, publisher):
    """bids/asks: [[price, size, ...], ...] (strings or numbers). An Item when depth within 1% of
    mid leans hard to one side (|imbalance| >= 0.35, both sides >= $20k), else None."""
    def levels(rows):
        out = []
        for row in rows if isinstance(rows, list) else ():
            if isinstance(row, (list, tuple)) and len(row) >= 2:
                px, sz = _f(row[0]), _f(row[1])
                if px and sz and px > 0 and sz > 0:
                    out.append((px, sz))
        return out
    b, a = levels(bids), levels(asks)
    if not b or not a:
        return None
    best_bid, best_ask = max(p for p, _ in b), min(p for p, _ in a)
    if best_bid >= best_ask:
        return None   # crossed or locked book: a bad snapshot
    mid = (best_bid + best_ask) / 2
    bid_usd = sum(p * s for p, s in b if p >= mid * (1 - BOOK_BAND))
    ask_usd = sum(p * s for p, s in a if p <= mid * (1 + BOOK_BAND))
    if bid_usd < BOOK_MIN_USD or ask_usd < BOOK_MIN_USD:
        return None
    imb = (bid_usd - ask_usd) / (bid_usd + ask_usd)
    if abs(imb) < BOOK_IMB:
        return None
    side = "buyers" if imb > 0 else "sellers"
    title = (f"{sym} order book leans to {side}: {_usd(bid_usd)} bids vs {_usd(ask_usd)} asks within 1% ({venue})")
    return Item(source=source, url=f"data://{source}/{sym}/{_hour(ctx.now)}", title=title,
                text=f"Resting depth within 1% of the {mid:,.6g} mid is {abs(imb) * 100:.0f}% one-sided.",
                published_ts=ctx.now, symbols=[sym], kind="market", publisher=publisher,
                facts={"bid_usd": round(bid_usd, 2), "ask_usd": round(ask_usd, 2), "imbalance": round(imb, 4),
                       "mid": mid}, catalyst="market_structure", direction=1 if imb > 0 else -1, confidence=0.5)


class _Book(Adapter):
    team, kind = "market", "market"
    assets = ("crypto",)
    interval_s = 1800.0
    per_symbol = True
    max_symbols_per_run = 6

    def book(self, ctx, sym):
        raise NotImplementedError

    def fetch(self, ctx):
        out, errors, ok = [], [], 0
        for sym in _crypto_syms(ctx)[: self.max_symbols_per_run]:
            try:
                it = self.book(ctx, sym)
            except RateLimited:
                raise
            except SourceError as e:   # one unknown pair must not cost the others
                errors.append(f"{sym}: {e}")
                continue
            ok += 1
            if it is not None:
                out.append(it)
        if errors and not ok:
            raise SourceError(errors[0][:200])
        for e in errors:
            ctx.log(e)
        return out


class CoinbaseBook(_Book):
    name = "coinbase_book"
    domains = ("api.exchange.coinbase.com",)
    description = "Coinbase order-book depth within 1% of mid: flags books that lean hard to one side."

    def book(self, ctx, sym):
        data = ctx.http.json(f"https://api.exchange.coinbase.com/products/{sym}/book?level=2")
        if not isinstance(data, dict) or not isinstance(data.get("bids"), list) or not isinstance(data.get("asks"), list):
            raise SourceError("coinbase book: no bids/asks")
        return book_item(ctx, self.name, "Coinbase", sym, data["bids"], data["asks"], "coinbase.com")


class KrakenBook(_Book):
    name = "kraken_book"
    domains = ("api.kraken.com",)
    role, standby_for = "standby", "coinbase_book"
    description = "Kraken order-book depth within 1% of mid; stands in for Coinbase's book."

    def book(self, ctx, sym):
        base = crypto_base(sym).upper()
        pair = KRAKEN_FROM.get(base, base) + "USD"
        res = _kraken_result(ctx.http.json(f"https://api.kraken.com/0/public/Depth?pair={pair}&count=200"), "Depth")
        bk = next((v for v in res.values() if isinstance(v, dict)), None)
        if bk is None or not isinstance(bk.get("bids"), list) or not isinstance(bk.get("asks"), list):
            raise SourceError("kraken Depth: no bids/asks")
        return book_item(ctx, self.name, "Kraken", sym, bk["bids"], bk["asks"], "kraken.com")


# ---------------------------------------------------------------------------- sentiment

class FearGreedCrypto(Adapter):
    name, team, kind = "fear_greed", "sentiment", "sentiment"
    assets = ("crypto",)
    interval_s = 3600.0
    domains = ("api.alternative.me",)
    description = "The crypto Fear & Greed Index, read contrarian: extreme fear leans bullish, extreme greed bearish."

    def fetch(self, ctx):
        data = ctx.http.json("https://api.alternative.me/fng/?limit=2")
        rows = data.get("data") if isinstance(data, dict) else None
        if not isinstance(rows, list) or not rows:
            err = (data.get("metadata") or {}).get("error") if isinstance(data, dict) and isinstance(data.get("metadata"), dict) else None
            raise SourceError(f"fear & greed: {err or 'no data'}"[:150])
        cur = rows[0] if isinstance(rows[0], dict) else {}
        v, ts = _f(cur.get("value")), _f(cur.get("timestamp"))
        if v is None or not 0 <= v <= 100 or ts is None:
            raise SourceError("fear & greed: unreadable latest reading")
        if ts > ctx.now + FUTURE_SKEW or ts < ctx.now - 2 * DAY:
            return []   # stale (or nonsense) reading: say nothing rather than repeat old news
        ts = min(ts, ctx.now)
        label = clean(cur.get("value_classification") or "")[:30]
        prev = _f(rows[1].get("value")) if len(rows) > 1 and isinstance(rows[1], dict) else None
        d = 1 if v <= 20 else (-1 if v >= 80 else 0)
        title = f"Crypto Fear & Greed Index at {v:.0f}" + (f" ({label})" if label else "")
        if prev is not None:
            title += f", {'up' if v > prev else 'down' if v < prev else 'unchanged'} from {prev:.0f} the day before"
        syms = ["BTC-USD"] + [s for s in _crypto_syms(ctx) if s != "BTC-USD"]
        return [Item(source=self.name, url=f"data://fear_greed/{_day(ts)}", title=title,
                     text="A market-wide mood reading (0 = extreme fear, 100 = extreme greed), read contrarian.",
                     published_ts=ts, symbols=syms, kind="sentiment", publisher="alternative.me",
                     facts={"value": v, "classification": label, "previous": prev, "scope": "market"},
                     catalyst="sentiment", direction=d, confidence=0.45)]


# ---------------------------------------------------------------------------- discovery

MAX_NEW_LISTINGS = 25   # more "new" pairs than this in one run means the listing glitched: reseed instead


def listing_items(ctx, source, venue, current, official=True):
    """current: {product_id: base} of the venue's USD/USDC pairs now. The first run only seeds
    ctx.state["known"]; later a base with no USD/USDC pair before -> one 'listing' Item. Known ids
    are a running union, so a pair that drops out of one response is not "new" next time."""
    known = ctx.state.get("known")
    seeded = isinstance(known, list)
    known = set(known) if seeded else set()
    new = {pid: base for pid, base in current.items() if pid not in known}
    ctx.state["known"] = sorted(known | set(current))
    if not seeded:
        ctx.log(f"seeded {len(current)} {venue} pairs")
        return []
    old_bases = {pid.split("-")[0] for pid in known}
    fresh = {}
    for pid, base in sorted(new.items()):
        if base not in old_bases:
            fresh.setdefault(base, pid)
    if len(fresh) > MAX_NEW_LISTINGS:
        ctx.log(f"{len(fresh)} new {venue} pairs at once: treated as a listing glitch, reseeded")
        return []
    out = []
    for base, pid in fresh.items():
        out.append(Item(source=source, url=f"data://{source}/{pid}", title=f"{venue} lists {base}-USD",
                        text=f"A new {venue} trading pair appeared: {pid}.", published_ts=ctx.now,
                        symbols=[f"{base}-USD"], kind="discovery", publisher=venue.lower() + ".com",
                        facts={"product_id": pid}, catalyst="listing", direction=1, confidence=0.8,
                        official=official))
    return out


class CoinbaseListings(Adapter):
    name, team, kind = "coinbase_listings", "discovery", "discovery"
    assets = ("crypto",)
    interval_s = 3600.0
    official = True
    domains = ("api.exchange.coinbase.com",)
    description = "New Coinbase USD/USDC pairs (including coins the board does not follow yet)."

    def fetch(self, ctx):
        cur = {}
        for p in _coinbase_products(ctx):
            quote, base = str(p.get("quote_currency") or "").upper(), _ticker(p.get("base_currency"))
            if quote in ("USD", "USDC") and base and str(p.get("status") or "").lower() != "delisted":
                cur[p["id"].upper()] = base
        if not cur:
            raise SourceError("coinbase /products: no USD or USDC pairs")
        return listing_items(ctx, self.name, "Coinbase", cur)


class KrakenListings(Adapter):
    name, team, kind = "kraken_listings", "discovery", "discovery"
    assets = ("crypto",)
    interval_s = 3600.0
    official = True
    domains = ("api.kraken.com",)
    role, standby_for = "standby", "coinbase_listings"
    description = "New Kraken USD pairs; stands in for Coinbase listings."

    def fetch(self, ctx):
        pairs = _kraken_pairs(ctx.http.json("https://api.kraken.com/0/public/AssetPairs"))
        cur = {sym: sym.split("-")[0] for sym, p in pairs.items() if p["status"] != "delisted"}
        return listing_items(ctx, self.name, "Kraken", cur)


class CoinGeckoTrending(Adapter):
    name, team, kind = "coingecko_trending", "discovery", "discovery"
    assets = ("crypto",)
    interval_s = 3600.0
    domains = ("api.coingecko.com",)
    description = "Coins trending in CoinGecko searches (top 500 by market cap only)."

    def fetch(self, ctx):
        data = ctx.http.json("https://api.coingecko.com/api/v3/search/trending")
        coins = data.get("coins") if isinstance(data, dict) else None
        if not isinstance(coins, list):
            raise SourceError("coingecko trending: no 'coins' list")
        out, seen = [], set()
        for i, c in enumerate(coins):
            it = c.get("item") if isinstance(c, dict) else None
            if not isinstance(it, dict):
                continue
            base, cid = _ticker(it.get("symbol")), str(it.get("id") or "").strip()[:80]
            mcr = _f(it.get("market_cap_rank"))
            if not base or not cid or base in seen or mcr is None or not 0 < mcr <= 500:
                continue
            seen.add(base)
            score = _f(it.get("score"))
            rank = int(score) + 1 if score is not None and score >= 0 else i + 1
            pc = (it.get("data") or {}).get("price_change_percentage_24h") if isinstance(it.get("data"), dict) else None
            ch = _f(pc.get("usd")) if isinstance(pc, dict) else None
            name = clean(it.get("name") or "")[:60] or base
            title = f"{name} ({base}) is trending on CoinGecko (#{rank} in searches" + (
                f", {_pct(ch)} in 24h)" if ch is not None else ")")
            out.append(Item(source=self.name, url=f"data://coingecko_trending/{cid}/{_day(ctx.now)}", title=title,
                            text=f"{name} is among the most-searched coins on CoinGecko; market-cap rank {int(mcr)}.",
                            published_ts=ctx.now, symbols=[f"{base}-USD"], kind="discovery", publisher="coingecko.com",
                            facts={"rank": rank, "market_cap_rank": int(mcr), "change_24h": ch, "coingecko_id": cid},
                            catalyst="sentiment", direction=0, confidence=0.4))
        return out


ADAPTERS = [
    CoinDeskRSS, CointelegraphRSS, DecryptRSS, CryptoSlateRSS,
    CoinbaseStatus, KrakenStatus, CoinbaseIncidents, KrakenIncidents,
    DefiLlamaTVL, DefiLlamaChains,
    KrakenFuturesFunding, DeribitFunding, CoinbaseBook, KrakenBook,
    FearGreedCrypto,
    CoinbaseListings, KrakenListings, CoinGeckoTrending,
]
