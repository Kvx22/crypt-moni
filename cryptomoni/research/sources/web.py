"""Search-like and social sources: news searches, regulator press feeds, crowd chatter.

The specialised adapters (exchange status, filings, funding rates) see a few things very
clearly. These see a lot of things roughly: a news search per coin or stock, the SEC's own
press releases, what StockTwits and Reddit are saying, the CNN fear-and-greed reading.
They mostly return plain text items (headlines, posts) and leave the labelling to the
extractor; the two aggregate readings (StockTwits mood, CNN fear and greed) arrive
pre-labelled.

Every adapter is defensive in the same way: an entry with a missing field or an odd type is
skipped, a response that is not the documented shape at all raises SourceError (so the
worker can switch to the standby), and a per-symbol adapter keeps going when one symbol's
request fails, failing the run only when every request failed.

Search pairs (primary -> standby): google_news -> bing_news, google_risk -> bing_risk,
sec_press -> sec_litigation, stocktwits -> reddit_stocks, reddit_crypto -> hn_search,
fear_greed (crypto module) -> cnn_fear_greed, coingecko_trending (crypto module) ->
google_listings. cnbc_rss, marketwatch_rss, theblock_rss and searxng are dormant reserve.
"""
from __future__ import annotations

import re
import time
import urllib.parse

from ..aliases import CRYPTO_NAMES, STOCK_NAMES
from ..http import RateLimited, SourceError
from ..text import clean, domain, parse_feed, parse_ts
from ..types import Item
from .base import Adapter, crypto_base

DAY = 86400.0
BROWSER_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
              "Chrome/128.0.0.0 Safari/537.36")
TEXT_MAX = 1500
FUTURE_SLACK_S = 3600.0          # clocks disagree a little; an item "from the future" beyond this is junk


# ---------------------------------------------------------------------------- small helpers

def _hour(ts):
    return time.strftime("%Y-%m-%dT%H", time.gmtime(ts))


def _day(ts):
    return time.strftime("%Y-%m-%d", time.gmtime(ts))


def _num(x):
    """A finite float from a number or numeric string, else None."""
    if isinstance(x, bool):
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if v == v and abs(v) != float("inf") else None


def _fresh(ts, now, max_age_s):
    """Unknown dates pass (many feeds omit them); known ones must be recent and not far in the future."""
    return ts is None or (now - max_age_s <= ts <= now + FUTURE_SLACK_S)


def _ordered(ctx):
    """The context's symbols, the ones advisors asked about first."""
    syms = ctx.symbols if isinstance(ctx.symbols, dict) else {}
    req = [s for s in (ctx.requested or []) if s in syms]
    return req + [s for s in syms if s not in req]


def display_name(ctx, symbol, asset):
    """Best search name for a symbol: the aliases' first name, else a built-in table, else the ticker."""
    names = (ctx.names or {}).get(symbol) if isinstance(ctx.names, dict) else None
    if isinstance(names, (list, tuple)):
        for n in names:
            if isinstance(n, str) and n.strip():
                return n.strip()
    if asset == "crypto":
        base = crypto_base(symbol)
        return (CRYPTO_NAMES.get(base) or [base])[0]
    return (STOCK_NAMES.get(symbol) or [symbol])[0]


def news_query(ctx, symbol, asset):
    """The catalyst search for one symbol (Google News syntax; Bing reads it the same way)."""
    name = display_name(ctx, symbol, asset)
    if asset == "crypto":
        return f'"{name}" (crypto OR token OR coin)'
    return f'"{name}" OR "{symbol} stock"'


RISK_WORDS = "(hack OR exploit OR lawsuit OR SEC OR delist OR halt OR fraud OR bankruptcy)"


def risk_query(ctx, symbol, asset):
    return f'"{display_name(ctx, symbol, asset)}" {RISK_WORDS}'


LISTING_QUERY = '(Coinbase OR Kraken OR Robinhood) (lists OR listing OR "adds trading")'


def google_news_url(q, when="2d"):
    return ("https://news.google.com/rss/search?q=" + urllib.parse.quote_plus(q) + f"+when:{when}"
            + "&hl=en-US&gl=US&ceid=US:en")


def bing_news_url(q):
    return "https://www.bing.com/news/search?q=" + urllib.parse.quote_plus(q) + "&format=rss"


# the outlet is whatever follows the LAST " - " (or en/em dash, or " | "); it may itself hold a
# hyphen without spaces ("Business-Standard")
_SUFFIX = re.compile(r"^(?P<head>.{10,})\s[-–—|]\s+(?P<pub>(?:(?!\s[-–—|]\s).){1,60})$")


def split_publisher(title, known=""):
    """'Solana jumps 8% - CoinDesk' -> ('Solana jumps 8%', 'CoinDesk'). Google appends the outlet to
    every headline; when the outlet is known from <source> only that exact suffix is removed."""
    title = clean(title)
    known = clean(known)
    if known:
        for sep in (" - ", " – ", " — ", " | "):
            if title.endswith(sep + known) and len(title) - len(sep + known) >= 10:
                return title[: -len(sep + known)].rstrip(), known
    m = _SUFFIX.match(title)
    if m and len(m.group("pub").split()) <= 6:
        return m.group("head").rstrip(), m.group("pub").strip()
    return title, ""


def bing_target(link):
    """Bing News RSS links go through apiclick.aspx?...&url=<quoted article URL>: return the article URL."""
    try:
        p = urllib.parse.urlsplit(link)
        q = urllib.parse.parse_qs(p.query)
    except (TypeError, ValueError):
        return link
    if "bing.com" in (p.hostname or ""):
        for key in ("url", "u"):
            for v in q.get(key, []):
                v = v.strip()
                if v.lower().startswith(("http://", "https://")):
                    return v
    return link


def _entries(ctx, url, headers=None):
    xml = ctx.http.text(url, headers=headers) if headers else ctx.http.text(url)
    return parse_feed(xml)


def _feed_items(entries, source, kind, now, max_age_s, symbols=(), seen=None, engine="", publisher="",
                official=False, max_items=60):
    """Feed entries -> Items. engine "google" strips the outlet suffix and takes the publisher from
    <source>; "bing" unwraps the redirect link; anything else is a plain site feed."""
    seen = set() if seen is None else seen
    out = []
    for e in entries:
        if len(out) >= max_items:
            break
        try:
            link, title = e.get("link") or "", e.get("title") or ""
            ts = e.get("published_ts")
            if not _fresh(ts, now, max_age_s):
                continue
            pub = publisher
            if engine == "google":
                src_dom = domain(e.get("source_url"))
                named = e.get("source_name") if src_dom and not src_dom.endswith("google.com") else ""
                title, suffix = split_publisher(title, named)
                pub = src_dom if src_dom and not src_dom.endswith("google.com") else suffix
            elif engine == "bing":
                link = bing_target(link)
                title = clean(title)
                pub = domain(link)
            else:
                title = clean(title)
                pub = pub or domain(link)
            if not link or not title or link in seen:
                continue
            seen.add(link)
            out.append(Item(source=source, url=link, title=title[:500], text=(e.get("summary") or "")[:TEXT_MAX],
                            published_ts=ts, symbols=list(symbols), kind=kind, publisher=(pub or "")[:120],
                            official=official))
        except (AttributeError, TypeError):
            continue
    return out


def _per_symbol(ctx, adapter, one):
    """Run one(symbol, asset, seen) -> items over this run's symbols. A failing symbol is skipped; the
    run fails only when every request failed. A rate limit stops the run at once (the domain is paused)."""
    syms = [s for s in _ordered(ctx) if adapter.covers(ctx.symbols.get(s))][: max(1, int(adapter.max_symbols_per_run))]
    out, seen, errors = [], set(), []
    for sym in syms:
        try:
            out += one(sym, ctx.symbols[sym], seen)
        except RateLimited:
            if out:
                ctx.log(f"rate limited after {len(out)} items; keeping them")
                return out
            raise
        except SourceError as e:
            errors.append(e)
            ctx.log(f"{sym}: {e}")
    if syms and len(errors) == len(syms):
        raise errors[-1]
    return out


# ---------------------------------------------------------------------------- news searches

class GoogleNews(Adapter):
    name, team, kind = "google_news", "catalysts", "news"
    assets = ("crypto", "stock")
    per_symbol, max_symbols_per_run, interval_s = True, 6, 1800
    domains = ("news.google.com",)
    role = "primary"
    description = "Google News search per coin and stock (last 2 days)."
    when, max_age_s, engine = "2d", 2.5 * DAY, "google"

    def query(self, ctx, symbol, asset):
        return news_query(ctx, symbol, asset)

    def url(self, q):
        return google_news_url(q, self.when)

    def fetch(self, ctx):
        def one(sym, asset, seen):
            entries = _entries(ctx, self.url(self.query(ctx, sym, asset)))
            return _feed_items(entries, self.name, self.kind, ctx.now, self.max_age_s, [sym], seen,
                               engine=self.engine, max_items=20)
        return _per_symbol(ctx, self, one)


class BingNews(GoogleNews):
    name = "bing_news"
    domains = ("www.bing.com",)
    role, standby_for = "standby", "google_news"
    description = "Bing News search per coin and stock: stands in for Google News."
    engine = "bing"

    def url(self, q):
        return bing_news_url(q)


class GoogleRisk(GoogleNews):
    name, team = "google_risk", "risk"
    interval_s = 2700
    description = "Google News search for hacks, lawsuits, halts, fraud and delistings per symbol (last 3 days)."
    when, max_age_s = "3d", 3.5 * DAY

    def query(self, ctx, symbol, asset):
        return risk_query(ctx, symbol, asset)


class BingRisk(GoogleRisk):
    name = "bing_risk"
    domains = ("www.bing.com",)
    role, standby_for = "standby", "google_risk"
    description = "Bing News risk search per symbol: stands in for the Google risk search."
    engine = "bing"

    def url(self, q):
        return bing_news_url(q)


class GoogleListingNews(Adapter):
    name, team, kind = "google_listings", "discovery", "discovery"
    assets = ("crypto",)
    interval_s = 3600
    domains = ("news.google.com",)
    role, standby_for = "standby", "coingecko_trending"
    description = "Google News search for new exchange listings (Coinbase, Kraken, Robinhood)."

    def fetch(self, ctx):
        entries = _entries(ctx, google_news_url(LISTING_QUERY, "2d"))
        return _feed_items(entries, self.name, self.kind, ctx.now, 2.5 * DAY, (), engine="google", max_items=40)


# ---------------------------------------------------------------------------- regulator feeds

class SECPress(Adapter):
    name, team, kind = "sec_press", "risk", "official"
    assets = ("crypto", "stock")
    interval_s = 3600
    domains = ("www.sec.gov",)
    official = True
    role = "primary"
    description = "SEC press releases (enforcement actions, charges, approvals), last 7 days."
    feed_url = "https://www.sec.gov/news/pressreleases.rss"

    def fetch(self, ctx):
        entries = _entries(ctx, self.feed_url)
        return _feed_items(entries, self.name, self.kind, ctx.now, 7 * DAY, (), publisher="sec.gov",
                           official=True, max_items=60)


class SECLitigation(SECPress):
    name = "sec_litigation"
    role, standby_for = "standby", "sec_press"
    description = "SEC litigation releases, last 7 days: stands in for SEC press releases."
    feed_url = "https://www.sec.gov/rss/litigation/litreleases.xml"


# ---------------------------------------------------------------------------- plain site feeds (dormant)

class _SiteRSS(Adapter):
    team, kind = "catalysts", "news"
    interval_s = 1800
    role = "dormant"
    feed_url = ""

    def fetch(self, ctx):
        entries = _entries(ctx, self.feed_url)
        return _feed_items(entries, self.name, self.kind, ctx.now, 3 * DAY, (), publisher=domain(self.feed_url))


class CNBCMarkets(_SiteRSS):
    name, assets, domains = "cnbc_rss", ("stock",), ("www.cnbc.com",)
    feed_url = "https://www.cnbc.com/id/100003114/device/rss/rss.html"
    description = "CNBC top market news (reserve)."


class MarketWatchRSS(_SiteRSS):
    name, assets, domains = "marketwatch_rss", ("stock",), ("feeds.content.dowjones.io",)
    feed_url = "https://feeds.content.dowjones.io/public/rss/mw_topstories"
    description = "MarketWatch top stories (reserve)."

    def fetch(self, ctx):
        entries = _entries(ctx, self.feed_url)
        return _feed_items(entries, self.name, self.kind, ctx.now, 3 * DAY, (), publisher="marketwatch.com")


class TheBlockRSS(_SiteRSS):
    name, assets, domains = "theblock_rss", ("crypto",), ("www.theblock.co",)
    feed_url = "https://www.theblock.co/rss.xml"
    description = "The Block crypto news (reserve)."


# ---------------------------------------------------------------------------- crowd sentiment

def stocktwits_code(symbol, asset):
    """StockTwits stream id: stocks by ticker, coins as BASE.X ("SOL-USD" -> "SOL.X")."""
    return f"{crypto_base(symbol)}.X" if asset == "crypto" else symbol.upper()


def stocktwits_mood(payload, now, window_s=6 * 3600):
    """{messages:[...]} -> (bulls, bears) among tagged messages inside the window. Raises SourceError
    when the payload has no message list at all."""
    msgs = payload.get("messages") if isinstance(payload, dict) else None
    if not isinstance(msgs, list):
        raise SourceError("stocktwits: no message list in the response")
    bulls = bears = 0
    ids = set()
    for m in msgs:
        if not isinstance(m, dict):
            continue
        ts = parse_ts(m.get("created_at"))
        if ts is None or not (now - window_s <= ts <= now + FUTURE_SLACK_S):
            continue
        if m.get("id") is not None:
            if m["id"] in ids:
                continue
            ids.add(m["id"])
        ent = m.get("entities")
        sent = ent.get("sentiment") if isinstance(ent, dict) else None
        basic = sent.get("basic") if isinstance(sent, dict) else None
        if basic == "Bullish":
            bulls += 1
        elif basic == "Bearish":
            bears += 1
    return bulls, bears


class StockTwits(Adapter):
    name, team, kind = "stocktwits", "sentiment", "sentiment"
    assets = ("crypto", "stock")
    per_symbol, max_symbols_per_run, interval_s = True, 5, 2700
    domains = ("api.stocktwits.com",)
    role = "primary"
    description = "StockTwits: share of bullish vs bearish tagged posts per symbol over the last 6 hours."
    MIN_TAGGED, FULL_SAMPLE = 8, 30

    def fetch(self, ctx):
        def one(sym, asset, seen):
            code = stocktwits_code(sym, asset)
            data = ctx.http.json(f"https://api.stocktwits.com/api/2/streams/symbol/{urllib.parse.quote(code)}.json")
            bulls, bears = stocktwits_mood(data, ctx.now)
            n = bulls + bears
            if n < self.MIN_TAGGED:
                return []
            share = bulls / n
            direction = 1 if share >= 0.7 else -1 if share <= 0.35 else 0
            conf = 0.35 + 0.25 * min(1.0, (n - self.MIN_TAGGED) / (self.FULL_SAMPLE - self.MIN_TAGGED))
            mood = {1: "leaning bullish", -1: "leaning bearish", 0: "mixed"}[direction]
            title = f"StockTwits on {code}: {share:.0%} bullish across {n} tagged posts in 6 hours ({mood})"
            return [Item(source=self.name, url=f"data://stocktwits/{sym}/{_hour(ctx.now)}", title=title,
                         text=f"{bulls} bullish and {bears} bearish tagged posts in the last 6 hours.",
                         published_ts=ctx.now, symbols=[sym], kind=self.kind, publisher="stocktwits.com",
                         facts=dict(bulls=bulls, bears=bears, n=n), catalyst="sentiment", direction=direction,
                         risk_flags=[], confidence=round(conf, 3))]
        return _per_symbol(ctx, self, one)


def reddit_posts(payload, sub, source, now, max_age_s=DAY, min_score=5, seen=None):
    """A subreddit listing ({data:{children:[{data:{...}}]}}) -> Items for recent posts with some upvotes."""
    data = payload.get("data") if isinstance(payload, dict) else None
    kids = data.get("children") if isinstance(data, dict) else None
    if not isinstance(kids, list):
        raise SourceError(f"reddit r/{sub}: not a listing")
    seen = set() if seen is None else seen
    out = []
    for k in kids:
        d = k.get("data") if isinstance(k, dict) else None
        if not isinstance(d, dict):
            continue
        title, perma = clean(d.get("title")), d.get("permalink")
        ts, score = _num(d.get("created_utc")), _num(d.get("score"))
        if not title or not isinstance(perma, str) or not perma.startswith("/") or ts is None:
            continue
        if score is None or score < min_score or not _fresh(ts, now, max_age_s):
            continue
        url = "https://www.reddit.com" + perma
        if url in seen:
            continue
        seen.add(url)
        out.append(Item(source=source, url=url, title=title[:500], text=clean(d.get("selftext"))[:TEXT_MAX],
                        published_ts=ts, symbols=[], kind="sentiment", publisher=f"reddit:r/{sub}",
                        facts=dict(score=int(score), comments=int(_num(d.get("num_comments")) or 0))))
    return out


class RedditCrypto(Adapter):
    name, team, kind = "reddit_crypto", "sentiment", "sentiment"
    assets = ("crypto",)
    interval_s = 1800
    domains = ("www.reddit.com",)
    role = "primary"
    description = "New posts on the big crypto subreddits (last 24 h, score 5+); names are mapped by the labeller."
    subs = ("CryptoCurrency", "Bitcoin", "ethereum")

    def fetch(self, ctx):
        if not any(self.covers(a) for a in (ctx.symbols or {}).values()):
            return []     # nothing subscribed it could be about: save the request
        out, seen, errors = [], set(), []
        for sub in self.subs:
            try:
                data = ctx.http.json(f"https://www.reddit.com/r/{sub}/new.json?limit=50")
                out += reddit_posts(data, sub, self.name, ctx.now, seen=seen)
            except RateLimited:
                if out:
                    return out
                raise
            except SourceError as e:
                errors.append(e)
                ctx.log(f"r/{sub}: {e}")
        if len(errors) == len(self.subs):
            raise errors[-1]
        return out


class RedditStocks(RedditCrypto):
    name = "reddit_stocks"
    assets = ("stock",)
    role, standby_for = "standby", "stocktwits"
    description = "New posts on r/stocks and r/wallstreetbets: stands in for StockTwits."
    subs = ("stocks", "wallstreetbets")


class HNSearch(Adapter):
    name, team, kind = "hn_search", "sentiment", "sentiment"
    assets = ("crypto", "stock")
    per_symbol, max_symbols_per_run, interval_s = True, 6, 1800
    domains = ("hn.algolia.com",)
    role, standby_for = "standby", "reddit_crypto"
    description = "Hacker News stories naming a coin or company (last 3 days): stands in for Reddit crypto."

    def fetch(self, ctx):
        def one(sym, asset, seen):
            name = display_name(ctx, sym, asset)
            data = ctx.http.json("https://hn.algolia.com/api/v1/search_by_date?query=" + urllib.parse.quote(name)
                                 + "&tags=story&hitsPerPage=20")
            return hn_items(data, name, sym, self.name, ctx.now, seen)
        return _per_symbol(ctx, self, one)


def hn_items(payload, name, symbol, source, now, seen=None, max_age_s=3 * DAY):
    """Algolia HN search hits -> Items. Keeps only stories whose title names the coin/company (the search
    also matches words in comments and URLs, which is mostly noise)."""
    hits = payload.get("hits") if isinstance(payload, dict) else None
    if not isinstance(hits, list):
        raise SourceError("hn: no hits list in the response")
    seen = set() if seen is None else seen
    pat = re.compile(r"(?<![^\W_])" + re.escape(name) + r"(?![^\W_])", re.I)
    out = []
    for h in hits:
        if not isinstance(h, dict):
            continue
        title, ts, oid = clean(h.get("title")), _num(h.get("created_at_i")), h.get("objectID")
        if not title or ts is None or not _fresh(ts, now, max_age_s) or not pat.search(title):
            continue
        url = h.get("url") if isinstance(h.get("url"), str) and h["url"].lower().startswith(("http://", "https://")) else ""
        if not url:
            if oid is None or not str(oid).isdigit():
                continue
            url = f"https://news.ycombinator.com/item?id={oid}"
        if url in seen:
            continue
        seen.add(url)
        out.append(Item(source=source, url=url, title=title[:500], text="", published_ts=ts, symbols=[symbol],
                        kind="sentiment", publisher="news.ycombinator.com",
                        facts=dict(points=int(_num(h.get("points")) or 0), comments=int(_num(h.get("num_comments")) or 0))))
    return out


def contrarian(score):
    """Fear/greed 0..100 -> direction: extreme fear is a buy signal, extreme greed a warning."""
    return 1 if score <= 20 else -1 if score >= 80 else 0


class CNNFearGreed(Adapter):
    name, team, kind = "cnn_fear_greed", "sentiment", "sentiment"
    assets = ("crypto", "stock")
    interval_s = 3600
    domains = ("production.dataviz.cnn.io",)
    role, standby_for = "standby", "fear_greed"
    description = "CNN Fear & Greed index, read contrarian: stands in for the crypto fear and greed index."
    url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"

    def fetch(self, ctx):
        data = ctx.http.json(self.url, headers={"User-Agent": BROWSER_UA, "Accept": "application/json",
                                                "Referer": "https://www.cnn.com/", "Origin": "https://www.cnn.com"})
        fg = data.get("fear_and_greed") if isinstance(data, dict) else None
        score = _num(fg.get("score")) if isinstance(fg, dict) else None
        if score is None or not 0 <= score <= 100:
            raise SourceError("cnn fear & greed: no score in the response")
        syms = _ordered(ctx)
        if not syms:
            return []
        rating = clean(fg.get("rating")) or "no rating"
        ts = parse_ts(fg.get("timestamp")) or ctx.now
        ts = ts if ts <= ctx.now + FUTURE_SLACK_S else ctx.now
        d = contrarian(score)
        note = {1: "extreme fear, which has tended to precede rebounds",
                -1: "extreme greed, which has tended to precede pullbacks", 0: "no extreme"}[d]
        return [Item(source=self.name, url=f"data://cnn_fear_greed/{_day(ts)}",
                     title=f"CNN Fear & Greed index at {score:.0f} ({rating})",
                     text=f"US market mood reads {rating} ({score:.0f} of 100): {note}.", published_ts=ts,
                     symbols=syms, kind=self.kind, publisher="cnn.com", facts=dict(score=round(score, 1), rating=rating),
                     catalyst="sentiment", direction=d, risk_flags=[], confidence=0.45)]


# ---------------------------------------------------------------------------- self-hosted search (dormant)

class SearXNG(Adapter):
    name, team, kind = "searxng", "catalysts", "search"
    assets = ("crypto", "stock")
    per_symbol, max_symbols_per_run, interval_s = True, 6, 1800
    domains = ()
    role = "dormant"
    description = "Your own SearXNG instance (set research.searxng_url): a meta-search per symbol."

    def fetch(self, ctx):
        base = (ctx.cfg or {}).get("searxng_url") if isinstance(ctx.cfg, dict) else None
        if not isinstance(base, str) or not base.lower().startswith(("http://", "https://")):
            # not configured: stay idle (no request, no failure) rather than make a grown slot look broken
            ctx.log("searxng: set research.searxng_url to your instance to use it")
            return []
        base = base.rstrip("/")

        def one(sym, asset, seen):
            name = display_name(ctx, sym, asset)
            q = f'"{name}" crypto' if asset == "crypto" else f'"{name}" stock'
            data = ctx.http.json(f"{base}/search?q={urllib.parse.quote_plus(q)}&format=json&time_range=day")
            return searx_items(data, sym, self.name, ctx.now, seen)
        return _per_symbol(ctx, self, one)


def searx_items(payload, symbol, source, now, seen=None, max_age_s=2 * DAY):
    res = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(res, list):
        raise SourceError("searxng: no results list in the response")
    seen = set() if seen is None else seen
    out = []
    for r in res[:30]:
        if not isinstance(r, dict):
            continue
        url, title = r.get("url"), clean(r.get("title"))
        if not isinstance(url, str) or not url.lower().startswith(("http://", "https://")) or not title or url in seen:
            continue
        ts = parse_ts(r.get("publishedDate")) if r.get("publishedDate") else None
        if not _fresh(ts, now, max_age_s):
            continue
        seen.add(url)
        out.append(Item(source=source, url=url, title=title[:500], text=clean(r.get("content"))[:TEXT_MAX],
                        published_ts=ts, symbols=[symbol], kind="search", publisher=domain(url),
                        facts=dict(engine=str(r.get("engine") or "")[:40])))
    return out


ADAPTERS = [GoogleNews, BingNews, GoogleRisk, BingRisk, SECPress, SECLitigation, StockTwits, RedditStocks,
            RedditCrypto, HNSearch, CNNFearGreed, GoogleListingNews, CNBCMarkets, MarketWatchRSS, TheBlockRSS, SearXNG]
