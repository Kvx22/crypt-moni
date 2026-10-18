"""python -m unittest tests.test_research_sources_web"""
import json
import os
import sys
import time
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from cryptomoni import feeds  # noqa: E402
from cryptomoni.research import taxonomy  # noqa: E402
from cryptomoni.research.http import RateLimited, SourceError  # noqa: E402
from cryptomoni.research.sources import sim, web  # noqa: E402
from cryptomoni.research.sources.base import FetchContext  # noqa: E402
from cryptomoni.research.types import EVIDENCE, TEAMS, Item  # noqa: E402

T0 = 1792300589.0   # 2026-10-18T05:16:29Z
HOUR = "2026-10-18T05"


def rfc822(ts):
    return time.strftime("%a, %d %b %Y %H:%M:%S GMT", time.gmtime(ts))


def iso(ts):
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))


class FakeHttp:
    """routes: {url: payload | Exception}; str payloads are served as text, anything else as JSON."""
    def __init__(self, routes):
        self.routes, self.calls, self.headers = dict(routes), [], []

    def _get(self, url, kw):
        self.calls.append(url)
        self.headers.append(kw.get("headers"))
        if url not in self.routes:
            raise SourceError(f"HTTP 404 for {url}")
        v = self.routes[url]
        if isinstance(v, Exception):
            raise v
        return v

    def text(self, url, **kw):
        v = self._get(url, kw)
        return v if isinstance(v, str) else json.dumps(v)

    def json(self, url, **kw):
        v = self._get(url, kw)
        return json.loads(v) if isinstance(v, str) else json.loads(json.dumps(v))


def ctx(routes, symbols=None, state=None, now=T0, requested=(), names=None, cfg=None, world=None):
    return FetchContext(http=FakeHttp(routes), now=now,
                        symbols=dict(symbols if symbols is not None else {"SOL-USD": "crypto"}),
                        requested=list(requested), state=state if state is not None else {},
                        cfg=cfg or {}, names=names or {}, world=world)


# ---------------------------------------------------------------------------- fixtures

GOOGLE_SOL_URL = ("https://news.google.com/rss/search?q=%22Solana%22+%28crypto+OR+token+OR+coin%29+when:2d"
                  "&hl=en-US&gl=US&ceid=US:en")

GOOGLE_RSS = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<rss version="2.0" xmlns:media="http://search.yahoo.com/mrss/"><channel>
<generator>NFE/5.0</generator><title>"Solana" (crypto OR token OR coin) when:2d - Google News</title>
<link>https://news.google.com/search?q=%22Solana%22&amp;hl=en-US&amp;gl=US&amp;ceid=US:en</link>
<language>en-US</language><webMaster>news-webmaster@google.com</webMaster>
<copyright>Copyright 2026 Google. All rights reserved.</copyright>
<lastBuildDate>{rfc822(T0)}</lastBuildDate>
<description>Google News</description>
<item><title>Solana ETF sees record inflows as SOL jumps 8% - CoinDesk</title>
<link>https://news.google.com/rss/articles/CBMiAAA?oc=5</link>
<guid isPermaLink="false">CBMiAAA</guid><pubDate>{rfc822(T0 - 3600)}</pubDate>
<description>&lt;a href="https://news.google.com/rss/articles/CBMiAAA?oc=5" target="_blank"&gt;Solana ETF sees record inflows as SOL jumps 8%&lt;/a&gt;&amp;nbsp;&amp;nbsp;&lt;font color="#6f6f6f"&gt;CoinDesk&lt;/font&gt;</description>
<source url="https://www.coindesk.com">CoinDesk</source></item>
<item><title>Why the Solana-Ethereum rivalry is heating up - Business-Standard</title>
<link>https://news.google.com/rss/articles/CBMiBBB?oc=5</link>
<guid isPermaLink="false">CBMiBBB</guid><pubDate>{rfc822(T0 - 7200)}</pubDate>
<description>Business-Standard</description></item>
<item><title>Solana validators vote on fee change - The Block</title>
<link>https://news.google.com/rss/articles/CBMiAAA?oc=5</link>
<guid isPermaLink="false">CBMiAAA-dup</guid><pubDate>{rfc822(T0 - 3700)}</pubDate>
<source url="https://www.theblock.co">The Block</source></item>
<item><title>Solana price prediction from last month - Some Blog</title>
<link>https://news.google.com/rss/articles/CBMiOLD?oc=5</link>
<guid isPermaLink="false">CBMiOLD</guid><pubDate>{rfc822(T0 - 30 * 86400)}</pubDate>
<source url="https://someblog.example">Some Blog</source></item>
</channel></rss>"""

BING_RSS = f"""<?xml version="1.0" encoding="utf-8" ?>
<rss version="2.0" xmlns:News="https://www.bing.com:443/news/search?q=Solana&amp;format=rss"><channel>
<title>Solana - BingNews</title><link>https://www.bing.com:443/news/search?q=Solana&amp;format=rss</link>
<description>Search results</description>
<item><title>Solana DEX volume tops Ethereum for a third week</title>
<link>http://www.bing.com/news/apiclick.aspx?ref=FexRss&amp;aid=&amp;tid=67A1B2C3&amp;url=https%3a%2f%2fwww.theblock.co%2fpost%2f123456%2fsolana-dex-volume&amp;c=1234567890&amp;mkt=en-us</link>
<description>Decentralised exchanges on Solana handled more volume than Ethereum...</description>
<pubDate>{rfc822(T0 - 5400)}</pubDate><News:Source>The Block</News:Source></item>
<item><title>Solana Mobile ships second phone</title>
<link>http://www.bing.com/news/apiclick.aspx?ref=FexRss&amp;aid=&amp;url=https%3a%2f%2fdecrypt.co%2f98765%2fsolana-mobile&amp;mkt=en-us</link>
<description>The second Solana phone ships this week.</description>
<pubDate>{rfc822(T0 - 9000)}</pubDate><News:Source>Decrypt</News:Source></item>
</channel></rss>"""


def stocktwits_payload(bulls, bears, untagged=3, old_bulls=0, now=T0):
    msgs, i = [], 600000000
    for k in range(bulls):
        i += 1
        msgs.append({"id": i, "body": f"$SOL.X loading up {k}", "created_at": iso(now - 600 - 60 * k),
                     "user": {"id": 1, "username": "a"}, "entities": {"sentiment": {"basic": "Bullish"}}})
    for k in range(bears):
        i += 1
        msgs.append({"id": i, "body": f"$SOL.X looks heavy {k}", "created_at": iso(now - 900 - 60 * k),
                     "entities": {"sentiment": {"basic": "Bearish"}}})
    for k in range(untagged):
        i += 1
        msgs.append({"id": i, "body": "watching", "created_at": iso(now - 300), "entities": {"sentiment": None}})
    for k in range(old_bulls):      # outside the 6 h window: ignored
        i += 1
        msgs.append({"id": i, "body": "old", "created_at": iso(now - 8 * 3600),
                     "entities": {"sentiment": {"basic": "Bullish"}}})
    msgs.append("junk entry")
    return {"response": {"status": 200}, "symbol": {"id": 7, "symbol": "SOL.X", "title": "Solana"},
            "cursor": {"more": True, "since": i, "max": 600000001}, "messages": msgs}


def reddit_listing(sub, now=T0):
    def post(i, title, age, score, **kw):
        d = {"subreddit": sub, "title": title, "selftext": kw.get("selftext", ""), "created_utc": now - age,
             "permalink": f"/r/{sub}/comments/abc{i}/post_{i}/", "score": score, "num_comments": 12,
             "url": f"https://www.reddit.com/r/{sub}/comments/abc{i}/post_{i}/"}
        return {"kind": "t3", "data": d}
    return {"kind": "Listing", "data": {"after": "t3_x", "dist": 4, "children": [
        post(1, f"{sub}: Solana fees are going up, thoughts?", 3600, 42, selftext="Long post about **fees**."),
        post(2, f"{sub}: low score post", 1800, 2),
        post(3, f"{sub}: old news", 3 * 86400, 900),
        {"kind": "t3", "data": {"title": "no permalink", "created_utc": now - 60, "score": 50}},
        "garbage",
    ]}}


HN_PAYLOAD = {"hits": [
    {"title": "Solana's new validator client is written in C", "url": "https://example.dev/solana-client",
     "created_at_i": int(T0 - 7200), "points": 120, "num_comments": 80, "objectID": "41000001"},
    {"title": "Ask HN: Is anyone building on Solana?", "url": None, "created_at_i": int(T0 - 3600),
     "points": 15, "objectID": "41000002"},
    {"title": "A story about Rust compilers", "url": "https://example.dev/rust", "created_at_i": int(T0 - 3600),
     "points": 300, "objectID": "41000003"},     # does not name the coin: dropped
    {"title": "Solana outage retrospective (2022)", "url": "https://example.dev/old", "created_at_i": int(T0 - 9 * 86400),
     "points": 50, "objectID": "41000004"},      # too old
    {"title": None, "objectID": "41000005"},
], "nbHits": 5, "page": 0, "hitsPerPage": 20}

CNN_PAYLOAD = {"fear_and_greed": {"score": 14.6, "rating": "extreme fear", "timestamp": "2026-10-18T04:59:58+00:00",
                                  "previous_close": 18.2, "previous_1_week": 30.1},
               "fear_and_greed_historical": {"timestamp": 1792299598000.0, "score": 14.6, "rating": "extreme fear",
                                             "data": []}}

SEARX_PAYLOAD = {"query": '"Solana" crypto', "number_of_results": 0, "results": [
    {"url": "https://www.coindesk.com/markets/2026/10/18/solana-rallies", "title": "Solana rallies on ETF hopes",
     "content": "SOL rose 8% on Saturday...", "publishedDate": iso(T0 - 4000), "engine": "bing news"},
    {"url": "https://decrypt.co/solana-thing", "title": "Solana thing", "content": "", "engine": "duckduckgo"},
    {"url": "https://www.coindesk.com/markets/2026/10/18/solana-rallies", "title": "Solana rallies (dup)",
     "engine": "google"},
    {"url": "ftp://bad.example/x", "title": "bad scheme"},
    {"url": "https://old.example/x", "title": "Old Solana post", "publishedDate": iso(T0 - 10 * 86400)},
]}


class Base(unittest.TestCase):
    def assertGoodItem(self, it, now=T0):
        self.assertIsInstance(it, Item)
        self.assertTrue(it.title.strip())
        self.assertTrue(it.url.startswith(("http://", "https://", "data://", "sim://")), it.url)
        self.assertIn(it.kind, taxonomy.FRESH_S)
        self.assertIn(it.kind, EVIDENCE)
        if it.published_ts is not None:
            self.assertLessEqual(it.published_ts, now + 3600)
        if it.catalyst is not None:
            self.assertIn(it.catalyst, taxonomy.CATALYSTS)
            self.assertIn(it.direction, taxonomy.DIRECTIONS)
            self.assertTrue(0 < it.confidence <= 1)
        for f in it.risk_flags:
            self.assertIn(f, taxonomy.RISK_FLAGS)
        json.dumps(it.d())


# ---------------------------------------------------------------------------- registry

class TestRegistry(Base):
    def test_web_roles_and_names(self):
        by = {c.name: c for c in web.ADAPTERS}
        self.assertEqual(len(by), len(web.ADAPTERS))
        want = {"google_news": ("catalysts", "primary", None), "bing_news": ("catalysts", "standby", "google_news"),
                "google_risk": ("risk", "primary", None), "bing_risk": ("risk", "standby", "google_risk"),
                "sec_press": ("risk", "primary", None), "sec_litigation": ("risk", "standby", "sec_press"),
                "stocktwits": ("sentiment", "primary", None), "reddit_stocks": ("sentiment", "standby", "stocktwits"),
                "reddit_crypto": ("sentiment", "primary", None), "hn_search": ("sentiment", "standby", "reddit_crypto"),
                "cnn_fear_greed": ("sentiment", "standby", "fear_greed"),
                "google_listings": ("discovery", "standby", "coingecko_trending"),
                "cnbc_rss": ("catalysts", "dormant", None), "marketwatch_rss": ("catalysts", "dormant", None),
                "theblock_rss": ("catalysts", "dormant", None), "searxng": ("catalysts", "dormant", None)}
        for name, (team, role, sb) in want.items():
            c = by[name]
            self.assertEqual((c.team, c.role, c.standby_for), (team, role, sb), name)
            self.assertIn(c.team, TEAMS)
            self.assertIn(c.kind, EVIDENCE)
            self.assertFalse(c.sim)
            self.assertTrue(c.description)
        self.assertTrue(by["sec_press"].official and by["sec_litigation"].official)
        self.assertEqual(by["google_news"].interval_s, 1800)
        self.assertEqual(by["google_risk"].interval_s, 2700)
        self.assertEqual(by["stocktwits"].interval_s, 2700)
        self.assertEqual(by["stocktwits"].max_symbols_per_run, 5)
        self.assertEqual(by["google_news"].max_symbols_per_run, 6)
        for n in ("google_news", "bing_news", "google_risk", "bing_risk", "stocktwits", "hn_search", "searxng"):
            self.assertTrue(by[n].per_symbol, n)
        for n in ("sec_press", "reddit_crypto", "reddit_stocks", "google_listings", "cnn_fear_greed"):
            self.assertFalse(by[n].per_symbol, n)

    def test_sim_registry(self):
        by = {c.name: c for c in sim.SIM_ADAPTERS}
        self.assertEqual(set(by), {"sim_news", "sim_wire", "sim_risk", "sim_social", "sim_listings"})
        self.assertEqual(by["sim_wire"].standby_for, "sim_news")
        self.assertEqual(by["sim_wire"].role, "standby")
        self.assertEqual({by[n].team for n in by}, {"catalysts", "risk", "sentiment", "discovery"})
        self.assertTrue(all(c.sim for c in by.values()))

    def test_catalog_twenty_and_twenty(self):
        from cryptomoni.research import sources
        ads = sources.catalog({})
        self.assertEqual(sources.IMPORT_ERRORS, {})
        counts = sources.count_roles(ads)
        self.assertEqual((counts["primary"], counts["standby"]), (20, 20))
        prim = {a.name: a for a in ads if a.role == "primary"}
        for a in ads:
            if a.role == "standby":
                self.assertIn(a.standby_for, prim, a.name)
                self.assertEqual(prim[a.standby_for].team, a.team, a.name)


# ---------------------------------------------------------------------------- google / bing

class TestNewsSearch(Base):
    def test_google_query_url_exact(self):
        c = ctx({}, names={"SOL-USD": ["Solana", "SOL"]})
        self.assertEqual(web.google_news_url(web.news_query(c, "SOL-USD", "crypto")), GOOGLE_SOL_URL)
        self.assertEqual(web.news_query(ctx({}, names={"NVDA": ["NVIDIA"]}), "NVDA", "stock"), '"NVIDIA" OR "NVDA stock"')
        self.assertTrue(web.risk_query(c, "SOL-USD", "crypto").startswith('"Solana" (hack OR exploit'))

    def test_google_news_items(self):
        c = ctx({GOOGLE_SOL_URL: GOOGLE_RSS}, names={"SOL-USD": ["Solana"]})
        items = web.GoogleNews().fetch(c)
        self.assertEqual(len(items), 2)            # duplicate link and month-old item dropped
        a, b = items
        for it in items:
            self.assertGoodItem(it)
            self.assertEqual(it.symbols, ["SOL-USD"])
            self.assertEqual(it.source, "google_news")
            self.assertEqual(it.kind, "news")
            self.assertIsNone(it.catalyst)
        self.assertEqual(a.title, "Solana ETF sees record inflows as SOL jumps 8%")
        self.assertEqual(a.publisher, "coindesk.com")
        self.assertEqual(a.url, "https://news.google.com/rss/articles/CBMiAAA?oc=5")
        self.assertAlmostEqual(a.published_ts, T0 - 3600, delta=1)
        # no <source>: the outlet comes from the " - Publisher" suffix (hyphenated outlet kept whole)
        self.assertEqual(b.title, "Why the Solana-Ethereum rivalry is heating up")
        self.assertEqual(b.publisher, "Business-Standard")

    def test_split_publisher(self):
        self.assertEqual(web.split_publisher("Solana jumps 8% - CoinDesk"), ("Solana jumps 8%", "CoinDesk"))
        self.assertEqual(web.split_publisher("Bitcoin - the long view - Bloomberg.com", "Bloomberg.com"),
                         ("Bitcoin - the long view", "Bloomberg.com"))
        self.assertEqual(web.split_publisher("No suffix here at all"), ("No suffix here at all", ""))
        self.assertEqual(web.split_publisher("Short - X"), ("Short - X", ""))   # head too short to split

    def test_google_risk_uses_3d_window(self):
        c = ctx({}, names={"SOL-USD": ["Solana"]})
        r = web.GoogleRisk()
        url = r.url(r.query(c, "SOL-USD", "crypto"))
        self.assertIn("+when:3d&", url)
        self.assertIn("hack+OR+exploit+OR+lawsuit+OR+SEC+OR+delist+OR+halt+OR+fraud+OR+bankruptcy", url)
        c.http.routes[url] = GOOGLE_RSS
        items = r.fetch(c)
        self.assertTrue(items and all(it.source == "google_risk" for it in items))

    def test_bing_unwraps_apiclick(self):
        q = web.news_query(ctx({}, names={"SOL-USD": ["Solana"]}), "SOL-USD", "crypto")
        url = web.bing_news_url(q)
        self.assertTrue(url.startswith("https://www.bing.com/news/search?q=%22Solana%22") and url.endswith("&format=rss"))
        c = ctx({url: BING_RSS}, names={"SOL-USD": ["Solana"]})
        items = web.BingNews().fetch(c)
        self.assertEqual([it.url for it in items], ["https://www.theblock.co/post/123456/solana-dex-volume",
                                                    "https://decrypt.co/98765/solana-mobile"])
        self.assertEqual([it.publisher for it in items], ["theblock.co", "decrypt.co"])
        for it in items:
            self.assertGoodItem(it)
            self.assertEqual(it.symbols, ["SOL-USD"])
        self.assertEqual(web.bing_target("https://example.com/x?url=https%3a%2f%2fevil.example"),
                         "https://example.com/x?url=https%3a%2f%2fevil.example")   # only Bing links are unwrapped

    def test_per_symbol_failures(self):
        names = {"SOL-USD": ["Solana"], "ETH-USD": ["Ethereum"]}
        c = ctx({GOOGLE_SOL_URL: GOOGLE_RSS}, symbols={"ETH-USD": "crypto", "SOL-USD": "crypto"}, names=names,
                requested=["SOL-USD"])
        items = web.GoogleNews().fetch(c)            # ETH 404s, SOL works: the run keeps SOL's items
        self.assertEqual(len(items), 2)
        self.assertEqual(c.http.calls[0], GOOGLE_SOL_URL)   # requested symbol first
        with self.assertRaises(SourceError):
            web.GoogleNews().fetch(ctx({}, names=names))
        with self.assertRaises(RateLimited):
            web.GoogleNews().fetch(ctx({GOOGLE_SOL_URL: RateLimited("429", 60)}, names=names))
        with self.assertRaises(SourceError):         # not a feed at all
            web.GoogleNews().fetch(ctx({GOOGLE_SOL_URL: "<html><body>captcha</body></html>"}, names=names))

    def test_listings_and_sec(self):
        lurl = web.google_news_url(web.LISTING_QUERY, "2d")
        self.assertIn("%28Coinbase+OR+Kraken+OR+Robinhood%29", lurl)
        items = web.GoogleListingNews().fetch(ctx({lurl: GOOGLE_RSS}))
        self.assertTrue(items)
        self.assertTrue(all(it.symbols == [] and it.kind == "discovery" for it in items))
        sec_rss = f"""<?xml version="1.0"?><rss version="2.0"><channel><title>Press Releases</title>
<link>https://www.sec.gov/news/pressreleases</link>
<item><title>SEC Charges Crypto Lender With Fraud</title><link>https://www.sec.gov/newsroom/press-releases/2026-120</link>
<pubDate>{rfc822(T0 - 86400)}</pubDate><description>The SEC today charged...</description></item>
<item><title>Old release</title><link>https://www.sec.gov/newsroom/press-releases/2026-001</link>
<pubDate>{rfc822(T0 - 40 * 86400)}</pubDate></item></channel></rss>"""
        items = web.SECPress().fetch(ctx({"https://www.sec.gov/news/pressreleases.rss": sec_rss}))
        self.assertEqual(len(items), 1)
        it = items[0]
        self.assertGoodItem(it)
        self.assertTrue(it.official)
        self.assertEqual((it.kind, it.publisher, it.symbols), ("official", "sec.gov", []))
        items = web.SECLitigation().fetch(ctx({"https://www.sec.gov/rss/litigation/litreleases.xml": sec_rss}))
        self.assertEqual(items[0].source, "sec_litigation")


# ---------------------------------------------------------------------------- crowd

ST_URL = "https://api.stocktwits.com/api/2/streams/symbol/SOL.X.json"


class TestStockTwits(Base):
    def run_st(self, payload):
        return web.StockTwits().fetch(ctx({ST_URL: payload}))

    def test_bullish_aggregate(self):
        items = self.run_st(stocktwits_payload(bulls=21, bears=5, old_bulls=10))
        self.assertEqual(len(items), 1)
        it = items[0]
        self.assertGoodItem(it)
        self.assertEqual(it.facts, {"bulls": 21, "bears": 5, "n": 26})
        self.assertEqual((it.catalyst, it.direction, it.symbols), ("sentiment", 1, ["SOL-USD"]))
        self.assertEqual(it.url, f"data://stocktwits/SOL-USD/{HOUR}")
        self.assertTrue(0.35 <= it.confidence <= 0.6)

    def test_thresholds(self):
        self.assertEqual(self.run_st(stocktwits_payload(7, 0, old_bulls=20)), [])   # < 8 tagged inside 6 h
        self.assertEqual(self.run_st(stocktwits_payload(7, 13))[0].direction, -1)   # 0.35 exactly -> bearish
        self.assertEqual(self.run_st(stocktwits_payload(3, 5))[0].direction, 0)     # 0.375 -> mixed
        self.assertEqual(self.run_st(stocktwits_payload(6, 4))[0].direction, 0)     # 0.60 -> mixed
        self.assertEqual(self.run_st(stocktwits_payload(7, 3))[0].direction, 1)     # 0.70 exactly -> bullish
        self.assertAlmostEqual(self.run_st(stocktwits_payload(8, 0))[0].confidence, 0.35)
        self.assertAlmostEqual(self.run_st(stocktwits_payload(60, 0))[0].confidence, 0.6)

    def test_codes_and_bad_shape(self):
        self.assertEqual(web.stocktwits_code("SOL-USD", "crypto"), "SOL.X")
        self.assertEqual(web.stocktwits_code("NVDA", "stock"), "NVDA")
        with self.assertRaises(SourceError):
            self.run_st({"response": {"status": 404}, "errors": [{"message": "Symbol not found"}]})


class TestRedditHN(Base):
    def test_reddit_crypto(self):
        routes = {f"https://www.reddit.com/r/{s}/new.json?limit=50": reddit_listing(s)
                  for s in ("CryptoCurrency", "Bitcoin")}
        c = ctx(routes)        # r/ethereum 404s: the run keeps the other two
        items = web.RedditCrypto().fetch(c)
        self.assertEqual(len(items), 2)
        it = items[0]
        self.assertGoodItem(it)
        self.assertEqual(it.url, "https://www.reddit.com/r/CryptoCurrency/comments/abc1/post_1/")
        self.assertEqual((it.publisher, it.kind, it.symbols), ("reddit:r/CryptoCurrency", "sentiment", []))
        self.assertEqual(it.facts["score"], 42)
        self.assertIn("fees", it.text)
        with self.assertRaises(SourceError):
            web.RedditCrypto().fetch(ctx({}))
        self.assertEqual(web.RedditCrypto().fetch(ctx(routes, symbols={"NVDA": "stock"})), [])   # nothing to cover

    def test_reddit_stocks(self):
        routes = {f"https://www.reddit.com/r/{s}/new.json?limit=50": reddit_listing(s)
                  for s in ("stocks", "wallstreetbets")}
        items = web.RedditStocks().fetch(ctx(routes, symbols={"NVDA": "stock"}))
        self.assertEqual({it.publisher for it in items}, {"reddit:r/stocks", "reddit:r/wallstreetbets"})
        with self.assertRaises(SourceError):
            web.reddit_posts({"error": 403}, "stocks", "reddit_stocks", T0)

    def test_hn(self):
        url = "https://hn.algolia.com/api/v1/search_by_date?query=Solana&tags=story&hitsPerPage=20"
        items = web.HNSearch().fetch(ctx({url: HN_PAYLOAD}, names={"SOL-USD": ["Solana"]}))
        self.assertEqual([it.url for it in items], ["https://example.dev/solana-client",
                                                    "https://news.ycombinator.com/item?id=41000002"])
        for it in items:
            self.assertGoodItem(it)
            self.assertEqual(it.symbols, ["SOL-USD"])
        with self.assertRaises(SourceError):
            web.hn_items({"message": "bad"}, "Solana", "SOL-USD", "hn_search", T0)


class TestCNNAndSearx(Base):
    def test_cnn(self):
        c = ctx({web.CNNFearGreed.url: CNN_PAYLOAD}, symbols={"SPY": "stock", "SOL-USD": "crypto"})
        items = web.CNNFearGreed().fetch(c)
        self.assertEqual(len(items), 1)
        it = items[0]
        self.assertGoodItem(it)
        self.assertEqual((it.direction, it.catalyst), (1, "sentiment"))   # extreme fear reads bullish
        self.assertEqual(set(it.symbols), {"SPY", "SOL-USD"})
        self.assertEqual(it.facts["score"], 14.6)
        self.assertIn("Mozilla", c.http.headers[0]["User-Agent"])
        greed = {"fear_and_greed": {"score": 85, "rating": "extreme greed", "timestamp": "2026-10-18T04:59:58+00:00"}}
        self.assertEqual(web.CNNFearGreed().fetch(ctx({web.CNNFearGreed.url: greed}))[0].direction, -1)
        mid = {"fear_and_greed": {"score": 50, "rating": "neutral"}}
        self.assertEqual(web.CNNFearGreed().fetch(ctx({web.CNNFearGreed.url: mid}))[0].direction, 0)
        with self.assertRaises(SourceError):
            web.CNNFearGreed().fetch(ctx({web.CNNFearGreed.url: {"oops": 1}}))

    def test_searxng(self):
        a = web.SearXNG()
        c = ctx({})
        self.assertEqual(a.fetch(c), [])            # not configured: idle, no request
        self.assertEqual(c.http.calls, [])
        url = 'https://searx.example/search?q=%22Solana%22+crypto&format=json&time_range=day'
        c = ctx({url: SEARX_PAYLOAD}, cfg={"searxng_url": "https://searx.example/"}, names={"SOL-USD": ["Solana"]})
        items = a.fetch(c)
        self.assertEqual([it.url for it in items], ["https://www.coindesk.com/markets/2026/10/18/solana-rallies",
                                                    "https://decrypt.co/solana-thing"])
        self.assertEqual([it.publisher for it in items], ["coindesk.com", "decrypt.co"])
        for it in items:
            self.assertGoodItem(it)
            self.assertEqual((it.kind, it.symbols), ("search", ["SOL-USD"]))


# ---------------------------------------------------------------------------- simulator

SYMS = ["BTC-USD", "ETH-USD", "SOL-USD"]


class TestSim(Base):
    def setUp(self):
        self.world = feeds.SimFeed(SYMS, start_ts=T0, history=200)

    def sctx(self, state=None, symbols=None):
        return ctx({}, symbols=symbols or {s: "crypto" for s in SYMS}, state=state, now=self.world.clock,
                   world=self.world)

    def test_items_and_world_untouched(self):
        before = json.dumps(self.world.rng.bit_generator.state, default=str)
        idio = self.world.idio.copy()
        got = {}
        for cls in sim.SIM_ADAPTERS:
            st, items = {}, []
            for _ in range(200):
                items += cls().fetch(self.sctx(st))
            got[cls.name] = items
            for it in items:
                self.assertGoodItem(it, now=self.world.clock)
                self.assertTrue(it.url.startswith("sim://"))
                self.assertIn(it.catalyst, taxonomy.CATALYSTS)
        self.assertEqual(json.dumps(self.world.rng.bit_generator.state, default=str), before)
        self.assertTrue(np.array_equal(idio, self.world.idio))
        for n in ("sim_news", "sim_wire", "sim_social", "sim_risk"):
            self.assertTrue(got[n], n)
        self.assertTrue(all(2 <= len(sim.SimNews().fetch(self.sctx({"runs": k}))) <= 4 for k in range(5)))
        pubs = {it.publisher for it in got["sim_news"]}
        self.assertTrue(pubs <= {f"sim-wire-{i}" for i in range(1, 7)} and len(pubs) >= 3)
        self.assertTrue({it.publisher for it in got["sim_wire"]}.isdisjoint(pubs))
        for it in got["sim_risk"]:
            self.assertTrue(it.risk_flags and it.direction == -1)
            if "halt" in it.risk_flags:
                self.assertTrue(it.official)

    def test_deterministic(self):
        a = [it.d() for it in sim.SimNews().fetch(self.sctx({"runs": 4}))]
        b = [it.d() for it in sim.SimNews().fetch(self.sctx({"runs": 4}))]
        self.assertEqual(a, b)
        c = [it.d() for it in sim.SimNews().fetch(self.sctx({"runs": 5}))]
        self.assertNotEqual(a, c)

    def test_news_is_noisily_predictive(self):
        agree = total = 0
        st = {}
        for _ in range(300):
            for it in sim.SimNews().fetch(self.sctx(st)):
                lean = sim.planted(self.world, it.symbols[0])
                self.assertTrue(lean)
                total += 1
                agree += (it.direction > 0) == (lean > 0)
        share = agree / total
        self.assertTrue(0.62 < share < 0.78, share)     # 70% agreement: informative, not an oracle
        # and the planted lean really is the simulator's: same sign as phase * mean idio
        i = SYMS.index("SOL-USD")
        self.assertEqual(np.sign(sim.planted(self.world, "SOL-USD")),
                         np.sign(self.world.phase() * self.world.idio.mean(0)[i]))

    def test_listings_only_unsubscribed(self):
        st, items = {}, []
        for _ in range(80):
            items += sim.SimListings().fetch(self.sctx(st, symbols={"BTC-USD": "crypto"}))
        self.assertTrue(items)
        for it in items:
            self.assertIn(it.symbols[0], ("ETH-USD", "SOL-USD"))
            self.assertEqual((it.catalyst, it.direction), ("listing", 1))
        self.assertEqual(len({it.symbols[0] for it in items}), len(items))   # each coin listed once

    def test_no_world(self):
        with self.assertRaises(SourceError):
            sim.SimNews().fetch(ctx({}))


if __name__ == "__main__":
    unittest.main()
