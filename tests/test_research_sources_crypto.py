"""python -m unittest tests.test_research_sources_crypto"""
import json
import os
import sys
import time
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from cryptomoni.research import taxonomy  # noqa: E402
from cryptomoni.research.http import RateLimited, SourceError  # noqa: E402
from cryptomoni.research.sources import crypto  # noqa: E402
from cryptomoni.research.sources.base import Adapter, FetchContext  # noqa: E402
from cryptomoni.research.types import EVIDENCE, TEAMS, Item  # noqa: E402

T0 = 1792300589.0   # 2026-10-18T05:16:29Z
HOUR = "2026-10-18T05"
DAY = "2026-10-18"


def rfc822(ts):
    return time.strftime("%a, %d %b %Y %H:%M:%S GMT", time.gmtime(ts))


class FakeHttp:
    """routes: {url: payload | Exception}. Payload str -> .text, anything else -> .json (via a JSON round trip,
    so the adapters see exactly what json.loads would give them)."""
    def __init__(self, routes):
        self.routes, self.calls = dict(routes), []

    def _get(self, url):
        self.calls.append(url)
        if url not in self.routes:
            raise SourceError(f"HTTP 404 for {url}")
        v = self.routes[url]
        if isinstance(v, Exception):
            raise v
        return v

    def text(self, url, **kw):
        v = self._get(url)
        return v if isinstance(v, str) else json.dumps(v)

    def json(self, url, **kw):
        v = self._get(url)
        return json.loads(v) if isinstance(v, str) else json.loads(json.dumps(v))


def ctx(routes, symbols=None, state=None, now=T0, requested=()):
    return FetchContext(http=FakeHttp(routes), now=now,
                        symbols=dict(symbols if symbols is not None else {"BTC-USD": "crypto", "SOL-USD": "crypto"}),
                        requested=list(requested), state=state if state is not None else {})


class Base(unittest.TestCase):
    def assertGoodItem(self, it, now=T0):
        self.assertIsInstance(it, Item)
        self.assertTrue(it.title.strip())
        self.assertTrue(it.url.startswith(("http://", "https://", "data://")), it.url)
        self.assertIn(it.kind, EVIDENCE)
        if it.published_ts is not None:
            self.assertLessEqual(it.published_ts, now)
        if it.catalyst is not None:
            self.assertIn(it.catalyst, taxonomy.CATALYSTS)
            self.assertIn(it.direction, taxonomy.DIRECTIONS)
            self.assertTrue(0 < it.confidence <= 1)
        for f in it.risk_flags:
            self.assertIn(f, taxonomy.RISK_FLAGS)
        json.dumps(it.d())   # storable


# ---------------------------------------------------------------------------- catalog

class TestCatalog(Base):
    SLOTS = {   # name: (class, team, kind, interval, role, standby_for)
        "coindesk_rss": ("CoinDeskRSS", "catalysts", "news", 900, "primary", None),
        "cointelegraph_rss": ("CointelegraphRSS", "catalysts", "news", 900, "standby", "coindesk_rss"),
        "decrypt_rss": ("DecryptRSS", "catalysts", "news", 900, "primary", None),
        "cryptoslate_rss": ("CryptoSlateRSS", "catalysts", "news", 900, "standby", "decrypt_rss"),
        "coinbase_status": ("CoinbaseStatus", "risk", "official", 600, "primary", None),
        "kraken_status": ("KrakenStatus", "risk", "official", 600, "standby", "coinbase_status"),
        "coinbase_incidents": ("CoinbaseIncidents", "risk", "official", 900, "primary", None),
        "kraken_incidents": ("KrakenIncidents", "risk", "official", 900, "standby", "coinbase_incidents"),
        "defillama_tvl": ("DefiLlamaTVL", "filings", "market", 3600, "primary", None),
        "defillama_chains": ("DefiLlamaChains", "filings", "market", 3600, "standby", "defillama_tvl"),
        "kraken_funding": ("KrakenFuturesFunding", "market", "market", 1800, "primary", None),
        "deribit_funding": ("DeribitFunding", "market", "market", 1800, "standby", "kraken_funding"),
        "coinbase_book": ("CoinbaseBook", "market", "market", 1800, "primary", None),
        "kraken_book": ("KrakenBook", "market", "market", 1800, "standby", "coinbase_book"),
        "fear_greed": ("FearGreedCrypto", "sentiment", "sentiment", 3600, "primary", None),
        "coinbase_listings": ("CoinbaseListings", "discovery", "discovery", 3600, "primary", None),
        "kraken_listings": ("KrakenListings", "discovery", "discovery", 3600, "standby", "coinbase_listings"),
        "coingecko_trending": ("CoinGeckoTrending", "discovery", "discovery", 3600, "primary", None),
    }

    def test_slot_table(self):
        got = {c.name: c for c in crypto.ADAPTERS}
        self.assertEqual(set(got), set(self.SLOTS))
        self.assertEqual(len(crypto.ADAPTERS), len(got))
        for name, (cls, team, kind, iv, role, sb) in self.SLOTS.items():
            a = got[name]()
            self.assertIsInstance(a, Adapter)
            self.assertEqual(type(a).__name__, cls)
            self.assertEqual((a.team, a.kind, a.interval_s, a.role, a.standby_for), (team, kind, iv, role, sb), name)
            self.assertEqual(tuple(a.assets), ("crypto",))
            self.assertIn(a.team, TEAMS)
            self.assertTrue(a.domains and a.description, name)
            if sb:
                self.assertEqual(got[sb].team, a.team)
                self.assertEqual(got[sb].role, "primary")

    def test_flags(self):
        self.assertTrue(crypto.CoinbaseBook.per_symbol and crypto.KrakenBook.per_symbol)
        self.assertEqual(crypto.CoinbaseBook.max_symbols_per_run, 6)
        for c in (crypto.CoinbaseStatus, crypto.CoinbaseIncidents, crypto.CoinbaseListings):
            self.assertTrue(c.official)

    def test_catalog_loads_module(self):
        from cryptomoni.research import sources
        names = {a.name for a in sources.catalog({})}
        self.assertNotIn("crypto", sources.IMPORT_ERRORS)
        self.assertTrue(set(self.SLOTS) <= names)


# ---------------------------------------------------------------------------- news

def coindesk_rss(now=T0):
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<rss xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:content="http://purl.org/rss/1.0/modules/content/"
     xmlns:media="http://search.yahoo.com/mrss/" version="2.0">
<channel><title>CoinDesk: Bitcoin, Ethereum, Crypto News and Price Data</title>
<link>https://www.coindesk.com</link><description>Leader in cryptocurrency news</description>
<item><title><![CDATA[Solana Validators Approve Fee Change in Close Vote]]></title>
<link>https://www.coindesk.com/tech/2026/10/18/solana-validators-approve-fee-change</link>
<guid isPermaLink="false">a1b2c3</guid><dc:creator><![CDATA[Jane Doe]]></dc:creator>
<description><![CDATA[<p>The proposal passed with <b>52%</b> of stake.</p>]]></description>
<pubDate>{rfc822(now - 3600)}</pubDate><category>Tech</category>
<media:content url="https://cdn.example/img.jpg" medium="image"/></item>
<item><title>Bitcoin Steadies Near $120K &amp; Ether Lags</title>
<link>https://www.coindesk.com/markets/2026/10/18/bitcoin-steadies</link>
<pubDate>{rfc822(now - 7200)}</pubDate><description>Markets recap.</description></item>
<item><title>Old Story From Last Week</title><link>https://www.coindesk.com/old</link>
<pubDate>{rfc822(now - 5 * 86400)}</pubDate></item>
<item><title>Story From The Future</title><link>https://www.coindesk.com/future</link>
<pubDate>{rfc822(now + 6 * 3600)}</pubDate></item>
<item><title>   </title><link>https://www.coindesk.com/no-title</link><pubDate>{rfc822(now)}</pubDate></item>
<item><title>No link at all</title><pubDate>{rfc822(now)}</pubDate></item>
<item><title>Bad link scheme</title><link>javascript:alert(1)</link></item>
<item><title>Undated but fine</title><link>https://www.coindesk.com/undated</link></item>
<item><title>Bitcoin Steadies Near $120K &amp; Ether Lags</title>
<link>https://www.coindesk.com/markets/2026/10/18/bitcoin-steadies</link></item>
</channel></rss>"""


DECRYPT_ATOM = f"""<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"><title>Decrypt</title>
<link rel="self" href="https://decrypt.co/feed"/><link rel="alternate" href="https://decrypt.co"/>
<entry><title>Novacoin Mainnet Goes Live</title><link rel="alternate" href="https://decrypt.co/300000/novacoin-mainnet"/>
<id>https://decrypt.co/?p=300000</id><updated>2026-10-18T04:00:00Z</updated>
<summary type="html">&lt;p&gt;The launch went smoothly.&lt;/p&gt;</summary></entry>
</feed>"""


class TestNews(Base):
    def test_rss_happy_and_junk(self):
        c = ctx({crypto.CoinDeskRSS.feed_url: coindesk_rss()})
        items = crypto.CoinDeskRSS().fetch(c)
        titles = [i.title for i in items]
        self.assertEqual(titles, ["Solana Validators Approve Fee Change in Close Vote",
                                  "Bitcoin Steadies Near $120K & Ether Lags", "Undated but fine"])
        a = items[0]
        self.assertGoodItem(a)
        self.assertEqual((a.source, a.kind, a.publisher, a.symbols), ("coindesk_rss", "news", "coindesk.com", []))
        self.assertEqual(a.text, "The proposal passed with 52% of stake.")
        self.assertAlmostEqual(a.published_ts, T0 - 3600)
        self.assertIsNone(a.catalyst)          # text items are left to the labeller
        self.assertIsNone(items[2].published_ts)

    def test_atom_and_standby_feeds(self):
        items = crypto.DecryptRSS().fetch(ctx({"https://decrypt.co/feed": DECRYPT_ATOM}))
        self.assertEqual(len(items), 1)
        self.assertEqual((items[0].url, items[0].publisher), ("https://decrypt.co/300000/novacoin-mainnet", "decrypt.co"))
        for cls, url, pub in ((crypto.CointelegraphRSS, "https://cointelegraph.com/rss", "cointelegraph.com"),
                              (crypto.CryptoSlateRSS, "https://cryptoslate.com/feed/", "cryptoslate.com")):
            items = cls().fetch(ctx({url: coindesk_rss()}))
            self.assertEqual(len(items), 3)
            self.assertTrue(all(i.publisher == pub and i.source == cls.name for i in items))

    def test_max_items_and_bad_xml(self):
        xml = "<rss><channel>" + "".join(
            f"<item><title>Story {i}</title><link>https://x.example/{i}</link></item>" for i in range(100)) + "</channel></rss>"
        self.assertEqual(len(crypto.rss_items(ctx({"u": xml}), "u", "s", "x.example")), 60)
        with self.assertRaises(SourceError):
            crypto.CoinDeskRSS().fetch(ctx({crypto.CoinDeskRSS.feed_url: "<html><body>Access denied"}))
        with self.assertRaises(SourceError):
            crypto.CoinDeskRSS().fetch(ctx({}))


# ---------------------------------------------------------------------------- exchange status

def cb_product(pid, status="online", trading_disabled=False, cancel_only=False, msg="", **kw):
    base, quote = pid.split("-")
    d = {"id": pid, "base_currency": base, "quote_currency": quote, "quote_increment": "0.01",
         "base_increment": "0.00000001", "display_name": f"{base}/{quote}", "min_market_funds": "1",
         "margin_enabled": False, "post_only": False, "limit_only": False, "cancel_only": cancel_only,
         "status": status, "status_message": msg, "trading_disabled": trading_disabled,
         "fx_stablecoin": False, "max_slippage_percentage": "0.03000000", "auction_mode": False,
         "high_bid_limit_percentage": ""}
    d.update(kw)
    return d


CB_PRODUCTS = "https://api.exchange.coinbase.com/products"


def cb_products(**over):
    base = [cb_product("BTC-USD"), cb_product("SOL-USD"), cb_product("ETH-USD"), cb_product("ETH-BTC"),
            cb_product("SOL-USDC"), cb_product("DOGE-USD")]
    return [over.get(p["id"], p) for p in base]


class TestCoinbaseStatus(Base):
    def test_state_diff(self):
        a, st = crypto.CoinbaseStatus(), {}
        self.assertEqual(a.fetch(ctx({CB_PRODUCTS: cb_products()}, state=st)), [])   # all fine: nothing to say
        self.assertEqual(st["last"], {"BTC-USD": "ok", "SOL-USD": "ok"})
        halted = cb_products(**{"SOL-USD": cb_product("SOL-USD", trading_disabled=True, msg="<b>Network upgrade</b>")})
        items = a.fetch(ctx({CB_PRODUCTS: halted}, state=st, now=T0 + 600))
        self.assertEqual(len(items), 1)
        it = items[0]
        self.assertGoodItem(it, T0 + 600)
        self.assertEqual((it.catalyst, it.direction, it.risk_flags, it.confidence, it.official, it.symbols, it.kind),
                         ("outage", -1, ["halt"], 0.95, True, ["SOL-USD"], "official"))
        self.assertEqual(it.url, f"data://coinbase_status/SOL-USD/halted/{HOUR}")
        self.assertIn("Network upgrade", it.text)
        self.assertTrue(it.facts["trading_disabled"])
        # same state again: silent
        self.assertEqual(a.fetch(ctx({CB_PRODUCTS: halted}, state=st, now=T0 + 1200)), [])
        # back online: one "resumed" item
        items = a.fetch(ctx({CB_PRODUCTS: cb_products()}, state=st, now=T0 + 1800))
        self.assertEqual(len(items), 1)
        self.assertEqual((items[0].catalyst, items[0].direction, items[0].risk_flags), ("outage", 1, []))
        self.assertIn("/SOL-USD/ok/", items[0].url)

    def test_delisted_cancel_only_and_first_look(self):
        prods = cb_products(**{"SOL-USD": cb_product("SOL-USD", status="delisted"),
                               "BTC-USD": cb_product("BTC-USD", cancel_only=True)})
        items = crypto.CoinbaseStatus().fetch(ctx({CB_PRODUCTS: prods}))
        by = {i.symbols[0]: i for i in items}
        self.assertEqual((by["SOL-USD"].catalyst, by["SOL-USD"].risk_flags), ("delisting", ["delisting"]))
        self.assertEqual((by["BTC-USD"].catalyst, by["BTC-USD"].risk_flags), ("outage", ["halt"]))

    def test_only_subscribed_and_odd_payloads(self):
        prods = cb_products(**{"DOGE-USD": cb_product("DOGE-USD", trading_disabled=True)})
        self.assertEqual(crypto.CoinbaseStatus().fetch(ctx({CB_PRODUCTS: prods})), [])
        # junk rows skipped; a subscribed symbol missing from the list keeps its state
        st = {"last": {"SOL-USD": "halted"}}
        odd = [None, 5, {"no": "id"}, cb_product("BTC-USD")]
        self.assertEqual(crypto.CoinbaseStatus().fetch(ctx({CB_PRODUCTS: odd}, state=st)), [])
        self.assertEqual(st["last"]["SOL-USD"], "halted")
        for bad in ({"message": "Internal error"}, [], [1, 2]):
            with self.assertRaises(SourceError):
                crypto.CoinbaseStatus().fetch(ctx({CB_PRODUCTS: bad}))
        # stocks in the context are ignored
        self.assertEqual(crypto.CoinbaseStatus().fetch(ctx({CB_PRODUCTS: cb_products()}, symbols={"NVDA": "stock"})), [])


KR_PAIRS = "https://api.kraken.com/0/public/AssetPairs"


def kr_pairs(**status):
    def p(key, alt, ws, base, quote):
        return {"altname": alt, "wsname": ws, "aclass_base": "currency", "base": base, "aclass_quote": "currency",
                "quote": quote, "lot": "unit", "cost_decimals": 5, "pair_decimals": 1, "lot_decimals": 8,
                "lot_multiplier": 1, "leverage_buy": [2, 3], "leverage_sell": [2, 3],
                "fees": [[0, 0.4], [10000, 0.35]], "fee_volume_currency": "ZUSD", "margin_call": 80,
                "margin_stop": 40, "ordermin": "0.0001", "costmin": "0.5", "tick_size": "0.1",
                "status": status.get(ws.split("/")[0], "online")}
    pairs = [("XXBTZUSD", "XBTUSD", "XBT/USD", "XXBT", "ZUSD"), ("SOLUSD", "SOLUSD", "SOL/USD", "SOL", "ZUSD"),
             ("XDGUSD", "XDGUSD", "XDG/USD", "XXDG", "ZUSD"), ("XETHZUSD", "ETHUSD", "ETH/USD", "XETH", "ZUSD"),
             ("SOLEUR", "SOLEUR", "SOL/EUR", "SOL", "ZEUR"), ("XETHXXBT", "ETHXBT", "ETH/XBT", "XETH", "XXBT")]
    pairs += [(f"{b}USD", f"{b}USD", f"{b}/USD", b, "ZUSD") for b in status.pop("_extra", [])]
    return {"error": [], "result": {k: p(*x) for k, *rest in [(x[0], x) for x in pairs] for x in [rest[0]]}}


class TestKrakenStatus(Base):
    def test_pairs_mapping(self):
        pairs = crypto._kraken_pairs(kr_pairs())
        self.assertEqual(set(pairs), {"BTC-USD", "SOL-USD", "DOGE-USD", "ETH-USD"})
        self.assertEqual(pairs["BTC-USD"]["pair"], "XXBTZUSD")
        # without wsname the legacy codes still map
        payload = {"error": [], "result": {"XXBTZUSD": {"base": "XXBT", "quote": "ZUSD", "status": "online"}}}
        self.assertEqual(list(crypto._kraken_pairs(payload)), ["BTC-USD"])

    def test_state_diff(self):
        a, st, syms = crypto.KrakenStatus(), {}, {"BTC-USD": "crypto", "DOGE-USD": "crypto", "SOL-USD": "crypto"}
        self.assertEqual(a.fetch(ctx({KR_PAIRS: kr_pairs()}, syms, st)), [])
        items = a.fetch(ctx({KR_PAIRS: kr_pairs(XBT="cancel_only", XDG="delisted", SOL="limit_only")}, syms, st))
        by = {i.symbols[0]: i for i in items}
        self.assertEqual(set(by), {"BTC-USD", "DOGE-USD"})       # limit_only is still tradable
        self.assertEqual(by["BTC-USD"].risk_flags, ["halt"])
        self.assertEqual(by["DOGE-USD"].catalyst, "delisting")
        self.assertTrue(by["BTC-USD"].url.startswith("data://kraken_status/BTC-USD/halted/"))
        self.assertEqual(by["BTC-USD"].facts["status"], "cancel_only")
        items = a.fetch(ctx({KR_PAIRS: kr_pairs()}, syms, st))
        self.assertEqual(sorted(i.symbols[0] for i in items), ["BTC-USD", "DOGE-USD"])
        self.assertTrue(all(i.direction == 1 for i in items))

    def test_errors(self):
        for bad in ({"error": ["EGeneral:Too many requests"]}, {"error": [], "result": {}}, [], {"error": []},
                    {"error": [], "result": {"SOLEUR": {"wsname": "SOL/EUR"}}}):
            with self.assertRaises(SourceError):
                crypto.KrakenStatus().fetch(ctx({KR_PAIRS: bad}))


# ---------------------------------------------------------------------------- incidents

def statuspage_rss(now=T0, host="status.coinbase.com", name="Coinbase"):
    def item(title, updates, ts, slug):
        desc = "".join(f"&lt;p&gt;&lt;small&gt;Oct &lt;var data-var='date'&gt;18&lt;/var&gt;, &lt;var data-var='time'&gt;"
                       f"05:00&lt;/var&gt; UTC&lt;/small&gt;&lt;br&gt;&lt;strong&gt;{s}&lt;/strong&gt; - {m}&lt;/p&gt;"
                       for s, m in updates)
        return (f"<item><title>{title}</title><description>{desc}</description><pubDate>{rfc822(ts)}</pubDate>"
                f"<link>https://{host}/incidents/{slug}</link><guid>https://{host}/incidents/{slug}</guid></item>")
    items = [
        item("Delayed withdrawals for Solana", [("Investigating", "We are investigating delayed SOL withdrawals.")],
             now - 1800, "aaa"),
        item("Degraded performance on trading", [("Resolved", "This incident has been resolved."),
                                                  ("Monitoring", "A fix has been implemented."),
                                                  ("Investigating", "Some users cannot place trades.")], now - 3 * 3600, "bbb"),
        item("Elevated errors on the mobile app", [("Identified", "The issue has been identified.")], now - 600, "ccc"),
        item("Scheduled maintenance for card services", [("Scheduled", "Card payments will be unavailable.")],
             now - 900, "ddd"),
        item("Old incident from last month", [("Resolved", "Resolved.")], now - 30 * 86400, "eee"),
        "<item><title>Undated</title><link>https://example.com/u</link><description>x</description></item>",
    ]
    return (f'<?xml version="1.0" encoding="UTF-8"?><rss version="2.0"><channel><title>{name} Status - Incident History'
            f'</title><link>https://{host}</link><description>Statuspage</description>{"".join(items)}</channel></rss>')


class TestIncidents(Base):
    def test_coinbase_incidents(self):
        items = crypto.CoinbaseIncidents().fetch(ctx({"https://status.coinbase.com/history.rss": statuspage_rss()}))
        by = {i.url.rsplit("/", 1)[1]: i for i in items}
        self.assertEqual(set(by), {"aaa", "bbb", "ccc", "ddd"})
        for i in items:
            self.assertGoodItem(i)
            self.assertEqual((i.catalyst, i.kind, i.official, i.symbols, i.publisher),
                             ("outage", "official", True, [], "status.coinbase.com"))
        sol = by["aaa"]
        self.assertEqual((sol.direction, sol.risk_flags), (-1, ["outage"]))
        self.assertNotIn("scope", sol.facts)                       # names a coin: the labeller maps it
        self.assertIn("Delayed withdrawals for Solana", sol.title)
        res = by["bbb"]
        self.assertEqual((res.direction, res.risk_flags, res.facts["status"]), (0, [], "resolved"))
        self.assertEqual(res.facts.get("scope"), "exchange")
        app = by["ccc"]
        self.assertEqual((app.direction, app.risk_flags, app.facts["scope"]), (-1, [], "exchange"))
        self.assertEqual((by["ddd"].direction, by["ddd"].risk_flags), (0, []))

    def test_kraken_incidents_and_bad_feed(self):
        url = "https://status.kraken.com/history.rss"
        items = crypto.KrakenIncidents().fetch(ctx({url: statuspage_rss(host="status.kraken.com", name="Kraken")}))
        self.assertEqual(len(items), 4)
        self.assertTrue(all(i.source == "kraken_incidents" and i.title.startswith("Kraken status:") for i in items))
        with self.assertRaises(SourceError):
            crypto.KrakenIncidents().fetch(ctx({url: '{"not": "xml"}'}))


# ---------------------------------------------------------------------------- TVL

LLAMA = "https://api.llama.fi/protocols"


def llama_protocols():
    def p(name, sym, tvl, c1, c7, cat="Lending", **kw):
        d = {"id": str(abs(hash(name)) % 9999), "name": name, "address": None, "symbol": sym, "url": "https://x.example",
             "description": "", "chain": "Multi-Chain", "logo": "", "audits": "2", "gecko_id": None, "cmcId": None,
             "category": cat, "chains": ["Ethereum"], "module": "x/index.js", "twitter": "x", "slug": name.lower(),
             "tvl": tvl, "chainTvls": {"Ethereum": tvl}, "change_1h": 0.1, "change_1d": c1, "change_7d": c7,
             "mcap": None}
        d.update(kw)
        return d
    return [
        p("Aave V3", "AAVE", 12.34e9, 1.2, 11.4),
        p("Aave V2", "AAVE", 0.4e9, -1.0, -20.0),
        p("Jito", "JTO", 2.1e9, -2.0, -18.0, "Liquid Staking"),
        p("Tiny Swap", "TINY", 3e6, 50.0, 400.0, "Dexes"),
        p("Binance CEX", "BNB", 150e9, 0.0, 0.0, "CEX"),
        p("Uniswap V3", "UNI", 4.2e9, None, None, "Dexes"),
        p("Weird", None, "lots", 1, 1),
        p("Glitchy", "GLT", 900e6, 0, 950.0),
        "not a dict",
    ]


class TestDefiLlama(Base):
    SYMS = {"AAVE-USD": "crypto", "JTO-USD": "crypto", "UNI-USD": "crypto", "TINY-USD": "crypto",
            "BNB-USD": "crypto", "GLT-USD": "crypto", "BTC-USD": "crypto"}

    def test_protocols(self):
        items = crypto.DefiLlamaTVL().fetch(ctx({LLAMA: llama_protocols()}, self.SYMS))
        by = {i.symbols[0]: i for i in items}
        self.assertEqual(set(by), {"AAVE-USD", "JTO-USD", "UNI-USD", "GLT-USD"})   # TINY < $50M, BNB is a CEX
        aave = by["AAVE-USD"]
        self.assertGoodItem(aave)
        self.assertEqual(aave.title, "Aave V3 TVL $12.3B, +11% on the week (DefiLlama)")
        self.assertEqual((aave.catalyst, aave.kind, aave.direction, aave.url),
                         ("other", "market", 1, f"data://defillama/AAVE-USD/{DAY}"))
        self.assertTrue(0.5 <= aave.confidence <= 0.8)
        self.assertEqual(aave.facts["tvl"], 12.34e9)
        self.assertEqual(by["JTO-USD"].direction, -1)
        self.assertEqual((by["UNI-USD"].direction, by["UNI-USD"].confidence), (0, 0.5))
        self.assertEqual(by["UNI-USD"].title, "Uniswap V3 TVL $4.2B (DefiLlama)")
        self.assertEqual(by["GLT-USD"].direction, 0)              # +950% is a data glitch, not a signal
        with self.assertRaises(SourceError):
            crypto.DefiLlamaTVL().fetch(ctx({LLAMA: {"error": "rate limited"}}))

    def test_chains_history(self):
        url = "https://api.llama.fi/v2/chains"
        def chains(eth, sol):
            return [{"gecko_id": "ethereum", "tvl": eth, "tokenSymbol": "ETH", "cmcId": "1027", "name": "Ethereum",
                     "chainId": 1}, {"gecko_id": "solana", "tvl": sol, "tokenSymbol": "SOL", "cmcId": "5426",
                     "name": "Solana", "chainId": None}, {"gecko_id": None, "tvl": 9e9, "tokenSymbol": None,
                     "name": "Base", "chainId": 8453}, {"tvl": "x", "tokenSymbol": "ETH", "name": "Fake"}]
        a, st, syms = crypto.DefiLlamaChains(), {}, {"ETH-USD": "crypto", "SOL-USD": "crypto"}
        first = a.fetch(ctx({url: chains(60e9, 10e9)}, syms, st, now=T0 - 7 * 86400))
        self.assertEqual(len(first), 2)
        self.assertTrue(all(i.direction == 0 and i.facts["change_1d"] is None for i in first))
        a.fetch(ctx({url: chains(61e9, 10e9)}, syms, st, now=T0 - 86400))
        items = {i.symbols[0]: i for i in a.fetch(ctx({url: chains(61.5e9, 12e9)}, syms, st))}
        sol = items["SOL-USD"]
        self.assertAlmostEqual(sol.facts["change_1d"], 20.0, places=3)
        self.assertAlmostEqual(sol.facts["change_7d"], 20.0, places=3)
        self.assertEqual(sol.direction, 1)
        self.assertEqual(items["ETH-USD"].direction, 0)
        self.assertEqual(sol.url, f"data://defillama/SOL-USD/{DAY}")
        self.assertLessEqual(len(st["days"]["Solana"]), 9)
        self.assertEqual(set(st["days"]), {"Ethereum", "Solana"})


# ---------------------------------------------------------------------------- funding

KF = "https://futures.kraken.com/derivatives/api/v3/tickers"


def kf_tickers(rates, extra=()):
    """rates: {base: relative hourly rate}; Kraken reports fundingRate as an absolute amount = rel * markPrice."""
    px = {"XBT": 120000.0, "ETH": 4500.0, "SOL": 200.0}
    out = []
    for base, rel in rates.items():
        mark = px.get(base, 10.0)
        out.append({"tag": "perpetual", "pair": f"{base}:USD", "symbol": f"PF_{base}USD", "markPrice": mark,
                    "bid": mark - 0.1, "bidSize": 1.0, "ask": mark + 0.1, "askSize": 1.0, "vol24h": 1000.0,
                    "volumeQuote": 1e6, "openInterest": 12345.6, "open24h": mark, "high24h": mark, "low24h": mark,
                    "last": mark, "lastTime": "2026-10-18T05:16:00.000Z", "lastSize": 0.1, "suspended": False,
                    "fundingRate": rel * mark, "fundingRatePrediction": rel * mark, "postOnly": False,
                    "indexPrice": mark, "change24h": 0.5})
    out += list(extra)
    return {"result": "success", "serverTime": "2026-10-18T05:16:29.000Z", "tickers": out}


NORMAL = {f"C{i}": 1.25e-5 * (1 + (i % 5 - 2) * 0.1) for i in range(20)}   # ~0.01%/8h, a little spread


class TestFunding(Base):
    def test_kraken_crowded_longs(self):
        rates = dict(NORMAL, XBT=1.3e-5, SOL=2.5e-4)      # SOL at 0.2%/8h: crowded
        extra = [{"tag": "month", "symbol": "FF_XBTUSD_261127", "markPrice": 1.0},
                 {"tag": "perpetual", "symbol": "PF_BADUSD", "fundingRate": "nan?", "markPrice": 5},
                 {"tag": "perpetual", "symbol": "PF_ETHUSD", "fundingRate": 1, "markPrice": 0}, "junk"]
        st = {}
        items = crypto.KrakenFuturesFunding().fetch(ctx({KF: kf_tickers(rates, extra)}, state=st))
        self.assertEqual(len(items), 1)
        it = items[0]
        self.assertGoodItem(it)
        self.assertEqual((it.symbols, it.catalyst, it.direction, it.confidence, it.kind),
                         (["SOL-USD"], "market_structure", -1, 0.55, "market"))
        self.assertEqual(it.url, f"data://kraken_funding/SOL-USD/{HOUR}")
        self.assertIn("longs crowded", it.title)
        self.assertIn("(Kraken Futures)", it.title)
        self.assertAlmostEqual(it.facts["funding"], 2.5e-4 * 8, places=8)
        self.assertGreaterEqual(it.facts["z"], 2)
        self.assertEqual(it.facts["oi"], 12345.6)
        self.assertEqual(st["hist"]["BTC-USD"], [round(1.3e-5 * 8, 12)] if False else st["hist"]["BTC-USD"])
        self.assertEqual(len(st["hist"]["SOL-USD"]), 1)
        # still crowded 30 min later: not repeated; after 6 h it is
        self.assertEqual(crypto.KrakenFuturesFunding().fetch(ctx({KF: kf_tickers(rates)}, state=st, now=T0 + 1800)), [])
        self.assertEqual(len(crypto.KrakenFuturesFunding().fetch(ctx({KF: kf_tickers(rates)}, state=st,
                                                                     now=T0 + 7 * 3600))), 1)

    def test_deeply_negative_and_own_history(self):
        a, st = crypto.KrakenFuturesFunding(), {}
        # warm up SOL's own history at a high but steady level: the cross-section says extreme, its own past does not
        hi = dict(NORMAL, SOL=2.0e-4)
        for k in range(15):
            a.fetch(ctx({KF: kf_tickers(dict(hi, SOL=2.0e-4 * (1 + 0.05 * (k % 3 - 1))))}, state=st, now=T0 + k * 1800))
        st.pop("ext", None)
        self.assertEqual(a.fetch(ctx({KF: kf_tickers(hi)}, state=st, now=T0 + 16 * 1800)), [])
        self.assertLessEqual(len(st["hist"]["SOL-USD"]), crypto.FUND_HIST)
        # deeply negative BTC (no own history needed beyond the cross-section): +1
        neg = dict(NORMAL, XBT=-3e-4)
        items = crypto.KrakenFuturesFunding().fetch(ctx({KF: kf_tickers(neg)}))
        self.assertEqual([(i.symbols, i.direction) for i in items], [(["BTC-USD"], 1)])
        self.assertIn("shorts crowded", items[0].title)

    def test_normal_market_is_silent_and_bad_shape(self):
        self.assertEqual(crypto.KrakenFuturesFunding().fetch(ctx({KF: kf_tickers(dict(NORMAL, XBT=1.2e-5, SOL=1.3e-5))})), [])
        for bad in ({"result": "error", "error": "apiLimitExceeded"}, {"tickers": []}, [],
                    {"tickers": [{"tag": "month", "symbol": "FI_XBTUSD_261127"}]}):
            with self.assertRaises(SourceError):
                crypto.KrakenFuturesFunding().fetch(ctx({KF: bad}))

    def test_deribit(self):
        u = crypto.DeribitFunding.URL.format

        def row(name, f8, oi=1000.0):
            return {"volume_usd": 1e6, "volume": 10.0, "quote_currency": "USD", "price_change": 0.5,
                    "open_interest": oi, "mid_price": 100.0, "mark_price": 100.0, "low": 99.0, "last": 100.0,
                    "instrument_name": name, "high": 101.0, "funding_8h": f8, "estimated_delivery_price": 100.0,
                    "current_funding": 0.0, "creation_timestamp": 1792300589000, "bid_price": 99.9,
                    "base_currency": name.split("-")[0].split("_")[0], "ask_price": 100.1}
        usdc = {"jsonrpc": "2.0", "result": [row(f"C{i}_USDC-PERPETUAL", 1e-4 * (1 + (i % 3 - 1) * 0.1)) for i in range(8)]
                + [row("SOL_USDC-PERPETUAL", 2.4e-3), row("BTC_USDC-PERPETUAL", 9e-3),
                   {"instrument_name": "SOL_USDC-27DEC26", "funding_8h": None}, {"odd": True}],
                "usIn": 1, "usOut": 2, "usDiff": 1, "testnet": False}
        btc = {"jsonrpc": "2.0", "result": [row("BTC-PERPETUAL", 1.0e-4, 9e8),
                                            {"instrument_name": "BTC-27DEC26", "mark_price": 121000.0}]}
        routes = {u("BTC"): btc, u("ETH"): SourceError("www.deribit.com: HTTP 500"), u("USDC"): usdc}
        items = crypto.DeribitFunding().fetch(ctx(routes))
        self.assertEqual([(i.symbols, i.direction, i.source) for i in items], [(["SOL-USD"], -1, "deribit_funding")])
        self.assertIn("(Deribit)", items[0].title)
        self.assertEqual(items[0].facts["instrument"], "SOL_USDC-PERPETUAL")   # BTC kept the inverse perp, not the USDC twin
        with self.assertRaises(SourceError):
            crypto.DeribitFunding().fetch(ctx({}))
        with self.assertRaises(RateLimited):
            crypto.DeribitFunding().fetch(ctx({u("BTC"): RateLimited("www.deribit.com: HTTP 429", 60)}))


# ---------------------------------------------------------------------------- books

def cb_book(mid, bid_usd, ask_usd, levels=10):
    """Coinbase level-2 shape: price/size strings plus an order count, spread over 0.9% either side,
    plus far-away levels that must not count."""
    bids = [[f"{mid * (1 - 0.0005 - 0.0009 * k):.4f}", f"{bid_usd / levels / mid:.8f}", 3] for k in range(levels)]
    asks = [[f"{mid * (1 + 0.0005 + 0.0009 * k):.4f}", f"{ask_usd / levels / mid:.8f}", 2] for k in range(levels)]
    bids.append([f"{mid * 0.95:.4f}", "1000000", 1])
    asks.append([f"{mid * 1.05:.4f}", "1000000", 1])
    return {"bids": bids, "asks": asks, "sequence": 123456789, "auction_mode": False, "auction": None,
            "time": "2026-10-18T05:16:29.123456Z"}


def cb_book_url(sym):
    return f"https://api.exchange.coinbase.com/products/{sym}/book?level=2"


class TestBooks(Base):
    def test_coinbase_threshold(self):
        routes = {cb_book_url("SOL-USD"): cb_book(200.0, 900_000, 200_000),     # leans to buyers
                  cb_book_url("BTC-USD"): cb_book(120000.0, 500_000, 400_000),  # balanced
                  cb_book_url("ETH-USD"): cb_book(4500.0, 15_000, 2_000),       # too thin
                  cb_book_url("DOGE-USD"): cb_book(0.25, 50_000, 300_000)}      # leans to sellers
        syms = {s: "crypto" for s in ("BTC-USD", "SOL-USD", "ETH-USD", "DOGE-USD")}
        items = {i.symbols[0]: i for i in crypto.CoinbaseBook().fetch(ctx(routes, syms))}
        self.assertEqual(set(items), {"SOL-USD", "DOGE-USD"})
        sol = items["SOL-USD"]
        self.assertGoodItem(sol)
        self.assertEqual((sol.direction, sol.confidence, sol.catalyst, sol.kind), (1, 0.5, "market_structure", "market"))
        self.assertEqual(sol.url, f"data://coinbase_book/SOL-USD/{HOUR}")
        self.assertAlmostEqual(sol.facts["bid_usd"], 900_000, delta=1)
        self.assertAlmostEqual(sol.facts["imbalance"], 7 / 11, places=3)
        self.assertIn("leans to buyers", sol.title)
        self.assertEqual(items["DOGE-USD"].direction, -1)

    def test_partial_failure_and_odd_books(self):
        routes = {cb_book_url("SOL-USD"): cb_book(200.0, 900_000, 200_000),
                  cb_book_url("ETH-USD"): {"bids": [["4600", "100", 1]], "asks": [["4500", "100", 1]]},   # crossed
                  cb_book_url("ADA-USD"): {"bids": [["x", "1", 1], [], "bad"], "asks": []},
                  cb_book_url("DOT-USD"): {"message": "NotFound"}}
        syms = {s: "crypto" for s in ("BTC-USD", "SOL-USD", "ETH-USD", "ADA-USD", "DOT-USD")}
        c = ctx(routes, syms)
        items = crypto.CoinbaseBook().fetch(c)         # BTC-USD 404s, DOT-USD has no book: logged, not fatal
        self.assertEqual([i.symbols[0] for i in items], ["SOL-USD"])
        self.assertTrue(c.state.get("_log"))
        with self.assertRaises(SourceError):
            crypto.CoinbaseBook().fetch(ctx({}, {"BTC-USD": "crypto"}))
        with self.assertRaises(RateLimited):
            crypto.CoinbaseBook().fetch(ctx({cb_book_url("BTC-USD"): RateLimited("429", 60)}, {"BTC-USD": "crypto"}))
        # caps itself at max_symbols_per_run
        many = {f"C{i}-USD": "crypto" for i in range(10)}
        c = ctx({}, many)
        try:
            crypto.CoinbaseBook().fetch(c)
        except SourceError:
            pass
        self.assertEqual(len(c.http.calls), 6)

    def test_kraken_book(self):
        def kr(mid, bid_usd, ask_usd):
            b = cb_book(mid, bid_usd, ask_usd)
            return {"error": [], "result": {"XXBTZUSD": {"asks": [[p, s, 1792300580] for p, s, _ in b["asks"]],
                                                         "bids": [[p, s, 1792300580] for p, s, _ in b["bids"]]}}}
        u = "https://api.kraken.com/0/public/Depth?pair={}USD&count=200".format
        routes = {u("XBT"): kr(120000.0, 100_000, 600_000), u("XDG"): kr(0.25, 400_000, 400_000),
                  u("SOL"): {"error": ["EQuery:Unknown asset pair"]}}
        syms = {"BTC-USD": "crypto", "DOGE-USD": "crypto", "SOL-USD": "crypto"}
        items = crypto.KrakenBook().fetch(ctx(routes, syms))
        self.assertEqual([(i.symbols, i.direction, i.source) for i in items], [(["BTC-USD"], -1, "kraken_book")])
        self.assertIn("(Kraken)", items[0].title)


# ---------------------------------------------------------------------------- fear & greed

FNG = "https://api.alternative.me/fng/?limit=2"


def fng(v, prev, ts=1792281600):   # 2026-10-18T00:00:00Z
    return {"name": "Fear and Greed Index", "data": [
        {"value": str(v), "value_classification": "Extreme Fear" if v <= 25 else "Neutral", "timestamp": str(ts),
         "time_until_update": "67410"},
        {"value": str(prev), "value_classification": "Fear", "timestamp": str(ts - 86400)}],
        "metadata": {"error": None}}


class TestFearGreed(Base):
    def test_contrarian(self):
        items = crypto.FearGreedCrypto().fetch(ctx({FNG: fng(15, 30)}, {"SOL-USD": "crypto", "NVDA": "stock"}))
        self.assertEqual(len(items), 1)
        it = items[0]
        self.assertGoodItem(it)
        self.assertEqual((it.symbols, it.direction, it.catalyst, it.kind, it.confidence),
                         (["BTC-USD", "SOL-USD"], 1, "sentiment", "sentiment", 0.45))
        self.assertEqual(it.url, f"data://fear_greed/{DAY}")
        self.assertEqual(it.title, "Crypto Fear & Greed Index at 15 (Extreme Fear), down from 30 the day before")
        self.assertEqual((it.facts["value"], it.facts["previous"]), (15.0, 30.0))
        self.assertEqual(crypto.FearGreedCrypto().fetch(ctx({FNG: fng(85, 70)}))[0].direction, -1)
        self.assertEqual(crypto.FearGreedCrypto().fetch(ctx({FNG: fng(50, 50)}))[0].direction, 0)

    def test_odd(self):
        self.assertEqual(crypto.FearGreedCrypto().fetch(ctx({FNG: fng(50, 50, ts=1792281600 - 5 * 86400)})), [])
        one = {"data": [{"value": "40", "timestamp": "1792281600"}], "metadata": {"error": None}}
        self.assertIsNone(crypto.FearGreedCrypto().fetch(ctx({FNG: one}))[0].facts["previous"])
        for bad in ({"data": [], "metadata": {"error": "Limit exceeded"}}, {"data": [{"value": "abc"}]}, [],
                    {"data": [{"value": "140", "timestamp": "1792281600"}]}):
            with self.assertRaises(SourceError):
                crypto.FearGreedCrypto().fetch(ctx({FNG: bad}))


# ---------------------------------------------------------------------------- listings & trending

class TestListings(Base):
    def test_coinbase_listings_diff(self):
        a, st = crypto.CoinbaseListings(), {}
        self.assertEqual(a.fetch(ctx({CB_PRODUCTS: cb_products()}, state=st)), [])     # first run seeds
        self.assertIn("SOL-USDC", st["known"])
        self.assertNotIn("ETH-BTC", st["known"])
        new = cb_products() + [cb_product("ZEPH-USD"), cb_product("ZEPH-USDC"), cb_product("ETH-USDC"),
                               cb_product("ZEPH-EUR"), cb_product("OLD-USD", status="delisted")]
        items = a.fetch(ctx({CB_PRODUCTS: new}, state=st, now=T0 + 3600))
        self.assertEqual(len(items), 1)                   # ETH already had a USD pair; EUR and delisted don't count
        it = items[0]
        self.assertGoodItem(it, T0 + 3600)
        self.assertEqual((it.title, it.symbols, it.catalyst, it.direction, it.confidence, it.kind, it.official),
                         ("Coinbase lists ZEPH-USD", ["ZEPH-USD"], "listing", 1, 0.8, "discovery", True))
        self.assertEqual(it.facts, {"product_id": "ZEPH-USD"})
        self.assertEqual(a.fetch(ctx({CB_PRODUCTS: new}, state=st, now=T0 + 7200)), [])
        # a short, partial listing then the full one again: nothing is "new"
        a.fetch(ctx({CB_PRODUCTS: [cb_product("BTC-USD")]}, state=st))
        self.assertEqual(a.fetch(ctx({CB_PRODUCTS: new}, state=st)), [])

    def test_glitch_guard(self):
        st = {"known": ["BTC-USD"]}
        flood = [cb_product(f"N{i}X-USD") for i in range(40)]
        self.assertEqual(crypto.CoinbaseListings().fetch(ctx({CB_PRODUCTS: flood}, state=st)), [])
        self.assertIn("N5X-USD", st["known"])

    def test_kraken_listings(self):
        a, st = crypto.KrakenListings(), {}
        self.assertEqual(a.fetch(ctx({KR_PAIRS: kr_pairs()}, state=st)), [])
        items = a.fetch(ctx({KR_PAIRS: kr_pairs(_extra=["NOVA"])}, state=st))
        self.assertEqual([(i.title, i.symbols, i.source) for i in items], [("Kraken lists NOVA-USD", ["NOVA-USD"], "kraken_listings")])


CG = "https://api.coingecko.com/api/v3/search/trending"


def cg_trending():
    def coin(i, cid, sym, name, mcr, ch):
        return {"item": {"id": cid, "coin_id": 1000 + i, "name": name, "symbol": sym, "market_cap_rank": mcr,
                         "thumb": "https://coin-images.coingecko.com/x.png", "small": "", "large": "", "slug": cid,
                         "price_btc": 1.2e-7, "score": i,
                         "data": {"price": 0.0123, "price_btc": "0.000000123", "price_change_percentage_24h":
                                  {"aed": ch, "usd": ch, "btc": ch}, "market_cap": "$1,234,567,890",
                                  "total_volume": "$98,765,432", "sparkline": "https://x/svg",
                                  "content": None}}}
    return {"coins": [coin(0, "novacoin", "nova", "Novacoin", 87, 12.44), coin(1, "obscure-thing", "OBS", "Obscure", None, 5.0),
                      coin(2, "deep-small", "DSM", "Deep Small", 1200, 50.0), coin(3, "solana", "SOL", "Solana", 6, -1.2),
                      coin(4, "weird", "W!RD", "Weird", 50, 1.0), {"item": None}, "junk"],
            "nfts": [{"id": "x", "name": "Some NFT"}], "categories": [{"id": 1, "name": "Meme"}]}


class TestTrending(Base):
    def test_trending(self):
        items = crypto.CoinGeckoTrending().fetch(ctx({CG: cg_trending()}))
        self.assertEqual([i.symbols for i in items], [["NOVA-USD"], ["SOL-USD"]])
        nova = items[0]
        self.assertGoodItem(nova)
        self.assertEqual((nova.catalyst, nova.kind, nova.direction), ("sentiment", "discovery", 0))
        self.assertEqual(nova.facts, {"rank": 1, "market_cap_rank": 87, "change_24h": 12.44, "coingecko_id": "novacoin"})
        self.assertEqual(nova.title, "Novacoin (NOVA) is trending on CoinGecko (#1 in searches, +12% in 24h)")
        self.assertEqual(nova.url, f"data://coingecko_trending/novacoin/{DAY}")
        self.assertEqual(items[1].facts["rank"], 4)
        with self.assertRaises(SourceError):
            crypto.CoinGeckoTrending().fetch(ctx({CG: {"status": {"error_code": 429}}}))
        self.assertEqual(crypto.CoinGeckoTrending().fetch(ctx({CG: {"coins": []}})), [])


if __name__ == "__main__":
    unittest.main()
