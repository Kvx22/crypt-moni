"""python -m unittest tests.test_research_text"""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from cryptomoni.research import aliases, text  # noqa: E402
from cryptomoni.research.aliases import Aliases, discover  # noqa: E402
from cryptomoni.research.http import SourceError  # noqa: E402
from cryptomoni.research.text import (clean, domain, fingerprint, hamming, html_to_text,  # noqa: E402
                                      parse_feed, parse_ts, same_story, tokens)

T0 = 1792300589.0   # 2026-10-18T05:16:29Z

GOOGLE_NEWS_RSS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<rss xmlns:media="http://search.yahoo.com/mrss/" version="2.0"><channel>
<generator>NFE/5.0</generator><title>"solana" - Google News</title>
<link>https://news.google.com/search?q=solana</link><language>en-US</language>
<item><title>Solana ETF sees record inflows - CoinDesk</title>
<link>https://news.google.com/rss/articles/CBMiAAA?oc=5</link>
<guid isPermaLink="false">CBMiAAA</guid><pubDate>Sat, 18 Oct 2026 05:16:29 GMT</pubDate>
<description>&lt;a href="https://news.google.com/rss/articles/CBMiAAA?oc=5" target="_blank"&gt;Solana ETF sees record inflows&lt;/a&gt;&amp;nbsp;&amp;nbsp;&lt;font color="#6f6f6f"&gt;CoinDesk&lt;/font&gt;</description>
<source url="https://www.coindesk.com">CoinDesk</source>
<media:title>shadowing title</media:title></item>
<item><title></title><link></link></item>
</channel></rss>"""

PLAIN_RSS = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:content="http://purl.org/rss/1.0/modules/content/" xmlns:dc="http://purl.org/dc/elements/1.1/">
<channel><title>CoinDesk: Bitcoin, Ethereum, Crypto News</title><link>https://www.coindesk.com</link>
<item><title><![CDATA[Bitcoin <b>tops</b> $100K]]></title><link>https://www.coindesk.com/markets/2026/10/18/btc</link>
<guid isPermaLink="true">https://www.coindesk.com/markets/2026/10/18/btc</guid>
<dc:date>2026-10-18T05:16:29Z</dc:date>
<category>Markets</category><category>Bitcoin</category><category>Markets</category>
<content:encoded><![CDATA[<p>Bitcoin rose 5% on Saturday.</p><script>alert(1)</script>]]></content:encoded></item>
<item><title>Permalink only</title><guid>https://example.com/a/1</guid><description>x</description></item>
</channel></rss>"""

ATOM = """<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"><title>SEC filings</title>
<link rel="self" href="https://www.sec.gov/feed.atom"/><link href="https://www.sec.gov/"/>
<entry><title type="html">8-K - NVIDIA CORP (0001045810)</title>
<link rel="edit" href="https://www.sec.gov/edit/1"/>
<link rel="alternate" type="text/html" href="https://www.sec.gov/Archives/edgar/data/1045810/x-index.htm"/>
<id>urn:tag:sec.gov,2008:accession-number=0001045810-26-000123</id>
<updated>2026-10-18T07:16:29+02:00</updated><published>2026-10-18T05:16:29.500Z</published>
<summary type="html">&lt;b&gt;Filed:&lt;/b&gt; 2026-10-18 &amp;lt;b&amp;gt;Item 2.02&amp;lt;/b&amp;gt;</summary>
<category term="8-K" label="form type"/>
<source><title>Reuters</title><link rel="alternate" href="https://www.reuters.com/"/></source></entry>
</feed>"""

RDF = """<?xml version="1.0"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" xmlns="http://purl.org/rss/1.0/"
 xmlns:dc="http://purl.org/dc/elements/1.1/">
<channel rdf:about="https://example.org/"><title>Example RDF</title><link>https://example.org/</link></channel>
<item rdf:about="https://example.org/story/1"><title>Ethereum upgrade ships</title>
<link>https://example.org/story/1</link><description>The upgrade went live.</description>
<dc:date>2026-10-18T05:16:29+00:00</dc:date><dc:subject>ethereum</dc:subject></item>
</rdf:RDF>"""

ARTICLE_HTML = """<!doctype html><html><head><title>Fallback title | Site</title>
<meta property="og:title" content="Solana validators vote on fee change">
<style>p{color:red}</style><script>var x = "<p>not text</p>";</script></head>
<body><header><nav><ul><li><a href="/">Home</a></li><li><a href="/m">Markets</a></li></ul></nav></header>
<div class="sidebar"><p><a href="/1">Trending story one that is long enough to count as text</a></p></div>
<article><h1>Solana validators vote</h1>
<p>Solana validators began voting on Friday on a proposal that would change how transaction fees are split.</p>
<div class="ad">Advertisement</div>
<p>The proposal needs two thirds of stake to pass, and early tallies suggest it will clear that bar.</p>
<p><a href="/share">Share</a></p></article>
<footer><p>Copyright 2026 Example Media. All rights reserved worldwide forever.</p></footer>
<svg><title>icon</title></svg></body></html>"""

DENSE_HTML = """<html><head><title>Exchange halts withdrawals</title></head><body>
<div id="menu"><a href="/a">News</a> <a href="/b">Prices</a> <a href="/c">Learn</a></div>
<div id="main"><div class="body">
<p>The exchange halted withdrawals late on Thursday, citing a wallet maintenance issue it did not explain.</p>
<p>Users reported balances were still visible, but no transfers had gone out for several hours since the halt.</p>
<h2>What we know</h2><p>Short.</p>
<li>Withdrawals paused across all supported networks, including Bitcoin and Ethereum.</li>
</div></div>
<div id="related"><p>Related: something else entirely different from the story here.</p></div>
</body></html>"""


class CleanAndDates(unittest.TestCase):
    def test_clean(self):
        self.assertEqual(clean("<p>Hi&nbsp; <b>there</b>\n\n &amp; you</p>"), "Hi there & you")
        self.assertEqual(clean("&lt;p&gt;double &amp;amp; escaped&lt;/p&gt;"), "double & escaped")
        self.assertEqual(clean("a <script>evil()</script> b <!-- c -->"), "a b")
        self.assertEqual(clean("1 < 2 and 3 > 2"), "1 < 2 and 3 > 2")
        for junk in (None, 5, "", "   ", ["x"]):
            self.assertEqual(clean(junk), "")
        self.assertEqual(clean(b"caf\xc3\xa9"), "caf\u00e9")

    def test_parse_ts_formats(self):
        for s in ("Sat, 18 Oct 2026 05:16:29 GMT", "Sat, 18 Oct 2026 05:16:29 +0000",
                  "18 Oct 2026 01:16:29 -0400", "Sat, 18 Oct 2026 05:16:29 -0000",
                  "2026-10-18T05:16:29Z", "2026-10-18T05:16:29+00:00", "2026-10-18T07:16:29+02:00",
                  "2026-10-18 05:16:29", "2026-10-18T05:16:29UTC", "2026-10-18T00:16:29-0500"):
            self.assertEqual(parse_ts(s), T0, s)
        self.assertAlmostEqual(parse_ts("2026-10-18T05:16:29.123456789Z"), T0 + 0.123456, places=5)
        self.assertEqual(parse_ts("2026-10-18"), T0 - (5 * 3600 + 16 * 60 + 29))
        self.assertEqual(parse_ts(" 2026-10-18T05:16:29Z \n"), T0)

    def test_parse_ts_boundaries_and_junk(self):
        self.assertEqual(parse_ts("2024-02-29T00:00:00Z"), 1709164800.0)   # leap day
        self.assertEqual(parse_ts("2026-12-31T23:59:59Z") + 1, parse_ts("2027-01-01T00:00:00Z"))
        for bad in (None, "", "garbage", "2026-02-30T00:00:00Z", "2026-13-01", "2026-10-18T25:00:00Z",
                    "Sun, 99 Foo 2026", "x" * 500, True, -5, "0000-01-01T00:00:00Z", [], "1969-12-31T00:00:00Z"):
            self.assertIsNone(parse_ts(bad), bad)
        self.assertEqual(parse_ts(T0), T0)


class Feeds(unittest.TestCase):
    def test_google_news_rss(self):
        es = parse_feed(GOOGLE_NEWS_RSS)
        self.assertEqual(len(es), 1)   # the empty item is skipped
        e = es[0]
        self.assertEqual(e["title"], "Solana ETF sees record inflows - CoinDesk")
        self.assertEqual(e["link"], "https://news.google.com/rss/articles/CBMiAAA?oc=5")
        self.assertEqual(e["guid"], "CBMiAAA")
        self.assertEqual(e["published_ts"], T0)
        self.assertEqual((e["source_name"], e["source_url"]), ("CoinDesk", "https://www.coindesk.com"))
        self.assertEqual(e["summary"], "Solana ETF sees record inflows CoinDesk")
        self.assertEqual(set(e), {"title", "link", "summary", "published_ts", "guid", "source_name",
                                  "source_url", "categories"})

    def test_plain_rss_with_content_encoded(self):
        es = parse_feed(PLAIN_RSS)
        self.assertEqual(len(es), 2)
        e = es[0]
        self.assertEqual(e["title"], "Bitcoin tops $100K")
        self.assertEqual(e["summary"], "Bitcoin rose 5% on Saturday.")
        self.assertEqual(e["published_ts"], T0)
        self.assertEqual(e["categories"], ["Markets", "Bitcoin"])
        self.assertEqual(e["source_name"], "CoinDesk: Bitcoin, Ethereum, Crypto News")
        self.assertEqual(e["source_url"], "https://www.coindesk.com")
        self.assertEqual(es[1]["link"], "https://example.com/a/1")   # permalink guid stands in for link
        self.assertIsNone(es[1]["published_ts"])

    def test_atom(self):
        (e,) = parse_feed(ATOM)
        self.assertEqual(e["link"], "https://www.sec.gov/Archives/edgar/data/1045810/x-index.htm")
        self.assertEqual(e["title"], "8-K - NVIDIA CORP (0001045810)")
        self.assertTrue(e["guid"].startswith("urn:tag:sec.gov"))
        self.assertAlmostEqual(e["published_ts"], T0 + 0.5)   # <published> beats <updated>
        self.assertEqual(e["summary"], "Filed: 2026-10-18 Item 2.02")
        self.assertEqual(e["categories"], ["8-K"])
        self.assertEqual((e["source_name"], e["source_url"]), ("Reuters", "https://www.reuters.com/"))
        (e,) = parse_feed(ATOM.replace("<source>", "<x>").replace("</source>", "</x>"))
        self.assertEqual((e["source_name"], e["source_url"]), ("SEC filings", "https://www.sec.gov/"))

    def test_rdf(self):
        (e,) = parse_feed(RDF)
        self.assertEqual(e["title"], "Ethereum upgrade ships")
        self.assertEqual(e["link"], "https://example.org/story/1")
        self.assertEqual(e["guid"], "https://example.org/story/1")
        self.assertEqual(e["published_ts"], T0)
        self.assertEqual(e["categories"], ["ethereum"])
        self.assertEqual(e["source_name"], "Example RDF")

    def test_bytes_and_bom(self):
        self.assertEqual(len(parse_feed(("\ufeff" + PLAIN_RSS).encode("utf-8"))), 2)
        latin = '<?xml version="1.0" encoding="ISO-8859-1"?><rss><channel><item><title>caf\xe9</title></item></channel></rss>'
        self.assertEqual(parse_feed(latin.encode("latin-1"))[0]["title"], "caf\u00e9")

    def test_repair(self):
        broken = "<rss><channel><item><title>AT&T & Bitcoin\x01 &nbsp;news</title>" \
                 "<link>https://x.com/a?b=1&c=2</link></item></channel></rss>"
        (e,) = parse_feed(broken)
        self.assertEqual(e["title"], "AT&T & Bitcoin news")
        self.assertEqual(e["link"], "https://x.com/a?b=1&c=2")

    def test_entity_bomb_is_defused(self):
        bomb = ('<?xml version="1.0"?><!DOCTYPE lolz [<!ENTITY lol "lol"><!ENTITY lol2 "&lol;&lol;&lol;">]>'
                '<rss><channel><item><title>t &lol2;</title><link>https://x.com/</link></item></channel></rss>')
        (e,) = parse_feed(bomb)
        self.assertEqual(e["link"], "https://x.com/")

    def test_long_summary_cut(self):
        body = " ".join(["word"] * 1000)
        xml = f"<rss><channel><item><title>t</title><description>{body}</description></item></channel></rss>"
        s = parse_feed(xml)[0]["summary"]
        self.assertLessEqual(len(s), text.SUMMARY_MAX)
        self.assertTrue(s.endswith("\u2026"))

    def test_bad_input_raises_source_error(self):
        for bad in ("", "not xml at all", "<rss><channel><item>", None, b"\x00\x01", "<<<>>>"):
            with self.assertRaises(SourceError):
                parse_feed(bad)
        with self.assertRaises(SourceError):   # valid XML, but a web page
            parse_feed("<html><body><p>hi</p></body></html>")

    def test_empty_feed_and_odd_entries(self):
        self.assertEqual(parse_feed("<rss><channel><title>x</title></channel></rss>"), [])
        odd = ("<rss><channel><item><title>ok</title><link>javascript:alert(1)</link><pubDate>never</pubDate>"
               "<category/></item><item><description>only a summary here</description></item></channel></rss>")
        es = parse_feed(odd)
        self.assertEqual([e["link"] for e in es], ["", ""])
        self.assertEqual(es[1]["title"], "only a summary here")
        self.assertEqual(es[0]["categories"], [])


class Pages(unittest.TestCase):
    def test_article(self):
        title, body = html_to_text(ARTICLE_HTML)
        self.assertEqual(title, "Solana validators vote on fee change")
        self.assertIn("transaction fees are split", body)
        self.assertIn("two thirds of stake", body)
        self.assertEqual(len(body.split("\n")), 3)   # h1 + two paragraphs
        for junk in ("Home", "Trending", "not text", "Copyright", "Share", "icon", "Advertisement", "color"):
            self.assertNotIn(junk, body)

    def test_densest_container(self):
        title, body = html_to_text(DENSE_HTML)
        self.assertEqual(title, "Exchange halts withdrawals")
        lines = body.split("\n")
        self.assertTrue(lines[0].startswith("The exchange halted"))
        self.assertIn("What we know", lines)
        self.assertIn("Withdrawals paused across all supported networks, including Bitcoin and Ethereum.", lines)
        self.assertNotIn("Related", body)
        self.assertNotIn("Prices", body)

    def test_max_chars_and_junk(self):
        page = "<html><body><article>" + "".join(f"<p>{'Sentence number %d is here. ' % i * 4}</p>" for i in range(200)) + "</article></body></html>"
        _, body = html_to_text(page, max_chars=500)
        self.assertLessEqual(len(body), 500)
        self.assertEqual(html_to_text(""), ("", ""))
        self.assertEqual(html_to_text(None), ("", ""))
        t, b = html_to_text("<p>unclosed <b>tags <div>everywhere</p></span></article><<>>& stray text long enough to keep.")
        self.assertIsInstance(b, str)
        t, b = html_to_text("<html><head><title>T\u00e9st \u2603</title></head><body><p>" + "\u65e5\u672c\u8a9e\u306e\u8a18\u4e8b\u3067\u3059\u3002" * 9 + "</p></body></html>")
        self.assertEqual(t, "T\u00e9st \u2603")
        self.assertIn("\u65e5\u672c\u8a9e", b)


class UrlsAndTokens(unittest.TestCase):
    def test_domain(self):
        self.assertEqual(domain("https://www.coindesk.com/x"), "coindesk.com")
        self.assertEqual(domain("https://m.youtube.com/watch?v=1"), "youtube.com")
        self.assertEqual(domain("HTTP://user:pw@WWW.Example.COM:8080/p"), "example.com")
        self.assertEqual(domain("news.google.com/rss"), "news.google.com")
        self.assertEqual(domain("https://www.com/"), "www.com")
        self.assertEqual(domain("https://[::1]:80/"), "::1")
        for bad in (None, "", 7, "http://[bad", "   "):
            self.assertEqual(domain(bad), "")

    def test_tokens(self):
        self.assertEqual(tokens("$SOL up 5.5% on Solana's news; I said a 7"),
                         ["$sol", "up", "5.5%", "on", "solana's", "news", "said", "7"])
        self.assertEqual(tokens("1inch 1,000 BTC_USD $100M"), ["1inch", "1,000", "btc", "usd", "100m"])
        self.assertEqual(tokens("\u00dcn\u00efcode Stra\u00dfe \u65e5\u672c"), ["\u00fcn\u00efcode", "stra\u00dfe", "\u65e5\u672c"])
        self.assertEqual(tokens(None), [])
        self.assertEqual(tokens(""), [])


class Fingerprints(unittest.TestCase):
    STORY = ("Bitcoin climbed above $100,000 for the first time on Saturday as spot ETF inflows hit a record "
             "and short sellers were forced to cover. Analysts said the move was driven by steady institutional "
             "demand and a drop in exchange balances, which fell to a five-year low this week.")

    def test_syndicated_copies_match(self):
        a = fingerprint("Bitcoin tops $100K as ETF inflows hit record - CoinDesk", self.STORY)
        b = fingerprint("Bitcoin tops $100K as ETF inflows hit record | Yahoo Finance", "<p>" + self.STORY + "</p>")
        c = fingerprint("Bitcoin Tops $100K as ETF Inflows Hit Record", self.STORY + " Read more at our site.")
        self.assertEqual(len(a), 16)
        int(a, 16)
        self.assertTrue(same_story(a, b), hamming(a, b))
        self.assertTrue(same_story(a, c), hamming(a, c))

    def test_different_stories_do_not(self):
        a = fingerprint("Bitcoin tops $100K as ETF inflows hit record", self.STORY)
        others = [
            fingerprint("Solana validators vote on fee change", "Solana validators began voting on Friday on a "
                        "proposal that would change how transaction fees are split between validators and stakers."),
            fingerprint("Nvidia beats earnings estimates", "Nvidia reported revenue well above forecasts on "
                        "Wednesday, helped by data-centre demand, and guided higher for the next quarter."),
            fingerprint("Exchange halts withdrawals", "The exchange halted withdrawals late on Thursday, citing "
                        "a wallet maintenance issue it did not explain to users."),
        ]
        for o in others:
            self.assertFalse(same_story(a, o), hamming(a, o))
        self.assertFalse(same_story(others[0], others[1]))

    def test_stable_and_edge_cases(self):
        self.assertEqual(fingerprint("Same title", "same body"), fingerprint("Same title", "same body"))
        self.assertEqual(fingerprint("", ""), "0" * 16)
        self.assertEqual(fingerprint(None, None), "0" * 16)
        self.assertFalse(same_story(fingerprint("", ""), fingerprint("", "")))
        self.assertEqual(len(fingerprint("one")), 16)
        self.assertEqual(hamming("0" * 16, "f" * 16), 64)
        self.assertEqual(hamming("abc", "0" * 16), 64)
        self.assertEqual(hamming(None, "0" * 16), 64)
        self.assertEqual(hamming("zz" * 8, "0" * 16), 64)


SUBSCRIBED = {s: "crypto" for s in ("BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "DOGE-USD", "ADA-USD", "AVAX-USD",
                                    "LINK-USD", "LTC-USD", "BCH-USD", "DOT-USD", "AAVE-USD", "OP-USD", "POL-USD")}
SUBSCRIBED.update({s: "stock" for s in ("SPY", "NVDA", "IBIT", "V", "META", "DIA", "GDX", "AAPL")})


class AliasTables(unittest.TestCase):
    def test_tables_cover_the_universe(self):
        import json
        from cryptomoni import universe
        with open(os.path.join(ROOT, "config.json")) as f:
            crypto = json.load(f)["universe"]
        for s in crypto + universe.CRYPTO_CANDIDATES:
            self.assertIn(s.split("-")[0], aliases.CRYPTO_NAMES, s)
        with open(os.path.join(ROOT, "config.robinhood.json")) as f:
            stocks = json.load(f)["stocks"]["symbols"]
        for s in stocks + universe.STOCK_CANDIDATES:
            self.assertIn(s, aliases.STOCK_NAMES, s)
        for t in ("TAO", "TON", "TRX", "BNB", "PEPE", "WIF", "BONK", "SEI", "JUP", "PYTH", "ONDO"):
            self.assertIn(t, aliases.CRYPTO_NAMES)
        for names in list(aliases.CRYPTO_NAMES.values()) + list(aliases.STOCK_NAMES.values()):
            self.assertTrue(names and all(isinstance(n, str) and n for n in names))


class Find(unittest.TestCase):
    def setUp(self):
        self.a = Aliases(SUBSCRIBED)

    def f(self, t):
        return self.a.find(t)

    def test_basic_forms(self):
        self.assertEqual(self.f("Bitcoin and Ethereum rally"), [("BTC-USD", "name"), ("ETH-USD", "name")])
        self.assertEqual(self.f("$sol and $ETH"), [("SOL-USD", "cashtag"), ("ETH-USD", "cashtag")])
        self.assertEqual(self.f("Nvidia Corp (NASDAQ: NVDA) said"), [("NVDA", "exchange")])
        self.assertEqual(self.f("Shares (NASDAQ:NVDA) rose"), [("NVDA", "exchange")])
        self.assertEqual(self.f("NYSE: V rose"), [("V", "exchange")])
        self.assertEqual(self.f("BTC-USD, ETH/USDT and SOL / USD"),
                         [("BTC-USD", "pair"), ("ETH-USD", "pair"), ("SOL-USD", "pair")])
        self.assertEqual(self.f("SOL jumps"), [("SOL-USD", "ticker")])

    def test_strongest_how_is_reported(self):
        self.assertEqual(self.f("Nvidia said... later (NASDAQ: NVDA)"), [("NVDA", "exchange")])
        self.assertEqual(self.f("BTC then Bitcoin then $BTC"), [("BTC-USD", "cashtag")])

    def test_case_rules(self):
        self.assertEqual(self.f("the sol was hot"), [])
        self.assertEqual(self.f("bitcoin and SOLANA"), [("BTC-USD", "name"), ("SOL-USD", "name")])
        self.assertEqual(self.f("btc eth"), [])

    def test_longest_match_wins(self):
        self.assertEqual(self.f("Bitcoin Cash forks again"), [("BCH-USD", "name")])
        self.assertEqual(self.f("iShares Bitcoin Trust inflows"), [("IBIT", "name")])
        only_btc = Aliases({"BTC-USD": "crypto"})
        self.assertEqual(only_btc.find("Bitcoin Cash forks again"), [])
        self.assertEqual(only_btc.find("Ethereum Classic and Bitcoin"), [("BTC-USD", "name")])

    def test_ambiguous_need_context(self):
        for t in ("Click the LINK below", "Polygon meshes in games", "an avalanche hit the town", "DOT the i",
                  "OP ed", "DOGE cuts federal staff", "the meta of the game", "V for Vendetta", "DIA de los muertos",
                  "Apple pie recipe"):
            self.assertEqual(self.f(t), [], t)
        self.assertEqual(self.f("LINK token jumps"), [("LINK-USD", "ticker")])
        self.assertEqual(self.f("Avalanche network upgrade"), [("AVAX-USD", "name")])
        self.assertEqual(self.f("the price of Polygon fell"), [("POL-USD", "name")])
        self.assertEqual(self.f("OP staking rewards rise"), [("OP-USD", "ticker")])
        self.assertEqual(self.f("Visa (V) shares rose"), [("V", "ticker")])
        self.assertEqual(self.f("Meta shares fall"), [("META", "name")])
        self.assertEqual(self.f("$LINK"), [("LINK-USD", "cashtag")])
        self.assertEqual(self.f("OP-USD"), [("OP-USD", "pair")])
        self.assertEqual(self.f("Meta Platforms reports"), [("META", "name")])

    def test_context_window_is_bounded(self):
        far = "Polygon " + "one two three four five six seven eight " + "token"
        self.assertEqual(self.f(far), [])
        near = "Polygon one two three four five token"
        self.assertEqual(self.f(near), [("POL-USD", "name")])

    def test_only_subscribed_and_unique(self):
        self.assertEqual(self.f("Bittensor and Cardano and Cardano"), [("ADA-USD", "name")])
        self.assertEqual(self.f("Tesla (NASDAQ: TSLA)"), [])

    def test_word_boundaries(self):
        self.assertEqual(self.f("SOLAR panels, BTCX, XBTC, Bitcoiners"), [])
        self.assertEqual(self.f("(BTC)"), [("BTC-USD", "ticker")])
        self.assertEqual(self.f("BTC."), [("BTC-USD", "ticker")])

    def test_junk_input(self):
        for bad in (None, "", 5, ["BTC"], "\x00\x01", "$", "$$$", "(NASDAQ: )", "-USD"):
            self.assertEqual(self.a.find(bad), [], bad)
        self.assertEqual(Aliases(None).find("Bitcoin"), [])
        self.assertEqual(Aliases({None: "crypto", 5: "x"}).find("Bitcoin"), [])

    def test_large_text_is_fast(self):
        import time
        t = time.time()
        found = self.a.find("Bitcoin LINK word $DOT " * 20000)
        self.assertEqual(found, [("BTC-USD", "name"), ("DOT-USD", "cashtag")])
        deep = "<html><body>" + "<div>" * 3000 + "<p>" + "Plenty of words in one paragraph. " * 50 + "</p>" + "</div>" * 3000
        self.assertIn("Plenty", html_to_text(deep)[1])
        self.assertEqual(html_to_text("<p>Some sentence that is long enough to keep.</p>", max_chars="x")[1],
                         "Some sentence that is long enough to keep.")
        self.assertLess(time.time() - t, 5.0)

    def test_unicode(self):
        self.assertEqual(self.f("\u201cBitcoin\u201d \u2014 Solana\u2019s rise, \u00e9t\u00e9 SOL\u2026"),
                         [("BTC-USD", "name"), ("SOL-USD", "name")])

    def test_extra_names_and_unknown_symbols(self):
        a = Aliases({"ZZZ-USD": "crypto", "BTC-USD": "crypto", "ACME": "stock"},
                    extra_names={"ZZZ-USD": ["Zeta Chain Z"], "BTC-USD": ["digital gold"], "bad": None, 3: ["x"]})
        self.assertEqual(a.find("Zeta Chain Z and digital gold"), [("ZZZ-USD", "name"), ("BTC-USD", "name")])
        self.assertEqual(a.find("ZZZ and ACME"), [("ZZZ-USD", "ticker"), ("ACME", "ticker")])
        self.assertEqual(a.names("ZZZ-USD"), ["Zeta Chain Z", "ZZZ"])

    def test_names_and_asset(self):
        self.assertEqual(self.a.names("SOL-USD"), ["Solana", "SOL"])
        self.assertEqual(self.a.names("NVDA"), ["Nvidia", "NVDA"])
        self.assertEqual(self.a.names("BCH-USD"), ["Bitcoin Cash", "BCH"])
        self.assertEqual(self.a.names(None), [])
        self.assertEqual(self.a.asset("SOL-USD"), "crypto")
        self.assertEqual(self.a.asset("NVDA"), "stock")
        self.assertEqual(self.a.asset("TAO-USD"), "crypto")   # not subscribed: guessed
        self.assertEqual(self.a.asset("MSFT"), "stock")
        self.assertIsNone(self.a.asset("ZZZQ"))
        self.assertIsNone(self.a.asset(None))


class Discover(unittest.TestCase):
    def test_cashtags_exchange_and_names(self):
        t = ("$TAO and Bittensor rallied while $NVDA held; Berkshire (NYSE: BRK.B) and NYSE:KO. "
             "Also $1, $USD, $100M, $5B, $usdt and $BTC.")
        self.assertEqual(discover(t, {"BTC-USD"}),
                         [("TAO-USD", "crypto", "cashtag"), ("NVDA", "stock", "cashtag"),
                          ("BRK.B", "stock", "exchange"), ("KO", "stock", "exchange")])

    def test_unknown_cashtag_needs_crypto_context(self):
        self.assertEqual(discover("$ZZZ token surges on airdrop", set()), [("ZZZ-USD", "crypto", "cashtag")])
        self.assertEqual(discover("$ZZZ shares rose", set()), [])

    def test_known_are_skipped_and_names_respect_ambiguity(self):
        self.assertEqual(discover("Solana and Kaspa", {"SOL-USD"}), [("KAS-USD", "crypto", "name")])
        self.assertEqual(discover("Jupiter is a planet; Stellar results", set()), [])
        self.assertEqual(discover("Bitcoin Cash", {"BTC-USD"}), [("BCH-USD", "crypto", "name")])

    def test_junk(self):
        for bad in (None, "", 7, "$", "$1 $2 $100M $USD", "(NASDAQ: )"):
            self.assertEqual(discover(bad, set()), [], bad)
        self.assertEqual(discover("Kaspa", None), [("KAS-USD", "crypto", "name")])


if __name__ == "__main__":
    unittest.main()
