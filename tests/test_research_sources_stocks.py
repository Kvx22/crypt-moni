"""python -m unittest tests.test_research_sources_stocks"""
import calendar
import json
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from cryptomoni.research import taxonomy  # noqa: E402
from cryptomoni.research.http import RateLimited, SourceError  # noqa: E402
from cryptomoni.research.sources import stocks  # noqa: E402
from cryptomoni.research.sources.base import Adapter, FetchContext  # noqa: E402
from cryptomoni.research.types import EVIDENCE, TEAMS, Item  # noqa: E402


def utc(*a):
    return float(calendar.timegm(a + (0,) * (6 - len(a))))


NOW = utc(2026, 10, 20, 14, 0, 0)    # Tue 2026-10-20 10:00 New York (EDT)


class FakeHttp:
    """routes: {url: payload | Exception}. str payloads come back from .text, anything else from .json
    (after a JSON round trip). Every call's url and keyword arguments are recorded."""
    def __init__(self, routes):
        self.routes, self.calls, self.kw = dict(routes), [], []

    def _get(self, url, kw):
        self.calls.append(url)
        self.kw.append(kw)
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
        try:   # like Http.json: a body that isn't JSON is a SourceError
            return json.loads(v) if isinstance(v, str) else json.loads(json.dumps(v))
        except ValueError as e:
            raise SourceError(f"bad JSON ({e})") from e


def ctx(routes, symbols=None, state=None, now=NOW, requested=()):
    return FetchContext(http=FakeHttp(routes), now=now,
                        symbols=dict(symbols if symbols is not None else {"AAPL": "stock"}),
                        requested=list(requested), state=state if state is not None else {})


class Base(unittest.TestCase):
    def assertGoodItem(self, it, now=NOW):
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
        json.dumps(it.d())


# ---------------------------------------------------------------------------- fixtures: SEC

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
TICKERS = {
    "0": {"cik_str": 1045810, "ticker": "NVDA", "title": "NVIDIA CORP"},
    "1": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
    "2": {"cik_str": 1067983, "ticker": "BRK-B", "title": "BERKSHIRE HATHAWAY INC"},
    "3": {"cik_str": 789019, "ticker": "MSFT", "title": "MICROSOFT CORP"},
}


def subs_url(cik):
    return f"https://data.sec.gov/submissions/CIK{cik:010d}.json"


# (accession, filingDate, acceptanceDateTime [New York time despite the Z], form, items, primaryDocument, description)
AAPL_FILINGS = [
    ("0000320193-26-000105", "2026-10-19", "2026-10-19T16:30:42.000Z", "8-K", "2.02,9.01", "aapl-20261019.htm", "8-K"),
    ("0001140361-26-038411", "2026-10-19", "2026-10-19T18:35:01.000Z", "4", "", "xslF345X05/wk-form4_1792447000.xml", "FORM 4"),
    ("0001140361-26-038377", "2026-10-16", "2026-10-16T19:02:11.000Z", "4", "", "xslF345X05/wk-form4_1792188000.xml", "FORM 4"),
    ("0001193125-26-401122", "2026-10-15", "2026-10-15T08:00:12.000Z", "S-3", "", "d901234ds3.htm", "S-3ASR"),
    ("0000320193-26-000101", "2026-10-14", "2026-10-14T16:31:05.000Z", "8-K", "5.02", "aapl-20261014.htm", "8-K"),
    ("0000320193-26-000099", "2026-10-13", "2026-10-13T06:01:36.000Z", "D", "", "primary_doc.xml", ""),
    ("0000320193-26-000090", "2026-08-01", "2026-08-01T16:30:00.000Z", "10-Q", "", "aapl-20260627.htm", "10-Q"),
]


def submissions_json(name, cik, rows):
    cols = list(zip(*rows))
    return {
        "cik": str(cik), "entityType": "operating", "sic": "3571", "name": name, "tickers": ["AAPL"],
        "exchanges": ["Nasdaq"], "fiscalYearEnd": "0926",
        "filings": {"recent": {
            "accessionNumber": list(cols[0]), "filingDate": list(cols[1]), "reportDate": [""] * len(rows),
            "acceptanceDateTime": list(cols[2]), "act": ["34"] * len(rows), "form": list(cols[3]),
            "fileNumber": ["001-36743"] * len(rows), "filmNumber": [""] * len(rows), "items": list(cols[4]),
            "core_type": list(cols[3]), "size": [12345] * len(rows), "isXBRL": [0] * len(rows),
            "isInlineXBRL": [0] * len(rows), "primaryDocument": list(cols[5]),
            "primaryDocDescription": list(cols[6])},
            "files": []},
    }


AAPL_SUBS = submissions_json("Apple Inc.", 320193, AAPL_FILINGS)


def form4_xml(owner, director, officer, title, txs, plan=False):
    """A Form 4 ownershipDocument in SEC's real layout (values wrapped in <value>). txs: (code, shares, price)."""
    rows = "".join(f"""
        <nonDerivativeTransaction>
            <securityTitle><value>Common Stock</value></securityTitle>
            <transactionDate><value>2026-10-16</value></transactionDate>
            <transactionCoding>
                <transactionFormType>4</transactionFormType>
                <transactionCode>{code}</transactionCode>
                <equitySwapInvolved>0</equitySwapInvolved>
            </transactionCoding>
            <transactionAmounts>
                <transactionShares><value>{sh}</value></transactionShares>
                <transactionPricePerShare><value>{px}</value></transactionPricePerShare>
                <transactionAcquiredDisposedCode><value>{'D' if code == 'S' else 'A'}</value></transactionAcquiredDisposedCode>
            </transactionAmounts>
            <postTransactionAmounts><sharesOwnedFollowingTransaction><value>100000</value></sharesOwnedFollowingTransaction></postTransactionAmounts>
            <ownershipNature><directOrIndirectOwnership><value>D</value></directOrIndirectOwnership></ownershipNature>
        </nonDerivativeTransaction>""" for code, sh, px in txs)
    return f"""<?xml version="1.0"?>
<ownershipDocument>
    <schemaVersion>X0508</schemaVersion>
    <documentType>4</documentType>
    <periodOfReport>2026-10-16</periodOfReport>
    <aff10b5One>{1 if plan else 0}</aff10b5One>
    <issuer>
        <issuerCik>0000320193</issuerCik>
        <issuerName>Apple Inc.</issuerName>
        <issuerTradingSymbol>AAPL</issuerTradingSymbol>
    </issuer>
    <reportingOwner>
        <reportingOwnerId>
            <rptOwnerCik>0001214156</rptOwnerCik>
            <rptOwnerName>{owner}</rptOwnerName>
        </reportingOwnerId>
        <reportingOwnerAddress><rptOwnerStreet1>ONE APPLE PARK WAY</rptOwnerStreet1><rptOwnerCity>CUPERTINO</rptOwnerCity></reportingOwnerAddress>
        <reportingOwnerRelationship>
            <isDirector>{director}</isDirector>
            <isOfficer>{officer}</isOfficer>
            <isTenPercentOwner>0</isTenPercentOwner>
            <isOther>0</isOther>
            {f'<officerTitle>{title}</officerTitle>' if title else ''}
        </reportingOwnerRelationship>
    </reportingOwner>
    <nonDerivativeTable>{rows}
    </nonDerivativeTable>
    <derivativeTable>
        <derivativeTransaction>
            <securityTitle><value>Restricted Stock Unit</value></securityTitle>
            <transactionCoding><transactionFormType>4</transactionFormType><transactionCode>M</transactionCode></transactionCoding>
            <transactionAmounts><transactionShares><value>90000</value></transactionShares>
            <transactionPricePerShare><value>100</value></transactionPricePerShare></transactionAmounts>
        </derivativeTransaction>
    </derivativeTable>
    <ownerSignature><signatureName>/s/ Jane Roe, Attorney-in-Fact</signatureName><signatureDate>2026-10-19</signatureDate></ownerSignature>
</ownershipDocument>
"""


# director: a $250k open-market purchase (P), a small sale (S) and a $5M grant (A) that must not count
FORM4_BUY = form4_xml("DOE JOHN A", 1, 0, "", [("P", "2,500", "100.00"), ("S", "1000", "101.00"), ("A", "50000", "100.00")])
# CFO: a $2.1M sale under a 10b5-1 plan, plus tax withholding (F) that must not count
FORM4_SELL = form4_xml("ROE JANE", 0, 1, "Chief Financial Officer", [("S", "20000", "105.00"), ("F", "8000", "105.00")],
                       plan=True)
ARCH = "https://www.sec.gov/Archives/edgar/data/"
F4_BUY_URL = ARCH + "320193/000114036126038411/wk-form4_1792447000.xml"
F4_SELL_URL = ARCH + "320193/000114036126038377/wk-form4_1792188000.xml"


def atom_url(t, form=""):
    return (f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={t}&type={form}&dateb=&owner=include"
            "&count=40&output=atom")


def atom_entry(form, acc, cik, updated, filed, summary_items="", items_desc=""):
    nodash = acc.replace("-", "")
    href = f"https://www.sec.gov/Archives/edgar/data/{cik}/{nodash}/{acc}-index.htm"
    summ = f" &lt;b&gt;Filed:&lt;/b&gt; {filed} &lt;b&gt;AccNo:&lt;/b&gt; {acc} &lt;b&gt;Size:&lt;/b&gt; 349 KB"
    summ += "".join(f"&lt;br&gt;Item {c}: {n}" for c, n in summary_items)
    return f"""
<entry>
<category label="form type" scheme="https://www.sec.gov/" term="{form}" />
<content type="text/xml">
<accession-number>{acc}</accession-number>
<act>34</act>
<file-number>001-36743</file-number>
<filing-date>{filed}</filing-date>
<filing-href>{href}</filing-href>
<filing-type>{form}</filing-type>
<form-name>Current report</form-name>
{f'<items-desc>{items_desc}</items-desc>' if items_desc else ''}
<size>349 KB</size>
</content>
<id>urn:tag:sec.gov,2008:accession-number={acc}</id>
<link href="{href}" rel="alternate" type="text/html" />
<summary type="html">{summ}</summary>
<title>{form}  - Current report</title>
<updated>{updated}</updated>
</entry>"""


def atom_feed(entries):
    return f"""<?xml version="1.0" encoding="ISO-8859-1" ?>
<feed xmlns="http://www.w3.org/2005/Atom">
<author><email>webmaster@sec.gov</email><name>Webmaster</name></author>
<company-info>
<addresses><address type="mailing"><city>CUPERTINO</city></address></addresses>
<cik>0000320193</cik>
<conformed-name>Apple Inc.</conformed-name>
</company-info>
<id>https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&amp;CIK=0000320193</id>
<link href="https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&amp;CIK=0000320193" rel="alternate" type="text/html" />
<title>Apple Inc.  (0000320193)</title>
<updated>2026-10-20T09:59:59-04:00</updated>
{''.join(entries)}
</feed>"""


ATOM_ALL = atom_feed([
    atom_entry("8-K", "0000320193-26-000105", 320193, "2026-10-19T16:30:42-04:00", "2026-10-19",
               [("2.02", "Results of Operations and Financial Condition"), ("9.01", "Financial Statements and Exhibits")],
               "items 2.02 and 9.01"),
    atom_entry("4", "0001140361-26-038411", 320193, "2026-10-19T18:35:01-04:00", "2026-10-19"),
    atom_entry("S-3ASR", "0001193125-26-401122", 320193, "2026-10-15T08:00:12-04:00", "2026-10-15"),
    atom_entry("10-Q", "0000320193-26-000090", 320193, "2026-08-01T16:30:00-04:00", "2026-08-01"),
])
ATOM_FORM4 = atom_feed([
    atom_entry("4", "0001140361-26-038411", 320193, "2026-10-19T18:35:01-04:00", "2026-10-19"),
    atom_entry("424B2", "0001193125-26-401999", 320193, "2026-10-19T17:00:00-04:00", "2026-10-19"),   # type=4 prefix match
])
NO_MATCH = ("<!DOCTYPE html><html><head><title>EDGAR Search Results</title></head><body>"
            "<center><h1>No matching Ticker Symbol.</h1></center></body></html>")


def full_submission(acc, xml):
    """The <accession>.txt full submission the Atom standby reads: SGML header + the XML document."""
    return (f"<SEC-DOCUMENT>{acc}.txt : 20261019\n<SEC-HEADER>{acc}.hdr.sgml : 20261019\n"
            "ACCESSION NUMBER:\t\t" + acc + "\nCONFORMED SUBMISSION TYPE:\t4\n</SEC-HEADER>\n<DOCUMENT>\n<TYPE>4\n"
            "<SEQUENCE>1\n<FILENAME>wk-form4_1792447000.xml\n<TEXT>\n<XML>\n" + xml + "</XML>\n</TEXT>\n</DOCUMENT>\n"
            "</SEC-DOCUMENT>\n")


# ---------------------------------------------------------------------------- catalog

class TestCatalog(Base):
    SLOTS = {   # name: (class, team, kind, role, standby_for, per_symbol, max_per_run)
        "sec_filings": ("SECFilings", "filings", "filing", "primary", None, True, 6),
        "sec_filings_atom": ("SECFilingsAtom", "filings", "filing", "standby", "sec_filings", True, 6),
        "sec_insider": ("SECInsider", "filings", "filing", "primary", None, True, 4),
        "sec_insider_atom": ("SECInsiderAtom", "filings", "filing", "standby", "sec_insider", True, 4),
        "earnings_calendar": ("EarningsCalendar", "filings", "filing", "primary", None, False, None),
        "sec_8k_earnings": ("SEC8KEarnings", "filings", "filing", "standby", "earnings_calendar", True, 6),
        "yahoo_rss": ("YahooRSS", "catalysts", "news", "primary", None, True, 8),
        "nasdaq_rss": ("NasdaqRSS", "catalysts", "news", "standby", "yahoo_rss", True, 8),
        "yahoo_vix": ("YahooVIX", "market", "market", "primary", None, False, None),
        "cboe_vix": ("CboeVIX", "market", "market", "standby", "yahoo_vix", False, None),
    }

    def test_slot_table(self):
        got = {c.name: c for c in stocks.ADAPTERS}
        self.assertEqual(set(got), set(self.SLOTS))
        self.assertEqual(len(stocks.ADAPTERS), len(got))
        for name, (cls, team, kind, role, sb, per, cap) in self.SLOTS.items():
            a = got[name]()
            self.assertIsInstance(a, Adapter)
            self.assertEqual(type(a).__name__, cls)
            self.assertEqual((a.team, a.kind, a.role, a.standby_for, a.per_symbol), (team, kind, role, sb, per), name)
            if cap:
                self.assertEqual(a.max_symbols_per_run, cap, name)
            self.assertEqual(tuple(a.assets), ("stock",))
            self.assertIn(a.team, TEAMS)
            self.assertTrue(a.domains and a.description, name)
            if sb:
                self.assertEqual(got[sb].team, a.team)
                self.assertEqual(got[sb].role, "primary")
        self.assertEqual(stocks.SECFilings.interval_s, 3600)
        self.assertEqual(stocks.SECInsider.interval_s, 3600)
        self.assertEqual(stocks.EarningsCalendar.interval_s, 6 * 3600)
        self.assertEqual(stocks.YahooRSS.interval_s, 1800)
        self.assertEqual(stocks.YahooVIX.interval_s, 3600)
        for c in (stocks.SECFilings, stocks.SECInsider, stocks.SECFilingsAtom, stocks.SECInsiderAtom):
            self.assertTrue(c.official)

    def test_catalog_loads_module(self):
        from cryptomoni.research import sources
        names = {a.name for a in sources.catalog({})}
        self.assertNotIn("stocks", sources.IMPORT_ERRORS)
        self.assertTrue(set(self.SLOTS) <= names)


# ---------------------------------------------------------------------------- SEC: tickers, labels, submissions

class TestCikMap(Base):
    def test_cache_only_subscribed_and_etfs_skip(self):
        syms = {"AAPL": "stock", "SPY": "stock", "BRK.B": "stock", "BTC-USD": "crypto"}
        c = ctx({TICKERS_URL: TICKERS}, syms)
        m = stocks.sec_cik_map(c)
        self.assertEqual(m, {"AAPL": (320193, "Apple Inc."), "BRK.B": (1067983, "BERKSHIRE HATHAWAY INC")})
        self.assertEqual(sorted(k for k in c.state if k.startswith("cik:")), ["cik:AAPL", "cik:BRK.B", "cik:SPY"])
        self.assertIsNone(c.state["cik:SPY"][0])        # remembered as "no CIK" so it is not looked up every run
        self.assertNotIn("cik:NVDA", c.state)           # never the whole map
        # within 7 days: served from state, no request
        c2 = ctx({}, syms, state=c.state, now=NOW + 6 * 86400)
        self.assertEqual(stocks.sec_cik_map(c2), m)
        self.assertEqual(c2.http.calls, [])
        # after 7 days: refreshed
        c3 = ctx({TICKERS_URL: TICKERS}, syms, state=c.state, now=NOW + 8 * 86400)
        stocks.sec_cik_map(c3)
        self.assertEqual(c3.http.calls, [TICKERS_URL])

    def test_bad_map(self):
        with self.assertRaises(SourceError):
            stocks.sec_cik_map(ctx({TICKERS_URL: "not json at all"}))
        with self.assertRaises(SourceError):
            stocks.sec_cik_map(ctx({TICKERS_URL: {"0": {"oops": 1}}}))


class TestLabels(Base):
    def test_8k_items_and_forms(self):
        L = stocks.label_filing
        self.assertEqual(L("8-K", ["2.02", "9.01"])[:4], ("earnings", 0, [], 0.8))
        self.assertEqual(L("8-K", ["1.01"])[0], "partnership")
        self.assertEqual(L("8-K", ["2.01", "9.01"])[0], "mna")
        self.assertEqual(L("8-K", ["1.03"])[:3], ("legal", -1, ["bankruptcy"]))
        self.assertEqual(L("8-K", ["4.02"])[:4], ("filing", -1, [], 0.7))
        self.assertEqual(L("8-K", ["5.02"])[:3], ("filing", 0, []))
        self.assertEqual(L("8-K", ["7.01"])[:2], ("filing", 0))
        self.assertEqual(L("8-K", ["8.01"])[:2], ("filing", 0))
        self.assertEqual(L("8-K", ["3.01"])[:3], ("delisting", -1, ["delisting"]))
        self.assertEqual(L("8-K", ["2.02", "1.03"])[0], "legal")     # the worst item decides
        self.assertEqual(L("8-K", ["9.01"])[:2], ("filing", 0))
        for f in ("S-1", "S-3", "S-3ASR", "424B5"):
            self.assertEqual(L(f, [])[:3], ("unlock", -1, ["dilution"]), f)
        self.assertEqual(L("SC 13D", [])[:2], ("insider", 1))
        self.assertEqual(L("10-Q", [])[:2], ("filing", 0))
        self.assertEqual(L("10-K", [])[:2], ("filing", 0))
        self.assertIsNone(L("4", []))
        self.assertIsNone(L("D", []))
        for lab in (L("8-K", ["1.03"]), L("S-3", [])):
            for f in lab[2]:
                self.assertIn(f, taxonomy.RISK_FLAGS)

    def test_submissions_parse(self):
        name, fs = stocks.submissions(ctx({subs_url(320193): AAPL_SUBS}), 320193)
        self.assertEqual(name, "Apple Inc.")
        f = fs[0]
        self.assertEqual((f["form"], f["items"], f["accession"]), ("8-K", ["2.02", "9.01"], "0000320193-26-000105"))
        # acceptanceDateTime is New York time with a misleading Z: 16:30:42 EDT = 20:30:42 UTC
        self.assertEqual(f["ts"], utc(2026, 10, 19, 20, 30, 42))
        with self.assertRaises(SourceError):
            stocks.submissions(ctx({subs_url(320193): {"cik": "320193", "filings": {}}}), 320193)


class TestSECFilings(Base):
    def routes(self):
        return {TICKERS_URL: TICKERS, subs_url(320193): AAPL_SUBS}

    def test_new_filings(self):
        c = ctx(self.routes(), {"AAPL": "stock", "SPY": "stock"})
        items = stocks.SECFilings().fetch(c)
        for it in items:
            self.assertGoodItem(it)
            self.assertEqual((it.symbols, it.kind, it.publisher, it.official), (["AAPL"], "filing", "sec.gov", True))
        by = {it.facts["accession"]: it for it in items}
        # form 4 and Form D are not filings-team news; the 10-Q is older than 10 days
        self.assertEqual(set(by), {"0000320193-26-000105", "0001193125-26-401122", "0000320193-26-000101"})
        e = by["0000320193-26-000105"]
        self.assertEqual((e.catalyst, e.direction, e.confidence), ("earnings", 0, 0.8))
        self.assertEqual(e.facts["items"], ["2.02", "9.01"])
        self.assertEqual(e.url, ARCH + "320193/000032019326000105/0000320193-26-000105-index.htm")
        self.assertIn("8-K", e.title)
        self.assertIn("Item 2.02", e.text)
        self.assertIn("Item 9.01", e.text)
        self.assertEqual(e.published_ts, utc(2026, 10, 19, 20, 30, 42))
        s3 = by["0001193125-26-401122"]
        self.assertEqual((s3.catalyst, s3.direction, s3.risk_flags), ("unlock", -1, ["dilution"]))
        self.assertEqual(by["0000320193-26-000101"].catalyst, "filing")
        # no submissions request for an ETF without a CIK
        self.assertNotIn(subs_url(0), c.http.calls)
        self.assertEqual(sum(u.startswith("https://data.sec.gov/") for u in c.http.calls), 1)
        # second run: nothing new
        c2 = ctx(self.routes(), {"AAPL": "stock"}, state=c.state)
        self.assertEqual(stocks.SECFilings().fetch(c2), [])

    def test_partial_and_total_failure(self):
        routes = dict(self.routes())
        routes[subs_url(1045810)] = SourceError("data.sec.gov: HTTP 500")
        c = ctx(routes, {"NVDA": "stock", "AAPL": "stock"})
        items = stocks.SECFilings().fetch(c)
        self.assertTrue(items and all(it.symbols == ["AAPL"] for it in items))
        self.assertTrue(c.state.get("_log"))
        with self.assertRaises(SourceError):
            stocks.SECFilings().fetch(ctx(routes, {"NVDA": "stock"}))
        routes[subs_url(1045810)] = RateLimited("data.sec.gov: HTTP 429", 60)
        with self.assertRaises(RateLimited):
            stocks.SECFilings().fetch(ctx(routes, {"NVDA": "stock", "AAPL": "stock"}))

    def test_caps_itself(self):
        many = {f"T{i}": "stock" for i in range(10)}
        tick = {str(i): {"cik_str": 1000 + i, "ticker": f"T{i}", "title": f"T{i} Corp"} for i in range(10)}
        c = ctx({TICKERS_URL: tick}, many)
        with self.assertRaises(SourceError):
            stocks.SECFilings().fetch(c)
        self.assertEqual(sum(u.startswith("https://data.sec.gov/") for u in c.http.calls), 6)

    def test_atom_standby(self):
        c = ctx({atom_url("AAPL"): ATOM_ALL, atom_url("SPY"): NO_MATCH}, {"AAPL": "stock", "SPY": "stock"})
        items = stocks.SECFilingsAtom().fetch(c)
        by = {it.facts["accession"]: it for it in items}
        self.assertEqual(set(by), {"0000320193-26-000105", "0001193125-26-401122"})
        e = by["0000320193-26-000105"]
        self.assertGoodItem(e)
        self.assertEqual((e.catalyst, e.source, e.facts["items"]), ("earnings", "sec_filings_atom", ["2.02", "9.01"]))
        self.assertEqual(e.url, ARCH + "320193/000032019326000105/0000320193-26-000105-index.htm")
        self.assertEqual(e.published_ts, utc(2026, 10, 19, 20, 30, 42))
        self.assertEqual(by["0001193125-26-401122"].risk_flags, ["dilution"])
        self.assertEqual(stocks.SECFilingsAtom().fetch(ctx({atom_url("AAPL"): ATOM_ALL}, state=c.state)), [])
        with self.assertRaises(SourceError):
            stocks.SECFilingsAtom().fetch(ctx({atom_url("AAPL"): "<rss><channel></channel></rss>"}))


# ---------------------------------------------------------------------------- SEC: Form 4

class TestInsider(Base):
    def test_parse_form4(self):
        f = stocks.parse_form4(FORM4_BUY)
        self.assertEqual(f["buy"], dict(shares=2500.0, value=250000.0, price=100.0))   # the A grant is not a buy
        self.assertEqual(f["sell"]["shares"], 1000.0)
        self.assertEqual((f["owner"], f["display"], f["role"], f["is_officer"]), ("DOE JOHN A", "J. Doe", "Director", False))
        g = stocks.parse_form4(FORM4_SELL)
        self.assertIsNone(g["buy"])
        self.assertEqual(g["sell"], dict(shares=20000.0, value=2100000.0, price=105.0))   # F withholding ignored
        self.assertEqual((g["role"], g["is_officer"], g["plan"]), ("Chief Financial Officer", True, True))
        only_grant = stocks.parse_form4(form4_xml("DOE JOHN A", 1, 0, "", [("A", "50000", "100"), ("M", "9000", "50")]))
        self.assertIsNone(stocks.insider_item("sec_insider", "AAPL", "https://x", NOW, only_grant))
        with self.assertRaises(SourceError):
            stocks.parse_form4("<html><body>not a form</body></html>")

    def test_thresholds(self):
        small = stocks.parse_form4(form4_xml("SMITH ANN", 1, 0, "", [("P", "500", "100")]))       # $50k buy
        self.assertIsNone(stocks.insider_item("s", "AAPL", "https://x", NOW, small))
        dsale = stocks.parse_form4(form4_xml("SMITH ANN", 1, 0, "", [("S", "50000", "100")]))     # director, $5M sale
        self.assertIsNone(stocks.insider_item("s", "AAPL", "https://x", NOW, dsale))           # sales count for officers only
        osale = stocks.parse_form4(form4_xml("SMITH ANN", 0, 1, "CEO", [("S", "9000", "100")]))   # officer, $900k
        self.assertIsNone(stocks.insider_item("s", "AAPL", "https://x", NOW, osale))

    def test_sec_insider(self):
        routes = {TICKERS_URL: TICKERS, subs_url(320193): AAPL_SUBS, F4_BUY_URL: FORM4_BUY, F4_SELL_URL: FORM4_SELL}
        c = ctx(routes)
        items = stocks.SECInsider().fetch(c)
        self.assertIn(F4_BUY_URL, c.http.calls)                     # xslF345X05/ prefix stripped: the raw XML
        by = {it.facts["code"]: it for it in items}
        self.assertEqual(set(by), {"P", "S"})
        buy, sell = by["P"], by["S"]
        for it in items:
            self.assertGoodItem(it)
            self.assertEqual((it.catalyst, it.kind, it.symbols, it.official), ("insider", "filing", ["AAPL"], True))
        self.assertEqual(buy.title, "Director J. Doe bought $250k of AAPL (Form 4)")
        self.assertEqual((buy.direction, buy.confidence), (1, 0.7))
        self.assertEqual(buy.facts, dict(code="P", shares=2500.0, price=100.0, value=250000.0, owner="DOE JOHN A",
                                         title="Director", plan_10b5_1=False))
        self.assertEqual(buy.url, ARCH + "320193/000114036126038411/0001140361-26-038411-index.htm")
        self.assertEqual((sell.direction, sell.confidence, sell.facts["value"]), (-1, 0.4, 2100000.0))
        self.assertIn("Chief Financial Officer J. Roe sold $2.1M of AAPL", sell.title)
        self.assertIn("10b5-1", sell.text)
        # seen: no refetch next run
        c2 = ctx(routes, state=c.state)
        self.assertEqual(stocks.SECInsider().fetch(c2), [])
        self.assertNotIn(F4_BUY_URL, c2.http.calls)

    def test_broken_document_is_skipped(self):
        routes = {TICKERS_URL: TICKERS, subs_url(320193): AAPL_SUBS, F4_BUY_URL: "<html>oops", F4_SELL_URL: FORM4_SELL}
        c = ctx(routes)
        items = stocks.SECInsider().fetch(c)
        self.assertEqual([it.facts["code"] for it in items], ["S"])
        self.assertTrue(c.state.get("_log"))

    def test_atom_standby(self):
        txt = ARCH + "320193/000114036126038411/0001140361-26-038411.txt"
        c = ctx({atom_url("AAPL", "4"): ATOM_FORM4, txt: full_submission("0001140361-26-038411", FORM4_BUY)})
        items = stocks.SECInsiderAtom().fetch(c)
        self.assertEqual([(it.source, it.facts["code"], it.direction) for it in items], [("sec_insider_atom", "P", 1)])
        self.assertEqual(items[0].title, "Director J. Doe bought $250k of AAPL (Form 4)")
        self.assertFalse(any("401999" in u for u in c.http.calls))   # the 424B2 from the prefix match is not read


# ---------------------------------------------------------------------------- earnings

def nasdaq_url(day):
    return f"https://api.nasdaq.com/api/calendar/earnings?date={day}"


def nasdaq(rows):
    return {"data": {"asOf": "Tue, Oct 20, 2026", "headers": {"symbol": "Symbol", "name": "Company Name"},
                     "rows": rows}, "message": None, "status": {"rCode": 200, "bCodeMessage": None, "developerMessage": None}}


def nrow(sym, name, when, eps="$1.78", n="10"):
    return {"lastYearRptDt": "10/30/2025", "lastYearEPS": "$1.64", "time": when, "symbol": sym, "name": name,
            "marketCap": "$3,712,345,678,901", "fiscalQuarterEnding": "Sep/2026", "epsForecast": eps, "noOfEsts": n}


class TestEarnings(Base):
    def routes(self):
        return {
            nasdaq_url("2026-10-20"): nasdaq([nrow("KO", "Coca-Cola Company (The)", "time-pre-market")]),
            nasdaq_url("2026-10-21"): nasdaq([nrow("NVDA", "NVIDIA Corporation", "time-after-hours", "$1.25", "38"),
                                               nrow("TSLA", "Tesla, Inc.", "time-after-hours"), "junk"]),
            nasdaq_url("2026-10-22"): nasdaq([nrow("AAPL", "Apple Inc.", "time-pre-market")]),
            nasdaq_url("2026-10-23"): {"data": {"asOf": "Fri, Oct 23, 2026", "rows": None}, "message": "No data",
                                       "status": {"rCode": 200}},
            nasdaq_url("2026-10-26"): nasdaq([nrow("MSFT", "Microsoft Corporation", "time-not-supplied", "", "")]),
        }

    def test_week_ahead(self):
        syms = {"NVDA": "stock", "AAPL": "stock", "MSFT": "stock", "SPY": "stock"}
        c = ctx(self.routes(), syms)
        items = stocks.EarningsCalendar().fetch(c)
        # Tue 20 .. Fri 23 and Mon 26: today plus the next 4 weekdays, never the weekend
        self.assertEqual(c.http.calls, [nasdaq_url(d) for d in ("2026-10-20", "2026-10-21", "2026-10-22",
                                                                  "2026-10-23", "2026-10-26")])
        for kw in c.http.kw:
            h = kw["headers"]
            self.assertEqual((h["Accept"], h["Origin"], h["Referer"]),
                             ("application/json", "https://www.nasdaq.com", "https://www.nasdaq.com/"))
            self.assertIn("Mozilla/5.0", h["User-Agent"])
        by = {it.symbols[0]: it for it in items}
        self.assertEqual(set(by), {"NVDA", "AAPL", "MSFT"})
        n = by["NVDA"]
        self.assertGoodItem(n)
        self.assertEqual(n.title, "NVDA reports earnings Wed Oct 21 (after close)")
        self.assertEqual((n.catalyst, n.direction, n.confidence, n.kind), ("earnings", 0, 0.9, "filing"))
        self.assertEqual(n.url, "data://earnings/NVDA/2026-10-21")
        self.assertEqual((n.facts["date"], n.facts["time"]), ("2026-10-21", "after close"))
        self.assertIn("38 estimates", n.text)
        self.assertEqual(by["AAPL"].title, "AAPL reports earnings Thu Oct 22 (before the open)")
        self.assertEqual(by["MSFT"].title, "MSFT reports earnings Mon Oct 26")
        self.assertEqual(by["MSFT"].facts["time"], "unknown")
        # already announced: not repeated
        self.assertEqual(stocks.EarningsCalendar().fetch(ctx(self.routes(), syms, state=c.state)), [])

    def test_weekend_start(self):
        sat = utc(2026, 10, 24, 15, 0, 0)
        c = ctx({}, {"AAPL": "stock"}, now=sat)
        with self.assertRaises(SourceError):
            stocks.EarningsCalendar().fetch(c)
        self.assertEqual(c.http.calls, [nasdaq_url(d) for d in ("2026-10-26", "2026-10-27", "2026-10-28", "2026-10-29")])
        # Friday evening in New York is already Saturday in UTC: the calendar day is New York's
        fri_night = utc(2026, 10, 24, 2, 0, 0)    # Fri Oct 23 22:00 EDT
        c = ctx({}, {"AAPL": "stock"}, now=fri_night)
        with self.assertRaises(SourceError):
            stocks.EarningsCalendar().fetch(c)
        self.assertEqual(c.http.calls[0], nasdaq_url("2026-10-23"))

    def test_errors(self):
        self.assertEqual(stocks.EarningsCalendar().fetch(ctx({}, {"BTC-USD": "crypto"})), [])   # no stocks, no calls
        bad = {nasdaq_url(d): {"data": None, "message": None, "status": {"rCode": 400, "bCodeMessage": [{"code": 1001}]}}
               for d in ("2026-10-20", "2026-10-21", "2026-10-22", "2026-10-23", "2026-10-26")}
        with self.assertRaises(SourceError):
            stocks.EarningsCalendar().fetch(ctx(bad))
        r = self.routes()
        r[nasdaq_url("2026-10-20")] = RateLimited("api.nasdaq.com: HTTP 429", 60)
        with self.assertRaises(RateLimited):
            stocks.EarningsCalendar().fetch(ctx(r))
        r = self.routes()
        r[nasdaq_url("2026-10-20")] = SourceError("api.nasdaq.com: HTTP 403")
        items = stocks.EarningsCalendar().fetch(ctx(r, {"AAPL": "stock"}))     # one bad day costs only that day
        self.assertEqual([it.symbols for it in items], [["AAPL"]])

    def test_8k_standby(self):
        routes = {TICKERS_URL: TICKERS, subs_url(320193): AAPL_SUBS}
        c = ctx(routes)
        items = stocks.SEC8KEarnings().fetch(c)
        self.assertEqual(len(items), 1)          # the 10-14 8-K is item 5.02, not results
        it = items[0]
        self.assertGoodItem(it)
        self.assertEqual((it.catalyst, it.direction, it.confidence, it.source), ("earnings", 0, 0.8, "sec_8k_earnings"))
        self.assertIn("released quarterly results", it.title)
        self.assertEqual(stocks.SEC8KEarnings().fetch(ctx(routes, state=c.state)), [])
        # four days later the 2.02 is out of its 3-day window
        self.assertEqual(stocks.SEC8KEarnings().fetch(ctx(routes, now=NOW + 3 * 86400)), [])


# ---------------------------------------------------------------------------- news

def rss(items, title="Yahoo! Finance: AAPL News", site="http://finance.yahoo.com/q/h?s=AAPL"):
    body = "".join(f"""
    <item>
      <description>{d}</description>
      <guid isPermaLink="false">{g}</guid>
      <link>{l}</link>
      <pubDate>{p}</pubDate>
      <title>{t}</title>
    </item>""" for t, l, d, g, p in items)
    return f"""<?xml version="1.0" encoding="utf-8"?>
<rss version="2.0">
  <channel>
    <copyright>Copyright (c) 2026 Yahoo! Inc. All rights reserved.</copyright>
    <description>Latest Financial News for AAPL</description>
    <language>en-US</language>
    <lastBuildDate>Tue, 20 Oct 2026 13:58:02 +0000</lastBuildDate>
    <link>{site}</link>
    <title>{title}</title>{body}
  </channel>
</rss>"""


YAHOO_URL = "https://feeds.finance.yahoo.com/rss/2.0/headline?s={}&region=US&lang=en-US".format
YAHOO_AAPL = rss([
    ("Apple supplier signals strong iPhone 18 demand", "https://finance.yahoo.com/news/apple-supplier-signals-133000123.html",
     "Shares of the chip supplier rose after it raised its outlook.", "aa1", "Tue, 20 Oct 2026 13:30:00 +0000"),
    ("Apple faces EU fine over App Store rules &amp; steering", "https://www.reuters.com/technology/apple-eu-fine-2026-10-19/",
     "&lt;p&gt;The European Commission said on Monday...&lt;/p&gt;", "aa2", "Mon, 19 Oct 2026 16:05:00 +0000"),
    ("Old news: Apple event recap", "https://finance.yahoo.com/news/apple-event-recap-090000.html",
     "Recap.", "aa3", "Wed, 14 Oct 2026 09:00:00 +0000"),
    ("Undated Apple item", "https://finance.yahoo.com/news/apple-undated.html", "", "aa4", ""),
])
YAHOO_UNKNOWN = rss([("RSS feed not found", "http://finance.yahoo.com/", "Symbol not found.", "x", "Tue, 20 Oct 2026 13:00:00 +0000")],
                    title="Yahoo! Finance: RSS feed not found")


class TestNews(Base):
    def test_yahoo(self):
        c = ctx({YAHOO_URL("AAPL"): YAHOO_AAPL, YAHOO_URL("ZZZZ"): YAHOO_UNKNOWN}, {"AAPL": "stock", "ZZZZ": "stock"})
        items = stocks.YahooRSS().fetch(c)
        self.assertEqual([it.title for it in items], ["Apple supplier signals strong iPhone 18 demand",
                                                      "Apple faces EU fine over App Store rules & steering",
                                                      "Undated Apple item"])   # 6 days old: dropped
        a, b, u = items
        for it in items:
            self.assertGoodItem(it)
            self.assertEqual((it.symbols, it.kind, it.source, it.catalyst), (["AAPL"], "news", "yahoo_rss", None))
        self.assertEqual(a.publisher, "finance.yahoo.com")
        self.assertEqual(b.publisher, "reuters.com")       # the item names its outlet
        self.assertEqual(a.published_ts, utc(2026, 10, 20, 13, 30))
        self.assertNotIn("<p>", b.text)
        self.assertIsNone(u.published_ts)
        self.assertIn("User-Agent", c.http.kw[0]["headers"])

    def test_yahoo_failures(self):
        c = ctx({YAHOO_URL("AAPL"): YAHOO_AAPL, YAHOO_URL("MSFT"): "<html>Will be right back</html>"},
                {"MSFT": "stock", "AAPL": "stock"})
        self.assertEqual(len(stocks.YahooRSS().fetch(c)), 3)
        self.assertTrue(c.state.get("_log"))
        with self.assertRaises(SourceError):
            stocks.YahooRSS().fetch(ctx({YAHOO_URL("MSFT"): "<html>Will be right back</html>"}, {"MSFT": "stock"}))

    def test_nasdaq_standby(self):
        url = "https://www.nasdaq.com/feed/rssoutbound?symbol=AAPL"
        feed = rss([("Apple (AAPL) Q4 preview: what to expect", "https://www.nasdaq.com/articles/apple-aapl-q4-preview",
                     "Analysts expect...", "n1", "Tue, 20 Oct 2026 12:00:00 +0000")], title="Nasdaq AAPL",
                   site="https://www.nasdaq.com")
        items = stocks.NasdaqRSS().fetch(ctx({url: feed}))
        self.assertEqual([(it.source, it.publisher, it.symbols) for it in items], [("nasdaq_rss", "nasdaq.com", ["AAPL"])])


# ---------------------------------------------------------------------------- VIX

VIX_URL = "https://query1.finance.yahoo.com/v8/finance/chart/%5EVIX?interval=1d&range=3mo"


def sessions(end_day, n):
    """The last n weekdays up to and including end_day (date tuples), oldest first."""
    import datetime as dt
    d, out = dt.date(*end_day), []
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d -= dt.timedelta(days=1)
    return out[::-1]


def chart(points, symbol="^VIX", gran="1d"):
    """Yahoo chart JSON. points: [(ts, close or None)]."""
    ts = [int(t) for t, _ in points]
    cl = [c for _, c in points]
    return {"chart": {"result": [{
        "meta": {"currency": "USD", "symbol": symbol, "exchangeName": "CXI", "instrumentType": "INDEX",
                 "regularMarketPrice": cl[-1] if cl else None, "gmtoffset": -14400, "timezone": "EDT",
                 "dataGranularity": gran, "range": "3mo"},
        "timestamp": ts,
        "indicators": {"quote": [{"open": cl, "high": cl, "low": cl, "close": cl, "volume": [0] * len(cl)}],
                       "adjclose": [{"adjclose": cl}]}}], "error": None}}


def vix_chart(closes, end=(2026, 10, 20)):
    days = sessions(end, len(closes))
    return chart([(utc(d.year, d.month, d.day, 13, 30), c) for d, c in zip(days, closes)])


STOCKS = {"AAPL": "stock", "NVDA": "stock", "BTC-USD": "crypto"}


class TestVIX(Base):
    def fetch(self, closes, state=None, now=NOW, end=(2026, 10, 20)):
        return stocks.YahooVIX().fetch(ctx({VIX_URL: vix_chart(closes, end)}, STOCKS, state=state, now=now))

    def test_fear(self):
        st = {}
        items = self.fetch([18.0] * 20 + [19.0, 21.0, 24.0, 27.5, 31.2], state=st)
        self.assertEqual(len(items), 1)
        it = items[0]
        self.assertGoodItem(it)
        self.assertEqual(sorted(it.symbols), ["AAPL", "NVDA"])        # every subscribed stock, no crypto
        self.assertEqual((it.catalyst, it.direction, it.confidence, it.kind), ("macro", -1, 0.45, "market"))
        self.assertEqual(it.facts["vix"], 31.2)
        self.assertAlmostEqual(it.facts["chg_5d"], 31.2 / 18.0 - 1, places=3)
        self.assertIn("fear spiking", it.title)
        self.assertEqual(it.url, "data://vix/2026-10-20")
        self.assertEqual(self.fetch([18.0] * 20 + [19.0, 21.0, 24.0, 27.5, 31.2], state=st), [])   # once per session day

    def test_jump_calm_normal(self):
        jump = self.fetch([16.0] * 20 + [16.5, 17.0, 18.0, 19.0, 20.4])     # +27.5% in 5 sessions, still under 30
        self.assertEqual(jump[0].direction, -1)
        calm = self.fetch([12.8] * 25)
        self.assertEqual(calm[0].direction, 0)
        self.assertIn("complacency", calm[0].title)
        normal = self.fetch([17.0] * 24 + [18.0])
        self.assertEqual(normal[0].direction, 0)

    def test_null_closes_stale_and_errors(self):
        closes = [18.0] * 23 + [None, 32.0]
        items = self.fetch(closes)
        self.assertEqual(items[0].facts["vix"], 32.0)
        self.assertEqual(self.fetch([18.0] * 25, end=(2026, 10, 2)), [])   # newest bar 18 days old: stuck feed
        self.assertEqual(stocks.YahooVIX().fetch(ctx({}, {"BTC-USD": "crypto"})), [])
        err = {"chart": {"result": None, "error": {"code": "Not Found", "description": "No data found, symbol may be delisted"}}}
        with self.assertRaises(SourceError):
            stocks.YahooVIX().fetch(ctx({VIX_URL: err}, STOCKS))
        with self.assertRaises(SourceError):
            stocks.YahooVIX().fetch(ctx({VIX_URL: {"finance": {"error": "x"}}}, STOCKS))

    def test_cboe_csv(self):
        url = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"
        days = sessions((2026, 10, 19), 30)
        closes = [15.0] * 25 + [15.5, 16.0, 17.2, 18.4, 19.6]
        lines = ["DATE,OPEN,HIGH,LOW,CLOSE", "01/02/1990,17.240000,17.240000,17.240000,17.240000"]
        lines += [f"{d.month:02d}/{d.day:02d}/{d.year},{c - 0.3:.6f},{c + 0.5:.6f},{c - 0.6:.6f},{c:.6f}"
                  for d, c in zip(days, closes)]
        lines += ["10/21/2026,99,99,99,99", "garbage,row", ""]           # a future row and junk never count
        c = ctx({url: "﻿" + "\r\n".join(lines)}, STOCKS)
        items = stocks.CboeVIX().fetch(c)
        self.assertEqual(len(items), 1)
        it = items[0]
        self.assertGoodItem(it)
        self.assertEqual((it.source, it.publisher, it.direction), ("cboe_vix", "cboe.com", -1))   # 19.6 vs 15.0: +31%
        self.assertEqual(it.facts["vix"], 19.6)
        self.assertEqual(it.url, "data://vix/2026-10-19")
        with self.assertRaises(SourceError):
            stocks.CboeVIX().fetch(ctx({url: "<html>Access Denied</html>"}, STOCKS))


# ---------------------------------------------------------------------------- hourly prices

class TestYahooHourly(Base):
    URL = "https://query1.finance.yahoo.com/v8/finance/chart/AAPL?interval=1h&range=5d&includePrePost=false"

    def test_bars(self):
        # Mon 10-19 regular session (13:30Z .. 19:30Z) and Tue 10-20's first bar (still forming at 14:00Z)
        mon = [utc(2026, 10, 19, 13 + k, 30) for k in range(7)]
        pts = list(zip(mon, [252.1, 252.8, None, 253.4, 253.0, None, 254.2])) + [(utc(2026, 10, 20, 13, 30), 255.0)]
        http = FakeHttp({self.URL: chart(pts, "AAPL", "1h")})
        bars = stocks.yahoo_hourly(http, "AAPL", "5d", now=NOW)
        self.assertEqual(bars, [(int(mon[0]), 252.1), (int(mon[1]), 252.8), (int(mon[3]), 253.4),
                                (int(mon[4]), 253.0), (int(mon[6]), 254.2)])
        self.assertIn("User-Agent", http.kw[0]["headers"])
        # once the hour is over the bar is complete
        bars = stocks.yahoo_hourly(http, "AAPL", "5d", now=utc(2026, 10, 20, 14, 30))
        self.assertEqual(bars[-1], (int(utc(2026, 10, 20, 13, 30)), 255.0))

    def test_month_and_errors(self):
        url = self.URL.replace("range=5d", "range=1mo")
        http = FakeHttp({url: chart([(utc(2026, 10, 1, 13, 30), 240.0)], "AAPL", "1h")})
        self.assertEqual(stocks.yahoo_hourly(http, "AAPL", "1mo", now=NOW), [(int(utc(2026, 10, 1, 13, 30)), 240.0)])
        with self.assertRaises(SourceError):
            stocks.yahoo_hourly(FakeHttp({self.URL: {"chart": {"result": [], "error": None}}}), "AAPL", "5d", now=NOW)
        empty = {"chart": {"result": [{"meta": {"symbol": "AAPL"}, "timestamp": [], "indicators": {"quote": [{"close": []}]}}],
                           "error": None}}
        self.assertEqual(stocks.yahoo_hourly(FakeHttp({self.URL: empty}), "AAPL", "5d", now=NOW), [])


if __name__ == "__main__":
    unittest.main()
