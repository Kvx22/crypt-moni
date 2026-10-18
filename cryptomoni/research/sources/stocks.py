"""Stock and ETF research sources: SEC filings, insider trades, the earnings calendar,
per-ticker news and the VIX.

Each primary has a standby that reaches the same facts another way, so the worker pool can
fail over without losing a team:

  sec_filings        SEC submissions JSON: new 8-K / 10-Q / 10-K / S-3 / 13D, pre-labelled by item
    sec_filings_atom   the same filings from EDGAR's per-company Atom feed
  sec_insider        Form 4 open-market purchases and large officer sales (raw ownership XML)
    sec_insider_atom   the same, finding the Form 4s through the Atom feed
  earnings_calendar  Nasdaq's earnings calendar: who reports in the next week, and when
    sec_8k_earnings    8-K item 2.02 ("results of operations") in the last three days
  yahoo_rss          Yahoo Finance headlines per ticker
    nasdaq_rss         Nasdaq's per-ticker news feed
  yahoo_vix          the VIX from Yahoo's chart API: one market-wide reading per session day
    cboe_vix           the VIX history CSV straight from Cboe

Structured sources pre-label their items (catalyst, direction, risk flags, confidence) because
an 8-K item code or a Form 4 transaction code says more than the words around it. News items
are left for the extractor. `yahoo_hourly` is the hourly price helper the outcome logger uses.

Every parser here expects junk: a missing key skips that entry; a payload of the wrong shape
altogether raises SourceError so the worker can mark the source as failing.
"""
from __future__ import annotations

import csv
import datetime as dt
import io
import re
import time
import urllib.parse
import xml.etree.ElementTree as ET

from ..http import RateLimited, SourceError
from ..text import clean, domain, parse_feed, parse_ts
from ..types import Item
from .base import Adapter, sym_assets

DAY = 86400.0
CIK_TTL_S = 7 * DAY
FILING_WINDOW_S = 10 * DAY
NEWS_MAX_AGE_S = 3 * DAY
SEEN_CAP = 300

BROWSER_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
              "Chrome/128.0.0.0 Safari/537.36")
# Nasdaq's API refuses anything that does not look like its own web page asking.
NASDAQ_HEADERS = {"Accept": "application/json", "User-Agent": BROWSER_UA,
                  "Origin": "https://www.nasdaq.com", "Referer": "https://www.nasdaq.com/"}
NASDAQ_RSS_HEADERS = {"Accept": "application/rss+xml, application/xml;q=0.9, */*;q=0.5", "User-Agent": BROWSER_UA,
                      "Referer": "https://www.nasdaq.com/"}
# Yahoo's chart API answers unknown user agents with 429s; a browser string gets the normal limits.
YAHOO_HEADERS = {"User-Agent": BROWSER_UA}

WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


# ---------------------------------------------------------------------------- small helpers

def _num(x):
    """'1,234.5' / '$1.02' / 12 -> float, None if not a finite number."""
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        v = float(x)
    elif isinstance(x, str):
        s = x.strip().replace(",", "").replace("$", "")
        if not s:
            return None
        try:
            v = float(s)
        except ValueError:
            return None
    else:
        return None
    return v if v == v and abs(v) != float("inf") else None


def _money(v):
    """250000 -> '$250k', 1234567 -> '$1.2M'."""
    v = abs(float(v))
    if v >= 1e9:
        return f"${v / 1e9:.1f}B"
    if v >= 1e6:
        return f"${v / 1e6:.1f}M"
    if v >= 1e3:
        return f"${v / 1e3:.0f}k"
    return f"${v:.0f}"


def _et_offset_h(ts):
    """Hours New York is behind UTC at `ts` (US DST rules; no tz database needed, Windows has none)."""
    y = dt.datetime.fromtimestamp(ts, dt.timezone.utc).year

    def nth_sunday(month, n):
        d = dt.date(y, month, 1)
        d += dt.timedelta(days=(6 - d.weekday()) % 7)
        return d + dt.timedelta(weeks=n - 1)
    start = dt.datetime.combine(nth_sunday(3, 2), dt.time(7), dt.timezone.utc).timestamp()   # 2:00 EST
    end = dt.datetime.combine(nth_sunday(11, 1), dt.time(6), dt.timezone.utc).timestamp()    # 2:00 EDT
    return 4 if start <= ts < end else 5


def et_date(ts) -> dt.date:
    """The New York calendar date at unix time `ts`."""
    return dt.datetime.fromtimestamp(ts - _et_offset_h(ts) * 3600, dt.timezone.utc).date()


def _et_ts(day, hour, minute=0):
    """Unix time of a New York wall-clock time on `day`."""
    naive = dt.datetime(day.year, day.month, day.day, hour, minute, tzinfo=dt.timezone.utc).timestamp()
    return naive + _et_offset_h(naive + 5 * 3600) * 3600


def _ymd(s):
    try:
        return dt.date.fromisoformat(str(s).strip()[:10])
    except (TypeError, ValueError):
        return None


def _day_text(d):
    return f"{WEEKDAYS[d.weekday()]} {MONTHS[d.month - 1]} {d.day}"


def _past(ts, now):
    """A timestamp no later than now (feeds sometimes run a little ahead); None stays None."""
    return None if ts is None else min(ts, now)


def _remember(state, key, values, cap=SEEN_CAP):
    """Append to a capped list in state (newest last)."""
    old = state.get(key)
    old = old if isinstance(old, list) else []
    state[key] = (old + [v for v in values if v not in old])[-cap:]


def _seen(state, key):
    v = state.get(key)
    return set(v) if isinstance(v, list) else set()


def _stocks(ctx):
    return [s for s in sym_assets(ctx, "stock") if isinstance(s, str) and s.strip()]


def _per_ticker(ctx, fn):
    """Run fn(ticker) -> list[Item] for every subscribed stock in the context. One ticker
    failing is logged and skipped; if every ticker failed the first error is raised.
    A rate limit stops the run at once (the worker honours its Retry-After)."""
    out, errors, ok = [], [], 0
    for t in _stocks(ctx):
        try:
            out += fn(t)
            ok += 1
        except RateLimited:
            raise
        except SourceError as e:
            errors.append(e)
            ctx.log(f"{t}: {e}")
    if errors and not ok:
        raise errors[0]
    return out


def _xml_root(text, what):
    """Parse XML namespace-free (tags lose their {ns} prefix). DTDs are dropped first:
    nothing here needs one, and that defuses entity-expansion tricks."""
    if isinstance(text, bytes):
        text = text.decode("utf-8", "replace")
    if not isinstance(text, str) or not text.strip():
        raise SourceError(f"{what}: empty response")
    text = re.sub(r"<!DOCTYPE[^>]*>", "", text.lstrip("﻿ \t\r\n"), flags=re.I)
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        raise SourceError(f"{what}: not valid XML") from None
    for el in root.iter():
        if isinstance(el.tag, str) and el.tag.startswith("{"):
            el.tag = el.tag.split("}", 1)[1]
    return root


def _xt(el, path):
    """Stripped text at path ('' if missing). Form 4 wraps most values in <value>."""
    if el is None:
        return ""
    node = el.find(path)
    if node is None:
        return ""
    v = node.find("value")
    return "".join((v if v is not None else node).itertext()).strip()


def _flag(s):
    return str(s).strip().lower() in ("1", "true", "y", "yes")


# ---------------------------------------------------------------------------- SEC: tickers and submissions

SEC_TICKERS = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
SEC_ARCHIVE = "https://www.sec.gov/Archives/edgar/data/{cik}/{nodash}/"
SEC_ATOM = ("https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={t}&type={form}&dateb=&owner=include"
            "&count=40&output=atom")


def sec_cik_map(ctx) -> dict:
    """{TICKER: (cik, title)} for the context's subscribed stocks, from SEC's ticker map.

    Cached in ctx.state for 7 days, one key per ticker ("cik:AAPL" -> [cik, title, fetched_ts];
    cik None for tickers SEC doesn't list, which is most ETFs), so two shards of the same
    adapter never overwrite each other's cache. The whole map is only fetched when a
    subscribed ticker has no fresh entry."""
    tickers = [t.upper() for t in _stocks(ctx)]
    out, stale = {}, []
    for t in tickers:
        e = ctx.state.get("cik:" + t)
        if isinstance(e, list) and len(e) == 3 and isinstance(e[2], (int, float)) and ctx.now - e[2] < CIK_TTL_S:
            if isinstance(e[0], int) and e[0] > 0:
                out[t] = (e[0], str(e[1] or t))
        else:
            stale.append(t)
    if not stale:
        return out
    data = ctx.http.json(SEC_TICKERS)
    if isinstance(data, dict):
        rows = data.values()
    elif isinstance(data, list):
        rows = data
    else:
        raise SourceError("sec.gov: company_tickers.json is not a JSON object")
    found = {}
    for r in rows:
        if not isinstance(r, dict):
            continue
        t = str(r.get("ticker") or "").strip().upper()
        cik = _num(r.get("cik_str"))
        if t and cik and cik > 0 and t not in found:
            found[t] = (int(cik), clean(r.get("title")) or t)
    if not found:
        raise SourceError("sec.gov: company_tickers.json has no tickers")
    for t in stale:
        # SEC writes class shares with a dash (BRK-B); some symbol lists use a dot
        hit = found.get(t) or found.get(t.replace(".", "-"))
        ctx.state["cik:" + t] = [hit[0], hit[1], ctx.now] if hit else [None, "", ctx.now]
        if hit:
            out[t] = hit
    return out


def submissions(ctx, cik) -> tuple[str, list]:
    """(company name, recent filings) from data.sec.gov. Each filing is a dict with
    accession, form, filing_date, ts, items (list of '2.02' codes), doc, desc."""
    data = ctx.http.json(SEC_SUBMISSIONS.format(cik=int(cik)))
    recent = (data.get("filings") or {}).get("recent") if isinstance(data, dict) else None
    if not isinstance(recent, dict) or not isinstance(recent.get("accessionNumber"), list):
        raise SourceError("data.sec.gov: submissions JSON has no filings.recent")

    def col(name, i):
        v = recent.get(name)
        return v[i] if isinstance(v, list) and i < len(v) and v[i] is not None else ""
    out = []
    for i, acc in enumerate(recent["accessionNumber"][:400]):
        acc = str(acc or "").strip()
        if not re.fullmatch(r"\d{10}-\d{2}-\d{6}", acc):
            continue
        filed = _ymd(col("filingDate", i))
        ts = parse_ts(col("acceptanceDateTime", i)) or (
            _et_ts(filed, 17) if filed else None)     # no acceptance time: assume end of that day
        if ts is None:
            continue
        out.append(dict(accession=acc, form=str(col("form", i)).strip().upper(), filing_date=str(filed or ""),
                        ts=float(ts), items=re.findall(r"\d{1,2}\.\d{2}", str(col("items", i))),
                        doc=str(col("primaryDocument", i)).strip(), desc=clean(col("primaryDocDescription", i))))
    return clean(data.get("name")) or "", out


def filing_index_url(cik, accession):
    return SEC_ARCHIVE.format(cik=int(cik), nodash=accession.replace("-", "")) + f"{accession}-index.htm"


# 8-K item code -> (catalyst, direction, risk_flags, confidence, plain English). Listed in the
# order that decides a filing's label when it carries several items (worst news first).
ITEMS_8K = {
    "1.03": ("legal", -1, ["bankruptcy"], 0.9, "bankruptcy or receivership"),
    "3.01": ("delisting", -1, ["delisting"], 0.75, "notice of delisting or a listing-rule failure"),
    "4.02": ("filing", -1, [], 0.7, "earlier financial statements can no longer be relied on (restatement)"),
    "2.01": ("mna", 0, [], 0.6, "completed an acquisition or disposition"),
    "1.01": ("partnership", 0, [], 0.5, "entered a material agreement"),
    "2.02": ("earnings", 0, [], 0.8, "results of operations"),
    "5.02": ("filing", 0, [], 0.6, "executive or director change"),
    "7.01": ("filing", 0, [], 0.5, "Regulation FD disclosure"),
    "8.01": ("filing", 0, [], 0.5, "other events"),
}
ITEM_NAMES = {"9.01": "financial statements and exhibits", "5.07": "shareholder vote results",
              "2.03": "took on a material debt obligation", "2.05": "exit or restructuring costs",
              "2.06": "material impairment", "3.02": "unregistered sale of shares", "5.03": "bylaw or charter change"}


def label_filing(form, items):
    """(catalyst, direction, risk_flags, confidence, what) for a filing, or None to skip it."""
    form = (form or "").upper().strip()
    if form == "8-K":
        for code, lab in ITEMS_8K.items():
            if code in items:
                return lab
        names = [ITEM_NAMES[c] for c in items if c in ITEM_NAMES]
        return ("filing", 0, [], 0.4, names[0] if names else "current report")
    if form.startswith(("S-1", "S-3", "424B")):
        return ("unlock", -1, ["dilution"], 0.6, "share offering or shelf registration")
    if form in ("SC 13D", "SCHEDULE 13D"):
        return ("insider", 1, [], 0.5, "an activist or large holder disclosed a 5%+ stake")
    if form in ("10-Q", "10-K", "20-F", "40-F"):
        return ("filing", 0, [], 0.6, "annual report" if form != "10-Q" else "quarterly report")
    return None


def _filing_item(source, ticker, company, cik, f, lab):
    cat, d, flags, conf, what = lab
    form = f["form"]
    codes = [c for c in f["items"]]
    item_txt = ", ".join(codes)
    title = f"{company or ticker} ({ticker}) files {form}: {what}" + (f" (Item {item_txt})" if codes and form == "8-K" else "")
    parts = [f"{form} filed {f['filing_date'] or 'recently'}."]
    for c in codes:
        name = ITEMS_8K[c][4] if c in ITEMS_8K else ITEM_NAMES.get(c)
        if name:
            parts.append(f"Item {c}: {name}.")
    if f.get("desc"):
        parts.append(f["desc"])
    return Item(source=source, url=filing_index_url(cik, f["accession"]), title=title[:300], text=" ".join(parts),
                published_ts=f["ts"], symbols=[ticker], kind="filing", publisher="sec.gov",
                facts=dict(form=form, items=codes, accession=f["accession"], filing_date=f["filing_date"]),
                catalyst=cat, direction=d, risk_flags=list(flags), confidence=conf, official=True)


class SECFilings(Adapter):
    name, team, kind = "sec_filings", "filings", "filing"
    assets = ("stock",)
    interval_s = 3600.0
    per_symbol = True
    max_symbols_per_run = 6
    domains = ("www.sec.gov", "data.sec.gov")
    official = True
    description = "New SEC filings per company (8-K by item code, 10-Q/10-K, offerings, 13D) from EDGAR's submissions API."

    def fetch(self, ctx):
        ciks = sec_cik_map(ctx)

        def one(t):
            if t.upper() not in ciks:
                return []           # most ETFs have no CIK in the ticker map
            cik, title = ciks[t.upper()]
            name, filings = submissions(ctx, cik)
            key = "seen:" + t
            seen, out, new = _seen(ctx.state, key), [], []
            for f in filings:
                if f["accession"] in seen or not (0 <= ctx.now - f["ts"] <= FILING_WINDOW_S) and f["ts"] <= ctx.now:
                    continue
                f["ts"] = _past(f["ts"], ctx.now)
                if ctx.now - f["ts"] > FILING_WINDOW_S:
                    continue
                lab = label_filing(f["form"], f["items"])
                if lab is None:
                    continue
                new.append(f["accession"])
                out.append(_filing_item(self.name, t, name or title, cik, f, lab))
            _remember(ctx.state, key, new)
            return out
        return _per_ticker(ctx, one)


# ---------------------------------------------------------------------------- SEC: Atom feeds

def edgar_atom(ctx, ticker, form=""):
    """Entries of EDGAR's per-company Atom feed: dicts with accession, form, cik, url, ts,
    filing_date, items. [] when EDGAR doesn't know the ticker (it answers with an HTML page)."""
    url = SEC_ATOM.format(t=urllib.parse.quote(ticker), form=urllib.parse.quote(form))
    body = ctx.http.text(url, accept="application/atom+xml, application/xml;q=0.9, */*;q=0.5")
    head = (body or "")[:2000].lower()
    if "<feed" not in head and ("no matching" in (body or "").lower() or "<html" in head):
        return []
    root = _xml_root(body, "sec.gov atom")
    if root.tag != "feed":
        raise SourceError(f"sec.gov atom: not an Atom feed (root <{str(root.tag)[:20]}>)")
    out = []
    for e in root.findall("entry"):
        cat = e.find("category")
        content = e.find("content")
        f_form = ((cat.get("term") if cat is not None else "") or _xt(content, "filing-type")).strip().upper()
        link = next((l.get("href") for l in e.findall("link") if (l.get("href") or "").startswith("https://")), "")
        acc = _xt(content, "accession-number") or _xt(e, "id")
        m = re.search(r"\d{10}-\d{2}-\d{6}", acc) or re.search(r"\d{10}-\d{2}-\d{6}", link)
        cik = re.search(r"/edgar/data/(\d+)/", link)
        ts = parse_ts(_xt(e, "updated"))
        if not (f_form and m and cik and ts):
            continue
        filed = _xt(content, "filing-date")
        summary = clean(_xt(e, "summary")) + " " + _xt(content, "items-desc")
        out.append(dict(accession=m.group(0), form=f_form, cik=int(cik.group(1)), ts=float(ts),
                        url=filing_index_url(int(cik.group(1)), m.group(0)),
                        filing_date=filed if _ymd(filed) else "", desc="",
                        items=list(dict.fromkeys(re.findall(r"\b\d{1,2}\.\d{2}\b", summary)))))
    return out


class SECFilingsAtom(Adapter):
    name, team, kind = "sec_filings_atom", "filings", "filing"
    assets = ("stock",)
    interval_s = 3600.0
    per_symbol = True
    max_symbols_per_run = 6
    domains = ("www.sec.gov",)
    official = True
    role, standby_for = "standby", "sec_filings"
    description = "New SEC filings per company from EDGAR's Atom feed (standby for sec_filings)."

    def fetch(self, ctx):
        def one(t):
            key = "seen:" + t
            seen, out, new = _seen(ctx.state, key), [], []
            for f in edgar_atom(ctx, t):
                ts = _past(f["ts"], ctx.now)
                if f["accession"] in seen or ctx.now - ts > FILING_WINDOW_S:
                    continue
                lab = label_filing(f["form"], f["items"])
                if lab is None:
                    continue
                f["ts"] = ts
                new.append(f["accession"])
                out.append(_filing_item(self.name, t, "", f["cik"], f, lab))
            _remember(ctx.state, key, new)
            return out
        return _per_ticker(ctx, one)


# ---------------------------------------------------------------------------- SEC: Form 4 insider trades

BUY_MIN_USD = 100_000
SELL_MIN_USD = 1_000_000
_ENTITY = re.compile(r"\b(INC|LLC|L\.?P|LTD|CORP|CORPORATION|TRUST|FUND|CAPITAL|PARTNERS|HOLDINGS|GROUP|MANAGEMENT|CO)\b\.?", re.I)


def owner_display(name):
    """SEC's 'DOE JOHN A' -> 'J. Doe'; companies keep their (title-cased) name."""
    name = clean(name)
    parts = name.replace(",", " ").split()
    if not parts:
        return "An insider"
    if _ENTITY.search(name) or len(parts) == 1:
        return name.title() if name.isupper() else name
    return f"{parts[1][0].upper()}. {parts[0].title()}"


def parse_form4(xml_text) -> dict:
    """Ownership XML -> dict(owner, role, is_officer, buy=dict(shares, value, price), sell=..., plan).
    Only open-market purchases (P) and sales (S) count: grants, option exercises, tax
    withholding and gifts (A/M/F/G and the rest) say nothing about what insiders think."""
    m = re.search(r"<ownershipDocument\b.*?</ownershipDocument>", xml_text or "", re.S)
    root = _xml_root(m.group(0) if m else xml_text, "form 4")
    if root.tag != "ownershipDocument":
        raise SourceError(f"form 4: not an ownershipDocument (root <{str(root.tag)[:20]}>)")
    owner = root.find("reportingOwner")
    rel = owner.find("reportingOwnerRelationship") if owner is not None else None
    name = _xt(owner, "reportingOwnerId/rptOwnerName") if owner is not None else ""
    is_dir, is_off = _flag(_xt(rel, "isDirector")), _flag(_xt(rel, "isOfficer"))
    is_ten = _flag(_xt(rel, "isTenPercentOwner"))
    otitle = clean(_xt(rel, "officerTitle"))
    role = (otitle if is_off and otitle else "Officer" if is_off else "Director" if is_dir
            else "10% owner" if is_ten else "Insider")
    tot = {"P": [0.0, 0.0], "S": [0.0, 0.0]}     # code -> [shares, value]
    for tx in root.iter("nonDerivativeTransaction"):
        code = _xt(tx, "transactionCoding/transactionCode").upper()
        if code not in tot:
            continue
        sh = _num(_xt(tx, "transactionAmounts/transactionShares"))
        px = _num(_xt(tx, "transactionAmounts/transactionPricePerShare"))
        if not sh or not px or sh <= 0 or px <= 0:
            continue
        tot[code][0] += sh
        tot[code][1] += sh * px

    def side(code):
        sh, val = tot[code]
        return dict(shares=sh, value=round(val, 2), price=round(val / sh, 4)) if sh > 0 else None
    return dict(owner=name, display=owner_display(name), role=role, is_officer=is_off, is_director=is_dir,
                buy=side("P"), sell=side("S"), plan=_flag(_xt(root, "aff10b5One")))


def insider_item(source, ticker, url, ts, f4):
    """The Item for one parsed Form 4, or None when nothing in it clears the thresholds."""
    buy, sell = f4.get("buy"), f4.get("sell")
    if buy and buy["value"] >= BUY_MIN_USD:
        code, s, d, conf, verb = "P", buy, 1, 0.7, "bought"
    elif sell and sell["value"] >= SELL_MIN_USD and f4.get("is_officer"):
        code, s, d, conf, verb = "S", sell, -1, 0.4, "sold"      # sales are noisy: diversification, taxes, plans
    else:
        return None
    who = f"{f4['role']} {f4['display']}"
    title = f"{who} {verb} {_money(s['value'])} of {ticker} (Form 4)"
    text = (f"{who} {verb} {s['shares']:,.0f} shares of {ticker} at about ${s['price']:,.2f} "
            f"in open-market trades, about {_money(s['value'])} in total.")
    if f4.get("plan"):
        text += " The trade was made under a pre-arranged 10b5-1 plan."
    return Item(source=source, url=url, title=title, text=text, published_ts=ts, symbols=[ticker], kind="filing",
                publisher="sec.gov", catalyst="insider", direction=d, confidence=conf, official=True,
                facts=dict(code=code, shares=s["shares"], price=s["price"], value=s["value"], owner=f4["owner"],
                           title=f4["role"], plan_10b5_1=bool(f4.get("plan"))))


MAX_FORM4_PER_TICKER = 8   # XML fetches per ticker per run; the rest wait for the next run


def _insider_run(adapter, ctx, ticker, candidates, xml_url):
    """Shared by the two insider adapters: fetch and parse each unseen Form 4 in the window.
    candidates: dicts with accession, ts, url (index page); xml_url(c) -> raw document url."""
    key = "seen:" + ticker
    seen, out, done = _seen(ctx.state, key), [], []
    todo = [c for c in candidates if c["accession"] not in seen and ctx.now - _past(c["ts"], ctx.now) <= FILING_WINDOW_S]
    try:
        for c in todo[:MAX_FORM4_PER_TICKER]:
            try:
                f4 = parse_form4(ctx.http.text(xml_url(c)))
            except RateLimited:
                raise
            except SourceError as e:
                ctx.log(f"{ticker} {c['accession']}: {e}")   # a broken document is not worth retrying forever
                done.append(c["accession"])
                continue
            done.append(c["accession"])
            it = insider_item(adapter.name, ticker, c["url"], _past(c["ts"], ctx.now), f4)
            if it:
                out.append(it)
    finally:
        _remember(ctx.state, key, done)
    return out


class SECInsider(Adapter):
    name, team, kind = "sec_insider", "filings", "filing"
    assets = ("stock",)
    interval_s = 3600.0
    per_symbol = True
    max_symbols_per_run = 4
    domains = ("www.sec.gov", "data.sec.gov")
    official = True
    description = "Insider open-market buys (>= $100k) and large officer sales (>= $1M) from Form 4 filings."

    def fetch(self, ctx):
        ciks = sec_cik_map(ctx)

        def one(t):
            if t.upper() not in ciks:
                return []
            cik = ciks[t.upper()][0]
            _, filings = submissions(ctx, cik)
            cands = [dict(f, url=filing_index_url(cik, f["accession"])) for f in filings
                     if f["form"] == "4" and f["doc"]]

            def xml_url(c):
                doc = re.sub(r"^xslF345X\d+/", "", c["doc"])   # strip the stylesheet view to get the raw XML
                return SEC_ARCHIVE.format(cik=cik, nodash=c["accession"].replace("-", "")) + urllib.parse.quote(doc)
            return _insider_run(self, ctx, t, cands, xml_url)
        return _per_ticker(ctx, one)


class SECInsiderAtom(Adapter):
    name, team, kind = "sec_insider_atom", "filings", "filing"
    assets = ("stock",)
    interval_s = 3600.0
    per_symbol = True
    max_symbols_per_run = 4
    domains = ("www.sec.gov",)
    official = True
    role, standby_for = "standby", "sec_insider"
    description = "Insider trades from Form 4s found through EDGAR's Atom feed (standby for sec_insider)."

    def fetch(self, ctx):
        def one(t):
            # type=4 is a prefix match on EDGAR (it also returns 424B*): keep exact Form 4s only
            cands = [f for f in edgar_atom(ctx, t, "4") if f["form"] == "4"]

            def xml_url(c):
                # the Atom feed doesn't name the XML document; the full submission text carries it
                return SEC_ARCHIVE.format(cik=c["cik"], nodash=c["accession"].replace("-", "")) + f"{c['accession']}.txt"
            return _insider_run(self, ctx, t, cands, xml_url)
        return _per_ticker(ctx, one)


# ---------------------------------------------------------------------------- earnings

NASDAQ_EARNINGS = "https://api.nasdaq.com/api/calendar/earnings?date={d}"
WHEN = {"time-pre-market": "before the open", "time-after-hours": "after close"}


def _calendar_days(now, n=4):
    """Today (New York) if it is a weekday, then the next n weekdays."""
    today = et_date(now)
    days = [today] if today.weekday() < 5 else []
    d = today
    while len(days) < n + (1 if today.weekday() < 5 else 0):
        d += dt.timedelta(days=1)
        if d.weekday() < 5:
            days.append(d)
    return days


class EarningsCalendar(Adapter):
    name, team, kind = "earnings_calendar", "filings", "filing"
    assets = ("stock",)
    interval_s = 6 * 3600.0
    domains = ("api.nasdaq.com",)
    description = "Upcoming earnings dates for subscribed stocks from Nasdaq's earnings calendar (today + 4 weekdays)."

    def fetch(self, ctx):
        subs = {t.upper(): t for t in _stocks(ctx)}
        if not subs:
            return []
        out, errors, ok = [], [], 0
        sent = _seen(ctx.state, "sent")
        for day in _calendar_days(ctx.now):
            try:
                data = ctx.http.json(NASDAQ_EARNINGS.format(d=day.isoformat()), headers=NASDAQ_HEADERS)
            except RateLimited:
                raise
            except SourceError as e:
                errors.append(e)
                continue
            if not isinstance(data, dict) or "data" not in data:
                errors.append(SourceError("api.nasdaq.com: earnings JSON has no data"))
                continue
            ok += 1
            rows = (data.get("data") or {}).get("rows") if isinstance(data.get("data"), dict) else None
            for r in rows if isinstance(rows, list) else ():     # null rows: nobody reports that day
                if not isinstance(r, dict):
                    continue
                sym = subs.get(str(r.get("symbol") or "").strip().upper())
                if not sym:
                    continue
                url = f"data://earnings/{sym}/{day.isoformat()}"
                if url in sent:
                    continue
                when = WHEN.get(str(r.get("time") or "").strip())
                title = f"{sym} reports earnings {_day_text(day)}" + (f" ({when})" if when else "")
                eps, n_est = _num(r.get("epsForecast")), _num(r.get("noOfEsts"))
                text = f"{clean(r.get('name')) or sym} is scheduled to report quarterly results on {_day_text(day)}"
                text += f" {when}." if when else " (time not announced)."
                if eps is not None:
                    text += f" Consensus EPS forecast {r.get('epsForecast')}" + (f" from {n_est:.0f} estimates." if n_est else ".")
                out.append(Item(source=self.name, url=url, title=title, text=text, published_ts=ctx.now, symbols=[sym],
                                kind="filing", publisher="nasdaq.com", catalyst="earnings", direction=0, confidence=0.9,
                                facts=dict(date=day.isoformat(), time=when or "unknown", eps_forecast=eps,
                                           n_estimates=n_est, fiscal_quarter=str(r.get("fiscalQuarterEnding") or ""))))
                sent.add(url)
        if errors and not ok:
            raise errors[0]
        _remember(ctx.state, "sent", [it.url for it in out])
        return out


class SEC8KEarnings(Adapter):
    name, team, kind = "sec_8k_earnings", "filings", "filing"
    assets = ("stock",)
    interval_s = 3600.0
    per_symbol = True
    max_symbols_per_run = 6
    domains = ("www.sec.gov", "data.sec.gov")
    official = True
    role, standby_for = "standby", "earnings_calendar"
    description = "Quarterly results just released: 8-K item 2.02 in the last three days (standby for earnings_calendar)."

    def fetch(self, ctx):
        ciks = sec_cik_map(ctx)

        def one(t):
            if t.upper() not in ciks:
                return []
            cik, title = ciks[t.upper()]
            name, filings = submissions(ctx, cik)
            key = "seen:" + t
            seen, out = _seen(ctx.state, key), []
            for f in filings:
                ts = _past(f["ts"], ctx.now)
                if f["form"] != "8-K" or "2.02" not in f["items"] or f["accession"] in seen or ctx.now - ts > 3 * DAY:
                    continue
                co = name or title
                out.append(Item(source=self.name, url=filing_index_url(cik, f["accession"]),
                                title=f"{co} ({t}) released quarterly results (8-K Item 2.02)",
                                text=f"{co} filed an 8-K on {f['filing_date'] or 'a recent day'} with Item 2.02, results of "
                                     f"operations and financial condition. {f.get('desc') or ''}".strip(),
                                published_ts=ts, symbols=[t], kind="filing", publisher="sec.gov",
                                facts=dict(form="8-K", items=f["items"], accession=f["accession"], filing_date=f["filing_date"]),
                                catalyst="earnings", direction=0, confidence=0.8, official=True))
            _remember(ctx.state, key, [it.facts["accession"] for it in out])
            return out
        return _per_ticker(ctx, one)


# ---------------------------------------------------------------------------- news

_YAHOO_HOSTS = ("yahoo.com", "finance.yahoo.com", "feeds.finance.yahoo.com")
_NOT_FOUND = re.compile(r"rss feed not found|no results found", re.I)


def ticker_news(source, ctx, ticker, url, default_pub, own_hosts, headers=None, max_items=20):
    """Items from one per-ticker news feed. symbols=[ticker] (the feed is a per-ticker query)."""
    entries = parse_feed(ctx.http.text(url, headers=headers))
    out = []
    for e in entries:
        title, link = e.get("title") or "", e.get("link") or ""
        if not title or not link.startswith(("http://", "https://")) or _NOT_FOUND.search(title):
            continue
        ts = e.get("published_ts")
        if ts is not None:
            if ts > ctx.now + 3600 or ctx.now - ts > NEWS_MAX_AGE_S:
                continue          # a date far in the future is a broken feed; old news is not news
            ts = min(ts, ctx.now)
        # an item that names its outlet (<source>, or a link off-site) is credited to that outlet
        pub = default_pub
        for cand in (e.get("source_url"), link):
            dom = domain(cand)
            if dom and not dom.endswith(own_hosts):
                pub = dom
                break
        out.append(Item(source=source, url=link, title=title, text=e.get("summary") or "", published_ts=ts,
                        symbols=[ticker], kind="news", publisher=pub))
        if len(out) >= max_items:
            break
    return out


class YahooRSS(Adapter):
    name, team, kind = "yahoo_rss", "catalysts", "news"
    assets = ("stock",)
    interval_s = 1800.0
    per_symbol = True
    max_symbols_per_run = 8
    domains = ("feeds.finance.yahoo.com",)
    description = "Per-ticker headlines from Yahoo Finance's RSS feed."
    URL = "https://feeds.finance.yahoo.com/rss/2.0/headline?s={t}&region=US&lang=en-US"

    def fetch(self, ctx):
        return _per_ticker(ctx, lambda t: ticker_news(self.name, ctx, t, self.URL.format(t=urllib.parse.quote(t)),
                                                      "finance.yahoo.com", _YAHOO_HOSTS, YAHOO_HEADERS))


class NasdaqRSS(Adapter):
    name, team, kind = "nasdaq_rss", "catalysts", "news"
    assets = ("stock",)
    interval_s = 1800.0
    per_symbol = True
    max_symbols_per_run = 8
    domains = ("www.nasdaq.com",)
    role, standby_for = "standby", "yahoo_rss"
    description = "Per-ticker news from Nasdaq's RSS feed (standby for yahoo_rss)."
    URL = "https://www.nasdaq.com/feed/rssoutbound?symbol={t}"

    def fetch(self, ctx):
        return _per_ticker(ctx, lambda t: ticker_news(self.name, ctx, t, self.URL.format(t=urllib.parse.quote(t)),
                                                      "nasdaq.com", ("nasdaq.com",), NASDAQ_RSS_HEADERS))


# ---------------------------------------------------------------------------- VIX

VIX_FEAR, VIX_JUMP, VIX_CALM = 30.0, 0.25, 13.0
VIX_STALE_S = 6 * DAY      # a newest bar older than this means the feed is stuck, not that the market is calm


def vix_item(source, ctx, closes):
    """closes: [(session date, close)] oldest first. One market-wide Item for every subscribed
    stock per session day, or None (no stocks, already sent today, stale or too short)."""
    stocks = _stocks(ctx)
    closes = [(d, c) for d, c in closes if c and c > 0]
    if not stocks or len(closes) < 6:
        return None
    day, vix = closes[-1]
    if (et_date(ctx.now) - day).days * DAY > VIX_STALE_S or ctx.state.get("last_day") == day.isoformat():
        return None
    chg = vix / closes[-6][1] - 1.0
    if vix >= VIX_FEAR or chg >= VIX_JUMP:
        d, mood = -1, f"fear spiking: VIX {vix:.1f}" + (f", up {chg:+.0%} in 5 sessions" if chg >= VIX_JUMP else "")
        text = "Volatility is high or rising fast; stocks tend to fall together while fear spikes."
    elif vix <= VIX_CALM:
        d, mood = 0, f"complacency: VIX only {vix:.1f}"
        text = "Volatility is unusually low. Calm markets can turn quickly; complacency is a caution, not a signal."
    else:
        d, mood = 0, f"VIX {vix:.1f}, {chg:+.0%} over 5 sessions"
        text = "Volatility is in its normal range."
    ctx.state["last_day"] = day.isoformat()
    return Item(source=source, url=f"data://vix/{day.isoformat()}", title=f"Market volatility, {mood}",
                text=text, published_ts=min(ctx.now, _et_ts(day, 16)), symbols=stocks, kind="market",
                publisher="cboe.com" if source == "cboe_vix" else "finance.yahoo.com",
                catalyst="macro", direction=d, confidence=0.45,
                facts=dict(vix=round(vix, 2), chg_5d=round(chg, 4), date=day.isoformat()))


def _chart(data, what):
    """Yahoo chart JSON -> (timestamps, closes) lists of equal length."""
    chart = data.get("chart") if isinstance(data, dict) else None
    if not isinstance(chart, dict):
        raise SourceError(f"{what}: chart JSON has no chart")
    err = chart.get("error")
    if err:
        desc = err.get("description") if isinstance(err, dict) else err
        raise SourceError(f"{what}: {str(desc)[:120]}")
    res = chart.get("result")
    if not isinstance(res, list) or not res or not isinstance(res[0], dict):
        raise SourceError(f"{what}: chart JSON has no result")
    r = res[0]
    ts = r.get("timestamp") or []
    q = ((r.get("indicators") or {}).get("quote") or [{}])
    close = q[0].get("close") if q and isinstance(q[0], dict) else None
    if not isinstance(ts, list) or not isinstance(close, list):
        raise SourceError(f"{what}: chart JSON has no timestamps/closes")
    return ts, close


class YahooVIX(Adapter):
    name, team, kind = "yahoo_vix", "market", "market"
    assets = ("stock",)
    interval_s = 3600.0
    domains = ("query1.finance.yahoo.com",)
    description = "The VIX from Yahoo Finance: one market-wide fear/complacency reading per session day."
    URL = "https://query1.finance.yahoo.com/v8/finance/chart/%5EVIX?interval=1d&range=3mo"

    def fetch(self, ctx):
        if not _stocks(ctx):
            return []
        ts, close = _chart(ctx.http.json(self.URL, headers=YAHOO_HEADERS), "yahoo vix")
        by_day = {}
        for t, c in zip(ts, close):
            t, c = _num(t), _num(c)
            if t and c and t <= ctx.now:
                by_day[et_date(t)] = c
        it = vix_item(self.name, ctx, sorted(by_day.items()))
        return [it] if it else []


class CboeVIX(Adapter):
    name, team, kind = "cboe_vix", "market", "market"
    assets = ("stock",)
    interval_s = 3600.0
    domains = ("cdn.cboe.com",)
    role, standby_for = "standby", "yahoo_vix"
    description = "The VIX daily history CSV from Cboe (standby for yahoo_vix)."
    URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"

    def fetch(self, ctx):
        if not _stocks(ctx):
            return []
        body = ctx.http.text(self.URL, accept="text/csv, */*;q=0.5")
        rows = list(csv.reader(io.StringIO((body or "").lstrip("﻿"))))
        head = [h.strip().upper() for h in rows[0]] if rows else []
        if "DATE" not in head or "CLOSE" not in head:
            raise SourceError("cdn.cboe.com: VIX CSV has no DATE/CLOSE header")
        di, ci = head.index("DATE"), head.index("CLOSE")
        out = {}
        for r in rows[-60:]:
            if len(r) <= max(di, ci):
                continue
            try:
                day = dt.datetime.strptime(r[di].strip(), "%m/%d/%Y").date()
            except ValueError:
                continue
            c = _num(r[ci])
            if c and day <= et_date(ctx.now):
                out[day] = c
        if not out:
            raise SourceError("cdn.cboe.com: VIX CSV has no readable rows")
        it = vix_item(self.name, ctx, sorted(out.items()))
        return [it] if it else []


# ---------------------------------------------------------------------------- prices (not an adapter)

def yahoo_hourly(http, symbol, rng="5d", now=None) -> list:
    """[(bar_start_ts, close)] regular-session hourly bars from Yahoo's chart API, oldest first.
    Null closes (halts, thin bars) are dropped, and so is the still-forming last bar."""
    now = time.time() if now is None else now
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(str(symbol))}"
           f"?interval=1h&range={urllib.parse.quote(str(rng))}&includePrePost=false")
    ts, close = _chart(http.json(url, headers=YAHOO_HEADERS), f"yahoo chart {symbol}")
    out = {}
    for t, c in zip(ts, close):
        t, c = _num(t), _num(c)
        if t and c and c > 0 and t + 3600 <= now:
            out[int(t)] = float(c)
    return sorted(out.items())


ADAPTERS = [SECFilings, SECFilingsAtom, SECInsider, SECInsiderAtom, EarningsCalendar, SEC8KEarnings,
            YahooRSS, NasdaqRSS, YahooVIX, CboeVIX]
