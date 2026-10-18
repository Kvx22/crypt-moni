"""Text helpers for the research adapters: everything that turns raw internet text into
something the labeller can read.

  clean          HTML fragment -> plain text (tags gone, entities decoded, spaces collapsed)
  parse_ts       RSS (RFC-822) and Atom (ISO-8601) dates -> unix seconds, None if unreadable
  parse_feed     RSS 2.0, Atom 1.0 and RSS 1.0/RDF -> a list of plain entry dicts
  html_to_text   a web page -> (title, main text), without a browser
  domain         "https://www.coindesk.com/x" -> "coindesk.com"
  tokens         lowercase word tokens that keep $cashtags, numbers and percentages
  fingerprint    a 64-bit SimHash of a story, so syndicated copies of the same article
                 (same words, different outlet) can be counted once, not five times

Inputs come from the open internet, so every function accepts junk: a value of the wrong
type, broken markup or an impossible date gives an empty result, never an exception. The
one exception is parse_feed, which raises SourceError when the body is not a feed at all,
so the worker can mark the source as failing instead of silently reporting "no news".
"""
from __future__ import annotations

import datetime as dt
import email.utils
import hashlib
import html
import re
import urllib.parse
import xml.etree.ElementTree as ET
from html.parser import HTMLParser

from .http import SourceError

SUMMARY_MAX = 1500

_TAG = re.compile(r"<[A-Za-z/!?][^>]*>", re.S)
_HTML_TAG = re.compile(r"</?[A-Za-z][^>]*>", re.S)
_DROP_BLOCKS = re.compile(r"<(script|style|noscript)\b[^>]*>.*?</\1\s*>", re.S | re.I)
_COMMENT = re.compile(r"<!--.*?-->", re.S)
_WS = re.compile(r"\s+")


def _str(s):
    if isinstance(s, bytes):
        return s.decode("utf-8", "replace")
    return s if isinstance(s, str) else ""


def clean(s) -> str:
    """HTML fragment -> one line of plain text."""
    s = _str(s)
    if not s:
        return ""
    s = _TAG.sub(" ", _DROP_BLOCKS.sub(" ", _COMMENT.sub(" ", s)))
    s = html.unescape(s)
    # feeds often escape their HTML twice: after one unescape the tags reappear
    if "<" in s:
        s = html.unescape(_HTML_TAG.sub(" ", _DROP_BLOCKS.sub(" ", s)))
    return _WS.sub(" ", s.replace("\u00a0", " ")).strip()


def _cut(s, n):
    """Cut at a word boundary, marking the cut with an ellipsis, never longer than n."""
    if len(s) <= n:
        return s
    head = s[: n - 1]
    sp = head.rfind(" ")
    if sp > n * 0.8:
        head = head[:sp]
    return head.rstrip(" ,;:-") + "\u2026"


# ---------------------------------------------------------------------------- dates

_ISO = re.compile(
    r"^(\d{4})-(\d{2})-(\d{2})"
    r"(?:[Tt ](\d{2}):(\d{2})(?::(\d{2})(?:[.,](\d+))?)?)?"
    r"\s*(Z|z|UTC|GMT|[+-]\d{2}(?::?\d{2})?)?$")


def _ts(d):
    if d.tzinfo is None:   # feeds that leave the zone out almost always mean UTC
        d = d.replace(tzinfo=dt.timezone.utc)
    t = d.timestamp()
    return t if 0 < t < 32503680000 else None   # 1970 .. 3000: anything else is garbage


def parse_ts(s) -> float | None:
    """RFC-822 ("Sat, 18 Oct 2026 05:16:29 GMT") or ISO-8601 ("2026-10-18T05:16:29Z",
    "+00:00", fractional seconds, date only) -> unix seconds UTC. None if unreadable."""
    if isinstance(s, (int, float)) and not isinstance(s, bool):
        return float(s) if 0 < s < 32503680000 else None
    s = _str(s).strip()
    if not s or len(s) > 64:
        return None
    m = _ISO.match(s)
    try:
        if m:
            y, mo, d, h, mi, sec, frac, tz = m.groups()
            micro = int((frac or "0")[:6].ljust(6, "0"))
            zone = dt.timezone.utc
            if tz and tz[0] in "+-":
                digits = tz[1:].replace(":", "")
                off = dt.timedelta(hours=int(digits[:2]), minutes=int(digits[2:4] or 0))
                zone = dt.timezone(off if tz[0] == "+" else -off)
            return _ts(dt.datetime(int(y), int(mo), int(d), int(h or 0), int(mi or 0),
                                   int(sec or 0), micro, tzinfo=zone))
        return _ts(email.utils.parsedate_to_datetime(s))
    except (TypeError, ValueError, OverflowError, IndexError):
        return None


# ---------------------------------------------------------------------------- feeds

ATOM = "http://www.w3.org/2005/Atom"
_KNOWN_NS = {"", ATOM, "http://purl.org/rss/1.0/", "http://purl.org/dc/elements/1.1/",
             "http://purl.org/rss/1.0/modules/content/", "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
             "http://backend.userland.com/rss2", "http://my.netscape.com/rdf/simple/0.9/"}
_RDF_ABOUT = "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}about"
_DOCTYPE = re.compile(r"<!DOCTYPE(?:[^\[>]*\[.*?\])?[^>]*>", re.S | re.I)
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\ufffe\uffff]")
_BARE_AMP = re.compile(r"&(?!(?:amp|lt|gt|quot|apos|#\d{1,7}|#[xX][0-9a-fA-F]{1,6});)")


def _local(tag):
    """'{ns}name' -> ('ns', 'name'); comments and processing instructions -> ('?', '')."""
    if not isinstance(tag, str):
        return "?", ""
    if tag.startswith("{"):
        ns, _, name = tag[1:].partition("}")
        return ns, name
    return "", tag


_DECL_ENC = re.compile(rb"""^\s*<\?xml[^>]*?encoding\s*=\s*["']([A-Za-z0-9._-]{1,40})["']""")


def _decode(raw):
    """Bytes in the encoding the XML declaration names (UTF-8 if none, or one we don't know)."""
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16", "replace")
    raw = raw[3:] if raw.startswith(b"\xef\xbb\xbf") else raw
    m = _DECL_ENC.match(raw)
    try:
        return raw.decode(m.group(1).decode("ascii") if m else "utf-8", "replace")
    except LookupError:
        return raw.decode("utf-8", "replace")


def _parse_xml(text):
    text = _decode(text) if isinstance(text, bytes) else _str(text)
    # Feeds never need a DTD; dropping it also defuses entity-expansion bombs.
    text = _DOCTYPE.sub("", text.lstrip("\ufeff \t\r\n"))
    try:
        return ET.fromstring(text)
    except ET.ParseError:
        pass
    fixed = _BARE_AMP.sub("&amp;", _CTRL.sub("", text))
    try:
        return ET.fromstring(fixed)
    except ET.ParseError:
        raise SourceError("feed: not valid XML") from None


def _text(el):
    return "".join(el.itertext()) if el is not None else ""


def _http(u):
    u = _str(u).strip()
    return u if u.lower().startswith(("http://", "https://")) else ""


def _atom_link(links):
    best = ""
    for el in links:
        href, rel = _http(el.get("href")), (el.get("rel") or "alternate").strip().lower()
        if href and rel == "alternate" and el.get("type", "text/html").startswith(("text/html", "application/xhtml")):
            return href
        best = best or (href if rel in ("alternate", "") else "")
    return best or next((_http(el.get("href")) for el in links if _http(el.get("href"))), "")


def _entry(el, default_name, default_url):
    first, multi = {}, {}
    for c in el:
        ns, name = _local(c.tag)
        if ns not in _KNOWN_NS or not name:
            continue   # media:title, itunes:summary, ... would shadow the real fields
        first.setdefault(name, c)
        multi.setdefault(name, []).append(c)

    title = clean(_text(first.get("title")))
    links = multi.get("link", [])
    link = _atom_link([l for l in links if l.get("href")]) or next(
        (_http(_text(l)) for l in links if _http(_text(l))), "")
    guid_el = first.get("guid") if "guid" in first else first.get("id")
    guid = _text(guid_el).strip() or (el.get(_RDF_ABOUT) or "").strip()
    if not link and guid_el is not None and (guid_el.get("isPermaLink") or "true").lower() != "false":
        link = _http(guid)
    summary = ""
    for k in ("description", "summary", "encoded", "content"):
        summary = clean(_text(first.get(k)))
        if summary:
            break
    if not title:
        title = _cut(summary, 140)
    if not title and not link:
        return None
    published = None
    for k in ("pubDate", "published", "issued", "date", "updated", "modified", "created"):
        published = parse_ts(_text(first.get(k)))
        if published:
            break
    cats = []
    for c in multi.get("category", []) + multi.get("subject", []):
        v = clean(c.get("term") or c.get("label") or _text(c))
        if v and v not in cats:
            cats.append(v[:80])
    src_name, src_url = default_name, default_url
    src = first.get("source")
    if src is not None:
        if src.get("url") is not None:                      # RSS: <source url="...">Name</source>
            src_name, src_url = clean(_text(src)) or src_name, _http(src.get("url")) or src_url
        else:                                               # Atom: <source><title/><link/></source>
            kids = {_local(k.tag)[1]: k for k in src}
            src_name = clean(_text(kids.get("title"))) or src_name
            src_url = _atom_link([k for k in src if _local(k.tag)[1] == "link"]) or src_url
    return {"title": title, "link": link, "summary": _cut(summary, SUMMARY_MAX),
            "published_ts": published, "guid": guid[:500] or link or title,
            "source_name": src_name, "source_url": src_url, "categories": cats[:20]}


def parse_feed(xml_text) -> list[dict]:
    """RSS 2.0 / Atom 1.0 / RSS 1.0 (RDF) -> entry dicts. Entries without a title or a
    link are skipped. source_name/source_url come from the entry's <source> when it has
    one (Google News) and from the feed's own title/link otherwise.
    Raises SourceError when the body is not XML or not a feed."""
    root = _parse_xml(xml_text)
    root_name = _local(root.tag)[1].lower()
    if root_name not in ("rss", "feed", "rdf", "channel"):
        raise SourceError(f"feed: not RSS/Atom (root <{root_name[:20]}>)")
    channel = root
    for c in root:
        if _local(c.tag)[1] == "channel":
            channel = c
            break
    feed_name, links = "", []
    for c in channel:
        ns, name = _local(c.tag)
        if ns not in _KNOWN_NS:
            continue
        if name == "title" and not feed_name:
            feed_name = clean(_text(c))[:200]
        elif name == "link":
            links.append(c)
    # Atom lists rel="self" (the feed itself) next to rel="alternate" (the site): want the site
    feed_url = _atom_link([l for l in links if l.get("href")]) or next(
        (_http(_text(l)) for l in links if _http(_text(l))), "")
    out = []
    for el in root.iter():
        if _local(el.tag)[1] in ("item", "entry") and _local(el.tag)[0] in _KNOWN_NS:
            try:
                e = _entry(el, feed_name, feed_url)
            except Exception:   # one odd entry never costs the rest of the feed
                e = None
            if e:
                out.append(e)
    return out


# ---------------------------------------------------------------------------- pages

_IGNORE = {"script", "style", "noscript", "nav", "header", "footer", "aside", "form", "svg",
           "iframe", "template", "button", "select", "textarea", "object", "canvas"}
_BLOCK = {"p", "li", "h1", "h2", "h3", "h4", "h5", "h6", "blockquote", "pre", "dd", "dt", "figcaption"}
_CONTAINER = {"div", "section", "main", "article", "td", "body", "table", "ul", "ol"}
_VOID = {"br", "hr", "img", "meta", "link", "input", "source", "wbr", "area", "base", "col", "embed", "param", "track"}
_HEADING = {"h1", "h2", "h3", "h4", "h5", "h6"}


class _Page(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.skip = 0                 # depth inside ignored elements
        self.stack = [0]              # open containers (ids); 0 = the document
        self.next_id = 1
        self.articles = set()         # container ids that are <article>
        self.blocks = []              # (text, tag, container path, link chars)
        self.cur, self.cur_tag, self.cur_links, self.in_a = [], None, 0, 0
        self.title, self.og_title, self.in_title = "", "", False

    def _flush(self):
        t = _WS.sub(" ", "".join(self.cur)).strip()
        if t:
            # the nearest dozen containers are plenty to find the main one, and keep absurdly
            # nested pages linear
            self.blocks.append((t, self.cur_tag or "div", tuple(self.stack[-12:]), self.cur_links))
        self.cur, self.cur_tag, self.cur_links = [], None, 0

    def handle_starttag(self, tag, attrs):
        if tag == "meta":
            a = dict(attrs)
            key = (a.get("property") or a.get("name") or "").lower()
            if key in ("og:title", "twitter:title") and not self.og_title:
                self.og_title = clean(a.get("content"))
            return
        if tag in _VOID:
            if tag == "br" and not self.skip:
                self.cur.append(" ")
            return
        if tag in _IGNORE:
            self.skip += 1
        elif self.skip:
            return
        elif tag == "title":
            self.in_title = True
        elif tag in _CONTAINER:
            self._flush()
            self.stack.append(self.next_id)
            if tag == "article":
                self.articles.add(self.next_id)
            self.next_id += 1
        elif tag in _BLOCK:
            self._flush()
            self.cur_tag = tag
        elif tag == "a":
            self.in_a += 1

    def handle_endtag(self, tag):
        if tag in _IGNORE:
            self.skip = max(0, self.skip - 1)
        elif tag == "title":
            self.in_title = False
        elif self.skip:
            return
        elif tag in _CONTAINER:
            self._flush()
            if len(self.stack) > 1:
                self.stack.pop()
        elif tag in _BLOCK:
            self._flush()
        elif tag == "a":
            self.in_a = max(0, self.in_a - 1)

    def handle_data(self, data):
        if self.in_title:
            self.title += data
        elif not self.skip:
            self.cur.append(data)
            if self.in_a:
                self.cur_links += len(data.strip())


def _menu_like(text, tag, links):
    if links > 0.5 * len(text):
        return True
    return len(text) < 40 and tag not in _HEADING and not text.rstrip("\"'\u201d)").endswith((".", "!", "?", ":"))


def html_to_text(html_text, max_chars=4000) -> tuple[str, str]:
    """A web page -> (title, main text). The text comes from the page's <article> when it
    has one, else from the container holding the most paragraph text. Navigation, ads
    furniture and other short link-heavy blocks are dropped; paragraphs join with newlines."""
    s = _str(html_text)
    try:
        max_chars = max(1, int(max_chars))
    except (TypeError, ValueError):
        max_chars = 4000
    if not s.strip():
        return "", ""
    p = _Page()
    try:
        p.feed(s)
        p.close()
    except Exception:   # HTMLParser is lenient; this only guards against the truly weird
        pass
    p._flush()
    title = p.og_title or clean(p.title)
    blocks = [b for b in p.blocks if not _menu_like(b[0], b[1], b[3])]
    if not title:
        title = next((b[0] for b in p.blocks if b[1] == "h1"), "")

    def size(cid):
        return sum(len(b[0]) for b in blocks if cid in b[2])

    best = max(p.articles, key=size, default=None)
    if best is None or size(best) < 80:
        # score every container by the text under it, nearer containers counting more
        score = {}
        for text, tag, path, _ in blocks:
            n = len(text) if len(text) >= 40 else 0
            for depth, cid in enumerate(reversed(path)):
                score[cid] = score.get(cid, 0) + n / (1 + depth)
        best = max(score, key=score.get, default=0)
    out, total = [], 0
    for text, tag, path, _ in blocks:
        if best not in path or (out and text == out[-1]):
            continue
        out.append(text)
        total += len(text) + 1
        if total >= max_chars:
            break
    return title[:300], _cut("\n".join(out), max_chars)


# ---------------------------------------------------------------------------- urls, words

def domain(url) -> str:
    """Lowercase host without "www." / "m." and without port or credentials."""
    u = _str(url).strip()
    if not u:
        return ""
    if "//" not in u:
        u = "//" + u
    try:
        host = urllib.parse.urlsplit(u).hostname or ""
    except ValueError:
        return ""
    host = host.strip(".")
    for prefix in ("www.", "m."):
        if host.startswith(prefix) and host.count(".") >= 2:
            host = host[len(prefix):]
    return host


_TOKEN = re.compile(r"\$[^\W\d_][^\W_]{0,11}|\d+(?:[.,]\d+)*%?(?![^\W_])|[^\W_]+(?:['\u2019][^\W_]+)*")


def tokens(text) -> list[str]:
    """Lowercase word tokens: "$SOL up 5.5% on Solana's news" -> ['$sol', 'up', '5.5%', 'on', "solana's", 'news']."""
    return [t for t in (m.lower() for m in _TOKEN.findall(_str(text))) if len(t) > 1 or t.isdigit()]


# ---------------------------------------------------------------------------- near-duplicates

_OUTLET_SUFFIX = re.compile(r"\s+[-|\u2013\u2014]\s+[^-|\u2013\u2014]{1,60}$")
_ZERO = "0" * 16


def _norm_title(title):
    # "Bitcoin tops $100k - CoinDesk" and "... | Reuters": the outlet tag is not the story
    t = clean(title)
    m = _OUTLET_SUFFIX.search(t)
    if m and len(m.group(0).split()) <= 6 and len(t) - len(m.group(0)) >= 15:
        t = t[: m.start()]
    return t


def fingerprint(title, text="") -> str:
    """64-bit SimHash (16 hex chars) over word 3-shingles of the title and the first 600
    characters of the text. Copies of one story differ in a few bits; different stories in ~32."""
    words = tokens(_norm_title(title) + " " + clean(text)[:600])
    if not words:
        return _ZERO
    grams = [" ".join(words[i:i + 3]) for i in range(max(1, len(words) - 2))]
    acc = [0] * 64
    for g in grams:
        h = int.from_bytes(hashlib.blake2b(g.encode("utf-8"), digest_size=8).digest(), "big")
        for i in range(64):
            acc[i] += 1 if (h >> i) & 1 else -1
    return f"{sum(1 << i for i in range(64) if acc[i] > 0):016x}"


def hamming(a, b) -> int:
    """Bits that differ between two fingerprints (64 when either is unreadable)."""
    try:
        return bin(int(a, 16) ^ int(b, 16)).count("1") if len(a) == len(b) == 16 else 64
    except (TypeError, ValueError):
        return 64


def same_story(a, b, max_bits=8) -> bool:
    """True when two fingerprints are close enough to be copies of one story."""
    if a == _ZERO or b == _ZERO:
        return False
    return hamming(a, b) <= max_bits
