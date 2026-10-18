"""Forever search: a sandboxed worker that keeps searching the open web for news about
the names the boards hold or watch, and the supervisors that keep it running.

The idea in plain English
  The research adapters read sources we chose. Forever search reads everything else: it
  rotates through topic queries ("token unlock", "SEC charges", "guidance cut", ...) and
  one query per subscribed name across a few news-search backends, follows each new link
  (politely: robots.txt, size caps, never a private address), extracts the article text
  and stores it in its OWN small database. The research service later copies those rows
  into the research database as QUARANTINED items: they count for nothing until ordinary
  sources say the same thing.

The sandbox (option C of the design)
  The worker is a separate process started with a stripped environment (no API keys, no
  tokens, nothing named CM_*), working in data/forever, importing only the research http
  and text helpers and this file. It never opens the boards' databases, never reads .env
  and cannot place an order. It exits when its stdin closes (the parent died), so an
  orphan never keeps crawling. Everything is Windows-safe: no fork, no signals, no
  os.kill(pid, 0); liveness is Popen.poll() and a heartbeat row.

Pieces
  QuarantineDB      the worker's SQLite file: qitems (what it found) and qheart (beats)
  safe_env          the minimal environment the worker is started with
  Worker / main     the worker loop (`python -m cryptomoni.research.forever --db ... --name ... --plan ...`)
  ForeverSupervisor starts the worker, restarts it with back-off, switches to a standby plan
  SupervisorPair    two supervisors: A works, B watches A's beat and takes over if A stalls
  ingest            quarantine rows -> research items (team "discovery", quarantine=True)
  build_plan        the worker's plan: topic queries + one name query per subscribed symbol
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import threading
import time
import traceback
import urllib.parse
import zlib

from . import text as T
from .http import DEFAULT_UA, Blocked, Http, RateLimited, SourceError

QTEXT_MAX = 4000
HEARTBEAT_S = 10.0
HEALTHY_S = 60.0           # a heartbeat older than this means "not healthy"
HUNG_S = 180.0             # no heartbeat this long after start (or since the last one): restart the worker
STALLED_S = 900.0          # the loop made no progress this long although it beats: restart the worker
BACKOFF_S = (10.0, 30.0, 120.0, 600.0)
CRASH_WINDOW_S = 600.0     # 3 crashes inside this window -> next standby plan
CRASHES_TO_SWITCH = 3
TAKEOVER_S = 90.0          # B takes over when A's beat is older than this
LOG_MAX_BYTES = 5 * 1024 * 1024
BACKENDS = ("google_news", "bing_news", "searxng", "fake")
DEFAULT_BACKENDS = ["google_news", "bing_news", "searxng"]
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _uhash(url):
    # same as db.url_hash (the worker may not import db.py)
    return hashlib.sha1((url or "").strip().encode("utf-8", "ignore")).hexdigest()


def _now_iso():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())


# ---------------------------------------------------------------------------- quarantine database

QSCHEMA = """
CREATE TABLE IF NOT EXISTS qitems(id INTEGER PRIMARY KEY, url TEXT, url_hash TEXT UNIQUE, domain TEXT, query TEXT,
  backend TEXT, fetched_ts REAL, published_ts REAL, title TEXT, text TEXT, ingested INTEGER DEFAULT 0);
CREATE INDEX IF NOT EXISTS qitems_ing ON qitems(ingested, id);
CREATE TABLE IF NOT EXISTS qheart(name TEXT PRIMARY KEY, pid INTEGER, heartbeat REAL, stats TEXT);
"""


class QuarantineDB:
    """The forever worker's own SQLite file. The worker writes qitems and its heartbeat;
    the research service reads the beats and takes/marks qitems. Thread-safe."""

    def __init__(self, path):
        self.path = path
        d = os.path.dirname(os.path.abspath(path))
        os.makedirs(d, exist_ok=True)
        self.c = sqlite3.connect(path, timeout=30, check_same_thread=False, isolation_level=None)
        self.c.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        self.last_prune = 0.0
        with self.lock:
            self.c.execute("PRAGMA busy_timeout=30000")
            try:
                self.c.execute("PRAGMA journal_mode=WAL")
            except sqlite3.DatabaseError:
                pass   # e.g. a network drive without WAL support: the default journal still works
            self.c.executescript(QSCHEMA)

    def close(self):
        with self.lock:
            self.c.close()

    def add(self, rows):
        """rows: dicts with url, title and optionally domain, query, backend, fetched_ts,
        published_ts, text. Duplicate URLs and rows without url/title are skipped.
        Returns the number actually stored."""
        vals = []
        for r in rows or ():
            if not isinstance(r, dict):
                continue
            url, title = str(r.get("url") or "").strip(), str(r.get("title") or "").strip()
            if not url or not title:
                continue
            vals.append((url[:1000], _uhash(url), str(r.get("domain") or "")[:120], str(r.get("query") or "")[:300],
                         str(r.get("backend") or "")[:40], _num(r.get("fetched_ts"), time.time()),
                         _num(r.get("published_ts"), None), title[:500], str(r.get("text") or "")[:QTEXT_MAX]))
        if not vals:
            return 0
        with self.lock:
            before = self.c.total_changes
            self.c.execute("BEGIN IMMEDIATE")
            try:
                self.c.executemany("""INSERT OR IGNORE INTO qitems(url,url_hash,domain,query,backend,fetched_ts,published_ts,
                                      title,text) VALUES(?,?,?,?,?,?,?,?,?)""", vals)
                self.c.execute("COMMIT")
            except Exception:
                self.c.execute("ROLLBACK")
                raise
            return self.c.total_changes - before

    def has(self, url_hash):
        with self.lock:
            return self.c.execute("SELECT 1 FROM qitems WHERE url_hash=?", (url_hash,)).fetchone() is not None

    def beat(self, name, pid, stats, ts=None):
        with self.lock:
            self.c.execute("""INSERT INTO qheart(name,pid,heartbeat,stats) VALUES(?,?,?,?) ON CONFLICT(name) DO UPDATE SET
                              pid=excluded.pid, heartbeat=excluded.heartbeat, stats=excluded.stats""",
                           (str(name), int(pid or 0), time.time() if ts is None else float(ts),
                            json.dumps(stats or {}, default=str)))

    def hearts(self):
        """{name: dict(pid, heartbeat, stats)}"""
        with self.lock:
            rows = self.c.execute("SELECT * FROM qheart").fetchall()
        out = {}
        for r in rows:
            try:
                stats = json.loads(r["stats"] or "{}")
            except ValueError:
                stats = {}
            out[r["name"]] = dict(pid=r["pid"], heartbeat=r["heartbeat"] or 0.0, stats=stats if isinstance(stats, dict) else {})
        return out

    def take(self, limit=300):
        """Rows not yet ingested, oldest first (plain dicts)."""
        with self.lock:
            rows = self.c.execute("SELECT * FROM qitems WHERE ingested=0 ORDER BY id LIMIT ?", (int(limit),)).fetchall()
        return [dict(r) for r in rows]

    def mark(self, ids):
        ids = [int(i) for i in ids or ()]
        if not ids:
            return
        with self.lock:
            self.c.execute("BEGIN IMMEDIATE")
            try:
                self.c.executemany("UPDATE qitems SET ingested=1 WHERE id=?", [(i,) for i in ids])
                self.c.execute("COMMIT")
            except Exception:
                self.c.execute("ROLLBACK")
                raise

    def counts(self):
        with self.lock:
            r = self.c.execute("SELECT COUNT(*) n, SUM(ingested=0) pending FROM qitems").fetchone()
        return dict(n=r["n"] or 0, pending=r["pending"] or 0)

    def prune(self, keep_days=21, now=None):
        """Forget ingested rows older than keep_days (the research DB keeps its own copy).
        Their URLs can then be stored again, which the research DB dedupes anyway."""
        now = time.time() if now is None else now
        with self.lock:
            cur = self.c.execute("DELETE FROM qitems WHERE ingested=1 AND fetched_ts<?", (now - keep_days * 86400,))
            return cur.rowcount


def _num(v, default):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return f if f == f and abs(f) != float("inf") else default


# ---------------------------------------------------------------------------- sandbox environment

_KEEP = {"PATH", "SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT", "TEMP", "TMP", "TMPDIR", "HOME", "USERPROFILE",
         "LANG", "LC_ALL", "PYTHONIOENCODING", "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "SSL_CERT_FILE",
         "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE"}
_DENY = ("KEY", "SECRET", "TOKEN", "PASS", "CM_")


def _secretish(name):
    u = name.upper()
    return any(d in u for d in _DENY)


def safe_env(base=None, extra=None) -> dict:
    """The minimal environment the worker is started with: OS essentials, locale, proxy and
    CA settings only. Anything whose name mentions KEY / SECRET / TOKEN / PASS / CM_ is never
    passed, even if it is on the keep list or in `extra`. Names are matched case-blind
    (Windows environment names are)."""
    base = os.environ if base is None else base
    out = {}
    for k, v in dict(base).items():
        if isinstance(k, str) and isinstance(v, str) and k.upper() in _KEEP and not _secretish(k):
            out[k] = v
    for k, v in (extra or {}).items():
        if isinstance(k, str) and v is not None and not _secretish(k):
            out[k] = str(v)
    return out


# ---------------------------------------------------------------------------- search backends

def _result(title, link, snippet="", publisher="", published=None):
    return dict(title=title, link=link, snippet=snippet, publisher=publisher, published=published)


def google_news_url(query):
    return "https://news.google.com/rss/search?" + urllib.parse.urlencode(
        {"q": query, "hl": "en-US", "gl": "US", "ceid": "US:en"})


def bing_news_url(query):
    return "https://www.bing.com/news/search?" + urllib.parse.urlencode({"q": query, "format": "rss", "count": "30"})


def parse_google_news(xml_text):
    """Google News RSS -> results. Titles end in " - Publisher"; the link is a
    news.google.com redirect (never fetched); <source url=...> names the outlet."""
    out = []
    for e in T.parse_feed(xml_text):
        title, link = e.get("title") or "", e.get("link") or ""
        pub = e.get("source_name") or ""
        if pub and title.endswith(" - " + pub):
            title = title[: -len(pub) - 3].rstrip()
        snippet = e.get("summary") or ""
        if snippet.startswith(title):            # Google's description is usually "Title  Publisher"
            snippet = ""
        dom = T.domain(e.get("source_url") or "") or "news.google.com"
        if title and link:
            out.append(_result(title, link, snippet, dom, e.get("published_ts")))
    return out


def unwrap_bing(link):
    """Bing News RSS links go through bing.com/news/apiclick.aspx?...&url=<real url>:
    return the real article URL (or the link unchanged when it is not wrapped)."""
    try:
        p = urllib.parse.urlsplit(link)
    except ValueError:
        return link
    if (p.hostname or "").lower().endswith("bing.com"):
        for k, v in urllib.parse.parse_qsl(p.query, keep_blank_values=False):
            if k.lower() == "url" and v.lower().startswith(("http://", "https://")):
                return v
    return link


def parse_bing_news(xml_text):
    out = []
    for e in T.parse_feed(xml_text):
        link = unwrap_bing(e.get("link") or "")
        if e.get("title") and link:
            out.append(_result(e["title"], link, e.get("summary") or "", T.domain(link), e.get("published_ts")))
    return out


def searxng_search_url(base, query):
    return base.rstrip("/") + "/search?" + urllib.parse.urlencode(
        {"q": query, "format": "json", "categories": "news", "language": "en"})


def parse_searxng(data):
    """SearXNG JSON ({"results": [{url, title, content, publishedDate, engine}]}) -> results."""
    if not isinstance(data, dict) or not isinstance(data.get("results"), list):
        raise SourceError("searxng: unexpected response shape")
    out = []
    for r in data["results"]:
        if not isinstance(r, dict):
            continue
        url, title = str(r.get("url") or ""), T.clean(r.get("title") or "")
        if not title or not url.lower().startswith(("http://", "https://")):
            continue
        out.append(_result(title, url, T.clean(r.get("content") or ""), T.domain(url), T.parse_ts(r.get("publishedDate"))))
    return out


FAKE_PAGE = ("<html><head><title>{t}</title></head><body><nav>Home | Markets</nav><article>"
             "<h1>{t}</h1><p>{t}. This is a canned article body used only by the forever-search tests, "
             "long enough to pass the page extractor's paragraph filter.</p>"
             "<p>A second paragraph mentions the query again so the text is not empty.</p></article></body></html>")


def fake_search(query, n=3):
    """Canned results for tests (only when CM_FOREVER_FAKE=1 and the plan names "fake")."""
    tag = f"{zlib.crc32(query.encode('utf-8')):08x}"
    out = []
    for i in range(n):
        link = f"https://fake.example/{tag}/{i}"
        if i == 1:   # one Bing-style wrapped link, to exercise unwrapping
            link = "https://www.bing.com/news/apiclick.aspx?ref=FexRss&aid=&url=" + urllib.parse.quote(link, safe="") + "&c=1"
        out.append(_result(f"{query} story {i}", link, f"Snippet about {query} ({i})", "fake.example", 1_800_000_000.0 + i))
    return out


def fake_page(url):
    return T.html_to_text(FAKE_PAGE.format(t="Fake page " + url.rsplit("/", 2)[-2]))


# ---------------------------------------------------------------------------- budget

class Budget:
    """Token bucket: `per_hour` requests an hour, with a small burst (about two minutes' worth)."""

    def __init__(self, per_hour, clock=time.monotonic):
        self.clock = clock
        self.set_rate(per_hour)
        self.tokens = min(self.cap, 3.0)
        self.t = clock()

    def set_rate(self, per_hour):
        self.rate = max(1.0, _num(per_hour, 240.0)) / 3600.0
        self.cap = max(2.0, self.rate * 120.0)

    def wait_s(self):
        """Seconds until one request may go (0 = now; the token is then spent)."""
        now = self.clock()
        self.tokens = min(self.cap, self.tokens + (now - self.t) * self.rate)
        self.t = now
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return 0.0
        return (1.0 - self.tokens) / self.rate


# ---------------------------------------------------------------------------- the worker

def load_plan(path):
    """Read and normalise a plan file. Raises ValueError when it is unusable."""
    with open(path, "r", encoding="utf-8") as f:
        p = json.load(f)
    if not isinstance(p, dict):
        raise ValueError("plan is not a JSON object")
    q = [str(x).strip()[:200] for x in (p.get("queries") or []) if isinstance(x, (str, int, float)) and str(x).strip()]
    b = [x for x in (p.get("backends") or []) if x in BACKENDS]
    return dict(queries=q, backends=b, searxng_url=p.get("searxng_url") or None,
                per_hour=max(1.0, _num(p.get("per_hour"), 240.0)), page_fetch=bool(p.get("page_fetch", True)),
                max_page_bytes=int(max(16_384, min(_num(p.get("max_page_bytes"), 1_500_000), 10_000_000))),
                per_query=int(max(1, min(_num(p.get("per_query"), 10), 50))),
                user_agent=str(p.get("user_agent") or DEFAULT_UA)[:200])


class Worker:
    """One forever-search worker. `search` and `fetch_page` can be injected (tests);
    otherwise they use the real backends through a guarded Http client."""

    def __init__(self, qdb, name, plan_path, stop=None, search=None, fetch_page=None, log=None,
                 clock=time.time, sleep=None):
        self.qdb, self.name, self.plan_path = qdb, name, plan_path
        self.stop = stop or threading.Event()
        self.clock = clock
        self.sleep = sleep or self.stop.wait
        self.log = log or (lambda m: print(f"{_now_iso()} {name}: {m}", file=sys.stderr, flush=True))
        self.fake = os.environ.get("CM_FOREVER_FAKE") == "1"
        self._search, self._fetch_page = search, fetch_page
        self.stats = dict(fetched=0, stored=0, errors=0, blocked=0, dup=0, queries_done=0, started=clock(),
                          progress_ts=clock(), last_error="")
        self.plan, self.plan_mtime, self.plan_check = None, None, 0.0
        self.backend_until, self.backend_fails = {}, {}
        self.seen = set()
        self.pos = 0
        self.last_beat = 0.0
        self.budget = Budget(240)
        self._beat_lock = threading.Lock()
        self.http = self.page_http = self.sx_http = None

    # -- plan ---------------------------------------------------------------------------------
    def reload_plan(self, force=False):
        now = self.clock()
        if not force and now - self.plan_check < 60.0:
            return
        self.plan_check = now
        try:
            mtime = os.path.getmtime(self.plan_path)
        except OSError:
            mtime = None
        if not force and mtime == self.plan_mtime and self.plan is not None:
            return
        try:
            plan = load_plan(self.plan_path)
        except (OSError, ValueError) as e:
            if self.plan is None:
                self.plan = dict(queries=[], backends=[], searxng_url=None, per_hour=240.0, page_fetch=False,
                                 max_page_bytes=1_500_000, per_query=10, user_agent=DEFAULT_UA)
            self.log(f"plan unreadable ({e}); keeping the previous plan")
            self.plan_mtime = mtime
            return
        self.plan, self.plan_mtime = plan, mtime
        self.budget.set_rate(plan["per_hour"])
        self.http = Http(user_agent=plan["user_agent"], guard_private=True, timeout=20.0)
        self.page_http = Http(user_agent=plan["user_agent"], guard_private=True, respect_robots=True,
                              max_bytes=plan["max_page_bytes"], timeout=20.0)
        sx = plan.get("searxng_url")
        host = (urllib.parse.urlsplit(sx).hostname or "").lower() if isinstance(sx, str) else ""
        # a self-hosted SearXNG usually lives on this PC or the LAN: allow exactly that host
        self.sx_http = Http(user_agent=plan["user_agent"], guard_private=True, allow_hosts=(host,) if host else (),
                            timeout=20.0) if host else None
        self.log(f"plan loaded: {len(plan['queries'])} queries x {plan['backends']} at {plan['per_hour']:.0f}/h")

    # -- heartbeat ----------------------------------------------------------------------------
    def beat(self, force=False):
        now = self.clock()
        if not force and now - self.last_beat < HEARTBEAT_S:
            return
        with self._beat_lock:
            self.last_beat = now
            try:
                self.qdb.beat(self.name, os.getpid(), dict(self.stats, ts=now), ts=now)
            except sqlite3.Error as e:
                self.log(f"heartbeat failed: {e}")

    def _beat_loop(self):
        # beats even while the main loop waits on a slow page; progress_ts shows whether it is moving
        while not self.stop.wait(HEARTBEAT_S / 2):
            self.beat()

    # -- waiting and budget -------------------------------------------------------------------
    def wait(self, seconds):
        end = self.clock() + max(0.0, seconds)
        while not self.stop.is_set():
            left = end - self.clock()
            if left <= 0:
                return True
            self.sleep(min(left, 1.0))
            self.beat()
        return False

    def spend(self):
        """Wait for one request's worth of budget. False when told to stop."""
        while not self.stop.is_set():
            w = self.budget.wait_s()
            if w <= 0:
                return True
            if not self.wait(min(w, 5.0)):
                return False
        return False

    # -- backends -----------------------------------------------------------------------------
    def search(self, backend, query):
        if self._search:
            return self._search(backend, query)
        if backend == "fake":
            if not self.fake:
                raise SourceError("fake backend needs CM_FOREVER_FAKE=1")
            return fake_search(query)
        if backend == "google_news":
            return parse_google_news(self.http.text(google_news_url(query), accept="application/rss+xml, application/xml"))
        if backend == "bing_news":
            return parse_bing_news(self.http.text(bing_news_url(query), accept="application/rss+xml, application/xml"))
        if backend == "searxng":
            if not self.sx_http:
                return []
            return parse_searxng(self.sx_http.json(searxng_search_url(self.plan["searxng_url"], query)))
        raise SourceError(f"unknown backend {backend!r}")

    def fetch_page(self, url):
        """(title, text) of an article page, or raises SourceError/Blocked."""
        if self._fetch_page:
            return self._fetch_page(url)
        if self.fake and urllib.parse.urlsplit(url).hostname == "fake.example":
            return fake_page(url)
        r = self.page_http.get(url, accept="text/html,application/xhtml+xml")
        ct = (r.headers.get("Content-Type") or "").lower() if r.headers else ""
        if ct and "html" not in ct and "xml" not in ct:
            raise Blocked(f"{T.domain(url)}: not a web page ({ct.split(';')[0][:40]})")
        return T.html_to_text(r.text(), max_chars=QTEXT_MAX)

    def _usable(self, backend):
        if backend == "searxng" and not (self._search or self.sx_http):
            return False
        if backend == "fake" and not (self._search or self.fake):
            return False
        return self.backend_until.get(backend, 0.0) <= self.clock()

    def _fail(self, backend, e):
        n = self.backend_fails.get(backend, 0) + 1
        self.backend_fails[backend] = n
        wait = getattr(e, "retry_after", None) or min(3600.0, 60.0 * 2 ** min(n - 1, 6))
        self.backend_until[backend] = self.clock() + wait
        self.stats["last_error"] = f"{backend}: {str(e)[:160]}"
        self.log(f"{backend} failed ({e}); resting it {wait:.0f}s")

    # -- one result -----------------------------------------------------------------------------
    def handle(self, backend, query, res):
        """Store one search result. Returns True when a new qitem was stored."""
        if not isinstance(res, dict):
            return False
        title = T.clean(res.get("title") or "")[:500]
        link = str(res.get("link") or "").strip()
        if backend in ("bing_news", "fake"):
            link = unwrap_bing(link)
        if not title or not link.lower().startswith(("http://", "https://")):
            return False
        h = _uhash(link)
        if h in self.seen or self.qdb.has(h):
            self.seen.add(h)
            self.stats["dup"] += 1
            return False
        self.seen.add(h)
        if len(self.seen) > 200_000:
            self.seen = {h}
        snippet = T.clean(res.get("snippet") or "")
        body = ""
        host = (urllib.parse.urlsplit(link).hostname or "").lower()
        # Google News links are redirect stubs: store the title and snippet only
        if self.plan.get("page_fetch") and host != "news.google.com" and self.spend():
            try:
                _ptitle, body = self.fetch_page(link)
                self.stats["fetched"] += 1
            except Blocked as e:
                self.stats["blocked"] += 1
                self.log(f"page refused: {str(e)[:160]}")
            except SourceError as e:
                self.stats["errors"] += 1
                self.stats["last_error"] = str(e)[:160]
        txt = body if len(body or "") >= max(80, len(snippet)) else (snippet + ("\n" + body if body else ""))
        dom = T.domain(link) if backend != "google_news" else (res.get("publisher") or "news.google.com")
        n = self.qdb.add([dict(url=link, domain=dom or res.get("publisher") or "", query=query, backend=backend,
                               fetched_ts=self.clock(), published_ts=_num(res.get("published"), None),
                               title=title, text=txt.strip()[:QTEXT_MAX])])
        self.stats["stored"] += n
        return n > 0

    # -- one step -------------------------------------------------------------------------------
    def step(self):
        """Run one (query, backend) search. Returns False when there was nothing to do."""
        self.reload_plan()
        plan = self.plan or {}
        pairs = [(q, b) for q in plan.get("queries", []) for b in plan.get("backends", [])]
        if not pairs:
            return False
        for _ in range(len(pairs)):
            q, b = pairs[self.pos % len(pairs)]
            self.pos = (self.pos + 1) % len(pairs)
            if self._usable(b):
                break
        else:
            return False
        if not self.spend():
            return True
        try:
            results = self.search(b, q)
            if not isinstance(results, list):
                raise SourceError(f"{b}: unexpected results")
            self.backend_fails[b] = 0
        except RateLimited as e:
            self.stats["errors"] += 1
            self._fail(b, e)
            return True
        except SourceError as e:
            self.stats["errors"] += 1
            if isinstance(e, Blocked):
                self.stats["blocked"] += 1
            self._fail(b, e)
            return True
        stored = 0
        for res in results[: max(plan.get("per_query", 10) * 3, 10)]:
            if self.stop.is_set() or stored >= plan.get("per_query", 10):
                break
            try:
                stored += 1 if self.handle(b, q, res) else 0
            except Exception as e:  # noqa: BLE001  (one bad result never stops the worker)
                self.stats["errors"] += 1
                self.stats["last_error"] = f"result: {e!r}"[:160]
        self.stats["queries_done"] += 1
        self.stats["progress_ts"] = self.clock()
        return True

    def run(self, max_seconds=None):
        self.reload_plan(force=True)
        self.beat(force=True)
        t = threading.Thread(target=self._beat_loop, daemon=True, name="forever-beat")
        t.start()
        end = None if max_seconds is None else self.clock() + max_seconds
        while not self.stop.is_set():
            if end is not None and self.clock() >= end:
                break
            try:
                worked = self.step()
            except Exception as e:  # noqa: BLE001
                self.stats["errors"] += 1
                self.stats["last_error"] = repr(e)[:160]
                self.log("step failed: " + "".join(traceback.format_exception_only(type(e), e)).strip()[:300])
                worked = False
            self.stats["progress_ts"] = self.clock()
            if not worked:
                self.wait(5.0 if end is None else 0.2)
        self.stop.set()
        self.beat(force=True)
        return self.stats


def _watch_stdin(stop):
    """Sets `stop` when stdin reaches EOF: the parent closed the pipe or died.
    Reads the raw file descriptor: a daemon thread parked inside sys.stdin's buffered
    reader holds its lock and makes the interpreter abort at shutdown."""
    try:
        fd = sys.stdin.fileno()
        while os.read(fd, 1024):
            pass
    except (OSError, ValueError, AttributeError):
        pass
    stop.set()


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m cryptomoni.research.forever",
                                 description="Forever-search worker (sandboxed; started by the research service).")
    ap.add_argument("--db", required=True, help="quarantine database file")
    ap.add_argument("--name", default="primary", help="heartbeat name")
    ap.add_argument("--plan", required=True, help="plan JSON written by the research service")
    ap.add_argument("--max-seconds", type=float, default=None, help="exit after this long (tests)")
    ap.add_argument("--no-stdin", action="store_true", help="do not exit on stdin EOF (manual runs)")
    a = ap.parse_args(argv)
    # defence in depth: even if started by hand, drop anything that looks like a secret
    for k in [k for k in os.environ if _secretish(k) and k != "CM_FOREVER_FAKE"]:
        os.environ.pop(k, None)
    stop = threading.Event()
    if not a.no_stdin and sys.stdin is not None:
        threading.Thread(target=_watch_stdin, args=(stop,), daemon=True, name="forever-stdin").start()
    qdb = QuarantineDB(a.db)
    w = Worker(qdb, a.name, a.plan, stop=stop)
    w.log(f"started pid {os.getpid()}")
    try:
        stats = w.run(max_seconds=a.max_seconds)
    finally:
        qdb.close()
    w.log(f"stopped: {json.dumps({k: stats[k] for k in ('fetched', 'stored', 'errors', 'blocked', 'dup', 'queries_done')})}")
    return 0


# ---------------------------------------------------------------------------- supervisors

def _popen(argv, cwd=None, env=None, log_path=None):
    """Start the worker: stdin is a pipe we hold (closing it stops the worker), output goes
    to the log file, no console window on Windows."""
    if log_path:
        try:
            if os.path.getsize(log_path) > LOG_MAX_BYTES:
                os.replace(log_path, log_path + ".1")
        except OSError:
            pass
        out = open(log_path, "ab")
    else:
        out = subprocess.DEVNULL
    try:
        return subprocess.Popen(argv, cwd=cwd, env=env, stdin=subprocess.PIPE, stdout=out, stderr=subprocess.STDOUT,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    finally:
        if out is not subprocess.DEVNULL:
            out.close()   # the child has its own handle


class ForeverSupervisor:
    """Keeps one forever worker running.

    ensure(now) starts the worker when none is alive, restarts a dead or hung one after a
    back-off (10 s, 30 s, 2 min, then every 10 min), and after 3 crashes within 10 minutes
    moves to the next standby plan (standby 1, then standby 2). healthy = the child is
    alive and its heartbeat is fresher than 60 s."""

    def __init__(self, qdb_path, plan_path, python=sys.executable, cwd=None, log_path=None, name="primary",
                 standby_plans=(), clock=time.time, spawn=None, qdb=None):
        self.qdb_path, self.python, self.name, self.clock = os.path.abspath(qdb_path), python, name, clock
        self.cwd = cwd or os.path.dirname(self.qdb_path)
        self.log_path = log_path
        self.plans = [os.path.abspath(plan_path)] + [os.path.abspath(p) for p in standby_plans or ()]
        self.plan_index = 0
        self.spawn = spawn or _popen
        self.qdb = qdb
        self.proc, self.started_ts = None, None
        self.crashes, self.consec, self.restarts = [], 0, 0
        self.next_start, self.last_exit = 0.0, None
        self.events = []
        self.lock = threading.RLock()

    # -- helpers ------------------------------------------------------------------------------
    def _db(self):
        if self.qdb is None:
            self.qdb = QuarantineDB(self.qdb_path)
        return self.qdb

    def _event(self, now, title, body="", severity="info"):
        self.events = (self.events + [dict(ts=now, title=title, body=body, severity=severity, supervisor=self.name)])[-20:]

    def argv(self):
        return [self.python, "-m", "cryptomoni.research.forever", "--db", self.qdb_path, "--name", self.name,
                "--plan", self.plans[self.plan_index]]

    def env(self):
        pp = REPO_ROOT
        return safe_env(extra={"PYTHONPATH": pp, "PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1"})

    def alive(self):
        p = self.proc
        if p is None:
            return False
        try:
            return p.poll() is None
        except OSError:
            return False

    def heart(self):
        try:
            return self._db().hearts().get(self.name)
        except sqlite3.Error:
            return None

    def healthy(self, now=None):
        now = self.clock() if now is None else now
        h = self.heart() if self.alive() else None
        return bool(h and now - (h.get("heartbeat") or 0.0) <= HEALTHY_S and (h.get("pid") in (None, 0, getattr(self.proc, "pid", None))))

    def _kill(self, grace=5.0):
        p, self.proc = self.proc, None
        if p is None:
            return None
        try:
            if p.stdin:
                p.stdin.close()
        except OSError:
            pass
        try:
            return p.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            pass
        except Exception:  # noqa: BLE001
            pass
        for fn in (p.terminate, p.kill):
            try:
                fn()
                return p.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                continue
            except Exception:  # noqa: BLE001
                break
        return None

    def _crashed(self, now, why):
        started = now if self.started_ts is None else self.started_ts
        lived = now - started
        self.consec = 1 if lived >= CRASH_WINDOW_S else self.consec + 1
        self.crashes = [t for t in self.crashes if now - t <= CRASH_WINDOW_S] + [now]
        delay = BACKOFF_S[min(self.consec - 1, len(BACKOFF_S) - 1)]
        self.next_start = now + delay
        self._event(now, f"Forever search worker {self.name} stopped", f"{why}; restarting in {delay:.0f}s", "warn")
        if len(self.crashes) >= CRASHES_TO_SWITCH and self.plan_index + 1 < len(self.plans):
            self.plan_index += 1
            self.crashes = []
            self._event(now, f"Forever search switched to standby plan {self.plan_index}",
                        f"{CRASHES_TO_SWITCH} crashes within {CRASH_WINDOW_S / 60:.0f} minutes on the previous plan; "
                        f"now using {os.path.basename(self.plans[self.plan_index])}", "warn")

    # -- the job --------------------------------------------------------------------------------
    def ensure(self, now=None):
        """Make sure a worker runs. Returns a short state word: running / started / waiting / failed."""
        now = self.clock() if now is None else now
        with self.lock:
            if self.proc is not None:
                if self.alive():
                    h = self.heart()
                    started = now if self.started_ts is None else self.started_ts
                    beat = (h or {}).get("heartbeat") or 0.0
                    since = now - max(beat if (h and h.get("pid") == self.proc.pid) else 0.0, started)
                    progress = _num(((h or {}).get("stats") or {}).get("progress_ts"), started)
                    if since > HUNG_S:
                        self._kill(grace=1.0)
                        self._crashed(now, f"no heartbeat for {since:.0f}s (hung)")
                    elif h and h.get("pid") == getattr(self.proc, "pid", None) and now - progress > STALLED_S \
                            and now - started > STALLED_S:
                        self._kill(grace=1.0)
                        self._crashed(now, f"no progress for {now - progress:.0f}s (stalled)")
                    else:
                        return "running"
                else:
                    try:
                        code = self.proc.poll()
                    except OSError:
                        code = None
                    self.last_exit = code
                    self.proc = None
                    self._crashed(now, f"exit code {code}")
            if now < self.next_start:
                return "waiting"
            try:
                self.proc = self.spawn(self.argv(), cwd=self.cwd, env=self.env(), log_path=self.log_path)
            except Exception as e:  # noqa: BLE001  (python missing, cwd gone, ...)
                self.proc = None
                self.started_ts = now
                self._crashed(now, f"could not start: {e!r}"[:200])
                return "failed"
            self.started_ts = now
            self.restarts += 1
            return "started"

    def stop(self, timeout=None):
        """Close the worker's stdin (it exits on EOF), terminate it after 5 s. With
        `timeout`, gives up quietly if another thread holds the supervisor."""
        got = self.lock.acquire(timeout=timeout) if timeout is not None else self.lock.acquire()
        if not got:
            return False
        try:
            self._kill(grace=5.0)
            return True
        finally:
            self.lock.release()

    def status(self, now=None):
        now = self.clock() if now is None else now
        h = self.heart() or {}
        st = h.get("stats") or {}
        return dict(name=self.name, pid=getattr(self.proc, "pid", None), alive=self.alive(), healthy=self.healthy(now),
                    plan=os.path.basename(self.plans[self.plan_index]), plan_index=self.plan_index,
                    restarts=self.restarts, crashes_10m=len([t for t in self.crashes if now - t <= CRASH_WINDOW_S]),
                    next_start_in=max(0.0, self.next_start - now) if self.proc is None else 0.0, last_exit=self.last_exit,
                    heartbeat_age=round(now - h["heartbeat"], 1) if h.get("heartbeat") else None,
                    stats={k: st.get(k) for k in ("fetched", "stored", "errors", "blocked", "dup", "queries_done", "last_error")},
                    events=self.events[-5:])


class SupervisorPair:
    """Two supervisors for one job. A does the work: every tick it beats ("forever_supervisor:A"
    in kv) and runs ensure(). B beats too and watches A's beat; when A's beat is older than
    90 s (its thread hung or died), B takes over. B first adopts A's worker if it is still
    healthy, otherwise starts its own. When A beats again, it stops B's worker and carries on.

    tick_a / tick_b are called from two threads by the research service every 10 s; they can
    also be called directly (tests) with a fake clock and `spawn`."""

    def __init__(self, qdb_path, plan_path, cwd, log_path, standby_plans=(), kv=None, python=sys.executable,
                 spawn=None, clock=time.time):
        self.kv, self.clock = kv, clock
        self.qdb_path = qdb_path
        logs = [log_path, log_path]
        if log_path:
            root, ext = os.path.splitext(log_path)
            logs = [log_path, root + "-b" + (ext or ".log")]
        # each supervisor has its own worker name, so beats and health never mix
        self.a = ForeverSupervisor(qdb_path, plan_path, python=python, cwd=cwd, log_path=logs[0], name="forever-A",
                                   standby_plans=standby_plans, clock=clock, spawn=spawn)
        self.b = ForeverSupervisor(qdb_path, plan_path, python=python, cwd=cwd, log_path=logs[1], name="forever-B",
                                   standby_plans=standby_plans, clock=clock, spawn=spawn)
        self.holder = "A"
        self.takeovers = 0
        self.created = clock()
        self.mem = {}
        self.events = []
        self.lock = threading.Lock()
        self.last = {}

    # -- beats ----------------------------------------------------------------------------------
    def _beat(self, who, now):
        v = dict(ts=now, pid=os.getpid(), holder=self.holder)
        self.mem[who] = v
        if self.kv is not None:
            try:
                self.kv.kv_set("forever_supervisor:" + who, v)
            except Exception:  # noqa: BLE001  (a busy research DB must not stop supervision)
                pass

    def _beat_ts(self, who):
        v = None
        if self.kv is not None:
            try:
                v = self.kv.kv_get("forever_supervisor:" + who, None)
            except Exception:  # noqa: BLE001
                v = None
        if not isinstance(v, dict):
            v = self.mem.get(who)
        return _num((v or {}).get("ts"), None)

    def _event(self, now, title, body="", severity="info"):
        self.events = (self.events + [dict(ts=now, title=title, body=body, severity=severity)])[-20:]

    # -- ticks ----------------------------------------------------------------------------------
    def tick_a(self, now=None):
        now = self.clock() if now is None else now
        self._beat("A", now)
        with self.lock:
            was = self.holder
            self.holder = "A"
        if was == "B":
            self.b.stop(timeout=10.0)
            self._event(now, "Forever supervisor A is back", "A resumed supervision; B's worker was stopped.")
        self.last["A"] = self.a.ensure(now)
        return self.last["A"]

    def tick_b(self, now=None):
        now = self.clock() if now is None else now
        self._beat("B", now)
        ts = self._beat_ts("A")
        stale = (now - ts > TAKEOVER_S) if ts is not None else (now - self.created > TAKEOVER_S)
        if not stale:
            self.last["B"] = "watching"
            return "watching"
        with self.lock:
            was = self.holder
            self.holder = "B"
        if was != "B":
            self.takeovers += 1
            age = f"{now - ts:.0f}s" if ts is not None else "never beat"
            self._event(now, "Forever supervisor B took over", f"A's last beat: {age}.", "warn")
        if self.a.alive() and self.a.healthy(now):
            self.last["B"] = "adopted"     # A's worker is fine; just keep an eye on it
            return "adopted"
        if self.a.alive():
            self.a.stop(timeout=2.0)       # an unhealthy orphan of A's: stop it if A's lock allows
        self.last["B"] = self.b.ensure(now)
        return self.last["B"]

    # -- reporting ------------------------------------------------------------------------------
    def status(self, now=None):
        now = self.clock() if now is None else now
        ts_a, ts_b = self._beat_ts("A"), self._beat_ts("B")
        a, b = self.a.status(now), self.b.status(now)
        healthy = a["healthy"] or b["healthy"]
        try:
            q = self.a._db().counts()
        except sqlite3.Error:
            q = {}
        events = sorted(self.events + a["events"] + b["events"], key=lambda e: e.get("ts") or 0.0)[-8:]
        return dict(holder=self.holder, healthy=healthy, takeovers=self.takeovers,
                    beat_age_a=round(now - ts_a, 1) if ts_a is not None else None,
                    beat_age_b=round(now - ts_b, 1) if ts_b is not None else None,
                    state_a=self.last.get("A"), state_b=self.last.get("B"), a=a, b=b, quarantine=q, events=events)

    def stop(self):
        for s in (self.a, self.b):
            try:
                s.stop(timeout=15.0)
            except Exception:  # noqa: BLE001
                pass


# ---------------------------------------------------------------------------- service side

def ingest(qdb, rdb, now, limit=300) -> int:
    """Copy new quarantine rows into the research DB as quarantined "search" items of the
    discovery team, and mark them ingested. Returns how many items were added."""
    from .types import Item   # service side only: the worker never imports this
    rows = qdb.take(limit)
    if not rows:
        if now - getattr(qdb, "last_prune", 0.0) > 6 * 3600:
            qdb.last_prune = now
            try:
                qdb.prune(now=now)
            except sqlite3.Error:
                pass
        return 0
    items, ids = [], []
    for r in rows:
        ids.append(r["id"])
        try:
            url, title = str(r.get("url") or "").strip(), T.clean(r.get("title") or "")
            if not url or not title:
                continue
            backend = "".join(ch for ch in str(r.get("backend") or "web") if ch.isalnum() or ch == "_")[:30] or "web"
            items.append(Item(source="forever_" + backend, url=url, title=title, text=str(r.get("text") or "")[:QTEXT_MAX],
                              published_ts=_num(r.get("published_ts"), None), symbols=[], kind="search",
                              publisher=str(r.get("domain") or "") or T.domain(url), quarantine=True))
        except Exception:  # noqa: BLE001  (a malformed row is skipped and still marked)
            continue
    n = rdb.add_items(items, team="discovery", now=now) if items else 0
    qdb.mark(ids)
    return n


TOPIC_QUERIES = (
    "crypto exchange listing", "token unlock", "crypto exploit hack", "DeFi protocol exploit", "bridge hack",
    "SEC charges crypto", "SEC charges company", "crypto ETF flows", "spot ETF approval", "stablecoin depeg",
    "crypto exchange outage", "crypto delisting", "earnings surprise", "guidance cut", "guidance raised",
    "short seller report", "analyst downgrade", "trading halted", "chapter 11 bankruptcy", "secondary offering",
    "share buyback announced", "merger agreement", "class action lawsuit", "FDA approval", "mainnet launch",
)


def _base(symbol):
    s = str(symbol)
    return s.split("-")[0] if "-" in s else s


def build_plan(subscribed, names_fn, cfg) -> dict:
    """The worker's plan: the topic queries + one name query per subscribed symbol.
    subscribed = {symbol: asset}; names_fn(symbols) -> {symbol: [names]} (may be None);
    cfg = the research.forever block (backends, searxng_url, per_hour, page_fetch,
    max_page_bytes, per_query, extra_queries, topics)."""
    cfg = cfg if isinstance(cfg, dict) else {}
    subscribed = subscribed if isinstance(subscribed, dict) else {}
    topics = cfg.get("topics")
    queries = [str(q) for q in (topics if isinstance(topics, (list, tuple)) else TOPIC_QUERIES)]
    queries += [str(q) for q in (cfg.get("extra_queries") or []) if isinstance(q, str)]
    syms = sorted(str(s) for s in subscribed)
    names = {}
    if names_fn and syms:
        try:
            names = names_fn(syms) or {}
        except Exception:  # noqa: BLE001  (no names: fall back to tickers)
            names = {}
    for s in syms:
        asset = subscribed.get(s) or ("crypto" if "-" in s else "stock")
        ns = [n for n in (names.get(s) or []) if isinstance(n, str) and len(n.strip()) >= 2] if isinstance(names, dict) else []
        base = _base(s)
        # prefer a full name ("Solana") over a short ticker that matches other words
        name = next((n for n in ns if n.strip().lower() != base.lower()), ns[0] if ns else base).strip()
        queries.append(f'"{name}" crypto' if asset == "crypto" else f'"{name}" stock')
    seen, out = set(), []
    for q in queries:
        k = q.strip().lower()
        if k and k not in seen:
            seen.add(k)
            out.append(q.strip()[:200])
    backends = cfg.get("backends")
    backends = [b for b in backends if b in BACKENDS] if isinstance(backends, (list, tuple)) else list(DEFAULT_BACKENDS)
    return dict(queries=out[: int(_num(cfg.get("max_queries"), 400))], backends=backends or list(DEFAULT_BACKENDS),
                searxng_url=cfg.get("searxng_url") or None, per_hour=int(_num(cfg.get("per_hour"), 240)),
                page_fetch=bool(cfg.get("page_fetch", True)), max_page_bytes=int(_num(cfg.get("max_page_bytes"), 1_500_000)),
                per_query=int(_num(cfg.get("per_query"), 10)))


if __name__ == "__main__":
    sys.exit(main())
