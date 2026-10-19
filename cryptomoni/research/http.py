"""The polite HTTP client every research adapter and the forever-search worker use.

Rules it enforces for every request:
  * one User-Agent that names the project (SEC asks for a contact; set research.contact)
  * a per-domain minimum gap between requests, shared by every thread in the process
  * 429 / 503 back-off that honours Retry-After (the domain is paused, not hammered)
  * a size cap and a timeout, gzip/deflate handled, only http(s)
  * optional private-network guard (the forever-search worker turns it on) so a crawled
    link can never reach 127.0.0.1, your LAN, or the dashboard
Nothing here can place an order.
"""
from __future__ import annotations

import gzip
import ipaddress
import json
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import urllib.robotparser
import zlib

DEFAULT_UA = "crypt-moni-research/1.0 (+https://github.com/kvx22/crypt-moni)"
MAX_BYTES = 3 * 1024 * 1024

# seconds between two requests to the same domain (others default to `default_gap`)
DOMAIN_GAP = {
    "www.sec.gov": 0.25, "data.sec.gov": 0.25, "efts.sec.gov": 0.5,        # SEC allows 10 req/s; stay well under
    "news.google.com": 4.0, "www.bing.com": 4.0, "www.reddit.com": 6.0,      # search-like endpoints: slow and steady
    "api.stocktwits.com": 6.0, "hn.algolia.com": 2.0, "api.coingecko.com": 8.0,
    "query1.finance.yahoo.com": 1.0, "feeds.finance.yahoo.com": 2.0,
}


class SourceError(Exception):
    """A source failed (network, HTTP status, parse). The worker records it and backs off."""


class RateLimited(SourceError):
    def __init__(self, msg, retry_after=60.0):
        super().__init__(msg)
        self.retry_after = retry_after


class Blocked(SourceError):
    """Refused by the client's own guard (private address, scheme, robots.txt, size)."""


class Response:
    def __init__(self, url, status, headers, body):
        self.url, self.status, self.headers, self.body = url, status, headers, body

    def text(self):
        ct = self.headers.get("Content-Type", "") or ""
        enc = "utf-8"
        if "charset=" in ct:
            enc = ct.split("charset=", 1)[1].split(";")[0].strip().strip('"') or "utf-8"
        try:
            return self.body.decode(enc, errors="replace")
        except LookupError:
            return self.body.decode("utf-8", errors="replace")

    def json(self):
        try:
            return json.loads(self.text())
        except ValueError as e:
            raise SourceError(f"{urllib.parse.urlparse(self.url).netloc}: bad JSON ({e})") from e


def _private(host):
    """True if the host resolves to a loopback / private / link-local / reserved address."""
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError:
        return False   # unresolvable: urlopen will fail on its own
    for info in infos:
        ip = info[4][0].split("%")[0]
        try:
            a = ipaddress.ip_address(ip)
        except ValueError:
            continue
        if a.is_private or a.is_loopback or a.is_link_local or a.is_reserved or a.is_multicast or a.is_unspecified:
            return True
    return False


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None   # redirects are followed by Http.get itself so each hop is re-checked


def sec_user_agent(contact):
    """SEC's fair-access filter wants a plain 'name email' identity and answers anything fancier with HTTP 403."""
    contact = str(contact or "").strip()
    return f"crypt-moni {contact}" if contact else None


class Http:
    """Thread-safe. One instance per process is enough (the domain clock is shared)."""

    def __init__(self, user_agent=DEFAULT_UA, default_gap=1.0, timeout=20.0, max_bytes=MAX_BYTES,
                 guard_private=False, allow_hosts=(), respect_robots=False, sec_contact=None):
        self.sec_ua = sec_user_agent(sec_contact)
        self.ua, self.default_gap, self.timeout, self.max_bytes = user_agent, default_gap, timeout, max_bytes
        self.guard_private, self.allow_hosts, self.respect_robots = guard_private, set(allow_hosts), respect_robots
        self._next = {}           # domain -> earliest next request time
        self._paused = {}         # domain -> paused until (429/503)
        self._lock = threading.Lock()
        self._robots = {}         # domain -> (fetched_ts, RobotFileParser|None)
        self._opener = urllib.request.build_opener(_NoRedirect)
        self.stats = dict(requests=0, errors=0, bytes=0, rate_limited=0, blocked=0)

    # ---- politeness --------------------------------------------------------------------------
    def gap(self, host):
        return DOMAIN_GAP.get(host, self.default_gap)

    def _wait_turn(self, host):
        while True:
            with self._lock:
                now = time.monotonic()
                paused = self._paused.get(host, 0.0)
                if paused > now:
                    raise RateLimited(f"{host}: paused for {paused - now:.0f}s after a rate limit", paused - now)
                nxt = self._next.get(host, 0.0)
                if nxt <= now:
                    self._next[host] = now + self.gap(host)
                    return
                delay = nxt - now
            time.sleep(min(delay, 5.0))

    def pause(self, host, seconds):
        with self._lock:
            self._paused[host] = max(self._paused.get(host, 0.0), time.monotonic() + seconds)

    def paused_for(self, host):
        with self._lock:
            return max(0.0, self._paused.get(host, 0.0) - time.monotonic())

    # ---- guards ------------------------------------------------------------------------------
    def _check(self, url):
        p = urllib.parse.urlparse(url)
        if p.scheme not in ("http", "https"):
            raise Blocked(f"scheme {p.scheme!r} not allowed")
        host = (p.hostname or "").lower()
        if not host:
            raise Blocked("no host")
        if self.guard_private and host not in self.allow_hosts and (host in ("localhost",) or _private(host)):
            self.stats["blocked"] += 1
            raise Blocked(f"{host}: private or local address refused")
        if self.respect_robots and host not in self.allow_hosts and not self.robots_ok(url):
            self.stats["blocked"] += 1
            raise Blocked(f"{host}: robots.txt disallows {p.path or '/'}")
        return host

    def robots_ok(self, url):
        p = urllib.parse.urlparse(url)
        host = (p.hostname or "").lower()
        now = time.time()
        with self._lock:
            cached = self._robots.get(host)
        if cached is None or now - cached[0] > 24 * 3600:
            rp = urllib.robotparser.RobotFileParser()
            try:
                r = self._raw(f"{p.scheme}://{p.netloc}/robots.txt", host, max_bytes=512 * 1024, check=False)
                if r.status == 200:
                    rp.parse(r.text().splitlines())
                else:
                    rp = None          # no robots.txt (or an error page): allowed
            except SourceError:
                rp = None
            with self._lock:
                self._robots[host] = (now, rp)
            cached = (now, rp)
        rp = cached[1]
        return True if rp is None else rp.can_fetch(self.ua, url)

    # ---- requests ----------------------------------------------------------------------------
    def _raw(self, url, host, headers=None, max_bytes=None, check=True, data=None, method=None):
        if check:
            host = self._check(url)
        self._wait_turn(host)
        h = {"User-Agent": self.ua, "Accept-Encoding": "gzip, deflate", "Accept": "*/*"}
        h.update(headers or {})
        if self.sec_ua and (host == "sec.gov" or host.endswith(".sec.gov")):
            h["User-Agent"] = self.sec_ua
        req = urllib.request.Request(url, headers=h, data=data, method=method)
        cap = max_bytes or self.max_bytes
        self.stats["requests"] += 1
        try:
            with self._opener.open(req, timeout=self.timeout) as r:
                status, hdrs = r.status, r.headers
                body = r.read(cap + 1)
        except urllib.error.HTTPError as e:
            status, hdrs = e.code, e.headers
            try:
                body = e.read(64 * 1024)
            except OSError:
                body = b""
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as e:
            self.stats["errors"] += 1
            raise SourceError(f"{host}: {e}") from e
        if len(body) > cap:
            self.stats["blocked"] += 1
            raise Blocked(f"{host}: response over {cap // 1024} KB")
        enc = (hdrs.get("Content-Encoding") or "").lower() if hdrs else ""
        try:
            if enc == "gzip" or body[:2] == b"\x1f\x8b":
                body = gzip.decompress(body)
            elif enc == "deflate":
                body = zlib.decompress(body, -zlib.MAX_WBITS) if body[:1] != b"x" else zlib.decompress(body)
        except (OSError, zlib.error, EOFError):
            pass
        if len(body) > cap * 4:
            raise Blocked(f"{host}: decompressed response too large")
        self.stats["bytes"] += len(body)
        return Response(url, status, hdrs or {}, body)

    def get(self, url, headers=None, max_bytes=None, accept=None, redirects=3, ok=(200,), data=None, method=None):
        """GET (or POST with `data`). Follows up to `redirects` redirects, re-checking each hop.
        Raises RateLimited on 429/503, SourceError on other non-`ok` statuses."""
        h = dict(headers or {})
        if accept:
            h["Accept"] = accept
        for _ in range(redirects + 1):
            host = (urllib.parse.urlparse(url).hostname or "").lower()
            r = self._raw(url, host, h, max_bytes, data=data, method=method)
            if r.status in (301, 302, 303, 307, 308) and r.headers.get("Location"):
                url = urllib.parse.urljoin(url, r.headers["Location"])
                data, method = (None, None) if r.status == 303 else (data, method)
                continue
            if r.status in (429, 503):
                ra = r.headers.get("Retry-After") if r.headers else None
                try:
                    wait = float(ra) if ra else 120.0
                except ValueError:
                    wait = 120.0
                wait = min(max(wait, 30.0), 3600.0)
                self.pause(host, wait)
                self.stats["rate_limited"] += 1
                raise RateLimited(f"{host}: HTTP {r.status}, pausing {wait:.0f}s", wait)
            if r.status not in ok:
                self.stats["errors"] += 1
                raise SourceError(f"{host}: HTTP {r.status}")
            return r
        raise SourceError(f"too many redirects for {url[:120]}")

    def json(self, url, **kw):
        kw.setdefault("accept", "application/json")
        return self.get(url, **kw).json()

    def text(self, url, **kw):
        return self.get(url, **kw).text()
