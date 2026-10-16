"""Local dashboard server (stdlib only). Reads the SQLite store the scanner
writes, so it keeps working while the engine restarts. Binds to 127.0.0.1 by
default. The only write it can do is TRIP the kill switch (never reset it)."""
from __future__ import annotations

import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from .safety import KillSwitch
from .store import Store

STATIC = os.path.join(os.path.dirname(__file__), "static")


def make_handler(db, kill_file):
    kill = KillSwitch(kill_file)

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, body, ctype, code=200):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code=200):
            self._send(json.dumps(obj, default=float).encode(), "application/json", code)

        def do_GET(self):
            u = urlparse(self.path)
            q = parse_qs(u.query)
            if u.path in ("/", "/index.html"):
                with open(os.path.join(STATIC, "index.html"), "rb") as f:
                    return self._send(f.read(), "text/html; charset=utf-8")
            if not os.path.exists(db):
                return self._json(dict(error="no database yet: start the scanner"), 503)
            st = Store(db, readonly=True)
            try:
                raw = q.get("limit", ["200"])[0]
                lim = min(int(raw), 20000) if raw.isdigit() else 200
                if u.path == "/api/state":
                    hb = st.kv_get("heartbeat")
                    self._json(dict(snapshot=st.latest(), live=st.kv_get("live"),
                                    heartbeat_age=None if hb is None else time.time() - hb,
                                    kill=kill.engaged(), kill_reason=kill.reason(), version=st.counts()))
                elif u.path == "/api/history":
                    self._json(st.history(lim))
                elif u.path == "/api/trades":
                    self._json(st.trades(lim))
                elif u.path == "/api/events":
                    self._json(st.events(lim, q.get("kind", [None])[0]))
                elif u.path == "/api/alerts":
                    self._json(st.alerts(lim))
                else:
                    self._json(dict(error="not found"), 404)
            finally:
                st.close()

        def do_POST(self):
            if self.headers.get("X-CM") != "1":
                return self._json(dict(error="forbidden"), 403)
            if urlparse(self.path).path == "/api/kill":
                kill.trip("dashboard kill button")
                return self._json(dict(ok=True))
            self._json(dict(error="forbidden"), 403)

    return H


def serve(db, kill_file, port=8765, host="127.0.0.1"):
    srv = ThreadingHTTPServer((host, port), make_handler(db, kill_file))
    print(f"Dashboard on http://{'127.0.0.1' if host in ('0.0.0.0', '') else host}:{port}  (db={db})", flush=True)
    srv.serve_forever()
