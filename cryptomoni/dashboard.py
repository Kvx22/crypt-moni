"""Local dashboard server (stdlib only). Reads the SQLite store the scanner
writes, so it keeps working while the engine restarts. Binds to 127.0.0.1 by
default. Its only writes: TRIP the kill switch (never reset it), and approve or
reject proposed orders. Approving never sends anything by itself: the scanner
re-checks every approved order against fresh prices and the safety gate first."""
from __future__ import annotations

import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from .safety import KillSwitch
from .store import Store

STATIC = os.path.join(os.path.dirname(__file__), "static")


def make_handler(db, kill_file, token=None):
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
                elif u.path == "/api/outbox":
                    self._json(st.outbox(lim))
                else:
                    self._json(dict(error="not found"), 404)
            finally:
                st.close()

        def do_POST(self):
            if self.headers.get("X-CM") != "1":  # custom header: a cross-site page can't send it
                return self._json(dict(error="forbidden"), 403)
            path = urlparse(self.path).path
            if path == "/api/kill":
                kill.trip("dashboard kill button")
                return self._json(dict(ok=True))
            parts = path.strip("/").split("/")   # api/outbox/<id|all>/approve|reject
            if len(parts) != 4 or parts[:2] != ["api", "outbox"] or parts[3] not in ("approve", "reject"):
                return self._json(dict(error="forbidden"), 403)
            if token and self.headers.get("X-CM-Token") != token:
                return self._json(dict(error="approval token missing or wrong: open the dashboard with ?t=<token>"), 403)
            if parts[3] == "approve" and kill.engaged():
                return self._json(dict(error="kill switch engaged: nothing can be approved"), 409)
            st = Store(db)
            try:
                now = time.time()
                if parts[2] == "all":
                    if parts[3] != "approve":
                        return self._json(dict(error="forbidden"), 403)
                    n = 0
                    for r in st.outbox(200, "pending"):
                        if now <= r["expires_ts"]:
                            st.outbox_set(r["id"], "approved", now, decided_by="dashboard")
                            n += 1
                    return self._json(dict(ok=True, approved=n))
                if not parts[2].isdigit():
                    return self._json(dict(error="bad id"), 400)
                r = st.outbox_get(int(parts[2]))
                if not r:
                    return self._json(dict(error="no such order"), 404)
                if parts[3] == "approve":
                    if r["status"] != "pending" or now > r["expires_ts"]:
                        return self._json(dict(error=f"cannot approve: {r['status']}" + (" (expired)" if now > r["expires_ts"] else "")), 409)
                    st.outbox_set(r["id"], "approved", now, decided_by="dashboard")
                else:
                    if r["status"] not in ("pending", "approved"):
                        return self._json(dict(error=f"cannot reject: {r['status']}"), 409)
                    st.outbox_set(r["id"], "rejected", now, decided_by="dashboard")
                return self._json(dict(ok=True))
            finally:
                st.close()

    return H


def serve(db, kill_file, port=8765, host="127.0.0.1", token=None):
    srv = ThreadingHTTPServer((host, port), make_handler(db, kill_file, token))
    print(f"Dashboard on http://{'127.0.0.1' if host in ('0.0.0.0', '') else host}:{port}  (db={db})", flush=True)
    srv.serve_forever()
