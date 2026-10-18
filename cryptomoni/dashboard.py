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


def research_state(rdb_path, board=None, limit=60):
    """Everything the Research panel draws, from the research database (read-only)."""
    if not rdb_path or not os.path.exists(rdb_path):
        return dict(available=False)
    from .research.db import ResearchDB
    from .research.taxonomy import CATALYSTS, RISK_FLAGS
    r = ResearchDB(rdb_path, readonly=True)
    try:
        now = time.time()
        lease = r.lease()
        labelled = {x["note_id"] for x in r.q("SELECT note_id FROM labels")}
        notes = [dict(n.d(), labelled=n.id in labelled) for n in r.notes(limit)]
        for n in notes:
            n.pop("facts", None)
        return dict(available=True, now=now, service_age=None if not lease else now - (lease.get("heartbeat") or 0),
                    health=r.kv_get("health_now"), stages=r.stages(), evidence=(r.kv_get("stage_evidence") or {}).get("stages") or {},
                    notes=notes, requests=r.open_requests(30), discovered=r.discovered(limit=30),
                    proposals=r.proposals(board, limit=40), models=[m for m in r.models() if m["role"] in ("champion", "challenger")][:8],
                    events=r.events(40), outcome=r.kv_get("outcome_report"), board=board,
                    taxonomy=dict(catalysts=list(CATALYSTS), flags=list(RISK_FLAGS)))
    finally:
        r.close()


def make_handler(db, kill_file, token=None, research_db=None, board=None):
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
                    self._json(dict(snapshot=st.latest(), live=st.kv_get("live"), opps=st.kv_get("opps"),
                                    research=st.kv_get("research", {}), watchlist=st.kv_get("watchlist", []),
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
                elif u.path == "/api/research":
                    self._json(research_state(research_db, board))
                else:
                    self._json(dict(error="not found"), 404)
            finally:
                st.close()

        def _watch(self, op, sym, query):
            """Put a scout candidate on the board's watchlist (or take it off). Only names the scout actually
            surfaced can be added, so a request can't smuggle in an arbitrary symbol."""
            if token and self.headers.get("X-CM-Token") != token:
                return self._json(dict(error="token missing or wrong: open the dashboard with ?t=<token>"), 403)
            sym = sym.upper()
            st = Store(db)
            try:
                wl = st.kv_get("watchlist", [])
                if op == "add":
                    opps = st.kv_get("opps") or {}
                    found = next((i for a in opps.values() for i in a.get("items", []) if i["symbol"] == sym), None)
                    if not found and research_db and os.path.exists(research_db):
                        # an advisor's watch-only proposal can be promoted to the board too
                        from .research.db import ResearchDB
                        r = ResearchDB(research_db)
                        try:
                            p = next((x for x in r.proposals(board) if x["symbol"] == sym and x["status"] in ("shadow", "added")), None)
                            if p:
                                found = dict(symbol=sym, asset=p["asset"])
                                r.set_proposal(p["id"], "promoted")
                        finally:
                            r.close()
                    if not found:
                        return self._json(dict(error=f"{sym} is not on the scout's shortlist"), 404)
                    if not any(w["symbol"] == sym for w in wl):
                        wl.append(dict(symbol=sym, asset=found["asset"], added=time.time()))
                else:
                    wl = [w for w in wl if w["symbol"] != sym]
                st.kv_set("watchlist", wl)
                return self._json(dict(ok=True, watchlist=wl))
            finally:
                st.close()

        def _research(self, parts):
            """POST api/research/label/<note_id> (JSON body: catalyst, direction, flags) or
            api/research/stage/<name>/<on|off|shadow|auto>. Both need the token."""
            if token and self.headers.get("X-CM-Token") != token:
                return self._json(dict(error="token missing or wrong: open the dashboard with ?t=<token>"), 403)
            if not research_db or not os.path.exists(research_db):
                return self._json(dict(error="no research database yet"), 503)
            from .research.db import STAGE_ORDER, ResearchDB
            from .research.taxonomy import CATALYSTS, RISK_FLAGS
            r = ResearchDB(research_db)
            try:
                if parts[0] == "label" and len(parts) == 2 and parts[1].isdigit():
                    n = int(self.headers.get("Content-Length") or 0)
                    try:
                        body = json.loads(self.rfile.read(min(n, 4096)) or b"{}")
                    except ValueError:
                        return self._json(dict(error="bad JSON"), 400)
                    cat, d, flags = body.get("catalyst"), body.get("direction"), body.get("flags") or []
                    if cat not in CATALYSTS or d not in (-1, 0, 1) or not isinstance(flags, list) or any(f not in RISK_FLAGS for f in flags):
                        return self._json(dict(error="catalyst, direction (-1/0/1) and flags must come from the label set"), 400)
                    if not r.set_label(int(parts[1]), cat, d, flags, by="human"):
                        return self._json(dict(error="no such note"), 404)
                    return self._json(dict(ok=True))
                if parts[0] == "stage" and len(parts) == 3 and parts[1] in STAGE_ORDER and parts[2] in ("on", "off", "shadow", "auto"):
                    from .research.stages import set_manual
                    return self._json(dict(ok=True, message=set_manual(r, parts[1], "active" if parts[2] == "on" else parts[2])))
                return self._json(dict(error="forbidden"), 403)
            finally:
                r.close()

        def do_POST(self):
            if self.headers.get("X-CM") != "1":  # custom header: a cross-site page can't send it
                return self._json(dict(error="forbidden"), 403)
            path = urlparse(self.path).path
            if path == "/api/kill":
                kill.trip("dashboard kill button")
                return self._json(dict(ok=True))
            parts = path.strip("/").split("/")   # api/outbox/<id|all>/approve|reject
            if len(parts) == 4 and parts[:2] == ["api", "watch"] and parts[2] in ("add", "remove"):
                return self._watch(parts[2], parts[3], urlparse(self.path).query)
            if len(parts) >= 4 and parts[:2] == ["api", "research"]:
                return self._research(parts[2:])
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


def serve(db, kill_file, port=8765, host="127.0.0.1", token=None, research_db=None, board=None):
    srv = ThreadingHTTPServer((host, port), make_handler(db, kill_file, token, research_db, board))
    print(f"Dashboard on http://{'127.0.0.1' if host in ('0.0.0.0', '') else host}:{port}  (db={db})", flush=True)
    srv.serve_forever()
