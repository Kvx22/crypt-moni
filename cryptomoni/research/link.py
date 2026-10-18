"""The engine's narrow, fail-safe handle on the research database.

A board's engine uses this to: tell the research service what to look at (its
subscription), read the per-symbol research view, read which pieces are switched on
(stages), ask for more evidence (requests), and record what research WOULD have changed
(shadow rows) so the gates can judge it. Every call swallows its own errors: a broken
or missing research database never stops the scanner, it only means "no research".
"""
from __future__ import annotations

import os
import time

from .db import STAGE_ORDER, ResearchDB, research_path

DEFAULTS = dict(
    enabled=True, weight=0.35, thin_discount=0.85, thin_sources=2, boost_min_sources=4, boost_min_kinds=2,
    penalty_cap=0.35, max_watch_adds_per_advisor_day=1, watch_min_z=0.5, view_max_age_s=3 * 3600,
)


def research_cfg(cfg):
    out = dict(DEFAULTS)
    out.update((cfg or {}).get("research") or {})
    return out


def board_name(cfg):
    r = (cfg or {}).get("research") or {}
    if r.get("board"):
        return str(r["board"])
    return os.path.splitext(os.path.basename((cfg or {}).get("db", "board")))[0] or "board"


class ResearchLink:
    def __init__(self, cfg, log=None, db=None):
        self.rc = research_cfg(cfg)
        self.board = board_name(cfg)
        self.log = log                      # engine._ev(ts, kind, sev, title, body)
        self.err, self.err_ts = None, 0.0
        self._stages, self._stages_ts = {}, 0.0
        self.db = db
        if self.db is None:
            try:
                self.db = ResearchDB(research_path(cfg))
            except Exception as e:  # noqa: BLE001  (a bad path or a locked file must not stop the board)
                self._fail(e)

    @property
    def ok(self):
        return self.db is not None

    def _fail(self, e, now=None):
        now = time.time() if now is None else now
        self.err = f"{type(e).__name__}: {e}"[:300]
        if self.log and now - self.err_ts > 3600:
            self.log(now, "research", "warn", "Research database unavailable", f"The board carries on without research: {self.err}")
            self.err_ts = now

    def _safe(self, fn, default=None):
        if self.db is None:
            return default
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            self._fail(e)
            return default

    # ---- what the board wants researched -----------------------------------------------------
    def subscribe(self, symbols, now=None):
        """symbols: {symbol: asset}"""
        return self._safe(lambda: self.db.subscribe(self.board, dict(symbols), now), None)

    # ---- the view and the switches ----------------------------------------------------------
    def views(self, symbols, now=None):
        now = time.time() if now is None else now
        v = self._safe(lambda: self.db.view(symbols), {}) or {}
        # a view the service stopped refreshing is ignored (the service may be down)
        return {s: x for s, x in v.items() if now - (x.get("view_ts") or 0) <= self.rc["view_max_age_s"]}

    def stages(self, now=None):
        now = time.time() if now is None else now
        if now - self._stages_ts > 60 or not self._stages:
            st = self._safe(lambda: {k: v["state"] for k, v in self.db.stages().items()}, None)
            self._stages = st if st is not None else {k: "shadow" for k in STAGE_ORDER}
            self._stages_ts = now
        return self._stages

    def active(self, stage, now=None):
        return self.stages(now).get(stage) == "active"

    # ---- asking for evidence, recording the shadow -------------------------------------------
    def request_many(self, reqs, now=None):
        """reqs: [dict(symbol, asset, advisor, question, priority)]"""
        def go():
            for r in reqs:
                self.db.request(self.board, r["symbol"], r["asset"], r["advisor"], r["question"], r["priority"], now)
            return len(reqs)
        return self._safe(go, 0)

    def shadow(self, rows):
        """rows: [(ts, symbol, asset, kind, base, adj, detail)]; the board name is added here."""
        return self._safe(lambda: self.db.add_shadow([(r[0], self.board, *r[1:]) for r in rows]), None)

    def discovered(self, asset, now=None, max_age_s=7 * 86400):
        now = time.time() if now is None else now
        rows = self._safe(lambda: self.db.discovered(asset=asset), []) or []
        return [r for r in rows if r["status"] in ("new", "validated") and now - (r["last_seen"] or 0) <= max_age_s]

    def mark_discovered(self, symbol, status, detail=""):
        return self._safe(lambda: self.db.set_discovered(symbol, status, detail), None)

    def propose_watch(self, ts, symbol, asset, advisor, z, reason, active, price0=None, bench0=None):
        """An advisor proposes a watch-only add. Capped per advisor per UTC day."""
        def go():
            day0 = ts - ts % 86400
            n = self.db.q1("SELECT COUNT(*) n FROM proposals WHERE board=? AND advisor=? AND ts>=?", (self.board, advisor, day0))["n"]
            if n >= self.rc["max_watch_adds_per_advisor_day"]:
                return None
            return self.db.add_proposal(ts, self.board, symbol, asset, advisor, z, reason, "added" if active else "shadow", price0, bench0)
        return self._safe(go, None)

    def watch_only(self):
        """Names the advisors put on the watch-only list (active proposals), newest first."""
        return self._safe(lambda: self.db.proposals(self.board, "added"), []) or []

    def put_kv(self, key, value):
        return self._safe(lambda: self.db.kv_set(f"{key}:{self.board}", value), None)

    def service_heartbeat(self, now=None):
        now = time.time() if now is None else now
        lease = self._safe(lambda: self.db.lease(), None)
        return None if not lease else now - (lease.get("heartbeat") or 0)
