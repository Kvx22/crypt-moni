"""The research service's side of forever search: writes the worker's plan, runs the two
supervisors (A works, B watches A and takes over if A stalls), and ingests what the
sandboxed worker stored into the research database as quarantined items.

The worker itself (research/forever.py) runs as its own process with a stripped
environment (no keys), its own small database, and no access to the boards' databases.
"""
from __future__ import annotations

import json
import os
import threading
import time

from .db import research_path


class ForeverHost:
    def __init__(self, cfg, rdb, log=print, names_fn=None, subscribed_fn=None):
        from .forever import QuarantineDB, SupervisorPair
        self.cfg, self.rdb, self.log = cfg, rdb, log
        self.names_fn, self.subscribed_fn = names_fn, subscribed_fn
        self.rc = dict(enabled=True)
        self.rc.update(((cfg.get("research") or {}).get("forever")) or {})
        base = os.path.join(os.path.dirname(os.path.abspath(research_path(cfg))), "forever")
        os.makedirs(base, exist_ok=True)
        self.dir = base
        self.qdb_path = os.path.join(base, "quarantine.db")
        self.plan_path = os.path.join(base, "plan.json")
        self.qdb = QuarantineDB(self.qdb_path)
        self.write_plan()
        standby = []
        for k, backends in ((1, ["bing_news", "google_news"]), (2, ["google_news"])):
            p = os.path.join(base, f"plan-standby{k}.json")
            self._write(p, dict(self._plan(), backends=backends))
            standby.append(p)
        self.pair = SupervisorPair(self.qdb_path, self.plan_path, cwd=base, log_path=os.path.join(base, "forever.log"),
                                   standby_plans=standby, kv=rdb)
        self.stop_ev = threading.Event()
        self.threads = {}
        self.plan_ts = time.time()
        self._start_threads()

    def _plan(self):
        from .forever import build_plan
        subs = self.subscribed_fn() if self.subscribed_fn else {}
        return build_plan(subs, self.names_fn, self.rc)

    @staticmethod
    def _write(path, plan):
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(plan, f)
        os.replace(tmp, path)

    def write_plan(self):
        self._write(self.plan_path, self._plan())

    def _loop(self, which):
        fn = getattr(self.pair, "tick_" + which)
        while not self.stop_ev.wait(10.0):
            try:
                fn(time.time())
            except Exception as e:  # noqa: BLE001
                self.log(f"forever supervisor {which}: {e!r}")

    def _start_threads(self):
        for which in ("a", "b"):
            t = self.threads.get(which)
            if t is None or not t.is_alive():
                t = threading.Thread(target=self._loop, args=(which,), daemon=True, name=f"forever-sup-{which}")
                t.start()
                self.threads[which] = t

    def tick(self, now):
        """Called by the service loop: keeps both supervisor threads alive and the plan fresh."""
        self._start_threads()
        if now - self.plan_ts > 3600:
            self.plan_ts = now
            self.write_plan()

    def ingest(self, rdb, now):
        from .forever import ingest
        return ingest(self.qdb, rdb, now)

    def status(self):
        try:
            st = self.pair.status()
        except Exception as e:  # noqa: BLE001
            st = dict(error=repr(e))
        st["threads"] = {k: t.is_alive() for k, t in self.threads.items()}
        return st

    def stop(self):
        self.stop_ev.set()
        try:
            self.pair.stop()
        except Exception:  # noqa: BLE001
            pass
