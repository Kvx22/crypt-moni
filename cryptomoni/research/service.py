"""The research service: one process (shared by both boards) that runs the workers,
labels what they bring in, builds the per-symbol research view, logs outcomes, trains
its models and switches research pieces on one at a time when they have earned it.

How it runs
  * `python run.py up` starts it as a supervised child process (ServiceSupervisor). If the
    other board's `up` already runs one, this one stands by and takes over if that one dies
    (a lease row in the research database decides who works).
  * The child exits when its parent goes away (its stdin closes), so a closed board never
    leaves an orphan behind.
  * With the simulator feed it runs in-process instead, stepped by the engine's own clock.

Timed jobs (seconds, real or simulated clock)
  label 20 · view 60 · prices 1200 · outcomes 600 · reliability 3600 · stage gates 3600 ·
  extractor + outcome model training 86400 (the extractor also trains on first start) ·
  health 30 · prune 86400 · forever-search ingest 60
Nothing here can place an order.
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
import traceback
import uuid

from .db import ResearchDB, research_path
from .http import DEFAULT_UA, Http
from .taxonomy import BENCH

DEFAULTS = dict(
    max_workers=60, threads=8, contact=None, label_every=20, view_every=60, prices_every=1200, outcomes_every=600,
    reliability_every=3600, gates_every=3600, train_every=86400, health_every=30, prune_every=86400,
    ingest_every=60, plan_every=3600, alert_cooldown_h=6, digest_hour_utc=13, keep_days=90,
    forever=dict(enabled=True), model_min_conf=0.6,
)


def service_cfg(cfg):
    rc = dict(DEFAULTS)
    rc.update((cfg or {}).get("research") or {})
    return rc


def _sim(cfg):
    return (cfg.get("feed") or {}).get("kind") == "sim"


class Service:
    def __init__(self, cfg, db=None, world=None, clock=None, log=print, http=None, plan=True):
        self.cfg, self.rc = cfg, service_cfg(cfg)
        self.db = db or ResearchDB(research_path(cfg))
        self.world, self.clock, self.log = world, clock or time.time, log
        self.sim = world is not None or _sim(cfg)
        ua = DEFAULT_UA + (f" contact:{self.rc['contact']}" if self.rc.get("contact") else "")
        self.http = http or Http(user_agent=ua, sec_contact=self.rc.get("contact"))
        self.owner = f"{os.getpid()}-{uuid.uuid4().hex[:8]}"
        self.timers = {}
        self.alerted = {}
        self.model = None
        self.aliases = None
        self._alias_key = None
        self.forever = None
        self.errors = []
        from . import sources
        from .workers import WorkerPool
        self.adapters = sources.catalog(self.rc, sim=self.sim)
        self.pool = WorkerPool(self.db, self.http, self.adapters, self.rc, clock=self.clock, world=world,
                               names_fn=self.names_for)
        if plan:
            self.pool.plan()
        self.prices = self._price_keeper()
        self._load_model()

    # ---- helpers ---------------------------------------------------------------------------
    def _due(self, name, every, now):
        last = self.timers.get(name)
        if last is None or now - last >= every:
            self.timers[name] = now
            return True
        return False

    def _job(self, name, fn):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001  (one failing job never stops the service)
            msg = f"{name}: {type(e).__name__}: {e}"
            self.errors = (self.errors + [msg])[-20:]
            self.log(f"research job error {msg}\n{traceback.format_exc(limit=4)}")
            now = self.clock()
            if now - self.alerted.get("job:" + name, 0) > 3600:
                self.db.event("error", "warn", f"Research job '{name}' failed", msg)
                self.alerted["job:" + name] = now
            return None

    def subscribed(self):
        return self.db.subscribed(now=self.clock())

    def names_for(self, symbols):
        al = self.alias_table()
        return {s: al.names(s) for s in symbols}

    def alias_table(self):
        from .aliases import Aliases
        subs = self.subscribed()
        key = tuple(sorted(subs.items()))
        if self.aliases is None or key != self._alias_key:
            self.aliases, self._alias_key = Aliases(subs), key
        return self.aliases

    def _price_keeper(self):
        from .outcome import PriceKeeper
        if self.world is not None:
            w = self.world

            def sim_bars(symbol, since):
                from ..feeds import HOUR
                return [(b.ts, b.close) for b in w.candles(symbol, HOUR, since)]
            return PriceKeeper(self.db, {"crypto": sim_bars, "stock": sim_bars})
        from ..feeds import HOUR, LiveFeed
        from .sources.stocks import yahoo_hourly
        live = LiveFeed()

        def crypto(symbol, since):
            return [(b.ts, b.close) for b in live.candles(symbol, HOUR, since, None if since is None else time.time())]

        def stock(symbol, since):
            return yahoo_hourly(self.http, symbol, "1mo" if since is None else "5d")
        return PriceKeeper(self.db, {"crypto": crypto, "stock": stock})

    # ---- models ----------------------------------------------------------------------------
    def _load_model(self):
        from .classifier import TextModel
        ch = self.db.champion("extractor")
        if ch and ch.get("blob"):
            try:
                self.model = TextModel.from_blob(ch["blob"])
                return
            except Exception as e:  # noqa: BLE001
                self.log(f"research: extractor v{ch['version']} would not load ({e}); retraining")
        self.model = None

    def train_extractor(self, now):
        """Train a challenger on seed labels + your corrections + confident labels; promote if it wins."""
        from .classifier import TextModel, challenger_wins, evaluate
        from .seed_labels import SEED
        human = [dict(title=r["title"] or "", text=r["text"] or "", catalyst=r["catalyst"], direction=int(r["direction"] or 0),
                      risk_flags=__import__("json").loads(r["risk_flags"] or "[]"), weight=3.0, ts=r["ts"])
                 for r in self.db.labels()]
        seed = [dict(e, weight=1.0, ts=0.0) for e in SEED]
        weak = [dict(title=r["title"] or "", text="", catalyst=r["catalyst"], direction=int(r["direction"] or 0),
                     risk_flags=__import__("json").loads(r["risk_flags"] or "[]"), weight=0.4, ts=r["ts"])
                for r in self.db.q("""SELECT title, catalyst, direction, risk_flags, ts FROM notes WHERE label_by IN ('rules','source')
                                      AND confidence>=0.8 AND catalyst!='other' ORDER BY id DESC LIMIT 4000""")]
        # a fixed holdout: every 4th seed example plus the newest quarter of your corrections
        hold_seed = [e for i, e in enumerate(seed) if i % 4 == 0]
        train_seed = [e for i, e in enumerate(seed) if i % 4 != 0]
        human.sort(key=lambda e: e["ts"])
        k = len(human) // 4
        hold_h, train_h = (human[-k:], human[:-k]) if k else ([], human)
        holdout = hold_seed + hold_h
        chall = TextModel()
        chall.fit(train_seed + train_h + weak)
        cm = evaluate(chall, holdout)
        champ_m = evaluate(self.model, holdout) if self.model is not None else None
        wins, why = challenger_wins(champ_m, cm)
        v = self.db.save_model("extractor", chall.to_blob(), dict(cm, n_train=len(train_seed) + len(train_h) + len(weak),
                                                                  n_human=len(human), champion=champ_m, reason=why))
        if wins:
            self.db.promote_model("extractor", v)
            self.model = chall
            self.db.event("model", "info", f"Extractor v{v} promoted", f"{why}. Holdout: catalyst F1 {cm.get('cat_macro_f1', 0):.2f}, "
                          f"direction accuracy {cm.get('dir_acc', 0):.0%} on {cm.get('n', 0)} labelled examples.")
            from .pipeline import relabel_recent
            relabel_recent(self.db, self.model, self.alias_table(), now)
        else:
            self.db.set_model_role("extractor", v, "rejected")
        self.db.prune_model_blobs("extractor")
        return dict(version=v, promoted=wins, reason=why, metrics=cm)

    # ---- one pass of everything that is due -------------------------------------------------
    def step(self, now=None, run_workers=True):
        now = self.clock() if now is None else now
        rc = self.rc
        if run_workers:
            for slot in self.pool.due(now):
                self.pool.run_slot(slot, now)
            if self._due("supervise", 60, now):
                self._job("supervise", lambda: self.pool.supervise(now))   # it logs its own events
        if self.forever and self._due("ingest", rc["ingest_every"], now):
            self._job("ingest", lambda: self.forever.ingest(self.db, now))
        if self._due("label", rc["label_every"], now):
            self._job("label", lambda: self.label(now))
        if self._due("prices", rc["prices_every"], now):
            self._job("prices", lambda: self.refresh_prices(now))
        if self._due("outcomes", rc["outcomes_every"], now):
            from .outcome import mature
            self._job("outcomes", lambda: mature(self.db, now, rc))
        if self._due("view", rc["view_every"], now):
            self._job("view", lambda: self.build_view(now))
        if self._due("reliability", rc["reliability_every"], now):
            from .outcome import update_reliability
            self._job("reliability", lambda: update_reliability(self.db, now))
            self._job("requests", lambda: self.db.expire_requests(now=now))
        if self.model is None or self._due("train", rc["train_every"], now):
            self._job("train extractor", lambda: self.train_extractor(now))
            from .outcome import train_outcome_model
            self._job("train outcome model", lambda: self._outcome_report(train_outcome_model(self.db, now)))
        if self._due("gates", rc["gates_every"], now):
            from .stages import evaluate
            self._job("stage gates", lambda: evaluate(self.db, now, rc))   # it logs its own events
        if self._due("gaps", 600, now):
            self._job("gap check", lambda: self.gap_check(now))
        if self._due("health", rc["health_every"], now):
            self._job("health", lambda: self.health(now))
            self._job("alerts", self.alert_new_events)
        if self._due("prune", rc["prune_every"], now):
            self._job("prune", lambda: self.db.prune(rc["keep_days"], now))

    def label(self, now):
        from .pipeline import label_items
        total = dict(items=0, notes=0)
        for _ in range(5):   # drain a backlog in bounded batches
            r = label_items(self.db, self.alias_table(), self.model, now, self.rc)
            for k in total:
                total[k] += (r or {}).get(k, 0)
            if not r or r.get("items", 0) < 300:
                break
        return total

    def refresh_prices(self, now):
        subs = dict(self.subscribed())
        if any(a == "stock" for a in subs.values()):
            subs.setdefault(BENCH["stock"], "stock")
        subs.setdefault(BENCH["crypto"], "crypto")
        return self.prices.refresh(subs, now)

    def build_view(self, now):
        from .aggregate import build_view
        subs = self.subscribed()
        views = build_view(self.db, now, subs, self.rc)
        self.db.set_view(views, now)
        # requests that now have evidence are closed (the advisors see the new notes next round)
        for r in self.db.open_requests(500):
            v = views.get(r["symbol"])
            if v and v.get("n_notes", 0) > 0 and (v.get("last_note_ts") or 0) > r["ts"]:
                self.db.close_requests(r["symbol"], v["n_notes"], now)
        return len(views)

    def gap_check(self, now):
        from .aggregate import team_coverage
        cov = team_coverage(self.db, now, self.subscribed())
        # risk, discovery and filings only speak when something happens: quiet is normal for them, so
        # only a team with no working sources counts as a gap there (not a lack of fresh notes)
        for team in ("risk", "discovery", "filings"):
            if isinstance(cov.get(team), dict):
                cov[team]["symbols_covered"] = cov[team].get("symbols_total", 0)
        self.pool.gap_check(now, cov)   # it logs its own events
        return cov

    def _outcome_report(self, rep):
        if rep and rep.get("promoted"):
            self.db.event("model", "info", "Outcome model promoted",
                          f"Log-loss {rep.get('logloss_challenger', 0):.3f} vs {rep.get('logloss_champion', rep.get('logloss_baseline', 0)):.3f} "
                          f"on {rep.get('n_holdout', 0)} held-out notes.")
        self.db.kv_set("outcome_report", dict(rep or {}, ts=self.clock()))
        return rep

    # ---- events, alerts, health ----------------------------------------------------------------
    def alert_new_events(self):
        """Phone alerts for new warnings (failovers, degraded teams, diversity shortfalls, stage
        switches back to shadow, failing jobs), plus an info push when a stage switches on."""
        last = self.db.kv_get("alerted_event_id")
        rows = self.db.q("SELECT * FROM events WHERE id>? ORDER BY id LIMIT 200", (last or 0,))
        if last is None:   # first start: don't replay history to the phone
            self.db.kv_set("alerted_event_id", rows[-1]["id"] if rows else 0)
            return 0
        n = 0
        for e in rows:
            if e["severity"] in ("warn", "critical") or (e["kind"] == "stage" and e["severity"] == "info"):
                n += bool(self.alert(e["title"], e["title"], e["body"] or ""))
        if rows:
            self.db.kv_set("alerted_event_id", rows[-1]["id"])
        return n

    def alert(self, key, title, body):
        """Phone push (ntfy), at most once per key per `alert_cooldown_h`."""
        now = self.clock()
        if now - self.alerted.get(key, -1e18) < self.rc["alert_cooldown_h"] * 3600:
            return False
        self.alerted[key] = now
        if self.sim:
            return False
        from ..notify import push
        return push(self.cfg, f"Research: {title}"[:120], body[:900], (self.cfg.get("execution") or {}).get("dashboard_url"))

    def health(self, now):
        h = self.pool.health()
        last24 = now - 86400
        c = self.db.q1("""SELECT (SELECT COUNT(*) FROM items WHERE fetched_ts>?) items,
                          (SELECT COUNT(*) FROM notes WHERE fetched_ts>?) notes,
                          (SELECT COUNT(*) FROM requests WHERE status='open') open_requests,
                          (SELECT COUNT(*) FROM discovered WHERE status IN ('new','validated')) discovered""", (last24, last24))
        h.update(counts24=c, ts=now, owner=self.owner, errors=self.errors[-5:], forever=self.forever.status() if self.forever else None,
                 subscribed=len(self.subscribed()), stages={k: v["state"] for k, v in self.db.stages().items()},
                 models={m["name"]: dict(version=m["version"], metrics=m["metrics"]) for m in self.db.models() if m["role"] == "champion"},
                 http=dict(self.http.stats))
        self.db.kv_set("health_now", h)
        if self._due("health_hist", 600, now):
            self.db.add_health(dict(counts=h.get("counts"), counts24=c, teams=h.get("teams")), now)
        # stalls: an active slot that has not reported for 3x its interval
        for s in h.get("slots", []):
            if s.get("state") == "active" and s.get("stale"):
                self.alert(f"stall:{s['id']}", f"worker {s['adapter']} stalled", f"No report for {s.get('age_s', 0) / 60:.0f} min.")
        if self._due("digest", 3600, now) and time.gmtime(now).tm_hour == self.rc["digest_hour_utc"]:
            day = time.strftime("%Y-%m-%d", time.gmtime(now))
            if self.db.kv_get("digest_day") != day:
                self.db.kv_set("digest_day", day)
                cn = h.get("counts", {})
                body = (f"Last 24h: {c['items']} items, {c['notes']} notes, {c['open_requests']} open requests. Workers: "
                        f"{cn.get('active', 0)} active, {cn.get('standby', 0)} standby, {cn.get('failed', 0)} failed. Stages: "
                        + ", ".join(f"{k} {v}" for k, v in h["stages"].items()))
                self.db.event("digest", "info", "Research daily digest", body)
                if not self.sim:
                    from ..notify import push
                    push(self.cfg, "Research daily digest", body, (self.cfg.get("execution") or {}).get("dashboard_url"))
        return h

    # ---- the long-running process ---------------------------------------------------------------
    def acquire(self, now=None):
        return self.db.lease_try(self.owner, os.getpid(), now=self.clock() if now is None else now)

    def start_forever(self):
        fc = dict(enabled=True)
        fc.update(self.rc.get("forever") or {})
        if self.sim or not fc.get("enabled", True):
            return None
        from .forever_host import ForeverHost
        self.forever = ForeverHost(self.cfg, self.db, log=self.log, names_fn=self.names_for, subscribed_fn=self.subscribed)
        return self.forever

    def run(self, stop):
        """Blocking: work while holding the lease, stand by otherwise. `stop` is a threading.Event."""
        self.log(f"research service {self.owner}: {len(self.adapters)} adapters "
                 f"({sum(1 for a in self.adapters if a.role == 'primary')} primaries, {sum(1 for a in self.adapters if a.role == 'standby')} standbys)")
        holding = False
        pool_thread = pool_stop = None
        while not stop.is_set():
            now = self.clock()
            got = self._job("lease", lambda: self.acquire(now))
            if got and not holding:
                holding = True
                self.log("research service: holding the lease, workers starting")
                self.db.event("service", "info", "Research service started", f"pid {os.getpid()}, {len(self.adapters)} adapters")
                pool_stop = threading.Event()      # the pool's own switch: losing the lease stops the pool, not the service
                pool_thread = threading.Thread(target=self.pool.run_forever, args=(pool_stop, self.rc["threads"]), daemon=True,
                                               name="research-pool")   # it runs supervise() itself; the service runs the gap check
                pool_thread.start()
                self.start_forever()
            elif not got and holding:
                holding = False   # someone else took over (we were stuck too long): stand down
                self.log("research service: lost the lease, standing by")
                pool_stop.set()
                if self.forever:
                    self.forever.stop()
                    self.forever = None
            if holding:
                if self.forever:
                    self._job("forever", lambda: self.forever.tick(now))
                self.step(now, run_workers=False)
            stop.wait(5.0 if holding else 10.0)
        if pool_stop is not None:
            pool_stop.set()
        if self.forever:
            self.forever.stop()
        if holding:
            self.db.lease_release(self.owner)
        self.log("research service stopped")


def run_service(cfg):
    """Entry point of the child process (`python run.py research-service --config F`)."""
    stop = threading.Event()

    def watch_parent():
        try:
            while sys.stdin.buffer.read(1):
                pass
        except (OSError, ValueError, AttributeError):
            pass
        stop.set()   # stdin closed: the board that started us is gone

    if not sys.stdin or not sys.stdin.isatty():
        threading.Thread(target=watch_parent, daemon=True, name="parent-watch").start()
    svc = Service(cfg)
    try:
        svc.run(stop)
    except KeyboardInterrupt:
        stop.set()


class ServiceSupervisor:
    """Runs inside the board's `up` process: starts the research service as a child, restarts it
    with back-off if it dies, and reports a crash loop instead of spinning."""

    BACKOFF = (10, 30, 120, 600)

    def __init__(self, cfg_path, cfg, log, python=None, run_py=None):
        self.cfg_path, self.cfg, self.log = cfg_path, cfg, log
        self.python = python or sys.executable
        self.run_py = run_py or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "run.py"))
        self.proc, self.crashes, self.next_start = None, [], 0.0
        self.degraded = False
        board = os.path.splitext(os.path.basename(cfg.get("db", "board")))[0]
        self.log_path = os.path.join(os.path.dirname(research_path(cfg)) or ".", f"research-{board}.log")
        self.stop_ev = threading.Event()

    def _spawn(self):
        os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
        out = open(self.log_path, "a", encoding="utf-8", errors="replace")
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self.proc = subprocess.Popen([self.python, "-u", self.run_py, "research-service", "--config", self.cfg_path],
                                     stdin=subprocess.PIPE, stdout=out, stderr=subprocess.STDOUT, creationflags=flags)
        out.close()

    def tick(self, now=None):
        now = time.time() if now is None else now
        if self.proc is not None and self.proc.poll() is None:
            return "running"
        if self.proc is not None:
            code = self.proc.returncode
            try:
                self.proc.stdin.close()
            except (OSError, AttributeError):
                pass
            self.proc = None
            self.crashes = [t for t in self.crashes if now - t < 1800] + [now]
            k = min(len(self.crashes) - 1, len(self.BACKOFF) - 1)
            self.next_start = now + self.BACKOFF[k]
            if len(self.crashes) >= 5 and not self.degraded:
                self.degraded = True
                self.log(now, "research", "critical", "Research service keeps crashing",
                         f"It exited {len(self.crashes)} times in 30 min (last code {code}); retrying every 10 min. "
                         f"The board runs without research meanwhile. See {self.log_path}.")
            else:
                self.log(now, "research", "warn", "Research service exited", f"code {code}; restarting in {self.BACKOFF[k]}s")
        if now < self.next_start:
            return "waiting"
        try:
            self._spawn()
            if self.degraded and len([t for t in self.crashes if now - t < 1800]) == 0:
                self.degraded = False
            return "started"
        except OSError as e:
            self.next_start = now + 600
            self.log(now, "research", "warn", "Could not start the research service", str(e)[:300])
            return "failed"

    def run(self):
        while not self.stop_ev.wait(5.0):
            self.tick()

    def start(self):
        threading.Thread(target=self.run, daemon=True, name="research-supervisor").start()
        self.tick()
        return self

    def stop(self):
        self.stop_ev.set()
        p = self.proc
        if p is not None and p.poll() is None:
            try:
                p.stdin.close()            # the child stops itself when its stdin closes
                p.wait(10)
            except (OSError, subprocess.TimeoutExpired):
                p.terminate()
