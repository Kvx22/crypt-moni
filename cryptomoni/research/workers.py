"""Worker slots: who fetches what, when, and who covers for whom.

A "worker" is a slot: one source adapter (or one shard of a per-symbol adapter) with its own
timer, error count and heartbeat. The forty workers are twenty primaries and twenty standbys.

  * Primaries run on their adapter's interval, stretched or shrunk by how useful the source has
    proved (useful sources are polled up to twice as often, useless ones half as often).
  * A primary that fails three times in a row, or stops beating, is marked failed and its
    standby takes over. The failed primary gets one probe run every 30 minutes; when it works
    again it goes back to work and the standby goes back to waiting.
  * Errors back off exponentially (interval x 2^k, at most 6 h). A rate limit just waits the
    time the source asked for, and only counts as a failure after three in a row.
  * Per-symbol adapters (a news search per ticker) split the subscribed symbols into shards by
    a stable checksum, look at symbols an advisor asked about first, and rotate through the rest.
  * The gap detector checks each team's coverage. A team that is not covering its symbols
    gets more workers: first a dormant reserve adapter, then one more shard of a per-symbol
    primary, and if nothing is left, a once-a-day "diversity shortfall" note. Grown workers
    that turn out to be rarely useful are stopped again after two calm days.

Nothing here can place an order. One bad adapter or one bad reading never stops the loop:
every run, every check and every database write is caught and recorded.
"""
from __future__ import annotations

import copy
import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from .http import RateLimited, SourceError
from .sources import role_of
from .sources.base import FetchContext
from .types import TEAMS, Item

FAIL_AFTER = 3              # errors in a row before a slot is failed over
RATE_LIMITS_COUNT_AFTER = 3 # rate limits in a row before they count as errors
PROBE_S = 1800.0            # a failed slot is retried this often
MAX_BACKOFF_S = 6 * 3600.0
MIN_INTERVAL_S = 120.0
STALE_X = 4.0               # heartbeat older than this many intervals while due -> failed
REQUEST_SOON_S = 300.0      # per-symbol slots that can serve an open request run within 5 min
GAP_SHARE = 0.6             # a team should have fresh notes on at least 60% of its symbols
SHRINK_AFTER_S = 48 * 3600.0
SHRINK_USEFULNESS = 0.45
MAX_SHARDS = 4
SHORTFALL_EVERY_S = 86400.0
TICK_S = 5.0
SUPERVISE_EVERY_S = 60.0


def _num(x, default=0.0):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return default
    return x if x == x else default   # NaN -> default


def _dur(s):
    s = max(0.0, float(s))
    return f"{s / 3600:.1f} h" if s >= 5400 else f"{s / 60:.0f} min" if s >= 90 else f"{s:.0f} s"


def in_shard(symbol, shard, nshards):
    """Stable across processes and restarts (crc32, never the salted built-in hash)."""
    return nshards <= 1 or zlib.crc32(str(symbol).encode("utf-8")) % nshards == shard


def slot_id(adapter, shard=0, nshards=1):
    return f"{getattr(adapter, 'team', 'catalysts')}/{adapter.name}#{shard}of{nshards}"


def usefulness_mult(u):
    """Interval multiplier: usefulness 1.0 -> 0.5x (twice as often), 0.5 (unknown) -> 1x, 0 -> 2x."""
    if u is None:
        return 1.0
    u = min(1.0, max(0.0, _num(u, 0.5)))
    return min(2.0, max(0.5, 2.0 ** (1.0 - 2.0 * u)))


def _covers(adapter, asset):
    try:
        return bool(adapter.covers(asset))
    except Exception:
        return False


@dataclass(eq=False)        # identity equality: slots are hashable and tracked as objects
class Slot:
    id: str
    adapter: object
    role: str = "primary"   # primary | standby | grown
    shard: int = 0
    nshards: int = 1
    state: str = "active"   # active | standby | failed | stopped
    consecutive_err: int = 0
    last_ok: float | None = None
    last_err: float | None = None
    err: str = ""
    next_due: float = 0.0
    backoff_s: float = 0.0
    runs: int = 0
    items: int = 0
    standby_of: str | None = None
    grown: bool = False
    mult: float = 1.0       # interval multiplier from source usefulness (0.5..2)
    beat: float = 0.0       # last heartbeat: planned, run started, run finished
    rate_limited: int = 0   # rate limits in a row
    running: bool = False

    @property
    def name(self):
        return self.adapter.name

    @property
    def team(self):
        return getattr(self.adapter, "team", "catalysts")

    @property
    def interval(self):
        return max(MIN_INTERVAL_S, _num(getattr(self.adapter, "interval_s", 900.0), 900.0) * self.mult)

    def d(self):
        a = self.adapter
        return dict(id=self.id, adapter=self.name, team=self.team, role=self.role, shard=self.shard, nshards=self.nshards,
                    state=self.state, consecutive_err=self.consecutive_err, last_ok=self.last_ok, last_err=self.last_err,
                    err=self.err, next_due=self.next_due, backoff_s=self.backoff_s, runs=self.runs, items=self.items,
                    standby_of=self.standby_of, grown=self.grown, mult=round(self.mult, 3), interval_s=self.interval,
                    heartbeat=self.beat, running=self.running, per_symbol=bool(getattr(a, "per_symbol", False)),
                    domains=list(getattr(a, "domains", ()) or ()), description=getattr(a, "description", ""))


class WorkerPool:
    def __init__(self, db, http, adapters, cfg=None, clock=time.time, world=None, names_fn=None):
        self.db, self.http, self.cfg = db, http, dict(cfg or {})
        self.clock, self.world, self.names_fn = clock, world, names_fn
        self.adapters, seen = [], set()
        for a in adapters or []:
            if isinstance(getattr(a, "name", None), str) and a.name not in seen:
                seen.add(a.name)
                self.adapters.append(a)
        self.max_workers = max(1, int(_num(self.cfg.get("max_workers", 60), 60)))
        self.warmup_s = _num(self.cfg.get("gap_warmup_s", 2 * 3600), 2 * 3600)
        self.slots = []
        self.lock = threading.RLock()
        self._state_locks = {}     # adapter name -> lock for merging its persisted state
        self._pending = []         # events raised by runs, handed out by the next supervise()
        self._useful = {}          # adapter name -> usefulness (0.5 = unknown)
        self._last_gap, self._shortfall = {}, {}
        self._t0 = None
        self._stop = threading.Event()
        self.loop_error = ""

    # ------------------------------------------------------------------ helpers
    def _now(self, now):
        return self.clock() if now is None else now

    def _live(self):
        return [s for s in self.slots if s.state != "stopped"]

    def _primaries(self, name):
        return [s for s in self.slots if s.role == "primary" and s.name == name]

    def _standbys(self, pname):
        return [s for s in self.slots if s.role == "standby" and s.standby_of and
                getattr(s.adapter, "standby_for", None) == pname]

    def _save(self, s, now, detail=""):
        try:
            self.db.worker_set(s.id, s.name, s.team, s.role, s.shard, s.nshards, s.state, detail, int(s.grown), now=now)
        except Exception:
            pass

    def _sync(self, s, now, detail=None):
        try:
            self.db.worker_beat(s.id, state=s.state, detail=detail, now=now)
        except Exception:
            pass

    def _drop_stale_rows(self):
        try:
            self.db.drop_workers([s.id for s in self.slots])
        except Exception:
            pass

    def _event(self, kind, severity, title, body, now):
        try:
            self.db.event(kind, severity, title, body, ts=now)
        except Exception:
            pass
        return dict(ts=now, kind=kind, severity=severity, title=title, body=body)

    def refresh_usefulness(self):
        try:
            rows = self.db.sources()
        except Exception:
            return
        with self.lock:
            for s in self.slots:
                u = (rows.get(s.name) or {}).get("usefulness")
                self._useful[s.name] = 0.5 if u is None else _num(u, 0.5)
                s.mult = usefulness_mult(u)

    # ------------------------------------------------------------------ plan
    def plan(self, now=None):
        """One slot per primary (sharded if cfg["shards"][name] says so), one per standby; dormant: none."""
        now = self._now(now)
        shards = self.cfg.get("shards") or {}
        with self.lock:
            self._t0 = now
            self.slots = []
            first = {}
            for a in self.adapters:
                if role_of(a) != "primary":
                    continue
                n = int(min(MAX_SHARDS, max(1, _num(shards.get(a.name, 1), 1)))) if getattr(a, "per_symbol", False) else 1
                for k in range(n):
                    if len(self.slots) < self.max_workers:
                        self.slots.append(Slot(slot_id(a, k, n), a, "primary", k, n, "active", next_due=now, beat=now))
                        first.setdefault(a.name, self.slots[-1].id)
            for a in self.adapters:
                if role_of(a) != "standby" or len(self.slots) >= self.max_workers:
                    continue
                p = first.get(getattr(a, "standby_for", None))
                # a standby whose primary is disabled or missing simply runs in its place
                self.slots.append(Slot(slot_id(a), a, "standby", state="standby" if p else "active", standby_of=p,
                                       next_due=now, beat=now))
            slots = list(self.slots)
        self.refresh_usefulness()
        for s in slots:
            self._save(s, now, f"standby for {s.standby_of}" if s.state == "standby" else "")
        self._drop_stale_rows()
        return slots

    # ------------------------------------------------------------------ scheduling
    def _open_requests(self):
        try:
            return [(r["symbol"], r.get("asset")) for r in self.db.open_requests() if r.get("symbol")]
        except Exception:
            return []

    def due(self, now=None):
        """Active slots whose time has come, plus failed slots due a probe run. Never a running one."""
        now = self._now(now)
        reqs = self._open_requests()
        out = []
        with self.lock:
            for s in self.slots:
                if s.running:
                    continue
                if s.state == "active":
                    if (reqs and getattr(s.adapter, "per_symbol", False) and not s.consecutive_err and not s.rate_limited
                            and any((not asset or _covers(s.adapter, asset)) and in_shard(sym, s.shard, s.nshards)
                                    for sym, asset in reqs)):
                        s.next_due = min(s.next_due, now + REQUEST_SOON_S)
                    if s.next_due <= now:
                        out.append(s)
                elif s.state == "failed" and s.next_due <= now:
                    out.append(s)
        return out

    def _pick_symbols(self, s, covered, requested, state):
        """This shard's symbols: requested first, then the rest from a rotating cursor, capped."""
        mine = sorted(sym for sym in covered if in_shard(sym, s.shard, s.nshards))
        mine_set = set(mine)
        req = [sym for sym in dict.fromkeys(requested) if sym in mine_set]
        rest = [sym for sym in mine if sym not in set(req)]
        cap = max(1, int(_num(getattr(s.adapter, "max_symbols_per_run", 8), 8)))
        pick = req[:cap]
        room = cap - len(pick)
        if rest and room > 0:
            key = f"_cursor_{s.shard}of{s.nshards}"
            start = int(_num(state.get(key), 0)) % len(rest)
            take = min(room, len(rest))
            pick += [rest[(start + i) % len(rest)] for i in range(take)]
            state[key] = (start + take) % len(rest)
        return {sym: covered[sym] for sym in pick}

    def _state_lock(self, name):
        with self.lock:
            return self._state_locks.setdefault(name, threading.Lock())

    def _persist_state(self, name, base, work):
        """Write back only the keys this run changed, so two shards of one adapter don't clobber each other."""
        if not isinstance(work, dict) or work == base:
            return
        key = "adapter_state:" + name
        with self._state_lock(name):
            cur = self.db.kv_get(key, {})
            cur = cur if isinstance(cur, dict) else {}
            for k, v in work.items():
                if k not in base or base[k] != v:
                    cur[k] = v
            for k in set(base) - set(work):
                cur.pop(k, None)
            self.db.kv_set(key, cur)

    @staticmethod
    def _clean(raw):
        if raw is None:
            return []
        if isinstance(raw, Item):
            return [raw]
        try:
            raw = list(raw)
        except TypeError:
            raise ValueError(f"fetch() returned {type(raw).__name__}, not a list of items") from None
        out = []
        for it in raw:
            if isinstance(it, Item):
                out.append(it)
            elif isinstance(it, dict):
                try:
                    out.append(Item.from_d(it))
                except Exception:
                    pass
        return out

    def _fetch(self, s, now):
        """-> (new items stored, skipped). Raises whatever the adapter raises."""
        a = s.adapter
        try:
            subs = self.db.subscribed(now=now) or {}
        except Exception:
            subs = {}
        covered = {sym: asset for sym, asset in subs.items() if _covers(a, asset)}
        requested = [sym for sym, _ in self._open_requests()]
        base = self.db.kv_get("adapter_state:" + a.name, {})
        base = base if isinstance(base, dict) else {}
        work = copy.deepcopy(base)
        if getattr(a, "per_symbol", False):
            symbols = self._pick_symbols(s, covered, requested, work)
            if not symbols:
                self._persist_state(a.name, base, work)
                return 0, True
        else:
            symbols = {sym: covered[sym] for sym in requested if sym in covered}
            symbols.update(covered)
        names = {}
        if self.names_fn:
            try:
                names = self.names_fn(list(symbols)) or {}
            except Exception:
                names = {}
        ctx = FetchContext(http=self.http, now=now, symbols=symbols, requested=requested, state=work,
                           cfg=self.cfg, names=names if isinstance(names, dict) else {}, world=self.world)
        try:
            raw = a.fetch(ctx)
        finally:
            try:
                self._persist_state(a.name, base, ctx.state)
            except Exception:
                pass
        items = self._clean(raw)
        return (self.db.add_items(items, s.team, now=now) if items else 0), False

    def run_slot(self, slot, now=None):
        """Run one slot once and record the outcome. Returns the number of new items. Never raises."""
        now = self._now(now)
        n = 0
        try:
            with self.lock:
                slot.running, slot.beat = True, now
            self._sync(slot, now)
            try:
                n, skipped = self._fetch(slot, now)
            except RateLimited as e:
                self._rate_limited(slot, now, e)
            except Exception as e:
                self._failed_run(slot, now, e)
            else:
                self._ok(slot, now, n, skipped)
        except Exception:
            pass   # bookkeeping itself failed (database gone?): the next run tries again
        finally:
            slot.running = False
        return n

    def _ok(self, s, now, n, skipped):
        a = s.adapter
        with self.lock:
            s.consecutive_err = s.rate_limited = 0
            s.err, s.backoff_s = "", 0.0
            s.runs += 1
            s.items += n
            s.beat = now
            s.next_due = now + s.interval
            if not skipped:
                s.last_ok = now
            if s.state == "failed":
                self._pending += self._recover(s, now)
            detail = "nothing to look at in this shard" if skipped else f"ok: {n} new item{'s' if n != 1 else ''}"
        if not skipped:
            try:
                self.db.source_ok(a.name, s.team, getattr(a, "kind", "news"), n, s.interval, now=now)
            except Exception:
                pass
        self._sync(s, now, detail)

    def _backoff(self, s):
        return min(MAX_BACKOFF_S, s.interval * 2.0 ** min(s.consecutive_err, 30))

    def _rate_limited(self, s, now, e):
        ra = min(MAX_BACKOFF_S, max(1.0, _num(getattr(e, "retry_after", 60.0), 60.0)))
        with self.lock:
            s.rate_limited += 1
            s.runs += 1
            s.beat = now
            counted = s.rate_limited >= RATE_LIMITS_COUNT_AFTER
            wait = ra
            if counted:
                s.consecutive_err += 1
                s.last_err, s.err = now, f"rate limited {s.rate_limited} times in a row: {e}"[:300]
                wait = max(ra, self._backoff(s))
            if s.state == "failed":
                wait = max(wait, PROBE_S)
            s.backoff_s, s.next_due = wait, now + wait
        if counted:
            try:
                self.db.source_err(s.name, s.team, getattr(s.adapter, "kind", "news"), s.err, s.interval, now=now)
            except Exception:
                pass
        self._sync(s, now, f"rate limited, next try in {_dur(wait)}")

    def _failed_run(self, s, now, e):
        # a SourceError already says what went wrong in plain words; anything else is a bug worth naming
        msg = (str(e) or type(e).__name__) if isinstance(e, SourceError) else f"{type(e).__name__}: {e}"
        with self.lock:
            s.rate_limited = 0
            s.consecutive_err += 1
            s.runs += 1
            s.beat = now
            s.last_err, s.err = now, msg[:300]
            wait = PROBE_S if s.state == "failed" else self._backoff(s)
            s.backoff_s, s.next_due = wait, now + wait
            k = s.consecutive_err
        try:
            self.db.source_err(s.name, s.team, getattr(s.adapter, "kind", "news"), s.err, s.interval, now=now)
        except Exception:
            pass
        self._sync(s, now, f"error {k} in a row: {s.err}; next try in {_dur(wait)}")

    # ------------------------------------------------------------------ failover
    def _fail(self, s, now, why):
        """Mark a slot failed and bring in cover. Lock held. Returns events."""
        s.state = "failed"
        s.next_due, s.backoff_s = now + PROBE_S, PROBE_S
        self._sync(s, now, f"failed ({why}); probing every {_dur(PROBE_S)}")
        if s.role == "primary" or s.standby_of:
            pname = s.name if s.role == "primary" else s.adapter.standby_for
            cover = self._standbys(pname)
            if any(x.state == "active" for x in cover):
                return []          # another shard already failed over; the standby is on it
            spare = [x for x in cover if x.state == "standby"]
            if spare:
                sb = spare[0]
                sb.state, sb.next_due, sb.beat = "active", now, now   # fresh beat: it has been waiting, not stuck
                self._sync(sb, now, f"covering for {pname}")
                return [self._event("failover", "info", f"Standby {sb.name} took over from {s.name}: {why}"[:300],
                                    f"{s.name} failed ({why}). {sb.name} is doing its job until it works again; "
                                    f"{s.name} gets a test run every {_dur(PROBE_S)}.", now)]
            what = (f"{pname} and its standby {s.name} are both down" if s.role == "standby"
                    else f"{pname} is down and has no standby" if not cover else f"{pname} and its standbys are down")
            return [self._event("degraded", "warn", f"The {s.team} team is degraded: {what}",
                                f"Last error: {why}. The failed workers get a test run every {_dur(PROBE_S)}.", now)]
        return [self._event("failed", "warn", f"{s.name} stopped working: {why}"[:300],
                            f"It gets a test run every {_dur(PROBE_S)}.", now)]

    def _recover(self, s, now):
        """A failed slot just had a good run. Lock held. Returns events."""
        if s.role == "standby" and s.standby_of:
            pname = s.adapter.standby_for
            covering = any(p.state == "failed" for p in self._primaries(pname)) and not any(
                x.state == "active" for x in self._standbys(pname))
            s.state = "active" if covering else "standby"
            tail = f"it keeps covering for {pname}" if covering else "it is back on standby"
            return [self._event("recovered", "info", f"{s.name} is working again", f"{s.name} works again; {tail}.", now)]
        s.state = "active"
        if s.role == "primary" and not any(p.state == "failed" for p in self._primaries(s.name)):
            back = [x for x in self._standbys(s.name) if x.state == "active"]
            for x in back:
                x.state = "standby"
                self._sync(x, now, f"standby for {s.id}")
            if back:
                return [self._event("recovered", "info", f"{s.name} is working again",
                                    f"{s.name} is back at work; {', '.join(x.name for x in back)} went back on standby.", now)]
        return [self._event("recovered", "info", f"{s.name} is working again", f"{s.name} is back at work.", now)]

    def supervise(self, now=None):
        """Fail over slots that keep erroring or stopped beating. Returns events (including recoveries since last time)."""
        now = self._now(now)
        self.refresh_usefulness()
        with self.lock:
            events = []
            for s in list(self.slots):
                if s.state != "active":
                    continue
                if s.consecutive_err >= FAIL_AFTER:
                    why = s.err or "repeated errors"
                elif s.next_due <= now and now - s.beat > STALE_X * max(s.interval, s.backoff_s):
                    why = f"no heartbeat for {_dur(now - s.beat)}"
                else:
                    continue
                events += self._fail(s, now, why)
            events, self._pending = self._pending + events, []
        return events

    # ------------------------------------------------------------------ growth and shrink
    def _rename_shards(self, a, now):
        mine = sorted(self._primaries(a.name), key=lambda x: x.shard)
        for k, x in enumerate(mine):
            x.shard, x.nshards, x.id = k, len(mine), slot_id(a, k, len(mine))
        for sb in self._standbys(a.name):
            sb.standby_of = mine[0].id if mine else None
        for x in mine + self._standbys(a.name):
            self._save(x, now, "re-sharded")
        self._drop_stale_rows()
        return mine

    def _grow(self, team, now, why):
        """One remedy for a team with a gap. Lock held. Returns events."""
        live = self._live()
        if len(live) < self.max_workers:
            used = {s.name for s in live}
            for a in self.adapters:
                if role_of(a) == "dormant" and getattr(a, "team", None) == team and a.name not in used:
                    old = [s for s in self.slots if s.adapter is a]
                    s = old[0] if old else Slot(slot_id(a), a, "grown", grown=True)
                    if not old:
                        self.slots.append(s)
                    s.state, s.next_due, s.beat, s.consecutive_err, s.rate_limited = "active", now, now, 0, 0
                    s.mult = usefulness_mult(self._useful.get(a.name))
                    self._save(s, now, f"grown: {why}")
                    return [self._event("growth", "info", f"Started reserve source {a.name} for the {team} team",
                                        f"The {team} team had a gap ({why}).", now)]
            cands = [a for a in self.adapters if role_of(a) == "primary" and getattr(a, "team", None) == team
                     and getattr(a, "per_symbol", False)]
            cands = [(len(self._primaries(a.name)), i, a) for i, a in enumerate(cands)
                     if 0 < len(self._primaries(a.name)) < MAX_SHARDS and any(p.state == "active" for p in self._primaries(a.name))]
            if cands:
                n, _, a = min(cands, key=lambda c: c[:2])
                new = Slot(slot_id(a, n, n + 1), a, "primary", n, n + 1, "active", next_due=now, beat=now, grown=True,
                           mult=usefulness_mult(self._useful.get(a.name)))
                last = self._primaries(a.name)[-1]
                self.slots.insert(self.slots.index(last) + 1, new)
                self._rename_shards(a, now)
                return [self._event("growth", "info", f"Split {a.name} into {n + 1} shards for the {team} team",
                                    f"The {team} team had a gap ({why}); each shard now watches fewer symbols.", now)]
        last = self._shortfall.get(team)
        if last is None or now - last >= SHORTFALL_EVERY_S:
            self._shortfall[team] = now
            reason = (f"the worker limit ({self.max_workers}) is reached" if len(live) >= self.max_workers
                      else "there are no reserve sources or shards left to add")
            return [self._event("diversity", "warn", f"Diversity shortfall in the {team} team",
                                f"The {team} team has a gap ({why}) and {reason}. Consider adding sources for it.", now)]
        return []

    def _shrink(self, team, now):
        """Stop grown workers that are rarely useful once the team has had no gap for 48 h. Lock held."""
        if now - self._last_gap.get(team, self._t0 if self._t0 is not None else now) < SHRINK_AFTER_S:
            return []
        events, done = [], set()
        for s in list(self.slots):
            if s.team != team or not s.grown or s.state == "stopped" or s.name in done:
                continue
            u = self._useful.get(s.name, 0.5)
            if u >= SHRINK_USEFULNESS:
                continue
            done.add(s.name)
            if s.role == "primary":      # an extra shard: fold the last shard back in
                mine = sorted(self._primaries(s.name), key=lambda x: x.shard)
                last = mine[-1]
                if len(mine) <= 1 or not last.grown:
                    continue
                last.state = "stopped"
                self.slots.remove(last)
                self._rename_shards(s.adapter, now)
                title = f"Folded {s.name} back to {len(mine) - 1} shard{'s' if len(mine) > 2 else ''}"
            else:
                s.state = "stopped"
                self._sync(s, now, "stopped: rarely useful")
                title = f"Stopped reserve source {s.name}"
            events.append(self._event("shrink", "info", title, f"The {team} team has had no gap for "
                                      f"{_dur(SHRINK_AFTER_S)} and {s.name} was rarely useful (usefulness {u:.2f}).", now))
        return events

    def gap_check(self, now=None, coverage=None):
        """coverage: {team: dict(symbols_covered, symbols_total, ...)} from aggregate.team_coverage. Returns events."""
        now = self._now(now)
        coverage = coverage if isinstance(coverage, dict) else {}
        self.refresh_usefulness()
        with self.lock:
            warm = self._t0 is None or now - self._t0 >= self.warmup_s
            have = {getattr(a, "team", None) for a in self.adapters}
            have |= {t for t, c in coverage.items() if isinstance(c, dict) and _num(c.get("symbols_total")) > 0}
            teams = [t for t in TEAMS if t in have] + sorted(t for t in have if t not in TEAMS and isinstance(t, str))
            events = []
            for team in teams:
                c = coverage.get(team) if isinstance(coverage.get(team), dict) else {}
                total, covered = _num(c.get("symbols_total")), _num(c.get("symbols_covered"))
                active = sum(1 for s in self.slots if s.team == team and s.state == "active")
                if active == 0:
                    why = "no working sources"
                elif warm and total > 0 and covered < GAP_SHARE * total:
                    why = f"fresh notes on {covered:.0f} of {total:.0f} symbols in 24 h"
                else:
                    events += self._shrink(team, now)
                    continue
                self._last_gap[team] = now
                events += self._grow(team, now, why)
        return events

    # ------------------------------------------------------------------ the service loop
    def _job(self, s):
        self.run_slot(s, self.clock())

    def run_forever(self, stop_event=None, threads=8, coverage_fn=None, gap_every_s=900.0):
        """Every 5 s submit due slots to a thread pool (never the same slot twice at once); supervise every
        60 s; if coverage_fn(now) is given, gap-check every gap_every_s. Returns when stop_event is set."""
        if stop_event is not None:
            self._stop = stop_event
        stop = self._stop
        if not self.slots:
            self.plan()
        ex = ThreadPoolExecutor(max_workers=max(1, int(threads)), thread_name_prefix="research-slot")
        last_sup, last_gap = None, self.clock()
        try:
            while not stop.is_set():
                now = self.clock()
                try:
                    if last_sup is None or now - last_sup >= SUPERVISE_EVERY_S:
                        last_sup = now
                        self.supervise(now)
                    if coverage_fn is not None and now - last_gap >= gap_every_s:
                        last_gap = now
                        self.gap_check(now, coverage_fn(now))
                    for s in self.due(now):
                        with self.lock:
                            if s.running:
                                continue
                            s.running = True
                        try:
                            ex.submit(self._job, s)
                        except RuntimeError:   # executor shutting down
                            s.running = False
                            break
                except Exception as e:
                    self.loop_error = f"{type(e).__name__}: {e}"[:300]
                stop.wait(TICK_S)
        finally:
            ex.shutdown(wait=False, cancel_futures=True)

    def stop(self):
        self._stop.set()

    def health(self):
        with self.lock:
            slots = [s.d() for s in self.slots]
            live = self._live()
            counts = dict(active=sum(s.state == "active" for s in live), standby=sum(s.state == "standby" for s in live),
                          failed=sum(s.state == "failed" for s in live), grown=sum(bool(s.grown) for s in live),
                          total=len(live))
            teams = {}
            for s in live:
                t = teams.setdefault(s.team, dict(active=0, standby=0, failed=0, last_ok=None))
                if s.state in t:
                    t[s.state] += 1
                if s.last_ok is not None and (t["last_ok"] is None or s.last_ok > t["last_ok"]):
                    t["last_ok"] = s.last_ok
        return dict(slots=slots, counts=counts, teams=teams)
