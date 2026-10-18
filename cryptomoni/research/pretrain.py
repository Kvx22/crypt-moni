"""Pretraining: replay history so the advisors start trained instead of starting cold.

A fresh pick desk knows only its instincts and needs weeks of graded hours before its
weights mean much. The board already has (or can fetch) months of hourly bars, so this
module walks them hour by hour and runs the desk's own learning loop on them, exactly as
the live engine would have: the same 400-bar window, the same scanner metrics, the same
signal matrix and regime tilt, the same pending queue graded `horizon` hours later with
fwd = log(close then / close at the call), the same rounding of what is stored. Only the
wall clock is skipped.

Everything that does not depend on the desk (which coins were aligned each hour, their
signal matrix, which calls get graded when, and the forward returns) is computed once as a
"schedule". Desks then run over that schedule cheaply, which lets one pass compare several
desks on the same hours:

  * the challenger: a fresh desk that learns from the start; its state at the 80% mark is
    kept, and it keeps learning through the last 20% (the holdout);
  * the prior-only desk: instincts, never learns;
  * the champion: a copy of the live desk, learning through the holdout like the challenger.

Every desk is scored prequentially on the holdout (each call is scored before the desk
learns from it): rank IC of the combined score against what happened, and the relative
return of its top picks. `decide()` turns that report into keep-or-adopt, and `apply()`
does it on the engine thread. `PretrainJob` runs the whole thing in a background thread
with its own database connection, so the 24/7 loop never waits for it.
"""
from __future__ import annotations

import bisect
import datetime as dt
import math
import threading
import time

import numpy as np

from ..advisors import PickDesk, spearman, zscore
from ..feeds import DAY, HOUR, FeedError
from ..scanner import MIN_BARS, metrics, regime_tilt, signal_matrix
from ..stocks import MIN_DAYS, daily_metrics

WINDOW = 400          # the engine reads the last 400 stored hourly bars per symbol
STOCK_WINDOW = 400    # daily_metrics looks back at most 252 days
CHUNK = 290           # hourly bars per candles request
UNIFORM = [0.2] * 5


def day_str(ts):
    return dt.datetime.fromtimestamp(ts, dt.timezone.utc).date().isoformat()


# ---------------------------------------------------------------------------------- data
def backfill_hourly(feed, store, symbol, hours, now):
    """Fetch up to `hours` of completed hourly bars before `now` into the store, oldest
    first, in 290-bar chunks. Only bars not already stored are written. A feed error stops
    this symbol early (what was fetched so far is kept). Returns the number of bars added."""
    hours = max(0, int(hours))
    t = int(now - hours * HOUR) // HOUR * HOUR
    have = {b.ts for b in store.get_bars(symbol, HOUR, hours + 500)}
    budget, added = hours + 48, 0
    while t + HOUR <= now and budget > 0:
        end = min(t + (CHUNK - 1) * HOUR, int(now))
        try:
            got = feed.candles(symbol, HOUR, t, end)
        except FeedError:
            break
        bars = sorted((b for b in got or () if t <= b.ts <= end and b.ts + HOUR <= now), key=lambda b: b.ts)[:budget]
        budget -= len(bars)
        new, seen = [], set()
        for b in bars:
            if b.ts not in have and b.ts not in seen and b.close > 0:
                new.append(b)
                seen.add(b.ts)
        if new:
            store.put_bars(symbol, HOUR, new)
            have |= seen
            added += len(new)
        t = max(end, bars[-1].ts if bars else end) + HOUR
    return added


def probs_from_hist(hist):
    """Brain history rows [date, regime, conf] -> {date: probs}: `conf` on the regime and the
    rest spread evenly over the other four (the closest we can get to the live probs)."""
    out = {}
    for row in hist or ():
        try:
            d, r, c = str(row[0]), int(row[1]), float(row[2])
        except (TypeError, ValueError, IndexError):
            continue
        if not 0 <= r < 5 or not math.isfinite(c):
            continue
        c = min(max(c, 0.2), 1.0)
        p = [(1 - c) / 4] * 5
        p[r] = c
        out[d] = p
    return out


class _Probs:
    """Lookup of the brain's probs on a day: that day's entry, else the latest earlier one."""

    def __init__(self, probs_by_day):
        self.d = dict(probs_by_day or {})
        self.keys = sorted(self.d)

    def __call__(self, day):
        i = bisect.bisect_right(self.keys, day) - 1
        return self.d[self.keys[i]] if i >= 0 else UNIFORM


# ---------------------------------------------------------------------------------- schedule
def _grade_queue(steps, horizon, slack, cap):
    """Replays the engine's pending queue on the step positions: which calls get graded at
    which step, against which symbols. Purely data-driven, so it is computed once."""
    pending = []
    for j, st in enumerate(steps):
        keep, grades = [], []
        for k in pending:
            p = steps[k]
            age = st["pos"] - p["pos"]
            if age < horizon:
                keep.append(k)
                continue
            if age > horizon + slack:
                continue  # too stale to grade fairly
            idx = [i for i, s in enumerate(p["syms"]) if s in st["closes"]]
            if len(idx) < 4:
                continue
            fwd = np.array([math.log(st["closes"][p["syms"][i]] / p["close"][i]) for i in idx])
            grades.append((k, np.array(idx), fwd))
        st["grades"] = grades
        keep.append(j)
        pending = keep[-(horizon + cap):]
    return steps


def crypto_schedule(bars_by_sym, proxy, horizon, probs_by_day=None, start_index=None, progress=None, stop=None):
    """Hour-by-hour steps exactly as Engine.desk_step sees them. The timeline is the proxy's
    bars (the engine steps when the proxy closes a bar); a step needs >= 4 symbols with a bar
    at that hour and >= MIN_BARS bars of history."""
    syms_all = [s for s, b in bars_by_sym.items() if b]
    arr, at = {}, {}
    for s in syms_all:
        bs = sorted({b.ts: b for b in bars_by_sym[s] if b.close > 0}.values(), key=lambda b: b.ts)
        arr[s], at[s] = bs, {b.ts: i for i, b in enumerate(bs)}
    if proxy in arr:
        timeline = [b.ts for b in arr[proxy]]
    else:
        timeline = sorted({b.ts for s in syms_all for b in arr[s]})
    btc_ts = [b.ts for b in arr.get(proxy, [])]
    probs = _Probs(probs_by_day)
    steps = []
    t0 = int(start_index or 0)
    n = len(timeline)
    for ti in range(t0, n):
        if stop and stop():
            break
        if progress and ti % 24 == 0:
            progress(ti - t0, n - t0)
        ts = timeline[ti]
        view = {}
        for s in syms_all:
            i = at[s].get(ts)
            if i is not None and i + 1 >= MIN_BARS:
                view[s] = arr[s][max(0, i + 1 - WINDOW): i + 1]
        if len(view) < 4:
            continue
        btc = None
        if proxy in arr:
            j = bisect.bisect_right(btc_ts, ts)
            btc = arr[proxy][max(0, j - WINDOW): j] if j else None
        mets = {}
        for s, v in view.items():
            try:
                m = metrics(v, btc)
            except (ValueError, ZeroDivisionError, FloatingPointError, OverflowError):
                m = None
            if m is not None:
                mets[s] = m
        syms = [s for s in syms_all if s in mets]
        if len(syms) < 4:
            continue
        X = signal_matrix(mets, syms, regime_tilt(probs(day_str(ts + HOUR - DAY))))
        if not np.all(np.isfinite(X)):
            continue
        closes = {s: view[s][-1].close for s in syms}
        steps.append(dict(pos=ts // HOUR, ts=ts, syms=syms, close=[closes[s] for s in syms], closes=closes,
                          X=X, Xr=X.round(5)))
    return _grade_queue(steps, int(horizon), 6, 4)


def stock_schedule(closes_by_sym, volumes_by_sym, proxy, horizon, probs_by_day=None, progress=None, stop=None):
    """Day-by-day steps exactly as StockBook.step sees them: the proxy's trading days, symbols
    whose last completed day is that day with >= MIN_DAYS of history, daily_metrics over the
    series to date, grading by trading-day count."""
    series = {s: sorted((d, float(c)) for d, c in (closes_by_sym.get(s) or {}).items() if c and c > 0)
              for s in closes_by_sym}
    series = {s: v for s, v in series.items() if v}
    if proxy not in series:
        return []
    spy = dict(series[proxy])
    dates = {s: [d for d, _ in v] for s, v in series.items()}
    vals = {s: [c for _, c in v] for s, v in series.items()}
    at = {s: {d: i for i, d in enumerate(dates[s])} for s in series}
    probs = _Probs(probs_by_day)
    pdates = dates[proxy]
    steps = []
    for pi, day in enumerate(pdates):
        if stop and stop():
            break
        if progress and pi % 20 == 0:
            progress(pi, len(pdates))
        mets = {}
        for s in series:
            i = at[s].get(day)
            if i is None or i + 1 < MIN_DAYS:
                continue
            lo = max(0, i + 1 - STOCK_WINDOW)
            d, c = dates[s][lo:i + 1], vals[s][lo:i + 1]
            vol = [(volumes_by_sym.get(s) or {}).get(x, 0.0) or 0.0 for x in d]
            try:
                m = daily_metrics(d, c, spy, vol)
            except (ValueError, ZeroDivisionError, FloatingPointError, OverflowError):
                m = None
            if m:
                mets[s] = m
        syms = [s for s in series if s in mets]
        if len(syms) < 4:
            continue
        X = signal_matrix(mets, syms, regime_tilt(probs(day)))
        if not np.all(np.isfinite(X)):
            continue
        closes = {s: vals[s][at[s][day]] for s in syms}
        steps.append(dict(pos=pi, day=day, syms=syms, close=[closes[s] for s in syms], closes=closes, X=X, Xr=X.round(5)))
    return _grade_queue(steps, int(horizon), 5, 3)


# ---------------------------------------------------------------------------------- desks over a schedule
class _Runner:
    """One desk walking the schedule: predicts at each step, is scored on each grade before
    it learns from it (prequential), learns unless frozen."""

    def __init__(self, desk, learn=True):
        self.desk, self.learning = desk, learn
        self.calls = {}           # step index -> (S rounded as stored, combined score)
        self.ic, self.top = [], []

    def predict(self, j, st):
        S = self.desk.predict(st["X"])
        Z = np.array([zscore(s) for s in S])
        self.calls[j] = (S.round(6), self.desk.w @ Z)

    def grade(self, k, idx, fwd, steps):
        call = self.calls.pop(k, None)   # a call is graded at most once
        if call is None:
            return
        S, comb = call
        c = comb[idx]
        rel = fwd - fwd.mean()
        n = min(self.desk.top_k, max(1, len(fwd) // 3))
        self.ic.append(spearman(c, fwd))
        self.top.append(float(rel[np.argsort(-c, kind="stable")[:n]].mean()))
        if self.learning:
            self.desk.learn(steps[k]["Xr"][idx], S[:, idx], fwd)

    def forget(self, before):
        """Calls the queue dropped without grading (stale, too few symbols) are never needed."""
        for k in [k for k in self.calls if k < before]:
            del self.calls[k]


def _mean(x):
    return float(np.mean(x)) if len(x) else 0.0


def _check(desk):
    """PickDesk.load broadcasts a mis-sized state without complaint: refuse it here."""
    d, n = desk.prior.shape[1], desk.prior.shape[0]
    shapes = dict(Sxx=(d, d), Sxy=(d,), W=(n, d), w=(n,), ic=(n,), hit=(n,), excess=(n,), score=(n,))
    for k, shape in shapes.items():
        x = np.asarray(getattr(desk, k), float)
        if x.shape != shape or not np.all(np.isfinite(x)):
            raise ValueError(f"desk state {k} has shape {x.shape}, expected {shape}")
    return desk


def _desk(kwargs, state=None):
    d = PickDesk(**(kwargs or {}))
    if state:
        d.load(state)
        _check(d)
    return d


def run_schedule(steps, desk_kwargs, champion_state=None, split=None, progress=None, stop=None):
    """Walk the schedule with a fresh desk. With `split` (e.g. 0.8): the graded rounds after
    that share form the holdout, where the challenger, a prior-only desk and (when a
    champion state is given) the champion are all scored prequentially on the same calls.
    Returns dict(desk, rounds, hours, ic, challenger_state, holdout, cancelled)."""
    main = _Runner(_desk(desk_kwargs))
    n_grades = sum(len(st["grades"]) for st in steps)
    lag = max((j - k for j, st in enumerate(steps) for k, _, _ in st["grades"]), default=0)
    cut = int(n_grades * split) if split else 0
    others, challenger_state, g, h0, cancelled = {}, None, 0, None, False
    for j, st in enumerate(steps):
        if stop and stop():
            cancelled = True
            break
        if progress and j % 24 == 0:
            progress(j, len(steps))
        for k, idx, fwd in st["grades"]:
            for r in others.values():
                r.grade(k, idx, fwd, steps)
            main.grade(k, idx, fwd, steps)
            g += 1
            if g == cut:   # the 80% mark: keep the challenger, start the comparison desks
                challenger_state, h0 = main.desk.state(), len(main.ic)
                others["prior"] = _Runner(_desk(desk_kwargs), learn=False)
                if champion_state:
                    try:
                        others["champion"] = _Runner(_desk(desk_kwargs, champion_state))
                    except (KeyError, ValueError, TypeError, np.linalg.LinAlgError):
                        pass  # an unreadable live state: compare against instincts only
                for r in others.values():  # calls still waiting to be graded, made with each desk's own weights
                    for kk in sorted(main.calls):
                        r.predict(kk, steps[kk])
        for r in (main, *others.values()):
            r.predict(j, st)
            r.forget(j - lag)
    out = dict(desk=main.desk, rounds=main.desk.rounds, hours=len(steps), ic=main.ic, top=main.top,
               challenger_state=challenger_state, cancelled=cancelled, holdout=None)
    if h0 is not None:
        block = lambda ic, top: dict(ic=_mean(ic), top=_mean(top), n=len(ic))  # noqa: E731
        hold = dict(challenger=block(main.ic[h0:], main.top[h0:]))
        for name, r in others.items():
            hold[name] = block(r.ic, r.top)
        hold["champion"] = hold.get("champion")
        out["holdout"] = hold
    return out


# ---------------------------------------------------------------------------------- public replays
def replay_crypto(bars_by_sym, proxy, desk_kwargs, horizon, probs_by_day=None, start_index=None, progress=None, stop=None):
    """Replay the hourly desk over history with a fresh desk.
    Returns dict(desk=PickDesk, rounds, hours, ic=[spearman(combined, fwd) per graded hour])."""
    steps = crypto_schedule(bars_by_sym, proxy, horizon, probs_by_day, start_index, progress, stop)
    r = run_schedule(steps, desk_kwargs, stop=stop)
    return dict(desk=r["desk"], rounds=r["rounds"], hours=r["hours"], ic=r["ic"])


def replay_stocks(closes_by_sym, volumes_by_sym, proxy, desk_kwargs, horizon, probs_by_day=None, progress=None, stop=None):
    """The same for the stock desk on daily bars (horizon in trading days)."""
    steps = stock_schedule(closes_by_sym, volumes_by_sym, proxy, horizon, probs_by_day, progress, stop)
    r = run_schedule(steps, desk_kwargs, stop=stop)
    return dict(desk=r["desk"], rounds=r["rounds"], hours=r["hours"], ic=r["ic"])


def holdout_compare(steps, desk_kwargs, champion_state=None, split=0.8, progress=None, stop=None):
    """One replay from a fresh desk; the last 20% of graded rounds is the holdout.
    Returns dict(report, challenger_state, final_state) or None if cancelled or too little
    history. report: rounds, hours, holdout_rounds, ic_all, and per desk
    {"challenger"|"prior"|"champion": {ic, top, n}} (champion None without a state)."""
    r = run_schedule(steps, desk_kwargs, champion_state, split, progress, stop)
    if r["cancelled"]:
        return None
    hold = r["holdout"]
    report = dict(rounds=int(r["rounds"]), hours=int(r["hours"]), ic_all=_mean(r["ic"]),
                  holdout_rounds=int(hold["challenger"]["n"]) if hold else 0,
                  challenger=hold["challenger"] if hold else None, prior=hold["prior"] if hold else None,
                  champion=hold["champion"] if hold else None)
    if steps:
        first, last = steps[0], steps[-1]
        report["from"], report["to"] = first.get("ts", first.get("day")), last.get("ts", last.get("day"))
    return dict(report=report, challenger_state=r["challenger_state"], final_state=r["desk"].state())


# ---------------------------------------------------------------------------------- the decision
MIN_HOLDOUT = 10      # graded rounds in the holdout before any verdict counts


def decide(result, live_rounds, min_rounds_fresh=48):
    """-> (adopt, "final"|"none", reason). A fresh live desk adopts the replayed desk when
    the challenger beat instincts on the holdout by >= 0.005 mean IC; a mature one only
    when it beat both instincts and the live desk by >= 0.01."""
    rep = (result or {}).get("report") or {}
    ch, pr, cp = rep.get("challenger"), rep.get("prior"), rep.get("champion")
    if not (result or {}).get("final_state") or not ch or not pr:
        return False, "none", "not enough history to compare desks"
    if ch["n"] < MIN_HOLDOUT:
        return False, "none", f"only {ch['n']} graded rounds in the holdout (need {MIN_HOLDOUT})"
    gp = ch["ic"] - pr["ic"]
    if live_rounds < min_rounds_fresh:
        if gp >= 0.005:
            return True, "final", f"the replayed desk beat instincts by {gp:+.3f} IC on the holdout"
        return False, "none", f"the replayed desk did not beat instincts on the holdout ({gp:+.3f} IC): kept the prior"
    if not cp:
        return False, "none", "no live desk to compare against: kept the live desk"
    gc = ch["ic"] - cp["ic"]
    if gp >= 0.01 and gc >= 0.01:
        return True, "final", f"the replayed desk beat instincts by {gp:+.3f} and the live desk by {gc:+.3f} IC"
    return False, "none", f"the replayed desk did not clearly beat the live desk ({gc:+.3f} IC vs live, {gp:+.3f} vs instincts): kept the live desk"


def _body(section, label, adopt, reason):
    rep = section["report"]
    parts = [f"{label}: replayed {rep['hours']:,} steps, {rep['rounds']:,} graded rounds."]
    for k, name in (("challenger", "replayed desk"), ("prior", "instincts"), ("champion", "live desk")):
        x = rep.get(k)
        if x:
            parts.append(f"Holdout {name}: IC {x['ic']:+.3f}, top picks {x['top']:+.2%} vs the average.")
    parts.append(("Adopted the replayed desk: " if adopt else "No change: ") + reason + ".")
    return " ".join(parts)


def apply(engine, result, now):
    """Engine thread only. Apply the decision for each board in `result`, log one event,
    store kv "pretrain" for the dashboard. Returns {"crypto": (...), "stocks": (...)}."""
    fresh = int((result or {}).get("min_rounds_fresh", 48))
    if not result or result.get("error") or result.get("cancelled"):
        why = (result or {}).get("error") or ("cancelled" if (result or {}).get("cancelled") else "no result")
        engine._ev(now, "learn", "warn", "Advisor pretraining did not finish", str(why)[:300])
        engine.s.kv_set("pretrain", dict(ts=now, error=str(why)[:300]))
        return {}
    out, bodies, store = {}, [], dict(ts=now)
    boards = [("crypto", "desk", engine.desk, "Crypto")]
    if getattr(engine, "stocks", None) is not None:
        boards.append(("stocks", "stock_desk", engine.stocks.desk, "Stocks"))
    for name, key, desk, label in boards:
        sec = result.get(name)
        if not sec:
            continue
        adopt, which, reason = decide(sec, desk.rounds, fresh)
        if adopt:
            old = desk.state()
            try:
                _check(_desk(None, sec["final_state"]))   # validate on a scratch desk first
                desk.load(sec["final_state"])
                _check(desk)
                engine.s.kv_set(key, desk.state(), commit=False)
            except (KeyError, ValueError, TypeError, np.linalg.LinAlgError) as ex:
                desk.load(old)   # a half-loaded desk must never be left behind
                adopt, which, reason = False, "none", f"the replayed state would not load ({ex!r:.120})"
        out[name] = (adopt, which, reason)
        bodies.append(_body(sec, label, adopt, reason))
        store[name] = dict(sec["report"], adopted=adopt, which=which, reason=reason)
    hours = (result.get("crypto") or {}).get("report", {}).get("hours", 0)
    engine.s.kv_set("pretrain", store)
    engine._ev(now, "learn", "info", f"Advisors pretrained on {hours:,} hours of history",
               " ".join(bodies) or "Nothing to replay yet.", store)
    return out


# ---------------------------------------------------------------------------------- the background job
class PretrainJob:
    """Backfill + replay + compare in a daemon thread. `start()` and `apply()` run on the
    engine thread; the thread itself touches only its own Store connection and copies."""

    def __init__(self, engine, hours=2160, min_rounds_fresh=48):
        self.engine, self.hours, self.min_rounds_fresh = engine, int(hours), int(min_rounds_fresh)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._result = None
        self.status = dict(phase="idle", i=0, n=0)

    def running(self):
        return self._thread is not None and self._thread.is_alive()

    def start(self, now):
        """Snapshot what the replay needs (on the engine thread) and start. False if running."""
        if self.running():
            return False
        e = self.engine
        snap = dict(path=e.s.path, feed=e.feed, universe=list(e.universe), proxy=e.proxy,
                    desk_kwargs=dict(e.cfg.get("desk", {})), horizon=int(e.horizon), champion=e.desk.state(),
                    probs=probs_from_hist((e.brain or {}).get("hist")), stocks=None)
        st = getattr(e, "stocks", None)
        if st is not None:
            dk = dict((e.cfg.get("stocks") or {}).get("desk", {}))
            dk.setdefault("decay", 0.98)
            dk.setdefault("ridge", 300.0)
            snap["stocks"] = dict(closes={s: dict(st.closes.get(s, {})) for s in st.symbols},
                                  volumes={s: dict(st.volumes.get(s, {})) for s in st.symbols},
                                  proxy=st.proxy, horizon=int(st.horizon), desk_kwargs=dk, champion=st.desk.state(),
                                  probs=probs_from_hist((st.brain or {}).get("hist")))
        with self._lock:
            self._result = None
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, args=(snap, now), name="pretrain", daemon=True)
        self._thread.start()
        return True

    def cancel(self):
        self._stop.set()

    def poll(self):
        """The finished result (once), or None while running / before start."""
        with self._lock:
            r, self._result = self._result, None
        return r

    def _set(self, **kw):
        with self._lock:
            self.status = dict(self.status, **kw)

    def _run(self, snap, now):
        t0 = time.time()
        result = dict(started=now, min_rounds_fresh=self.min_rounds_fresh)
        store = None
        try:
            from ..store import Store
            store = Store(snap["path"])
            stop = self._stop.is_set
            backfilled, errors = 0, []
            for i, sym in enumerate(snap["universe"]):
                if stop():
                    break
                self._set(phase="backfill", i=i, n=len(snap["universe"]))
                try:
                    backfilled += backfill_hourly(snap["feed"], store, sym, self.hours, now)
                except Exception as ex:  # noqa: BLE001  (one bad symbol never stops the job)
                    errors.append(f"{sym}: {ex!r:.120}")
            bars = {}
            for sym in snap["universe"]:
                try:
                    b = store.get_bars(sym, HOUR, self.hours + WINDOW)
                except Exception as ex:  # noqa: BLE001
                    errors.append(f"{sym} read: {ex!r:.120}")
                    continue
                if b:
                    bars[sym] = b
            prog = lambda ph: (lambda i, n: self._set(phase=ph, i=i, n=n))  # noqa: E731
            steps = crypto_schedule(bars, snap["proxy"], snap["horizon"], snap["probs"], progress=prog("scan"), stop=stop)
            crypto = holdout_compare(steps, snap["desk_kwargs"], snap["champion"], progress=prog("replay"), stop=stop)
            if crypto is None or stop():
                result["cancelled"] = True
            else:
                crypto["report"].update(backfilled=backfilled, symbols=len(bars), errors=errors[:5])
                result["crypto"] = crypto
                sk = snap["stocks"]
                if sk:
                    ss = stock_schedule(sk["closes"], sk["volumes"], sk["proxy"], sk["horizon"], sk["probs"],
                                        progress=prog("stocks"), stop=stop)
                    stocks = holdout_compare(ss, sk["desk_kwargs"], sk["champion"], stop=stop)
                    if stocks is None:
                        result["cancelled"] = True
                    else:
                        result["stocks"] = stocks
            result["seconds"] = round(time.time() - t0, 1)
        except Exception as ex:  # noqa: BLE001  (the job reports failure; it never takes the engine down)
            result["error"] = repr(ex)[:300]
        finally:
            if store is not None:
                try:
                    store.close()
                except Exception:  # noqa: BLE001
                    pass
        # backwards-compatible top-level keys for the crypto board
        c = result.get("crypto") or {}
        result.update(report=c.get("report"), challenger_state=c.get("challenger_state"), final_state=c.get("final_state"))
        with self._lock:
            self._result = result
            self.status = dict(self.status, phase="done" if not result.get("error") else "failed")
