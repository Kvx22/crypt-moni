"""Switching research on, one piece at a time, only when the evidence says it helps.

Every piece of research the boards could act on starts in SHADOW: the engines compute what
they would have done with it and record that next to what they actually did, and the
outcome logger later fills in what prices did. Once an hour this module reads that record
and, for each piece, asks one plain question:

  penalties  did the names research would have marked down really lag their benchmark a day later?
  signals    does adding research to the advisors' scores rank coins better 4 hours ahead?
  watchlist  did the names the advisors proposed watching beat their benchmark over 7 days?
  cadence    did the 2-hour cadence paper book beat the main book without a worse drawdown?

The pieces form a ladder and climb it one at a time, in that order, at least three days
apart. A switched-on piece whose recent record turns clearly bad goes back to shadow (a
human hears about it). A piece a human switched OFF stays off; the gates never touch it.
States: "shadow" (recorded, not used), "active" (used), "off" (a human's choice).
"""
from __future__ import annotations

import copy
import math

import numpy as np

from .db import STAGE_ORDER
from .rdesk import spearman

STAGES = STAGE_ORDER
STATES = ("shadow", "active", "off")
LABEL = {"penalties": "Research penalties", "signals": "Research signals",
         "watchlist": "Watch-list proposals", "cadence": "The 2-hour cadence"}
DAY = 86400.0

DEFAULT_GATES = {
    "penalties": dict(min_days=7, min_events=30, max_mean_excess=0.0, min_hit=0.55, horizon="1d",
                      demote_mean_excess=0.005, demote_min_events=30),
    "signals": dict(min_days=7, min_obs=300, min_ic_gain=0.01, horizon="4h",
                    demote_ic_gain=-0.02, demote_min_obs=300),
    "watchlist": dict(min_days=10, min_proposals=10, min_mean_excess=0.0,
                      demote_mean_excess=-0.03, demote_min_proposals=10),
    "cadence": dict(min_days=14, min_return_gain=0.005, max_dd_worse=0.02, demote_return_gain=-0.01),
    "common": dict(min_gap_days=3, auto_promote=True, demote_window_days=14),
}


def gates_from(cfg):
    """DEFAULT_GATES with overrides from cfg["research"]["gates"] (or a bare gates dict)."""
    g = copy.deepcopy(DEFAULT_GATES)
    cfg = cfg or {}
    over = (cfg.get("research") or {}).get("gates") if "research" in cfg else cfg.get("gates", cfg)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and k in g:
            g[k].update(v)
        elif k in g["common"]:
            g["common"][k] = v
    return g


def _f(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _days(ts, now):
    return round((now - min(ts)) / DAY, 2) if ts else 0.0


# ---- evidence, one function per stage: (db, now, since, gate) -> dict -------------------------
def _penalties(db, now, since, g):
    rows = []
    for r in db.shadow(kind="penalty", since=since, matured=True):
        ex, base, adj = _f(r["excess"]), _f(r["base"]), _f(r["adj"])
        if ex is None or (r["detail"] or {}).get("horizon", "1d") != g["horizon"]:
            continue
        if base is not None and adj is not None and adj >= base:
            continue    # recorded, but research would not have marked it down
        rows.append((r["ts"], ex))
    ex = np.array([e for _, e in rows])
    return dict(n=len(rows), days=_days([t for t, _ in rows], now),
                mean_excess=float(ex.mean()) if len(ex) else None, hit=float((ex < 0).mean()) if len(ex) else None)


def _signals(db, now, since, g):
    hours = {}
    for r in db.shadow(kind="score", since=since, matured=True):
        ex, base, adj = _f(r["excess"]), _f(r["base"]), _f(r["adj"])
        if None in (ex, base, adj) or (r["detail"] or {}).get("horizon", "1d") != g["horizon"]:
            continue
        hours.setdefault((r["board"], int(r["ts"] // 3600)), []).append((r["ts"], base, adj, ex))
    gains, ib, ia, ts, n = [], [], [], [], 0
    for rows in hours.values():
        if len(rows) < 3:
            continue    # a rank correlation needs at least three names
        _, b, a, e = (np.array(c) for c in zip(*rows))
        cb, ca = spearman(b, e), spearman(a, e)
        ib.append(cb); ia.append(ca); gains.append(ca - cb)
        ts += [t for t, *_ in rows]; n += len(rows)
    m = lambda x: float(np.mean(x)) if x else None
    return dict(n=n, hours=len(gains), days=_days(ts, now), ic_base=m(ib), ic_adj=m(ia), ic_gain=m(gains))


def _watchlist(db, now, since, g):
    rows = db.q("SELECT ts, excess_7d FROM proposals WHERE excess_7d IS NOT NULL AND ts>=?", (since or 0.0,))
    rows = [(r["ts"], _f(r["excess_7d"])) for r in rows if _f(r["excess_7d"]) is not None and r["ts"] is not None]
    ex = np.array([e for _, e in rows])
    return dict(n=len(rows), days=_days([t for t, _ in rows], now), mean_excess=float(ex.mean()) if len(ex) else None)


def _cadence(db, now, since, g):
    boards = {}
    for r in db.q("SELECT key FROM kv WHERE key LIKE 'cadence:%'"):
        s = db.kv_get(r["key"], {})
        if not isinstance(s, dict):
            continue
        vals = {k: _f(s.get(k)) for k in ("days", "base_ret", "cad_ret", "base_dd", "cad_dd")}
        if None in vals.values():
            continue
        # drawdowns may be stored as -0.08 or 0.08: compare magnitudes
        boards[r["key"].split(":", 1)[1]] = dict(days=vals["days"], return_gain=vals["cad_ret"] - vals["base_ret"],
                                                 dd_worse=abs(vals["cad_dd"]) - abs(vals["base_dd"]))
    bs = list(boards.values())
    return dict(boards=boards, n=len(bs), days=min((b["days"] for b in bs), default=0.0),
                return_gain=min((b["return_gain"] for b in bs), default=None),
                dd_worse=max((b["dd_worse"] for b in bs), default=None))


EVIDENCE = {"penalties": _penalties, "signals": _signals, "watchlist": _watchlist, "cadence": _cadence}


def _pct(x):
    return "n/a" if x is None else f"{x:+.2%}"


def _verdict(stage, e, g):
    """(passes, plain-English reason) for promotion."""
    if stage == "penalties":
        ok = e["n"] >= g["min_events"] and e["days"] >= g["min_days"] and e["mean_excess"] <= g["max_mean_excess"] \
            and e["hit"] >= g["min_hit"] if e["n"] else False
        why = (f"{e['n']} penalised names over {e['days']:.1f} days did {_pct(e['mean_excess'])} against their benchmark "
               f"a day later on average; {0 if e['hit'] is None else e['hit']:.0%} lagged it "
               f"(needs {g['min_events']}+ over {g['min_days']}+ days, mean {_pct(g['max_mean_excess'])} or worse, "
               f"{g['min_hit']:.0%}+ lagging)")
    elif stage == "signals":
        ok = e["hours"] > 0 and e["n"] >= g["min_obs"] and e["days"] >= g["min_days"] and e["ic_gain"] >= g["min_ic_gain"]
        why = (f"adding research changed the 4-hour rank correlation by {e['ic_gain'] or 0:+.3f} "
               f"({e['n']} scores in {e['hours']} hours over {e['days']:.1f} days; needs {g['min_obs']}+ scores over "
               f"{g['min_days']}+ days and {g['min_ic_gain']:+.3f} or better)")
    elif stage == "watchlist":
        ok = e["n"] >= g["min_proposals"] and e["days"] >= g["min_days"] and e["mean_excess"] >= g["min_mean_excess"] \
            if e["n"] else False
        why = (f"{e['n']} proposed names over {e['days']:.1f} days did {_pct(e['mean_excess'])} against their benchmark "
               f"over 7 days (needs {g['min_proposals']}+ over {g['min_days']}+ days, mean {_pct(g['min_mean_excess'])} or better)")
    else:
        ok = e["n"] > 0 and e["days"] >= g["min_days"] and e["return_gain"] >= g["min_return_gain"] \
            and e["dd_worse"] <= g["max_dd_worse"]
        why = (f"the 2-hour book returned {_pct(e['return_gain'])} versus the main book with a drawdown "
               f"{_pct(e['dd_worse'])} deeper over {e['days']:.0f} days, worst board ({e['n']} boards reporting; needs "
               f"{g['min_days']}+ days, {_pct(g['min_return_gain'])} or better, drawdown at most {_pct(g['max_dd_worse'])} deeper)")
    return bool(ok), why


def _bad(stage, e, g):
    """True when an active stage's recent record has turned clearly bad."""
    if stage == "penalties":
        return e["n"] >= g["demote_min_events"] and e["mean_excess"] > g["demote_mean_excess"]
    if stage == "signals":
        return e["hours"] > 0 and e["n"] >= g["demote_min_obs"] and e["ic_gain"] < g["demote_ic_gain"]
    if stage == "watchlist":
        return e["n"] >= g["demote_min_proposals"] and e["mean_excess"] < g["demote_mean_excess"]
    return e["n"] > 0 and e["return_gain"] < g["demote_return_gain"]


def _last_promotion(db, st):
    ts = [db.kv_get("stage_last_promotion", None)] + [s.get("since") for s in st.values() if s.get("state") == "active"]
    ts = [t for t in (_f(x) for x in ts) if t is not None]
    return max(ts) if ts else None


def evaluate(db, now, cfg=None):
    """Compute every stage's evidence (kept in kv "stage_evidence"), demote active stages whose
    recent record turned bad, and promote at most one shadow stage. Returns the changes made."""
    g = gates_from(cfg)
    gc = g["common"]
    st = db.stages()
    recent_since = now - gc["demote_window_days"] * DAY
    evidence, changes = {}, []
    for stage in STAGES:
        s, gs = st[stage], g[stage]
        state = s.get("state") if s.get("state") in STATES else "shadow"
        # after a demotion the gate starts counting afresh, so one good old stretch can't flip it straight back
        since = _f(s.get("since")) if state == "shadow" and s.get("by") == "gate" else None
        try:
            e = EVIDENCE[stage](db, now, since, gs)
            e["pass"], e["reason"] = _verdict(stage, e, gs)
            if state == "active":
                # watch-list excess needs 7 days to mature: look back that much further
                rs = recent_since - 7 * DAY if stage == "watchlist" else recent_since
                r = EVIDENCE[stage](db, now, rs, gs)
                e["recent"], e["bad"] = r, bool(_bad(stage, r, gs))
        except Exception as ex:     # a malformed row must never stop the hourly run
            e = {"pass": False, "bad": False, "reason": f"could not read the evidence ({type(ex).__name__}: {ex})"}
        e["state"] = state
        evidence[stage] = e

    for stage in STAGES:
        e = evidence[stage]
        if e["state"] == "active" and e.get("bad"):
            _, why = _verdict(stage, e["recent"], g[stage])
            db.set_stage(stage, "shadow", evidence=e, by="gate", now=now)
            db.event("stage", "warn", f"{LABEL[stage]} switched back to shadow",
                     f"Over the last {gc['demote_window_days']} days {why}. It keeps running in shadow and can be "
                     "switched on again once the record recovers.", ts=now)
            changes.append(dict(stage=stage, **{"from": "active", "to": "shadow"}, reason="recent evidence turned bad", evidence=e))
            e["state"] = "shadow"

    demoted = {c["stage"] for c in changes}
    cand = next((s for s in STAGES if evidence[s]["state"] == "shadow"), None)
    last = _last_promotion(db, db.stages())
    for stage in STAGES:
        e = evidence[stage]
        if e["state"] != "shadow":
            continue
        if stage != cand:
            e["waiting"] = f"waits for {LABEL[cand].lower()} to be switched on first"
        elif stage in demoted:
            e["waiting"] = "just switched back to shadow"
        elif st[stage].get("by") == "human":
            e["waiting"] = "held in shadow by hand (set it to auto to let the gate decide)"
        elif not gc["auto_promote"]:
            e["waiting"] = "automatic switching is turned off"
        elif last is not None and now - last < gc["min_gap_days"] * DAY:
            e["waiting"] = f"the last switch was {(now - last) / DAY:.1f} days ago (needs {gc['min_gap_days']})"
        elif e["pass"]:
            db.set_stage(stage, "active", evidence=e, by="gate", now=now)
            db.kv_set("stage_last_promotion", now)
            db.event("stage", "info", f"{LABEL[stage]} switched on",
                     f"The shadow record passed its gate: {e['reason']}.", ts=now)
            changes.append(dict(stage=stage, **{"from": "shadow", "to": "active"}, reason=e["reason"], evidence=e))
            e["state"] = "active"
    db.kv_set("stage_evidence", dict(ts=now, stages=evidence))
    return changes


def set_manual(db, stage, state, now=None):
    """A human's switch, for the CLI and dashboard: shadow | active | off | auto (back under the gate)."""
    if stage not in STAGES:
        return f"Unknown stage '{stage}'. The stages are: {', '.join(STAGES)}."
    if state not in STATES + ("auto",):
        return f"Unknown state '{state}'. Use shadow, active, off or auto."
    label = LABEL[stage]
    if state == "auto":
        db.set_stage(stage, "shadow", by="auto", now=now)
        msg = f"{label}: back under the gate. It stays in shadow until its evidence passes."
    else:
        db.set_stage(stage, state, by="human", now=now)
        msg = {"active": f"{label}: switched on by hand.",
               "off": f"{label}: switched off by hand. The gate will not touch it until you set it to auto.",
               "shadow": f"{label}: held in shadow by hand. Set it to auto to let the gate decide."}[state]
    db.event("stage", "info", msg.split(":")[0] + f" set to {state} by hand", msg, ts=now)
    return msg
