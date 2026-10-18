"""The per-symbol research view: what the fresh notes add up to.

For each subscribed symbol the fresh notes are weighed (confidence x source reliability x
age decay), syndicated copies are folded into one STORY (the same wire piece on five sites is
one piece of evidence, not five), and then:
  * independent sources: distinct non-neutral stories from distinct publishers. Stories from
    the forever-search sandbox (quarantine) count only once ordinary evidence agrees with them.
  * score and features: tanh of weighted direction sums. Good news is only heard at full
    volume when it is corroborated (>= 4 independent sources across >= 2 evidence kinds);
    thin good news is scaled to a quarter. Bad news is never scaled down.
  * penalty: credible risk flags (official, a reliable source, or two independent stories
    naming the same flag) press on the score up to a cap; it is never a hard veto. Unconfirmed
    flags add at most a quarter of their severity.
  * halted: a fresh official halt that no newer official "resumed" note has cleared.
team_coverage() says how much of the watch list each team covered recently (gap detector).
"""
from __future__ import annotations

import math

from . import text
from .taxonomy import RISK_FLAGS, RESEARCH_SIGNALS, feature_of, severity
from .types import TEAMS

DEFAULTS = dict(boost_min_sources=4, boost_min_kinds=2, penalty_cap=0.35, penalty_half_life_h=24, thin_sources=2,
                quarantine_needs=1)
HALF_LIFE_H = dict(news=12, filing=48, market=3, sentiment=6, discovery=24, search=12, official=48)
SCALE = 1.5              # tanh scale of the weighted direction sums
THIN_SCALE = 0.25        # uncorroborated good news is heard, faintly
CREDIBLE_REL = 0.6       # a source this reliable makes its flags credible on its own
UNCONFIRMED_SHARE = 0.25
HALT_FRESH_S = 2 * 3600
TOP_N, FLAGS_N = 6, 5


def _decay(age_s, half_life_h):
    return 0.5 ** (max(0.0, age_s) / 3600.0 / max(1e-6, half_life_h))


def _reliability(db):
    out = {}
    try:
        for name, r in (db.sources() or {}).items():
            a, b = float(r.get("rel_a") or 0), float(r.get("rel_b") or 0)
            out[name] = a / (a + b) if a > 0 and b > 0 else 0.5
    except Exception:  # noqa: BLE001  (no sources table yet: everyone is average)
        pass
    return out


def _stories(notes, weight):
    """Fold syndicated copies (fingerprints within 8 bits) into stories, strongest note first."""
    stories = []
    for n in sorted(notes, key=lambda n: -weight[id(n)]):
        for s in stories:
            if any(text.same_story(n.cluster, c) for c in s["clusters"]):
                s["notes"].append(n)
                s["clusters"].add(n.cluster)
                s["publishers"].add(n.publisher or n.source)
                s["quarantine"] = s["quarantine"] and n.quarantine
                s["official"] = s["official"] or n.official
                break
        else:
            stories.append(dict(best=n, w=weight[id(n)], notes=[n], clusters={n.cluster}, publishers={n.publisher or n.source},
                                quarantine=n.quarantine, official=n.official))
    return stories


def _distinct(publisher_sets):
    """How many of these stories bring at least one publisher not already counted."""
    used, n = set(), 0
    for p in publisher_sets:
        if not p <= used:
            n += 1
        used |= p
    return n


def _independent(stories):
    return _distinct(s["publishers"] for s in stories if s["best"].direction)


def _halted(notes, now):
    halts = [n for n in notes if n.official and "halt" in n.risk_flags and now - n.ts < HALT_FRESH_S]
    if not halts:
        return False
    last = max(n.ts for n in halts)
    return not any(n.official and n.ts > last and n.direction > 0 and n.catalyst == "outage" for n in notes)


def _empty_view(sym, asset, now):
    return dict(symbol=sym, asset=asset, ts=now, n_notes=0, n_stories=0, n_independent=0, kinds=[], score=0.0, boost_ok=False,
                penalty=0.0, halted=False, thin=True, flags=[], features={k: 0.0 for k in RESEARCH_SIGNALS}, top=[],
                last_note_ts=None)


def _view(sym, asset, notes, now, cfg, rel):
    weight = {}
    for n in notes:
        hl = HALF_LIFE_H.get(n.evidence, 12)
        weight[id(n)] = max(0.0, n.confidence) * rel.get(n.source, 0.5) * _decay(now - n.ts, hl)
    stories = _stories(notes, weight)

    # quarantine stories count only when enough ordinary stories point the same way
    agree = {1: 0, -1: 0}
    for s in stories:
        if not s["quarantine"] and s["best"].direction:
            agree[s["best"].direction] += 1
    counted = [s for s in stories if not s["quarantine"] or (s["best"].direction and
                                                             agree[s["best"].direction] >= cfg["quarantine_needs"])]
    n_ind = _independent(counted)
    kinds = sorted({s["best"].evidence for s in counted if s["best"].direction and not s["quarantine"]})
    boost_ok = n_ind >= cfg["boost_min_sources"] and len(kinds) >= cfg["boost_min_kinds"]

    sums, total = {k: 0.0 for k in RESEARCH_SIGNALS}, 0.0
    for s in counted:
        b = s["best"]
        x = b.direction * s["w"]
        if x > 0 and not boost_ok:
            x *= THIN_SCALE
        feat = "r_fundamental" if b.evidence == "market" and b.catalyst == "other" else feature_of(b.catalyst)
        sums[feat] += x
        total += x

    # risk penalty: credible flags press harder; capped, never a veto
    flagged = [s for s in counted if s["best"].risk_flags]
    flag_pubs = {}
    for s in flagged:
        for f in s["best"].risk_flags:
            flag_pubs.setdefault(f, []).append(s["publishers"])
    corroborated = {f for f, pubs in flag_pubs.items() if _distinct(pubs) >= 2}
    raw, per_flag = 0.0, {}
    for s in flagged:
        b = s["best"]
        cred = s["official"] or rel.get(b.source, 0.5) >= CREDIBLE_REL or bool(corroborated & set(b.risk_flags))
        w2 = max(0.0, b.confidence) * _decay(now - b.ts, cfg["penalty_half_life_h"])
        sev = severity(b.risk_flags)
        raw += sev * w2 if cred else UNCONFIRMED_SHARE * sev * w2
        for f in b.risk_flags:
            p = per_flag.setdefault(f, dict(flag=f, severity=RISK_FLAGS.get(f, 0.0), credible=False, pubs=set(), w=-1.0,
                                            title="", url=""))
            p["credible"] = p["credible"] or bool(cred)
            p["pubs"] |= s["publishers"]
            if w2 > p["w"]:
                p.update(w=w2, title=b.title, url=b.url)
    cap = cfg["penalty_cap"]
    penalty = min(cap, raw)
    flags = sorted(per_flag.values(), key=lambda p: (-p["credible"], -p["severity"], -len(p["pubs"]), p["flag"]))[:FLAGS_N]

    features = {k: math.tanh(sums[k] / SCALE) for k in ("r_catalyst", "r_sentiment", "r_fundamental", "r_structure")}
    features["r_risk"] = -penalty / cap if cap > 0 else 0.0
    features["r_coverage"] = math.tanh(n_ind / 4.0)
    top = []
    for s in sorted(counted, key=lambda s: -s["w"])[:TOP_N]:
        b = s["best"]
        top.append(dict(id=b.id, title=b.title, url=b.url, publisher=b.publisher, catalyst=b.catalyst, direction=b.direction,
                        confidence=round(b.confidence, 3), evidence=b.evidence, ts=b.ts, quarantine=b.quarantine))
    return dict(symbol=sym, asset=asset, ts=now, n_notes=len(notes), n_stories=len(stories), n_independent=n_ind, kinds=kinds,
                score=round(math.tanh(total / SCALE), 4), boost_ok=boost_ok, penalty=round(penalty, 4),
                halted=_halted(notes, now), thin=n_ind < cfg["thin_sources"],
                flags=[dict(flag=p["flag"], severity=p["severity"], credible=p["credible"], sources=len(p["pubs"]),
                            title=p["title"], url=p["url"]) for p in flags],
                features={k: round(features[k], 4) for k in RESEARCH_SIGNALS}, top=top,
                last_note_ts=max(n.ts for n in notes))


def build_view(db, now, subscribed: dict, cfg=None, reliability=None) -> dict:
    """{symbol: view} for every subscribed symbol (see the module docstring for the fields)."""
    c = dict(DEFAULTS)
    for k in DEFAULTS:
        if (cfg or {}).get(k) is not None:
            c[k] = type(DEFAULTS[k])((cfg or {})[k])
    rel = dict(reliability) if reliability is not None else _reliability(db)
    subscribed = dict(subscribed or {})
    by_sym = {}
    for n in db.fresh_notes(now, list(subscribed)):
        by_sym.setdefault(n.symbol, []).append(n)
    out = {}
    for sym, asset in subscribed.items():
        notes = by_sym.get(sym)
        out[sym] = _view(sym, asset, notes, now, c, rel) if notes else _empty_view(sym, asset, now)
    return out


def team_coverage(db, now, subscribed, window_h=24) -> dict:
    """{team: dict(symbols_covered, symbols_total, notes, sources_active, assets)} over the last window_h.
    A team's symbols_total counts only the asset classes it has written about this past week
    (a filings team that only reads stock filings is not short of crypto notes)."""
    subscribed = dict(subscribed or {})
    since, week = now - window_h * 3600, now - 7 * 86400
    seen = db.q("SELECT team, symbol, asset, source, fetched_ts FROM notes WHERE fetched_ts >= ?", (week,))
    try:
        sources = db.sources()
    except Exception:  # noqa: BLE001
        sources = {}
    teams = list(TEAMS) + sorted({r["team"] for r in seen if r["team"] and r["team"] not in TEAMS})
    out = {}
    for team in teams:
        mine = [r for r in seen if r["team"] == team]
        assets = {r["asset"] for r in mine} or set(subscribed.values())
        recent = [r for r in mine if (r["fetched_ts"] or 0) >= since]
        covered = {r["symbol"] for r in recent if r["symbol"] in subscribed}
        active = {r["source"] for r in recent}
        active |= {n for n, s in sources.items() if s.get("team") == team and (s.get("last_ok") or 0) >= since}
        out[team] = dict(symbols_covered=len(covered), symbols_total=sum(1 for a in subscribed.values() if a in assets),
                         notes=len(recent), sources_active=len(active), assets=sorted(a for a in assets if a))
    return out
