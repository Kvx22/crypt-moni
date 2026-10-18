"""Items in, notes out.

Adapters store raw items (an article, a filing, a funding-rate reading). label_items() turns
each new item into one note per symbol it is about:
  * which symbols: the ones the adapter already knew, plus subscribed names, tickers and
    cashtags found in the text (for news-like items or items the adapter left unassigned);
  * what it says: structured adapters label their own items ("source"); text items get the
    rule labeller, and the extractor model when one is trained and confident ("model");
    risk flags are the rules' flags plus the model's very confident ones (rules are precise,
    the model adds recall);
  * how long it matters: published time (or fetched time when the published time is
    missing or implausible) plus a freshness window by evidence kind; flagged notes live longer.
Items about nothing we follow feed the discovery list instead.

relabel_recent() re-runs a newly promoted model over the last couple of days of notes it
or the rules labelled, so a better extractor improves the live view straight away.
"""
from __future__ import annotations

import time

from . import labels, text
from .aliases import discover
from .taxonomy import CATALYSTS, FLAG_FRESH_S, FRESH_S, RISK_FLAGS, severity
from .types import EVIDENCE, Note, dumps, loads

MODEL_MIN_CONF = 0.6     # the model's catalyst probability needed to overrule the rules
MODEL_FLAG_P = 0.7       # the model's flag probability needed to add a flag the rules missed
MAX_PAST_S = 30 * 86400  # a "published" time this far before we fetched it is not believable
FUTURE_SLACK_S = 600     # clock skew allowed between a publisher and us
TEXT_KINDS = ("news", "search", "official")


def _choose(title, body, model, min_conf):
    """(catalyst, direction, flags, confidence, label_by) for a text item: rules, or the model when confident."""
    r = labels.rule_label(title, body)
    cat, d, conf, by = r["catalyst"], r["direction"], r["confidence"], "rules"
    flags = list(r["risk_flags"])
    if model is not None:
        try:
            p = model.predict(title, body)
        except Exception:  # noqa: BLE001  (a broken model must not stop labelling)
            p = None
        if p:
            if p["cat_p"] >= min_conf:
                cat, d, by = p["catalyst"], int(p["direction"]), "model"
                conf = p["cat_p"] * p["dir_p"] ** 0.5
            flags += [f for f, q in (p.get("flag_p") or {}).items() if q >= MODEL_FLAG_P and f not in flags]
    flags = [f for f in RISK_FLAGS if f in flags]
    return cat, d, flags, round(max(0.0, min(1.0, conf)), 4), by


def _source_labels(it):
    cat = it.catalyst if it.catalyst in CATALYSTS else "other"
    try:
        d = max(-1, min(1, int(it.direction or 0)))
    except (TypeError, ValueError):
        d = 0
    flags = [f for f in RISK_FLAGS if f in (it.risk_flags or ())]
    try:
        conf = float(it.confidence) if it.confidence is not None else 0.7
    except (TypeError, ValueError):
        conf = 0.7
    return cat, d, flags, max(0.0, min(1.0, conf)), "source"


def _note_ts(published, fetched):
    if published is None:
        return fetched
    try:
        p = float(published)
    except (TypeError, ValueError):
        return fetched
    return fetched if p > fetched + FUTURE_SLACK_S or p < fetched - MAX_PAST_S else p


def _expires(ts, evidence, flags):
    return ts + (FLAG_FRESH_S if flags else FRESH_S.get(evidence, FRESH_S["news"]))


def _symbols(it, aliases):
    subs = aliases.symbols
    out = [s for s in (it.symbols or ()) if isinstance(s, str) and s in subs]
    if not it.symbols or it.kind in TEXT_KINDS:
        for s, _how in aliases.find(f"{it.title} {it.text}"):
            if s not in out:
                out.append(s)
    return out


def label_items(db, aliases, model=None, now=None, cfg=None, batch=300):
    """Label up to `batch` unlabelled items into notes. Returns dict(items, notes, discovered,
    unmapped, skipped, failed, by={label_by: notes})."""
    now = time.time() if now is None else now
    cfg = cfg or {}
    min_conf = float(cfg.get("model_min_conf", MODEL_MIN_CONF))
    rows = db.unlabelled_items(batch)
    subscribed = set(aliases.symbols)
    notes, done, failed, per_source, by = [], [], [], {}, {}
    n_disc = n_unmapped = n_skipped = 0
    for item_id, team, fetched_ts, it in rows:
        try:
            fetched_ts = float(fetched_ts if fetched_ts is not None else now)
            syms = _symbols(it, aliases)
            if not syms:
                done.append(item_id)
                if (it.facts or {}).get("scope") == "exchange":   # market-wide reading: nothing to file it under
                    n_skipped += 1
                    continue
                n_unmapped += 1
                found = {s: a for s, a, _ in discover(f"{it.title} {it.text}", subscribed)}
                if it.kind == "discovery":
                    for s in it.symbols or ():
                        if isinstance(s, str) and s and s not in subscribed:
                            found.setdefault(s, aliases.asset(s) or ("crypto" if "-" in s else "stock"))
                for s, a in found.items():
                    db.discover(s, a, it.source, reason=(it.title or "")[:200], now=now)
                n_disc += len(found)
                continue
            if it.catalyst:
                cat, d, flags, conf, label_by = _source_labels(it)
            else:
                cat, d, flags, conf, label_by = _choose(it.title, it.text, model, min_conf)
            ts = _note_ts(it.published_ts, fetched_ts)
            evidence = "official" if it.official else (it.kind if it.kind in EVIDENCE else "news")
            cluster = text.fingerprint(it.title, it.text)
            publisher = it.publisher or text.domain(it.url)
            for s in syms:
                notes.append(Note(symbol=s, asset=aliases.asset(s) or ("crypto" if "-" in s else "stock"), ts=ts,
                                  fetched_ts=fetched_ts, source=it.source, team=team, publisher=publisher, evidence=evidence,
                                  catalyst=cat, direction=d, confidence=conf, risk_flags=list(flags), severity=severity(flags),
                                  label_by=label_by, title=it.title, url=it.url, facts=dict(it.facts or {}), cluster=cluster,
                                  quarantine=bool(it.quarantine), official=bool(it.official),
                                  expires_ts=_expires(ts, evidence, flags), item_id=item_id))
                per_source[it.source] = per_source.get(it.source, 0) + 1
                by[label_by] = by.get(label_by, 0) + 1
            done.append(item_id)
        except Exception:  # noqa: BLE001  (one odd item must not block the queue: park it)
            failed.append(item_id)
    n = db.add_notes(notes)
    db.mark_labelled(done)
    db.mark_labelled(failed, state=2)
    db.bump_source_notes(per_source)
    return dict(items=len(rows), notes=n, discovered=n_disc, unmapped=n_unmapped, skipped=n_skipped, failed=len(failed), by=by)


def relabel_recent(db, model, aliases=None, now=None, days=2, cfg=None):
    """Re-run `model` over recent notes labelled by the rules or an earlier model (never touching
    human, source or seed labels). Returns the number of notes whose labels changed.
    `aliases` is accepted for the service's call signature; symbols are not re-mapped."""
    if model is None:
        return 0
    now = time.time() if now is None else now
    min_conf = float((cfg or {}).get("model_min_conf", MODEL_MIN_CONF))
    rows = db.q("""SELECT n.id, n.title, n.ts, n.evidence, n.catalyst, n.direction, n.risk_flags, n.confidence, n.label_by,
                          i.text AS body FROM notes n LEFT JOIN items i ON i.id = n.item_id
                   WHERE n.ts >= ? AND n.label_by NOT IN ('human', 'source', 'seed')
                   AND n.id NOT IN (SELECT note_id FROM labels)""", (now - days * 86400,))
    cache, upd = {}, []
    for r in rows:
        key = (r["title"] or "", r["body"] or "")
        if key not in cache:
            cache[key] = _choose(key[0], key[1], model, min_conf)
        cat, d, flags, conf, by = cache[key]
        old = (r["catalyst"], int(r["direction"] or 0), loads(r["risk_flags"], []) or [], r["label_by"])
        if old == (cat, d, flags, by) and abs(float(r["confidence"] or 0) - conf) < 1e-6:
            continue
        upd.append((cat, d, conf, dumps(flags), severity(flags), by, _expires(r["ts"], r["evidence"], flags), r["id"]))
    if upd:
        db.many("""UPDATE notes SET catalyst=?, direction=?, confidence=?, risk_flags=?, severity=?, label_by=?, expires_ts=?
                   WHERE id=?""", upd)
    return len(upd)
