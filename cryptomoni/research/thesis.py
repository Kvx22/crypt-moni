"""Written theses: a few plain sentences that say why a proposed order is on the board.

Display only. A thesis never changes a score, a size or a decision; the engine builds it
after the order is proposed and shows it next to the order the human approves or rejects.

write() is a fixed template over facts the board already has: the side and size, the desk's
verdict and regime, which advisors voted for it, and the research view (how many independent
sources over how many kinds of evidence, which way they lean, any risk flag and its capped
penalty). It never predicts and never promises: no "will", no "guaranteed", no hype.

llm_thesis() is optional: when research.thesis_llm = {"url", "model"} points at a model
running on this PC (an Ollama-style /api/generate), it asks that model for a three-sentence
draft from the same structured facts. Only 127.0.0.1 / localhost are allowed, the reply is
labelled "(AI draft, display only)", and any error simply means no draft.
"""
from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request

MAX_CHARS = 600

# what each advisor tends to like about a name (buy) or dislike (sell); keys from advisors.ADVISORS
_LIKES = {"kai": "momentum", "imani": "clean risk", "vera": "the crowd is not chasing it",
          "marguerite": "value", "idris": "the macro backdrop"}
_DISLIKES = {"kai": "fading momentum", "imani": "risk", "vera": "a crowded trade",
             "marguerite": "stretched value", "idris": "the macro backdrop"}

# words a thesis never prints, even inside a quoted headline (a headline using them is skipped)
_BANNED = re.compile(r"\b(will|guarantee[ds]?|moon(ing)?|100x|skyrocket\w*|can't lose|sure thing)\b", re.I)


def _names(advisors):
    if advisors is None:
        try:
            from ..advisors import ADVISORS
            advisors = [(a.key, a.name) for a in ADVISORS]
        except Exception:  # noqa: BLE001  (a thesis is decoration; fall back to the keys)
            advisors = []
    out = {}
    for a in advisors or ():
        try:
            k, n = (a.key, a.name) if hasattr(a, "key") else (a[0], a[1])
        except (TypeError, IndexError):
            continue
        out[str(k)] = str(n).split()[0] if str(n).strip() else str(k).title()
    return out


def _join(words):
    words = [w for w in words if w]
    if len(words) <= 1:
        return "".join(words)
    return ", ".join(words[:-1]) + " and " + words[-1]


def _num(v, default=0.0):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return f if f == f and abs(f) != float("inf") else default


def _short(s, n):
    s = " ".join(str(s or "").split())
    return s if len(s) <= n else s[: n - 1].rsplit(" ", 1)[0].rstrip(",;:-") + "…"


def _fit(parts, limit=MAX_CHARS):
    """Join sentences, dropping optional ones (marked by a leading '?') until it fits."""
    parts = list(parts)
    while True:
        s = " ".join(p.lstrip("?") for p in parts if p)
        if len(s) <= limit:
            return s
        opt = [i for i, p in enumerate(parts) if p and p.startswith("?")]
        if not opt:
            return _short(s, limit)
        parts.pop(opt[-1])


def _votes_sentence(votes, side, names):
    votes = votes if isinstance(votes, dict) else {}
    want = "y" if side == "buy" else "n"
    keys = [k for k in names if votes.get(k) == want] + [k for k in votes if k not in names and votes.get(k) == want]
    against = [k for k in names if votes.get(k) == ("n" if want == "y" else "y")]
    if not keys:
        return "No advisor voted for it on its own; it comes from the desk's blend."
    who = _join([names.get(k, str(k).title()) for k in keys])
    table = _LIKES if side == "buy" else _DISLIKES
    why = ", ".join(dict.fromkeys(table[k] for k in keys if k in table))
    verb = ("like it" if len(keys) > 1 else "likes it") if side == "buy" else "would sell"
    s = f"{who} {verb}" + (f" ({why})." if why else ".")
    if against:
        s += f" {_join([names.get(k, k) for k in against])} {'disagree' if len(against) > 1 else 'disagrees'}."
    return s


def _research_sentence(view):
    if not isinstance(view, dict) or not view:
        return "Research: thin, no research view for this name yet, so research did not move the score."
    n = int(_num(view.get("n_independent", view.get("n")), 0))
    kinds = view.get("kinds") or []
    kinds = kinds if isinstance(kinds, (list, tuple)) else []
    if view.get("thin") or n < 2:
        src = "no independent source" if n == 0 else f"only {n} independent source"
        return f"Research: thin ({src}), so it barely moved the score."
    score = _num(view.get("score"))
    lean = "leaning positive" if score > 0.1 else "leaning negative" if score < -0.1 else "mixed"
    quotes = []
    for t in view.get("top") or []:
        if not isinstance(t, dict) or t.get("quarantine"):
            continue
        title = _short(t.get("title"), 70)
        if not title or _BANNED.search(title):
            continue
        pub = _short(t.get("publisher"), 30)
        quotes.append(f"'{title}'" + (f" – {pub}" if pub else ""))
        if len(quotes) == 2:
            break
    s = f"Research: {n} independent sources over {len(kinds)} kind{'s' if len(kinds) != 1 else ''}, {lean}"
    s += f" ({'; '.join(quotes)})." if quotes else "."
    if not view.get("boost_ok", n >= 4 and len(kinds) >= 2) and score > 0:
        s += " Not enough corroboration for a full boost, so good news counts faintly."
    return s


def _risk_sentence(view):
    if not isinstance(view, dict):
        return "Risk: none flagged."
    parts = []
    if view.get("halted"):
        parts.append("The exchange reports trading halted on this name.")
    flags = [f for f in (view.get("flags") or []) if isinstance(f, dict) and f.get("flag")]
    pen = _num(view.get("penalty"))
    if flags:
        names = _join(list(dict.fromkeys(str(f["flag"]).replace("_", " ") for f in flags[:3])))
        cred = "credible" if any(f.get("credible") for f in flags) else "unconfirmed"
        parts.append(f"Risk: {names} flagged ({cred}); a capped penalty of {pen:.2f} applied to the score.")
    elif pen > 0:
        parts.append(f"Risk: a capped penalty of {pen:.2f} applied to the score.")
    else:
        parts.append("Risk: none flagged.")
    return " ".join(parts)


def write(symbol, side, verdict, votes, view, regime, advisors=None, notional=None) -> str:
    """A plain-English thesis of at most 600 characters. Display only."""
    side = "sell" if str(side).lower().startswith("s") else "buy"
    sym = str(symbol or "?")
    base = sym.split("-")[0] if "-" in sym else sym
    size = f" (${_num(notional):,.0f})" if notional is not None and _num(notional) > 0 else ""
    verdict = str(verdict or "no verdict").upper()
    regime = str(regime or "unknown").upper().replace("_", " ")
    article = "an" if regime[:1] in "AEIOU" else "a"
    head = f"{side.title()} {base}{size}: the desk rates it {verdict} in {article} {regime} market."
    s = _fit([head, "?" + _votes_sentence(votes, side, _names(advisors)), _research_sentence(view), _risk_sentence(view)])
    return s if not _BANNED.search(s) else _BANNED.sub("may", s)


# ---------------------------------------------------------------------------- optional local model

_LOCAL = {"127.0.0.1", "localhost", "::1"}


def _llm_cfg(cfg):
    cfg = cfg if isinstance(cfg, dict) else {}
    c = (cfg.get("research") or {}).get("thesis_llm") if isinstance(cfg.get("research"), dict) else None
    c = c if c is not None else cfg.get("thesis_llm")
    if not isinstance(c, dict) or not c.get("url") or not c.get("model"):
        return None
    return c


def local_url(url) -> bool:
    try:
        p = urllib.parse.urlsplit(str(url))
        return p.scheme == "http" and (p.hostname or "").lower() in _LOCAL
    except ValueError:
        return False


def _plain(facts):
    """Only structured facts: numbers, booleans, short strings, and lists/dicts of those."""
    if isinstance(facts, bool) or facts is None:
        return facts
    if isinstance(facts, (int, float)):
        return round(float(facts), 4)
    if isinstance(facts, str):
        return _short(facts, 160)
    if isinstance(facts, dict):
        return {str(k)[:40]: _plain(v) for k, v in list(facts.items())[:30]}
    if isinstance(facts, (list, tuple)):
        return [_plain(v) for v in list(facts)[:10]]
    return _short(str(facts), 80)


def _post(url, body, timeout):
    # no proxy: the model lives on this PC and the request must never leave it
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with opener.open(req, timeout=timeout) as r:
        return json.loads(r.read(256 * 1024).decode("utf-8", "replace"))


def llm_thesis(cfg, facts) -> str | None:
    """A three-sentence draft from a local model, or None (no config, non-local URL, any error)."""
    c = _llm_cfg(cfg)
    if c is None or not local_url(c["url"]):
        return None
    prompt = ("Write exactly 3 short plain-English sentences describing this proposed order for a human "
              "reviewer, using ONLY the facts below. Do not give advice, do not make predictions, do not use "
              "the words 'will' or 'guaranteed', no hype.\nFacts (JSON):\n"
              + json.dumps(_plain(facts if isinstance(facts, dict) else {}), sort_keys=True))
    try:
        r = _post(str(c["url"]), {"model": str(c["model"]), "prompt": prompt, "stream": False}, 20.0)
        text = " ".join(str(r.get("response") or "").split()) if isinstance(r, dict) else ""
    except Exception:  # noqa: BLE001  (a draft is decoration)
        return None
    if not text:
        return None
    prefix = "(AI draft, display only) "
    return prefix + _short(text, MAX_CHARS - len(prefix))
