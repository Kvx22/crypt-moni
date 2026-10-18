"""Did the research turn out right? This module keeps score.

Every note says something about one symbol at one moment. We record the price at that
moment (price0, plus the benchmark: BTC-USD for crypto, SPY for stocks), then look again
1h, 4h, 8h, 12h, 1d and 7d later and log the move against the benchmark (the excess log
return). From those outcomes we grade each source (reliability: how often its bullish or
bearish calls were right, recent calls counting more) and train a small outcome model
that guesses, from a note's labels, source and title, whether it will be right.

The clock rules, so nothing peeks at the future:
  - Prices are hourly bars stored by START time. A bar is only known at its END
    (start + 1h), so the price "at" a moment is the close of the latest bar that had
    ended by then, and only if that bar is recent (3 hours for crypto, 4 days for
    stocks, which have nights, weekends and holidays).
  - Crypto horizons are clock hours from the end of that price0 bar. Stock horizons are
    counted in regular-session hourly bars of the stock's own series (7 bars = one
    session, so "1d" is the same hour next session and "7d" is five sessions later).
  - An outcome is written only once its target bar has closed.

The outcome model is champion/challenger: a new model is trained on the oldest 75% of
the history, tested on the newest 25% (with a purge gap between the two so overlapping
outcomes cannot leak) and promoted only if it clearly beats (by a margin, and by more than
luck: a paired t-test on the holdout) the current champion, or a
"trust the source's track record" rule when there is no champion yet.
"""
from __future__ import annotations

import json
import math
import re
import time
import zlib
from bisect import bisect_left, bisect_right

import numpy as np

from .taxonomy import BENCH, CATALYSTS, HORIZON_BARS_STOCK, HORIZON_HOURS_CRYPTO, HORIZONS
from .types import EVIDENCE

try:  # the shared tokenizer, when the text module is installed; else a compatible local one
    from .text import tokens as _tokens
    TOKENIZER = "text"
except Exception:  # noqa: BLE001
    _TOK = re.compile(r"\$?[^\W_]+(?:[.,][0-9]+)*%?", re.UNICODE)
    TOKENIZER = "local"

    def _tokens(text):
        return [t for t in _TOK.findall((text or "").lower()) if len(t) > 1 or t.isdigit()]

HOUR = 3600
DAY = 86400
STALE_S = {"crypto": 3 * HOUR, "stock": 4 * DAY}   # how old the "price at ts" bar may be
FETCH_GAP_S = 20 * 60                              # refetch a symbol at most this often
WINDOW_DAYS = 30                                   # give up maturing things older than this
LABEL_BY = ("rules", "model", "human", "seed", "source")


def _pos(x):
    """x as a positive finite float, else None."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) and x > 0 else None


def _excess(p0, p1, b0, b1):
    """(ret, bench_ret, excess) as log returns, or None if any price is unusable."""
    p0, p1, b0, b1 = _pos(p0), _pos(p1), _pos(b0), _pos(b1)
    if None in (p0, p1, b0, b1):
        return None
    ret, bret = math.log(p1 / p0), math.log(b1 / b0)
    return ret, bret, ret - bret


def _vote(direction, excess, min_move):
    """+1 the call was right, -1 it was wrong, 0 the move was too small (or missing) to judge."""
    try:
        e = float(excess)
    except (TypeError, ValueError):
        return 0
    if not math.isfinite(e) or abs(e) < min_move or not direction:
        return 0
    return 1 if (e > 0) == (direction > 0) else -1


# ---- prices ------------------------------------------------------------------------------
class PriceKeeper:
    """Keeps the prices table topped up and answers "what was the price at ts".

    fetchers: {"crypto": f(symbol, since_ts | None) -> [(bar_start_ts, close)],
               "stock":  f(symbol, since_ts | None) -> [(bar_start_ts, close)]}
    Series read from the database are cached on this object; make a fresh keeper (as
    `mature` does) for a fresh view."""

    def __init__(self, db, fetchers=None, min_gap_s=FETCH_GAP_S):
        self.db, self.fetchers, self.min_gap_s = db, dict(fetchers or {}), min_gap_s
        self.last_fetch = {}      # symbol -> when we last asked (in memory, by design)
        self._cache = {}

    def refresh(self, symbols, now=None):
        """Fetch new hourly closes for {symbol: asset} plus the benchmarks. Never raises for
        one symbol: returns dict(fetched=[symbols asked], bars=n stored, errors={symbol: msg})."""
        now = time.time() if now is None else now
        want = {s: a for s, a in (symbols or {}).items() if a in BENCH}
        want.setdefault(BENCH["crypto"], "crypto")
        if "stock" in want.values():
            want.setdefault(BENCH["stock"], "stock")
        out = dict(fetched=[], bars=0, errors={})
        for sym, asset in sorted(want.items()):
            if now - self.last_fetch.get(sym, -1e18) < self.min_gap_s:
                continue
            self.last_fetch[sym] = now          # throttle failures too: a broken source is not hammered
            out["fetched"].append(sym)
            f = self.fetchers.get(asset)
            if f is None:
                out["errors"][sym] = f"no price fetcher for {asset}"
                continue
            try:
                bars = []
                for b in f(sym, self.db.last_price_ts(sym)) or ():
                    try:
                        t, c = int(b[0]), _pos(b[1])
                    except (TypeError, ValueError, IndexError):
                        continue
                    if c is not None and t + HOUR <= now:     # drop a still-forming bar
                        bars.append((t, c))
                self.db.put_prices(sym, bars)
            except Exception as e:  # noqa: BLE001
                out["errors"][sym] = (str(e) or type(e).__name__)[:200]
                continue
            if bars:
                self._cache.pop(sym, None)
                out["bars"] += len(bars)
        return out

    def series(self, symbol):
        """(starts, closes) sorted by start, unusable closes dropped. Cached."""
        s = self._cache.get(symbol)
        if s is None:
            rows = [(int(t), _pos(c)) for t, c in self.db.price_series(symbol)]
            rows = [(t, c) for t, c in rows if c is not None]
            s = self._cache[symbol] = ([t for t, _ in rows], [c for _, c in rows])
        return s

    def bar_index(self, symbol, ts, asset):
        """Index of the latest bar that had ENDED by ts (start + 1h <= ts), if recent enough."""
        starts, _ = self.series(symbol)
        i = bisect_right(starts, ts - HOUR) - 1
        if i < 0 or ts - (starts[i] + HOUR) > STALE_S.get(asset, STALE_S["crypto"]):
            return None
        return i

    def price_at(self, symbol, ts, asset):
        i = self.bar_index(symbol, ts, asset)
        return None if i is None else self.series(symbol)[1][i]

    def settled(self, symbol, ts, asset):
        """Index of the bar known at ts, but only once we are sure no later-arriving bar would
        have been the right one: either a later bar is already stored, or (crypto) it is the
        bar that ends on the hour at or before ts. Stops a lagging fetch fixing a stale price0."""
        i = self.bar_index(symbol, ts, asset)
        if i is None:
            return None
        starts, _ = self.series(symbol)
        if i < len(starts) - 1 or (asset == "crypto" and starts[i] + HOUR == int(ts // HOUR) * HOUR):
            return i
        return None

    def target(self, symbol, asset, ts, horizon, now):
        """(start, close) of the horizon bar for something that happened at ts, once it has
        closed by now; else None. Crypto: the bar ending `hours` after the end of the bar
        known at ts (one bar of slack if exactly that bar is missing). Stocks: the bar
        `HORIZON_BARS_STOCK[horizon]` bars later in the symbol's own series."""
        i0 = self.bar_index(symbol, ts, asset)
        if i0 is None or horizon not in HORIZONS:
            return None
        starts, closes = self.series(symbol)
        if asset == "stock":
            j = i0 + HORIZON_BARS_STOCK[horizon]
            if j >= len(starts):
                return None
        else:
            want = starts[i0] + HORIZON_HOURS_CRYPTO[horizon] * HOUR
            j = bisect_left(starts, want)
            if j >= len(starts) or starts[j] - want > HOUR:
                return None
        if starts[j] + HOUR > now:
            return None
        return starts[j], closes[j]

    def bench_close(self, bench, start, asset):
        """The benchmark's bar at the same start time, else its latest bar before (if recent)."""
        starts, closes = self.series(bench)
        i = bisect_right(starts, start) - 1
        if i < 0 or start - starts[i] > STALE_S.get(asset, STALE_S["crypto"]):
            return None
        return closes[i]

    def horizon_excess(self, symbol, asset, ts, horizon, p0, b0, now):
        """(ret, bench_ret, excess, known_ts) at `horizon`, or None if not yet knowable."""
        bench = BENCH.get(asset)
        tgt = bench and self.target(symbol, asset, ts, horizon, now)
        if not tgt:
            return None
        r = _excess(p0, tgt[1], b0, self.bench_close(bench, tgt[0], asset))
        return None if r is None else (*r, tgt[0] + HOUR)


# ---- maturing notes, shadow decisions and proposals ---------------------------------------------
def mature(db, now=None, cfg=None):
    """One pass: fill price0 on new notes, write outcomes whose horizon bar has closed, mature
    shadow decisions and score 7-day-old watch-list proposals. Returns the counts."""
    now = time.time() if now is None else now
    window = float((cfg or {}).get("outcome_window_days", WINDOW_DAYS)) * DAY
    pk = PriceKeeper(db)
    out = dict(price0=0, outcomes=0, shadow=0, proposals=0)

    # 1) price0 / bench0 at the moment of each recent note
    rows = []
    for n in db.q("SELECT id,symbol,asset,ts FROM notes WHERE price0 IS NULL AND ts>? AND ts<=?", (now - 7 * DAY, now)):
        bench = BENCH.get(n["asset"])
        if not bench:
            continue
        i, j = pk.settled(n["symbol"], n["ts"], n["asset"]), pk.settled(bench, n["ts"], n["asset"])
        if i is not None and j is not None:
            rows.append((pk.series(n["symbol"])[1][i], pk.series(bench)[1][j], n["id"]))
    db.set_price0(rows)
    out["price0"] = len(rows)

    # 2) outcomes at every horizon
    rows = []
    for h in HORIZONS:
        min_age = min(HORIZON_HOURS_CRYPTO[h], HORIZON_BARS_STOCK[h]) * HOUR
        pend = db.q("""SELECT n.id,n.symbol,n.asset,n.ts,n.price0,n.bench0 FROM notes n WHERE n.price0 IS NOT NULL
                       AND n.ts<=? AND n.ts>=? AND NOT EXISTS(SELECT 1 FROM outcomes o WHERE o.note_id=n.id AND o.horizon=?)""",
                    (now - min_age, now - window, h))
        for n in pend:
            r = pk.horizon_excess(n["symbol"], n["asset"], n["ts"], h, n["price0"], n["bench0"], now)
            if r:
                rows.append((n["id"], h, *r))
    db.add_outcomes(rows)
    out["outcomes"] = len(rows)

    # 3) shadow decisions (the engines record price0/bench0 in detail)
    rows = []
    for s in db.shadow(matured=False, since=now - window):
        d = s["detail"] if isinstance(s["detail"], dict) else {}
        r = pk.horizon_excess(s["symbol"], s["asset"], s["ts"], d.get("horizon", "1d"), d.get("price0"), d.get("bench0"), now)
        if r:
            rows.append((r[2], s["id"]))
    db.mature_shadow(rows)
    out["shadow"] = len(rows)

    # 4) watch-only proposals, a week on
    n = 0
    for p in db.q("""SELECT * FROM proposals WHERE status IN ('shadow','added') AND excess_7d IS NULL
                     AND ts<=? AND ts>=?""", (now - 7 * DAY, now - 7 * DAY - window)):
        r = pk.horizon_excess(p["symbol"], p["asset"], p["ts"], "7d", p["price0"], p["bench0"], now)
        if r:
            db.set_proposal(p["id"], p["status"], r[2])
            n += 1
    out["proposals"] = n
    return out


# ---- source reliability ------------------------------------------------------------------------
def update_reliability(db, now=None, horizon="1d", half_life_days=30, min_move=0.002):
    """Grade every source on its directional notes from the last 120 days. Recent calls weigh
    more (half-life `half_life_days`); moves under `min_move` are not counted either way.
    A source with no gradeable calls goes back to the prior (2, 2, usefulness 0.5)."""
    now = time.time() if now is None else now
    hl = max(1e-9, float(half_life_days)) * DAY
    rows = db.q("""SELECT n.source,n.team,n.ts,n.direction,o.excess FROM notes n JOIN outcomes o ON o.note_id=n.id
                   AND o.horizon=? WHERE n.direction!=0 AND n.ts>=? AND n.ts<=? AND o.ts<=?""",
                (horizon, now - 120 * DAY, now, now))
    acc = {}
    for r in rows:
        v = _vote(r["direction"], r["excess"], min_move)
        a = acc.setdefault(r["source"], dict(team=r["team"], s=0.0, f=0.0, n=0))
        if v:
            w = 0.5 ** (max(0.0, now - r["ts"]) / hl)
            a["s" if v > 0 else "f"] += w
            a["n"] += 1
    if acc:  # a source graded here must have a row to hold its grade (notes can outlive a renamed adapter)
        db.many("INSERT OR IGNORE INTO sources(name,team) VALUES(?,?)", [(k, v["team"]) for k, v in acc.items()])
    out = []
    for name in sorted(set(db.sources()) | set(acc)):
        a = acc.get(name, dict(s=0.0, f=0.0, n=0))
        rel_a, rel_b, n_eff = 2.0 + a["s"], 2.0 + a["f"], a["s"] + a["f"]
        mean = rel_a / (rel_a + rel_b)
        use = float(np.clip((mean - 0.5) * 2 * min(1.0, n_eff / 30) + 0.5, 0.05, 1.0))
        out.append(dict(source=name, rel_a=rel_a, rel_b=rel_b, usefulness=use, mean=mean, n=a["n"], n_eff=n_eff))
    db.set_reliability([(r["rel_a"], r["rel_b"], r["usefulness"], r["source"]) for r in out])
    return out


class _Rel(dict):
    def __missing__(self, key):
        return 0.5


def reliability(db):
    """{source: rel_a / (rel_a + rel_b)}; unknown sources read 0.5 (also via .get(s, 0.5))."""
    out = _Rel()
    for name, r in db.sources().items():
        a, b = _pos(r.get("rel_a")), _pos(r.get("rel_b"))
        out[name] = a / (a + b) if a and b else 0.5
    return out


def _pit_reliability(rows, known_ts, horizon, half_life_days=30, min_move=0.002):
    """Point-in-time reliability for training: each row sees only the outcomes that were known
    (matured) by its own ts, so the feature carries no future information. Sets row["reliability"]."""
    hl = max(1e-9, float(half_life_days)) * DAY
    lag = HORIZON_HOURS_CRYPTO.get(horizon, 24) * HOUR
    ev = []
    for r in rows:
        v = _vote(r["direction"], r["outcomes"].get(horizon), min_move)
        if v:
            ev.append((known_ts.get(r["id"]) or r["ts"] + lag, r["source"], v))
    ev.sort(key=lambda e: e[0])
    state, k = {}, 0                                         # source -> [success, failure, at]

    def decayed(src, t):
        s = state.setdefault(src, [0.0, 0.0, t])
        f = 0.5 ** (max(0.0, t - s[2]) / hl)
        s[0], s[1], s[2] = s[0] * f, s[1] * f, max(t, s[2])
        return s

    for r in sorted(rows, key=lambda r: r["ts"]):
        while k < len(ev) and ev[k][0] <= r["ts"]:
            s = decayed(ev[k][1], ev[k][0])
            s[0 if ev[k][2] > 0 else 1] += 1.0
            k += 1
        s = decayed(r["source"], r["ts"])
        r["reliability"] = (2.0 + s[0]) / (4.0 + s[0] + s[1])


# ---- the outcome model -------------------------------------------------------------------------
def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def _rowloss(p, y):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def _logloss(p, y):
    return float(np.mean(_rowloss(p, y))) if len(y) else float("nan")


def _truth(rows, horizon, fee):
    """1 if the note was right after costs (direction * excess > fee), else 0. NaN-safe: rows
    without a usable outcome must be filtered out before (see _has_outcome)."""
    return np.array([float(np.sign(float(r["direction"])) * float(r["outcomes"][horizon]) > fee) for r in rows])


def _has_outcome(r, horizon):
    try:
        return math.isfinite(float((r.get("outcomes") or {})[horizon]))
    except (KeyError, TypeError, ValueError):
        return False


DENSE = (["bias", "direction", "confidence", "severity", "reliability", "quarantine", "official"]
         + [f"cat:{c}" for c in CATALYSTS] + [f"catdir:{c}:{d}" for c in CATALYSTS for d in (-1, 1)]
         + [f"ev:{e}" for e in EVIDENCE] + [f"by:{b}" for b in LABEL_BY])
_DI = {k: i for i, k in enumerate(DENSE)}


class OutcomeModel:
    """Logistic regression for P(the note is right) at one horizon, trained by full-batch
    gradient descent with Adagrad. Features: catalyst (and catalyst x direction), direction,
    confidence, severity, evidence kind, source reliability, quarantine, official, who
    labelled it, and hashed title words. predict() returns edge = 2p - 1 (0 = no idea)."""

    def __init__(self, n_text=2 ** 12, l2=1e-2):
        self.n_text, self.l2 = int(n_text), float(l2)
        self.w = None
        self.horizon, self.fee = "1d", 0.002
        self.metrics = {}

    def _design(self, rows):
        n, d = len(rows), len(DENSE)
        X = np.zeros((n, d))
        ti, tv, tr = [], [], []
        for i, r in enumerate(rows):
            cat = r.get("catalyst") if r.get("catalyst") in CATALYSTS else "other"
            try:
                dirn = int(r.get("direction") or 0)
            except (TypeError, ValueError):
                dirn = 0
            dirn = (dirn > 0) - (dirn < 0)
            X[i, _DI["bias"]] = 1.0
            X[i, _DI["direction"]] = dirn
            X[i, _DI["confidence"]] = _unit(r.get("confidence"), 0.0)
            X[i, _DI["severity"]] = _unit(r.get("severity"), 0.0)
            X[i, _DI["reliability"]] = _unit(r.get("reliability"), 0.5) - 0.5
            X[i, _DI["quarantine"]] = float(bool(r.get("quarantine")))
            X[i, _DI["official"]] = float(bool(r.get("official")))
            X[i, _DI[f"cat:{cat}"]] = 1.0
            if dirn:
                X[i, _DI[f"catdir:{cat}:{dirn}"]] = 1.0
            if r.get("evidence") in EVIDENCE:
                X[i, _DI[f"ev:{r['evidence']}"]] = 1.0
            if r.get("label_by") in LABEL_BY:
                X[i, _DI[f"by:{r['label_by']}"]] = 1.0
            toks = sorted(set(_tokens(r.get("title") or "")))
            for t in toks:
                ti.append(zlib.crc32(t.encode("utf-8")) % self.n_text)
                tv.append(1.0 / math.sqrt(len(toks)))
                tr.append(i)
        return X, np.array(ti, np.int64), np.array(tv, float), np.array(tr, np.int64)

    def _score(self, des, w):
        X, ti, tv, tr = des
        d = X.shape[1]
        z = X @ w[:d]
        if len(ti):
            z = z + np.bincount(tr, weights=tv * w[d + ti], minlength=X.shape[0])
        return z

    def fit(self, rows, horizon="1d", fee=0.002, iters=300, lr=0.2):
        rows = [r for r in rows if _has_outcome(r, horizon) and int(r.get("direction") or 0) != 0]
        self.horizon, self.fee = horizon, float(fee)
        d = len(DENSE)
        self.w = np.zeros(d + self.n_text)
        if not rows:
            self.metrics = dict(n=0, horizon=horizon, fee=self.fee)
            return self
        y = _truth(rows, horizon, fee)
        des = self._design(rows)
        X, ti, tv, tr = des
        n = len(rows)
        reg = np.full_like(self.w, self.l2)
        reg[_DI["bias"]] = 0.0
        G = np.full_like(self.w, 1e-8)
        w = self.w
        for _ in range(iters):
            r = (_sigmoid(self._score(des, w)) - y) / n
            g = np.concatenate([X.T @ r, np.bincount(ti, weights=tv * r[tr], minlength=self.n_text)]) + reg * w
            G += g * g
            w -= lr * g / np.sqrt(G)
        p = _sigmoid(self._score(des, w))
        self.metrics = dict(n=n, horizon=horizon, fee=self.fee, base_rate=float(y.mean()),
                            logloss_train=_logloss(p, y), acc_train=float(np.mean((p > 0.5) == (y > 0.5))))
        return self

    def proba(self, rows):
        if self.w is None or not rows:
            return np.full(len(rows), 0.5)
        return _sigmoid(self._score(self._design(rows), self.w))

    def predict(self, rows):
        """edge = 2p - 1 per row, in [-1, 1]."""
        return 2.0 * self.proba(rows) - 1.0

    def to_blob(self):
        meta = dict(v=1, n_text=self.n_text, l2=self.l2, horizon=self.horizon, fee=self.fee, tok=TOKENIZER,
                    dense=len(DENSE), metrics=self.metrics, w=None if self.w is None else [float(x) for x in self.w])
        return zlib.compress(json.dumps(meta, separators=(",", ":")).encode("utf-8"))

    @classmethod
    def from_blob(cls, blob):
        """Raises ValueError on a blob this code cannot use (corrupt, or a different layout)."""
        try:
            meta = json.loads(zlib.decompress(bytes(blob)).decode("utf-8"))
        except Exception as e:  # noqa: BLE001
            raise ValueError(f"unreadable outcome model: {e}") from e
        m = cls(n_text=meta.get("n_text", 2 ** 12), l2=meta.get("l2", 1e-2))
        m.horizon, m.fee, m.metrics = meta.get("horizon", "1d"), float(meta.get("fee", 0.002)), meta.get("metrics") or {}
        if meta.get("w") is not None:
            m.w = np.asarray(meta["w"], float)
            if meta.get("tok") != TOKENIZER:
                raise ValueError("outcome model was trained with a different tokenizer")
            if meta.get("dense") != len(DENSE) or len(m.w) != len(DENSE) + m.n_text or not np.all(np.isfinite(m.w)):
                raise ValueError("outcome model layout does not match this version")
        return m


def _unit(x, default):
    """x clipped to [0, 1]; missing or NaN -> default."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return default
    return min(1.0, max(0.0, x)) if math.isfinite(x) else default


def _dedupe(rows):
    """Syndicated copies and story pile-ons are one piece of evidence: keep the first note per
    (symbol, cluster) and at most 3 per (symbol, UTC day). rows must be sorted by ts."""
    seen, per_day, out = set(), {}, []
    for r in rows:
        if r.get("cluster"):
            k = (r["symbol"], r["cluster"])
            if k in seen:
                continue
            seen.add(k)
        k = (r["symbol"], int(r["ts"] // DAY))
        if per_day.get(k, 0) >= 3:
            continue
        per_day[k] = per_day.get(k, 0) + 1
        out.append(r)
    return out


def model_name(horizon="1d"):
    """The models-table name: "outcome" for the default 1d model, "outcome_<h>" for others."""
    return "outcome" if horizon == "1d" else f"outcome_{horizon}"


def _paired_gain(p_new, p_ref, y):
    """How much lower the new model's holdout log-loss is (mean per note) and the paired t-stat
    of that gain. The t-stat stops a lucky few hundred notes from promoting a model on noise."""
    d = _rowloss(p_ref, y) - _rowloss(p_new, y)
    if len(d) < 2:
        return 0.0, 0.0
    sd = float(np.std(d, ddof=1))
    return float(d.mean()), float(d.mean() / (sd / math.sqrt(len(d)))) if sd > 1e-12 else 0.0


def train_outcome_model(db, now=None, horizon="1d", fee=0.002, min_rows=300, margin=0.002, min_t=2.0):
    """Train a challenger, test it against the champion (or the reliability-only rule) on the
    newest 25%, promote it only on a clear win (log-loss lower by >= margin AND a paired
    t-stat >= min_t), and (1d model) write edges for the last week's notes. The model is stored
    as model_name(horizon). Returns a report dict."""
    now = time.time() if now is None else now
    # logloss_champion is present only when there is a champion (callers fall back to logloss_baseline)
    rep = dict(horizon=horizon, n_rows=0, n_train=0, n_holdout=0, logloss_challenger=None, logloss_baseline=None,
               acc_challenger=None, promoted=False, version=None, edges=0, reason="")
    name = model_name(horizon)
    champ_row = db.champion(name)
    champ = None
    if champ_row and champ_row.get("blob"):
        try:
            champ = OutcomeModel.from_blob(champ_row["blob"])
        except ValueError as e:
            rep["reason"] = f"champion unusable ({e}); "

    # only outcomes already known at `now` (matters when replaying history on a simulated clock)
    known = {o["note_id"]: o["ts"] for o in db.q("SELECT note_id, ts FROM outcomes WHERE horizon=? AND ts<=?", (horizon, now))}
    rows = [r for r in db.note_rows("direction!=0 AND ts<=?", (now,)) if r["id"] in known and _has_outcome(r, horizon)]
    _pit_reliability(rows, known, horizon)
    rows = _dedupe(sorted(rows, key=lambda r: (r["ts"], r["id"])))
    rep["n_rows"] = len(rows)
    use = champ
    if len(rows) < min_rows:
        rep["reason"] += f"only {len(rows)} usable notes (need {min_rows})"
    else:
        cut = int(len(rows) * 0.75)
        hold = rows[cut:]
        # purge: drop training notes from the day before the holdout, and any whose outcome was
        # only known after the holdout began (stock horizons can straddle a weekend)
        start = hold[0]["ts"]
        train = [r for r in rows[:cut] if r["ts"] < start - DAY and known[r["id"]] <= start]
        y = _truth(hold, horizon, fee)
        ch = OutcomeModel().fit(train, horizon, fee)
        p_ch, p_base = ch.proba(hold), np.array([r["reliability"] for r in hold])
        p_ref = champ.proba(hold) if champ else p_base
        ref_name = "champion" if champ else "reliability-only baseline"
        gain, t = _paired_gain(p_ch, p_ref, y)
        ll_ch, ll_ref = _logloss(p_ch, y), _logloss(p_ref, y)
        if champ:
            rep["logloss_champion"] = ll_ref
        rep.update(n_train=len(train), n_holdout=len(hold), logloss_challenger=ll_ch, logloss_baseline=_logloss(p_base, y),
                   acc_challenger=float(np.mean((p_ch > 0.5) == (y > 0.5))), gain=gain, t=t)
        metrics = dict(ch.metrics, n_holdout=len(hold), logloss_holdout=ll_ch, logloss_ref=ll_ref, ref=ref_name, gain=gain,
                       t=t, acc_holdout=rep["acc_challenger"], holdout_start=hold[0]["ts"], trained_at=now)
        v = db.save_model(name, ch.to_blob(), metrics, now=now)
        rep["version"] = v
        # keep the store small: only the newest few unpromoted challengers keep their weights
        db.x("UPDATE models SET blob=NULL WHERE name=? AND role='challenger' AND version<?", (name, v - 2))
        vs = f"log-loss {ll_ch:.4f} vs {ll_ref:.4f} for the {ref_name} (t = {t:.1f})"
        if not train:
            rep["reason"] += "nothing left to train on after the purge"
        elif len(hold) < 60:
            rep["reason"] += f"holdout too small ({len(hold)} < 60)"
        elif gain >= margin and t >= min_t:
            db.promote_model(name, v)
            rep["promoted"], use = True, ch
            rep["reason"] += "promoted: " + vs
        else:
            rep["reason"] += f"kept the {ref_name}: " + vs

    if use is not None and horizon == "1d":     # notes.edge is the 1d edge; other horizons only report
        recent = db.note_rows("ts>? AND ts<=?", (now - 7 * DAY, now))
        rel = reliability(db)
        for r in recent:
            r["reliability"] = rel[r["source"]]
        directional = [r for r in recent if int(r.get("direction") or 0) != 0]
        edges = dict(zip((r["id"] for r in directional), use.predict(directional))) if directional else {}
        db.set_edge([(float(edges.get(r["id"], 0.0)), r["id"]) for r in recent])
        rep["edges"] = len(recent)
    return rep
