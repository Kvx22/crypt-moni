"""The advisors' research desk: how each of the five advisors weighs what the research
service found, learned the same way the pick desk learns the scanner's signals.

In plain terms:
  1. The research service boils each symbol's fresh notes down to six numbers in [-1, 1]
     (catalysts, sentiment, fundamentals, market structure, risk flags, coverage).
  2. Each advisor starts from an instinct about those six (Vera fades the crowd, Kai chases
     catalysts, Imani listens hardest to risk flags...) and learns from what prices did
     next, by ridge regression pulled toward that instinct. Stubborn advisors are pulled
     harder; old data is forgotten slowly.
  3. The engine adds a modest share of each advisor's research z-score to their scanner
     z-score (`overlay`). Until research arrives the overlay is exactly zero, and an
     advisor who likes a name nobody has researched is heard a little less.
  4. Credible risk flags come off the combined score as a CAPPED penalty (`combine`),
     never a veto; and wherever an advisor likes a thinly covered name the desk asks the
     research service to look harder (`thin_requests`).

This runs inside the board's engine: everything here is cheap and tolerates missing,
partial or malformed research by treating it as "no research" (zeros).
"""
from __future__ import annotations

import math

import numpy as np

from ..advisors import ADVISORS, zscore
from .taxonomy import RESEARCH_SIGNALS

RESEARCH_STYLE = {
    "vera":       {"r_catalyst": 0.2, "r_sentiment": -0.6, "r_fundamental": 0.2, "r_structure": -0.4, "r_risk": 0.3, "r_coverage": -0.1},
    "kai":        {"r_catalyst": 0.7, "r_sentiment": 0.4, "r_fundamental": 0.1, "r_structure": 0.2, "r_risk": 0.2, "r_coverage": 0.3},
    "imani":      {"r_catalyst": 0.2, "r_sentiment": -0.1, "r_fundamental": 0.3, "r_structure": 0.1, "r_risk": 0.9, "r_coverage": 0.2},
    "marguerite": {"r_catalyst": 0.3, "r_sentiment": -0.3, "r_fundamental": 0.7, "r_structure": 0.0, "r_risk": 0.4, "r_coverage": 0.1},
    "idris":      {"r_catalyst": 0.3, "r_sentiment": 0.0, "r_fundamental": 0.2, "r_structure": 0.5, "r_risk": 0.3, "r_coverage": 0.1},
}
D = len(RESEARCH_SIGNALS)
NO_VARIATION = 1e-6     # features that differ across coins by less than this carry no ranking information


def _num(v, default=0.0):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return f if math.isfinite(f) else default


def _avg_rank(x):
    """Ranks 0..n-1 with ties sharing their average rank."""
    x = np.asarray(x, float)
    r = np.empty(len(x)); r[np.argsort(x, kind="stable")] = np.arange(len(x))
    _, inv = np.unique(x, return_inverse=True)
    return (np.bincount(inv, weights=r) / np.bincount(inv))[inv]


def spearman(a, b):
    """Rank correlation, tie-aware, over the pairs where both values are finite (0 with < 3 pairs)."""
    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return 0.0
    ra, rb = _avg_rank(a[ok]), _avg_rank(b[ok])
    sa, sb = ra.std(), rb.std()
    return 0.0 if sa < 1e-12 or sb < 1e-12 else float(((ra - ra.mean()) * (rb - rb.mean())).mean() / (sa * sb))


def research_matrix(views, symbols):
    """(coins x 6) research features from the view; missing or junk values are 0, all clipped to [-1, 1]."""
    R = np.zeros((len(symbols), D))
    views = views if isinstance(views, dict) else {}
    for i, s in enumerate(symbols):
        v = views.get(s)
        f = v.get("features") if isinstance(v, dict) else None
        if isinstance(f, dict):
            R[i] = [_num(f.get(sig)) for sig in RESEARCH_SIGNALS]
    return np.clip(R, -1.0, 1.0)


def _has_variation(R):
    return R.ndim == 2 and R.shape[0] >= 2 and R.shape[1] == D and float(np.ptp(R, axis=0).max()) >= NO_VARIATION


class ResearchDesk:
    """Per-advisor weights over the six research features, learning online from forward returns."""

    def __init__(self, decay=0.995, ridge=500.0, prior_norm=0.15, ewma=0.03):
        self.decay, self.ridge, self.ewma = decay, ridge, ewma
        P = np.array([[RESEARCH_STYLE.get(a.key, {}).get(s, 0.0) for s in RESEARCH_SIGNALS] for a in ADVISORS], float)
        norm = np.linalg.norm(P, axis=1, keepdims=True)
        self.prior = np.divide(P, norm, out=np.zeros_like(P), where=norm > 0) * prior_norm
        self.alpha = np.array([ridge * (1 - a.open_mindedness) / a.open_mindedness for a in ADVISORS])
        self.Sxx = np.zeros((D, D)); self.Sxy = np.zeros(D); self.n_obs = 0.0
        self.W = self.prior.copy()
        self.ic = np.zeros(len(ADVISORS))
        self.rounds = 0

    def _refit(self):
        for a in range(len(ADVISORS)):
            self.W[a] = np.linalg.solve(self.Sxx + self.alpha[a] * np.eye(D), self.Sxy + self.alpha[a] * self.prior[a])

    # ---- scoring ------------------------------------------------------------
    def scores(self, R):
        """Raw research scores (advisors x coins)."""
        R = np.nan_to_num(np.asarray(R, float).reshape(-1, D))
        return self.W @ R.T

    def zr(self, R):
        """Row-wise z-scores of the research scores; all zeros while research has no spread across coins."""
        R = np.nan_to_num(np.asarray(R, float).reshape(-1, D))
        if not _has_variation(R):
            return np.zeros((len(ADVISORS), R.shape[0]))
        return np.array([zscore(s) for s in self.scores(R)])

    # ---- learning -----------------------------------------------------------
    def learn(self, R, fwd):
        """R: research features at decision time (coins x 6); fwd: realised forward log returns per
        coin. The IC is measured with the weights held BEFORE this update (no look-ahead). Coins
        with a non-finite return are dropped; a round with < 3 usable coins or no research spread
        teaches nothing and returns None. Otherwise returns [dict(key, ic)] per advisor."""
        R = np.nan_to_num(np.asarray(R, float).reshape(-1, D))
        fwd = np.asarray(fwd, float).ravel()
        if len(fwd) != R.shape[0]:
            return None
        ok = np.isfinite(fwd)
        R, fwd = R[ok], fwd[ok]
        if len(fwd) < 3 or not _has_variation(R):
            return None
        S = self.scores(R)
        rel = fwd - fwd.mean()
        y = np.clip(rel / max(float(rel.std()), 1e-6), -3, 3)
        self.Sxx = self.decay * self.Sxx + R.T @ R
        self.Sxy = self.decay * self.Sxy + R.T @ y
        self.n_obs = self.decay * self.n_obs + len(y)
        out = []
        for a, adv in enumerate(ADVISORS):
            ic = spearman(S[a], fwd)
            self.ic[a] = (1 - self.ewma) * self.ic[a] + self.ewma * ic
            out.append(dict(key=adv.key, ic=ic))
        self._refit()
        self.rounds += 1
        return out

    def learned_share(self):
        """How much of each advisor's research view now comes from data rather than instinct."""
        return (self.n_obs / (self.n_obs + self.alpha)).tolist()

    def drift(self):
        pn = np.linalg.norm(self.prior, axis=1)
        return (np.linalg.norm(self.W - self.prior, axis=1) / np.maximum(pn, 1e-12)).tolist()

    def explain(self, a, R, i, n=2):
        """The research signals (names from RESEARCH_SIGNALS) pushing advisor a's score UP on coin i."""
        contrib = self.W[a] * np.nan_to_num(np.asarray(R, float).reshape(-1, D)[i])
        order = np.argsort(-contrib, kind="stable")
        return [RESEARCH_SIGNALS[j] for j in order[:n] if contrib[j] > 0]

    # ---- persistence ----------------------------------------------------------
    def state(self):
        return dict(Sxx=self.Sxx.tolist(), Sxy=self.Sxy.tolist(), n_obs=self.n_obs, ic=self.ic.tolist(), rounds=self.rounds)

    def load(self, s):
        """Restore from state(); anything malformed leaves the desk at its instincts."""
        try:
            Sxx, Sxy = np.array(s["Sxx"], float), np.array(s["Sxy"], float)
            ic, n_obs = np.array(s.get("ic", [0.0] * len(ADVISORS)), float), _num(s.get("n_obs"))
            if Sxx.shape != (D, D) or Sxy.shape != (D,) or ic.shape != (len(ADVISORS),) \
                    or not (np.isfinite(Sxx).all() and np.isfinite(Sxy).all() and np.isfinite(ic).all()):
                return False
        except (TypeError, ValueError, KeyError, AttributeError):
            return False
        keep = (self.Sxx, self.Sxy, self.n_obs, self.ic, self.W.copy())
        self.Sxx, self.Sxy, self.n_obs, self.ic = Sxx, Sxy, max(n_obs, 0.0), ic
        try:
            self._refit()
        except np.linalg.LinAlgError:
            self.Sxx, self.Sxy, self.n_obs, self.ic, self.W = keep
            return False
        self.rounds = int(_num(s.get("rounds")))
        return True


# ---------------------------------------------------------------------------
def overlay(Z, zr, weight=0.35, thin=None, thin_discount=0.85, buy=0.5):
    """Scanner z-scores plus a share of research z-scores. Where a coin is thinly researched and
    the advisor already rates it a BUY, their (combined) score is discounted. Returns a new array."""
    Z = np.asarray(Z, float)
    Z2 = Z.copy()
    if zr is not None:
        zr = np.asarray(zr, float)
        if zr.shape == Z.shape:
            Z2 = Z2 + weight * np.nan_to_num(zr)
    if thin is not None and len(thin) == Z.shape[1]:
        mask = (Z >= buy) & np.asarray(thin, bool)[None, :]
        Z2[mask] *= thin_discount
    return Z2


def combine(w, Z, penalty=None):
    """The desk's combined score per coin: weighted advisor z-scores minus the research penalty."""
    c = np.asarray(w, float) @ np.asarray(Z, float)
    if penalty is not None:
        p = np.nan_to_num(np.asarray(penalty, float).ravel())
        if p.shape == c.shape:
            c = c - p
    return c


def _n_independent(view):
    return int(_num(view.get("n_independent"))) if isinstance(view, dict) else 0


def thin_requests(Z, symbols, views, buy=0.5, thin_sources=2):
    """Every (advisor, coin) the advisor rates a BUY while fewer than `thin_sources` independent
    sources cover it: a request for more evidence, strongest conviction first."""
    Z = np.asarray(Z, float)
    views = views if isinstance(views, dict) else {}
    out = []
    for i, sym in enumerate(list(symbols)[:Z.shape[1] if Z.ndim == 2 else 0]):
        n = _n_independent(views.get(sym))
        if n >= thin_sources:
            continue
        name = sym.split("-")[0]
        cover = "no independent source covers it yet" if n == 0 else \
            f"only {n} independent source{'s cover' if n > 1 else ' covers'} it"
        for a, adv in enumerate(ADVISORS):
            z = float(Z[a, i])
            if math.isfinite(z) and z >= buy:
                out.append(dict(symbol=sym, advisor_index=a, advisor=adv.key, z=z, n_independent=n,
                                question=f"{adv.name.split()[0]} likes {name} (z {z:+.1f}) but {cover}: "
                                         "check for news, filings and risk"))
    out.sort(key=lambda r: -r["z"])
    return out
