"""Hidden-Markov-model regime brain for crypto (BTC-USD daily closes).

Five regimes, ordered worst -> best: crash, bear, neutral, bull, euphoria.
Pure numpy: a diagonal-Gaussian HMM trained with EM, plus a CAUSAL forward
filter so nothing ever sees the future.

Ported from the Risk Board (riskboard/regime.py + regime_crypto.py). Only the
inputs are crypto-sized: calendar-day windows (crypto never closes),
sqrt(365) annualisation, a wider return clip (+-15%) and a 10% volatility floor.

Labelling rule (deterministic): after fitting, states are sorted by
"health" = mean(z-scored 30d return) + mean(z-scored drawdown). Lowest = crash,
highest = euphoria. The HMM finds the clusters; this rule only names them.
"""
from __future__ import annotations

import numpy as np

REGIMES = ["crash", "bear", "neutral", "bull", "euphoria"]
K = len(REGIMES)
FEATURES = ["ret1", "logvol10", "ret30", "dd365", "ext200"]
_HEALTH_COLS = (2, 3)  # ret30, dd365
WARMUP = 365


def make_features(close, start: int = 0) -> np.ndarray:
    """Feature matrix aligned to `close`; rows before WARMUP (or `start`) are NaN.
    Row t only uses closes up to t, so the matrix is causal row by row."""
    c = np.asarray(close, dtype=float)
    n = len(c)
    lr = np.full(n, np.nan)
    lr[1:] = np.diff(np.log(c))
    X = np.full((n, len(FEATURES)), np.nan)
    for t in range(max(WARMUP, start), n):
        w = lr[t - 9 : t + 1]
        vol = max(np.std(w, ddof=1) * np.sqrt(365), 0.10)
        X[t, 0] = np.clip(lr[t], -0.15, 0.15)
        X[t, 1] = np.log(vol)
        X[t, 2] = np.log(c[t] / c[t - 30])
        X[t, 3] = np.log(c[t] / c[t - 364 : t + 1].max())
        X[t, 4] = np.log(c[t] / c[t - 199 : t + 1].mean())
    return X


class HMM:
    """Diagonal-Gaussian HMM, canonical (labelled) parameters."""

    def __init__(self):
        self.pi = self.A = self.mu = self.var = None
        self.fmu = self.fsd = None  # feature standardisation
        self.loglik = None

    # ---- emissions / filtering -------------------------------------------
    def _logb(self, Z):
        d = Z[:, None, :] - self.mu[None]
        return -0.5 * (np.log(2 * np.pi * self.var)[None] + d**2 / self.var[None]).sum(2)

    def standardise(self, X):
        return (X - self.fmu) / self.fsd

    def filter(self, X):
        """Causal filtered posteriors P(state_t | x_1..t); (T,K)."""
        Z = self.standardise(np.asarray(X, float))
        lb = self._logb(Z)
        T = len(Z)
        out = np.empty((T, K))
        a = self.pi.copy()
        for t in range(T):
            pred = a if t == 0 else a @ self.A
            m = lb[t].max()
            a = pred * np.exp(lb[t] - m)
            s = a.sum()
            a = a / s if s > 0 else np.full(K, 1 / K)
            out[t] = a
        return out

    def predict_next(self, alpha):
        return np.asarray(alpha) @ self.A

    # ---- EM ---------------------------------------------------------------
    def _em(self, Z, pi, A, mu, var, n_iter, tol=1e-4):
        T = len(Z)
        prev = -np.inf
        for _ in range(n_iter):
            self.pi, self.A, self.mu, self.var = pi, A, mu, var
            lb = self._logb(Z)
            m = lb.max(1, keepdims=True)
            B = np.exp(lb - m)
            al = np.empty((T, K)); c = np.empty(T)
            a = pi * B[0]; c[0] = a.sum(); al[0] = a / c[0]
            for t in range(1, T):
                a = (al[t - 1] @ A) * B[t]; c[t] = a.sum(); al[t] = a / c[t]
            be = np.empty((T, K)); be[-1] = 1.0
            for t in range(T - 2, -1, -1):
                be[t] = (A @ (B[t + 1] * be[t + 1])) / c[t + 1]
            ll = float(np.log(c).sum() + m.sum())
            g = al * be
            g /= g.sum(1, keepdims=True)
            xi = (al[:-1, :, None] * A[None] * (B[1:] * be[1:])[:, None, :] / c[1:, None, None]).sum(0)
            pi = g[0] + 1e-6; pi /= pi.sum()
            A = xi + 1e-3; A /= A.sum(1, keepdims=True)
            w = g.sum(0)[:, None] + 1e-9
            mu = (g.T @ Z) / w
            var = np.maximum((g.T @ (Z**2)) / w - mu**2, 0.05)
            if abs(ll - prev) < tol * abs(ll):
                prev = ll
                break
            prev = ll
        self.pi, self.A, self.mu, self.var = pi, A, mu, var
        return prev

    def fit(self, X, n_init=4, n_iter=80, seed=0, warm: "HMM | None" = None):
        X = np.asarray(X, float)
        self.fmu, self.fsd = X.mean(0), X.std(0) + 1e-9
        Z = self.standardise(X)
        rng = np.random.default_rng(seed)
        starts = []
        if warm is not None and warm.mu is not None:
            starts.append((warm.pi.copy(), warm.A.copy(), self._remap(warm), warm.var.copy()))
        health = Z[:, _HEALTH_COLS[0]] + Z[:, _HEALTH_COLS[1]]
        order = np.argsort(health)
        for _ in range(n_init):
            # quantile-seeded centres + jitter, sticky transitions
            chunks = np.array_split(order, K)
            mu0 = np.stack([Z[ch].mean(0) for ch in chunks]) + rng.normal(0, 0.15, (K, Z.shape[1]))
            var0 = np.stack([Z[ch].var(0) + 0.1 for ch in chunks])
            A0 = np.full((K, K), 0.05 / (K - 1)); np.fill_diagonal(A0, 0.95)
            starts.append((np.full(K, 1 / K), A0, mu0, var0))
        best, best_ll = None, -np.inf
        for pi, A, mu, var in starts:
            ll = self._em(Z, pi, A, mu, var, n_iter)
            if ll > best_ll:
                best_ll, best = ll, (self.pi, self.A, self.mu, self.var)
        self.pi, self.A, self.mu, self.var = best
        self.loglik = best_ll
        self._canonicalise()
        return self

    def _remap(self, warm):
        # warm params are in another feature scaling; map means through raw space
        raw = warm.mu * warm.fsd + warm.fmu
        return (raw - self.fmu) / self.fsd

    def _canonicalise(self):
        h = self.mu[:, _HEALTH_COLS[0]] + self.mu[:, _HEALTH_COLS[1]]
        p = np.argsort(h)
        self.pi, self.mu, self.var = self.pi[p], self.mu[p], self.var[p]
        self.A = self.A[np.ix_(p, p)]

    # ---- persistence / diagnostics ----------------------------------------
    def to_dict(self):
        return {k: getattr(self, k).tolist() for k in ("pi", "A", "mu", "var", "fmu", "fsd")} | {"loglik": self.loglik}

    @classmethod
    def from_dict(cls, d):
        m = cls()
        for k in ("pi", "A", "mu", "var", "fmu", "fsd"):
            setattr(m, k, np.array(d[k], float))
        m.loglik = d.get("loglik")
        return m

    def describe(self):
        raw_mu = self.mu * self.fsd + self.fmu
        dur = 1.0 / np.maximum(1 - np.diag(self.A), 1e-6)
        return [dict(regime=REGIMES[k], daily_ret_pct=100 * raw_mu[k, 0], ann_vol_pct=100 * float(np.exp(raw_mu[k, 1])),
                     ret30_pct=100 * raw_mu[k, 2], dd365_pct=100 * raw_mu[k, 3], ext200_pct=100 * raw_mu[k, 4],
                     expected_days=float(dur[k])) for k in range(K)]


def _entropy(p):
    return float(-(p * np.log(p + 1e-12)).sum() / np.log(K))


class RegimeTracker:
    """Online wrapper: periodic warm-start refit, causal filtering, hysteresis.

    `update(close)` is the once-a-day step (it moves the hysteresis state).
    `peek(close)` reads the brain on a provisional series (today's live price
    appended) without changing any state, so the 24/7 scanner can show the
    intraday read between daily closes."""

    def __init__(self, refit_every=90, min_train=1095, max_train=2800, switch_prob=0.55,
                 switch_confirm=2, seed=0, filter_window=900, n_init=4, n_iter=80):
        self.refit_every, self.min_train, self.max_train = refit_every, min_train, max_train
        self.switch_prob, self.switch_confirm, self.seed = switch_prob, switch_confirm, seed
        self.filter_window, self.n_init, self.n_iter = filter_window, n_init, n_iter
        self.model: HMM | None = None
        self.last_fit_n = 0
        self.regime = 2
        self._pending, self._pending_n = None, 0
        self.fit_count = 0

    def ready(self, n):
        return n - WARMUP >= self.min_train or self.model is not None

    def _fit(self, valid, n):
        self.model = HMM().fit(valid[-self.max_train:], n_init=self.n_init, n_iter=self.n_iter,
                               seed=self.seed + self.fit_count, warm=self.model)
        self.last_fit_n = n
        self.fit_count += 1

    def _read(self, valid):
        alpha = self.model.filter(valid[-self.filter_window:])[-1]
        return alpha, self.model.predict_next(alpha)

    def update(self, close):
        """close: all daily closes up to and including the latest completed day."""
        close = np.asarray(close, float)
        n = len(close)
        refit = self.model is None or n - self.last_fit_n >= self.refit_every
        # between refits only the filter window is needed, so skip the older rows
        valid = make_features(close, start=0 if refit else n - self.filter_window)[WARMUP:]
        if refit:
            if len(valid) < self.min_train:
                return None
            self._fit(valid, n)
        alpha, nxt = self._read(valid)
        cand = int(alpha.argmax())
        if cand != self.regime:
            if alpha[cand] >= self.switch_prob:
                self._pending_n = self._pending_n + 1 if self._pending == cand else 1
                self._pending = cand
                if self._pending_n >= self.switch_confirm:
                    self.regime, self._pending, self._pending_n = cand, None, 0
            else:
                self._pending, self._pending_n = None, 0
        else:
            self._pending, self._pending_n = None, 0
        return dict(probs=alpha, next_probs=nxt, regime=self.regime, raw_regime=cand,
                    confidence=float(alpha.max()), entropy=_entropy(alpha), refit=refit,
                    table=self.model.describe() if refit else None)

    def peek(self, close):
        """Intraday read on a provisional series; never mutates the tracker."""
        if self.model is None:
            return None
        close = np.asarray(close, float)
        valid = make_features(close, start=len(close) - self.filter_window - 1)[WARMUP:]
        valid = valid[~np.isnan(valid).any(1)]
        alpha, nxt = self._read(valid)
        return dict(probs=alpha, next_probs=nxt, regime=self.regime, raw_regime=int(alpha.argmax()),
                    confidence=float(alpha.max()), entropy=_entropy(alpha))

    def state(self):
        return dict(model=self.model.to_dict() if self.model else None, last_fit_n=self.last_fit_n,
                    regime=self.regime, pending=self._pending, pending_n=self._pending_n, fit_count=self.fit_count)

    def load(self, s):
        self.model = HMM.from_dict(s["model"]) if s.get("model") else None
        self.last_fit_n, self.regime = s["last_fit_n"], s["regime"]
        self._pending, self._pending_n, self.fit_count = s["pending"], s["pending_n"], s["fit_count"]


def features_now(close):
    """Latest feature row in plain words, for the dashboard."""
    x = make_features(np.asarray(close, float), start=len(close) - 1)[-1]
    return dict(ret1=float(np.expm1(x[0])), vol10=float(np.exp(x[1])), ret30=float(np.expm1(x[2])),
                dd365=float(np.expm1(x[3])), ext200=float(np.expm1(x[4])))
