"""The five advisors. Original characters (not real people) written to stand for
five investing styles; the same board as the Risk Board, retrained for crypto.

They sit on two committees and adapt on both, all the time:

1. The exposure board (daily, from the BTC regime brain), ported unchanged:
   each advisor maps regime probabilities to an exposure vote (share of the
   risk budget to deploy). Learning: a per-regime Kelly-style exposure from
   realised BTC returns, blended with each prior by open-mindedness x data
   confidence; Hedge weights from each advisor's loss-averse payoff with a
   fixed-share floor; the chair caps the board at her own vote + 35 points.

2. The pick desk (hourly, from the scanner): each advisor scores every coin
   with a weight vector over the scanner's signals. They start from their
   instincts (a style prior) and learn from what actually happened H hours
   later, by ridge regression pulled toward that prior. Stubborn advisors
   (low open-mindedness) are pulled harder. The data is forgotten slowly
   (exponential decay), so when the market changes character the desk does too.
   Desk weights move by Hedge on each advisor's realised pick payoff.
   The chair vetoes BUY on any coin the risk score rates EXTREME.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .regime import REGIMES, K
from .scanner import SIGNALS, SIGNAL_NAMES


@dataclass
class Advisor:
    key: str
    name: str
    role: str
    color: str
    prior: tuple            # exposure by regime: crash..euphoria
    open_mindedness: float  # how far data may pull them from their instincts
    forward_look: float     # weight on next-step regime forecast
    style: dict             # signal -> prior weight on the pick desk
    lines: dict = field(default_factory=dict)       # exposure stance -> template
    pick_line: str = ""                             # pick-desk template

    def propose(self, probs, next_probs, learned, trust):
        table = (1 - self.open_mindedness * trust) * np.array(self.prior) + self.open_mindedness * trust * learned
        dist = (1 - self.forward_look) * np.asarray(probs) + self.forward_look * np.asarray(next_probs)
        return float(np.clip(dist @ table, 0.0, 1.0))


ADVISORS = [
    Advisor("vera", "Vera Holt", "The Contrarian", "#d9786b",
            (0.55, 0.25, 0.50, 0.40, 0.10), 0.5, 0.2,
            {"mom24": -0.6, "ret1h": -0.4, "rsi": -0.6, "drawdown": -0.4, "mom7d": -0.3, "volume": 0.2, "trend": -0.2},
            {"cut": "Everyone is long and loud. I'm taking chips off: {e:.0%} invested is plenty at {reg} {conf:.0%}.",
             "add": "Fear is where the pay is. {reg} at {conf:.0%} is a place to add slowly, to {e:.0%}.",
             "hold": "Nothing to fade here. {e:.0%} stays; I'm watching for the crowd to tilt."},
            "Everyone is chasing {avoid}. I'd rather own {pick}: {why}."),
    Advisor("kai", "Kai Arden", "The Founder", "#7a8fd9",
            (0.15, 0.35, 0.70, 0.95, 1.00), 0.2, 0.1,
            {"mom24": 0.6, "mom7d": 0.5, "breakout": 0.5, "volume": 0.4, "trend": 0.4, "vol": 0.2},
            {"cut": "Fine, I'll give ground to {e:.0%}. {reg} is not my market and I won't pretend otherwise.",
             "add": "I want more working: {e:.0%}. {reg} at {conf:.0%} won't last forever and I'd rather be early.",
             "hold": "Stay the course at {e:.0%}. Trends reward patience."},
            "{pick} is the one moving: {why}. That's where I want to be."),
    Advisor("imani", "Imani Okoro", "Chair · Risk Officer", "#c9a14a",
            (0.00, 0.15, 0.50, 0.70, 0.50), 0.3, 0.3,
            {"vol": -0.7, "drawdown": 0.3, "beta_tilt": -0.2, "trend": 0.3, "liquidity": 0.3, "volume": -0.1, "mom7d": 0.1},
            {"cut": "Regime reads {reg}, {conf:.0%} sure. I size to survive: {e:.0%}.",
             "add": "Regime reads {reg}, {conf:.0%} sure. I can support {e:.0%}, no more.",
             "hold": "No change in the risk picture. {e:.0%} holds."},
            "{pick} is the cleanest risk on the board: {why}. {avoid} I size at zero."),
    Advisor("marguerite", "Marguerite Lowe", "The Value Compounder", "#8fa06a",
            (0.40, 0.55, 0.60, 0.60, 0.30), 0.4, 0.1,
            {"liquidity": 0.6, "trend": 0.4, "vol": -0.4, "mom7d": -0.3, "rsi": -0.2},
            {"cut": "Prices have run ahead of what I'll pay. Trimming toward {e:.0%} in {reg}.",
             "add": "Cheaper prices are better prices. {reg}: I'll build to {e:.0%}.",
             "hold": "Compounding doesn't need action. {e:.0%} is fine."},
            "{pick}: {why}. I can hold that and sleep."),
    Advisor("idris", "Idris Calder", "The Macro Strategist", "#9b86c9",
            (0.05, 0.25, 0.55, 0.80, 0.60), 0.4, 0.5,
            {"beta_tilt": 0.8, "mom7d": 0.2, "trend": 0.2},
            {"cut": "The forecast leans toward worse regimes. {e:.0%} until the flows turn.",
             "add": "The transition odds improve. Moving to {e:.0%} ahead of the move.",
             "hold": "Transition odds are balanced. {e:.0%} for now."},
            "Bitcoin reads {reg}. {pick} fits that tape: {why}."),
]
CHAIR = "imani"
KEYS = [a.key for a in ADVISORS]


# ---------------------------------------------------------------------------
class ExposureBoard:
    """Ported from the Risk Board's Board: regime -> exposure, learning online."""

    def __init__(self, eta=0.05, kappa=1.0, fixed_share=0.02, decay=0.995, n0=60.0,
                 kelly_gamma=3.0, chair_slack=0.35, rf_annual=0.04 * 252 / 365):
        self.advisors = ADVISORS
        n = len(ADVISORS)
        self.w = np.full(n, 1.0 / n)
        self.score = np.zeros(n)
        self.eta, self.kappa, self.fs = eta, kappa, fixed_share
        self.decay, self.n0, self.gamma = decay, n0, kelly_gamma
        self.chair_slack, self.rf = chair_slack, rf_annual
        self.n_eff = np.zeros(K); self.m = np.zeros(K); self.v = np.full(K, 1e-4)
        self.vol_ewma = 0.03
        self.last_votes = None

    def learned_exposure(self):
        shrink = self.n_eff / (self.n_eff + self.n0)
        kelly = np.clip(self.m / (self.gamma * np.maximum(self.v, 1e-6)), 0, 1)
        return kelly, shrink

    def observe(self, probs_prev, ret):
        """Realised BTC return for the day AFTER `probs_prev`; updates stats and weights."""
        ex = ret - self.rf / 365
        p = np.asarray(probs_prev)
        self.n_eff = self.decay * self.n_eff + p
        wt = p / np.maximum(self.n_eff, 1e-9)
        self.m = (1 - wt) * self.m + wt * ex
        self.v = (1 - wt) * self.v + wt * (ex - self.m) ** 2
        self.vol_ewma = 0.97 * self.vol_ewma + 0.03 * abs(ret)
        if self.last_votes is not None:
            e = np.asarray(self.last_votes)
            u = e * ret - self.kappa * e * max(0.0, -ret)
            u = u / max(self.vol_ewma, 1e-4)
            self.score = 0.98 * self.score + 0.02 * u
            w = self.w * np.exp(self.eta * np.clip(u, -6, 6))
            w /= w.sum()
            self.w = (1 - self.fs) * w + self.fs / len(w)

    def decide(self, probs, next_probs, confidence, current_exposure=None, commit=True):
        kelly, shrink = self.learned_exposure()
        votes = np.array([a.propose(probs, next_probs, kelly, shrink) for a in self.advisors])
        if commit:
            self.last_votes = votes
        raw = float(self.w @ votes)
        ceiling = votes[KEYS.index(CHAIR)] + self.chair_slack
        dissent = float(np.sqrt(self.w @ (votes - raw) ** 2))
        ref = current_exposure if current_exposure is not None else raw
        stances = ["cut" if v < ref - 0.04 else "add" if v > ref + 0.04 else "hold" for v in votes]
        return dict(votes=votes.tolist(), weights=self.w.tolist(), raw=raw, chair_ceiling=float(ceiling),
                    board=float(min(raw, ceiling)), chair_capped=bool(raw > ceiling), dissent=dissent,
                    stances=stances, learned=kelly.tolist(), learned_trust=shrink.tolist(), scores=self.score.tolist())

    def minutes(self, decision, regime_idx, confidence):
        return [dict(key=a.key, name=a.name, stance=s, vote=v, text=a.lines[s].format(
                    e=v, reg=REGIMES[regime_idx].upper(), conf=confidence))
                for a, v, s in zip(self.advisors, decision["votes"], decision["stances"])]

    def state(self):
        return dict(w=self.w.tolist(), score=self.score.tolist(), n_eff=self.n_eff.tolist(), m=self.m.tolist(),
                    v=self.v.tolist(), vol=self.vol_ewma, last=None if self.last_votes is None else list(self.last_votes))

    def load(self, s):
        self.w = np.array(s["w"]); self.score = np.array(s["score"]); self.n_eff = np.array(s["n_eff"])
        self.m = np.array(s["m"]); self.v = np.array(s["v"]); self.vol_ewma = s["vol"]
        self.last_votes = None if s.get("last") is None else np.array(s["last"])


# ---------------------------------------------------------------------------
def _rank(x):
    r = np.empty(len(x)); r[np.argsort(x, kind="stable")] = np.arange(len(x)); return r


def spearman(a, b):
    if len(a) < 3:
        return 0.0
    ra, rb = _rank(np.asarray(a)), _rank(np.asarray(b))
    sa, sb = ra.std(), rb.std()
    return 0.0 if sa == 0 or sb == 0 else float(((ra - ra.mean()) * (rb - rb.mean())).mean() / (sa * sb))


def zscore(x):
    x = np.asarray(x, float)
    sd = x.std()
    return np.zeros_like(x) if sd < 1e-12 else (x - x.mean()) / sd


class PickDesk:
    """Per-advisor coin scoring that learns online from realised forward returns."""

    def __init__(self, decay=0.995, ridge=1000.0, prior_norm=0.15, eta=0.08, kappa=1.0,
                 fixed_share=0.03, top_k=3, ewma=0.03, buy=0.5, watch=0.2, avoid=-0.5):
        self.decay, self.ridge, self.eta, self.kappa, self.fs = decay, ridge, eta, kappa, fixed_share
        self.top_k, self.ewma = top_k, ewma
        self.buy, self.watch, self.avoid = buy, watch, avoid
        d, n = len(SIGNALS), len(ADVISORS)
        P = np.array([[a.style.get(s, 0.0) for s in SIGNALS] for a in ADVISORS], float)
        self.prior = P / np.linalg.norm(P, axis=1, keepdims=True) * prior_norm
        self.alpha = np.array([ridge * (1 - a.open_mindedness) / a.open_mindedness for a in ADVISORS])
        self.Sxx = np.zeros((d, d)); self.Sxy = np.zeros(d); self.n_obs = 0.0
        self.W = self.prior.copy()
        self.w = np.full(n, 1.0 / n)
        self.ic = np.zeros(n); self.hit = np.full(n, 0.5); self.excess = np.zeros(n); self.score = np.zeros(n)
        self.rounds = 0

    # ---- learning ---------------------------------------------------------
    def _refit(self):
        d = len(SIGNALS)
        for a in range(len(ADVISORS)):
            self.W[a] = np.linalg.solve(self.Sxx + self.alpha[a] * np.eye(d), self.Sxy + self.alpha[a] * self.prior[a])

    def learn(self, X, scores, fwd):
        """X: signal matrix at prediction time; scores: (advisors, coins) raw scores
        made then; fwd: realised forward log returns per coin. Returns a summary."""
        X, S, fwd = np.asarray(X, float), np.asarray(scores, float), np.asarray(fwd, float)
        rel = fwd - fwd.mean()
        disp = max(float(rel.std()), 1e-6)
        y = np.clip(rel / disp, -3, 3)
        self.Sxx = self.decay * self.Sxx + X.T @ X
        self.Sxy = self.decay * self.Sxy + X.T @ y
        self.n_obs = self.decay * self.n_obs + len(y)
        k = min(self.top_k, max(1, len(y) // 3))
        u = np.zeros(len(ADVISORS)); out = []
        for a in range(len(ADVISORS)):
            top = np.argsort(-S[a], kind="stable")[:k]
            ex = float(rel[top].mean())
            ic = spearman(S[a], fwd)
            hit = float((rel[top] > 0).mean())
            self.ic[a] = (1 - self.ewma) * self.ic[a] + self.ewma * ic
            self.hit[a] = (1 - self.ewma) * self.hit[a] + self.ewma * hit
            self.excess[a] += ex
            u[a] = (ex - self.kappa * max(0.0, -ex)) / disp
            out.append(dict(key=ADVISORS[a].key, ic=ic, excess=ex, hit=hit))
        self.score = 0.98 * self.score + 0.02 * u
        w = self.w * np.exp(self.eta * np.clip(u, -6, 6))
        w /= w.sum()
        self.w = (1 - self.fs) * w + self.fs / len(w)
        self._refit()
        self.rounds += 1
        return out

    # ---- deciding ---------------------------------------------------------
    def predict(self, X):
        """Raw scores (advisors x coins)."""
        return self.W @ np.asarray(X, float).T

    def decide(self, X, symbols, tiers):
        S = self.predict(X)
        Z = np.array([zscore(s) for s in S])
        combined = self.w @ Z
        coins = []
        for i, sym in enumerate(symbols):
            votes = {ADVISORS[a].key: ("y" if Z[a, i] >= self.buy else "n" if Z[a, i] <= self.avoid else "a")
                     for a in range(len(ADVISORS))}
            c = float(combined[i])
            verdict = self.verdict(c, tiers[i])
            coins.append(dict(symbol=sym, score=c, verdict=verdict, votes=votes,
                              z={ADVISORS[a].key: float(Z[a, i]) for a in range(len(ADVISORS))}))
        return S, coins

    def verdict(self, score, tier):
        if score >= self.buy:
            return "VETOED" if tier == "EXTREME" else "BUY"   # the chair's veto
        return "WATCH" if score >= self.watch else "AVOID" if score <= self.avoid else "NEUTRAL"

    def explain(self, a, X, i, n=2):
        """Top signals pushing advisor a's score on coin i."""
        contrib = self.W[a] * np.asarray(X)[i]
        order = np.argsort(-contrib)
        return [SIGNALS[j] for j in order[:n] if contrib[j] > 0]

    def learned_share(self):
        """How much of each advisor's view now comes from data rather than instinct."""
        return (self.n_obs / (self.n_obs + self.alpha)).tolist()

    def drift(self):
        return (np.linalg.norm(self.W - self.prior, axis=1) / np.linalg.norm(self.prior, axis=1)).tolist()

    def state(self):
        return dict(Sxx=self.Sxx.tolist(), Sxy=self.Sxy.tolist(), n_obs=self.n_obs, w=self.w.tolist(),
                    ic=self.ic.tolist(), hit=self.hit.tolist(), excess=self.excess.tolist(), score=self.score.tolist(),
                    rounds=self.rounds)

    def load(self, s):
        self.Sxx, self.Sxy, self.n_obs = np.array(s["Sxx"]), np.array(s["Sxy"]), s["n_obs"]
        self.w, self.ic, self.hit = np.array(s["w"]), np.array(s["ic"]), np.array(s["hit"])
        self.excess, self.score, self.rounds = np.array(s["excess"]), np.array(s["score"]), s["rounds"]
        if self.n_obs > 0:
            self._refit()


# ---------------------------------------------------------------------------
def describe_signal(sig, m, regime):
    """A signal in words, with the coin's real number."""
    f = {
        "mom24": lambda: f"{m['ret_24h']:+.1%} over 24h",
        "mom7d": lambda: f"{m['ret_7d']:+.1%} on the week",
        "ret1h": lambda: f"{m['ret_1h']:+.1%} in the last hour",
        "trend": lambda: f"{m['trend']:+.1%} versus its 3-day average",
        "breakout": lambda: f"at {m['range_pos']:.0%} of its 7-day range",
        "rsi": lambda: f"RSI {m['rsi']:.0f}",
        "volume": lambda: f"volume {m['vol_surge']:.1f}x normal",
        "vol": lambda: f"volatility {m['vol_24h']:.0%}",
        "drawdown": lambda: f"{m['dd']:+.1%} from its 12-day high",
        "liquidity": lambda: f"${m['usd_vol_24h'] / 1e6:,.0f}M traded in 24h",
        "beta_tilt": lambda: f"beta {m['beta']:.1f} to bitcoin with the brain at {regime}",
    }
    return f[sig]()


def pick_minutes(desk, X, symbols, mets, regime_name, scores):
    """One line per advisor about their favourite and least favourite coin."""
    lines = []
    for a, adv in enumerate(ADVISORS):
        order = np.argsort(-scores[a], kind="stable")
        i, j = int(order[0]), int(order[-1])
        sigs = desk.explain(a, X, i) or [SIGNALS[int(np.argmax(np.abs(desk.W[a])))]]
        why = " and ".join(describe_signal(s, mets[symbols[i]], regime_name) for s in sigs)
        text = adv.pick_line.format(pick=symbols[i].split("-")[0], avoid=symbols[j].split("-")[0], why=why, reg=regime_name)
        lines.append(dict(key=adv.key, name=adv.name, text=text, pick=symbols[i], avoid=symbols[j]))
    return lines


def learned_view(desk, a):
    """What advisor a has learned: weights vs instincts, per signal."""
    return [dict(signal=s, name=SIGNAL_NAMES[s], prior=float(desk.prior[a, j]), now=float(desk.W[a, j]))
            for j, s in enumerate(SIGNALS)]
