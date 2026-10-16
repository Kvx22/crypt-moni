"""The scanner: turns hourly bars into per-coin metrics, a deterministic risk
score, alerts, and the standardised signal matrix the advisors learn from.

Two reads per coin:
  * live    closed bars + the latest ticker price (what the board shows, alerts)
  * closed  closed bars only (what the advisors predict from and learn on), so
            learning never mixes a half-finished hour into its data
The risk score is fixed arithmetic, not learned: the advisors can disagree
with it, but they cannot talk it down.
"""
from __future__ import annotations

import math

import numpy as np

HOURS_YEAR = 24 * 365
MIN_BARS = 170          # 7 days + a little
SIGNALS = ["mom24", "mom7d", "ret1h", "trend", "breakout", "rsi", "volume", "vol", "drawdown", "liquidity", "beta_tilt"]
SIGNAL_NAMES = {
    "mom24": "24h momentum", "mom7d": "7-day momentum", "ret1h": "last-hour move", "trend": "trend vs 3-day average",
    "breakout": "position in 7-day range", "rsi": "RSI", "volume": "volume surge", "vol": "volatility",
    "drawdown": "drawdown depth", "liquidity": "liquidity", "beta_tilt": "bitcoin beta x regime",
}
TIERS = [(25, "LOW"), (45, "ELEVATED"), (65, "HIGH"), (101, "EXTREME")]


def ema(x, span):
    a = 2.0 / (span + 1)
    out = np.empty(len(x))
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = a * x[i] + (1 - a) * out[i - 1]
    return out


def rsi(close, n=14):
    d = np.diff(close)
    if len(d) < n + 1:
        return 50.0
    up, dn = np.clip(d, 0, None), np.clip(-d, 0, None)
    au, ad = up[:n].mean(), dn[:n].mean()
    for i in range(n, len(d)):  # Wilder smoothing
        au = (au * (n - 1) + up[i]) / n
        ad = (ad * (n - 1) + dn[i]) / n
    if ad == 0:
        return 100.0 if au > 0 else 50.0
    return float(100 - 100 / (1 + au / ad))


def metrics(bars, btc_bars=None, live_price=None):
    """Plain-number read of one coin. `bars` are closed hourly bars, oldest first."""
    if len(bars) < MIN_BARS:
        return None
    c = np.array([b.close for b in bars], float)
    v = np.array([b.volume for b in bars], float)
    if live_price:
        c = np.append(c, live_price)
        v = np.append(v, v[-1])
    p = c[-1]
    r = np.diff(np.log(c))
    sd7 = max(float(np.std(r[-168:], ddof=1)), 1e-6)
    hi7, lo7 = float(c[-169:-1].max()), float(c[-169:-1].min())
    beta = corr = 0.0
    if btc_bars is not None and len(btc_bars) >= MIN_BARS:
        bm = {b.ts: b.close for b in btc_bars}
        ts = [b.ts for b in bars]
        common = [i for i, t in enumerate(ts) if t in bm and i > 0 and ts[i - 1] in bm][-168:]
        if len(common) > 48:
            rc = np.array([math.log(bars[i].close / bars[i - 1].close) for i in common])
            rb = np.array([math.log(bm[ts[i]] / bm[ts[i - 1]]) for i in common])
            vb = rb.var()
            if vb > 0:
                beta = float(np.cov(rc, rb, ddof=0)[0, 1] / vb)
                sc = rc.std()
                corr = float(np.cov(rc, rb, ddof=0)[0, 1] / (sc * math.sqrt(vb))) if sc > 0 else 0.0
    vol24 = float(np.std(r[-24:], ddof=1) * math.sqrt(HOURS_YEAR))
    vol7 = float(sd7 * math.sqrt(HOURS_YEAR))
    usd24 = float((v[-24:] * c[-24:]).sum())
    m = dict(
        price=float(p),
        ret_1h=float(c[-1] / c[-2] - 1), ret_24h=float(c[-1] / c[-25] - 1), ret_7d=float(c[-1] / c[-169] - 1),
        vol_24h=vol24, vol_7d=vol7, vol_ratio=vol24 / vol7 if vol7 > 0 else 1.0,
        rsi=rsi(c[-120:]),
        vol_surge=float(v[-4:].mean() / max(v[-168:].mean(), 1e-12)),
        trend=float(p / ema(c[-240:], 72)[-1] - 1),
        hi7=hi7, lo7=lo7, range_pos=float(np.clip((p - lo7) / (hi7 - lo7), -0.25, 1.25)) if hi7 > lo7 else 0.5,
        dd=float(p / c[-300:].max() - 1),
        beta=beta, corr=corr, usd_vol_24h=usd24,
        tail=float(np.abs(r[-24:]).max() / sd7), z_1h=float(r[-1] / sd7),
        spark=[float(x) for x in c[-72:]],
    )
    m["risk"], m["tier"], m["risk_why"] = risk_score(m)
    return m


def risk_score(m):
    """0-100, higher = riskier. Fixed weights; returns (score, tier, top drivers)."""
    liq = math.log10(max(m["usd_vol_24h"], 1.0))
    parts = {
        "volatility": (0.30, np.clip((m["vol_24h"] - 0.35) / 1.25, 0, 1)),
        "drawdown": (0.20, np.clip(-m["dd"] / 0.35, 0, 1)),
        "vol expanding": (0.15, np.clip((m["vol_ratio"] - 1) / 1.5, 0, 1)),
        "tail moves": (0.15, np.clip((m["tail"] - 3) / 5, 0, 1)),
        "thin liquidity": (0.10, np.clip((7.5 - liq) / 2.5, 0, 1)),
        "high beta": (0.10, np.clip(m["beta"] - 1.0, 0, 1)),
    }
    score = float(100 * sum(w * x for w, x in parts.values()))
    tier = next(name for lim, name in TIERS if score < lim)
    why = [k for k, (w, x) in sorted(parts.items(), key=lambda kv: -kv[1][0] * kv[1][1]) if w * x >= 0.03][:3]
    return round(score, 1), tier, why


def tier_rank(t):
    return [n for _, n in TIERS].index(t)


def regime_tilt(probs):
    """-1 (all crash) .. +1 (all euphoria): how risk-on the brain reads."""
    return float(np.asarray(probs) @ np.array([-1.0, -0.5, 0.0, 0.5, 1.0]))


def signal_matrix(mets, symbols, tilt):
    """Cross-sectionally standardised signals (rows = symbols, cols = SIGNALS).
    Standardising across coins at each hour keeps the learning about *which coin*
    beats the others, not about whether crypto as a whole went up."""
    raw = []
    for s in symbols:
        m = mets[s]
        raw.append([math.log1p(m["ret_24h"]), math.log1p(m["ret_7d"]), math.log1p(m["ret_1h"]), math.log1p(m["trend"]),
                    m["range_pos"] - 0.5, (m["rsi"] - 50) / 50, math.log(max(m["vol_surge"], 1e-6)),
                    math.log(max(m["vol_24h"], 1e-6)), math.log1p(max(m["dd"], -0.99)),
                    math.log10(max(m["usd_vol_24h"], 1.0)), m["beta"]])
    X = np.array(raw, float)
    mu, sd = X.mean(0), X.std(0)
    Z = np.where(sd > 1e-12, (X - mu) / np.where(sd > 1e-12, sd, 1), 0.0)
    Z = np.clip(Z, -3, 3)
    Z[:, -1] = Z[:, -1] * tilt  # beta only matters in the direction the regime leans
    return Z


def alerts(sym, m, prev):
    """Alert candidates from one live read. `prev` = previous live read (or None).
    Returns [(kind, severity, title, body)]; the engine de-duplicates with cooldowns."""
    out = []
    name = sym.split("-")[0]
    z = m["z_1h"]
    if abs(z) >= 3:
        out.append(("move", "critical" if abs(z) >= 5 else "warn",
                    f"{name} {m['ret_1h']:+.1%} in the last hour", f"{z:+.1f} sigma vs its 7-day hourly volatility"))
    if m["price"] > m["hi7"] and (prev is None or prev["price"] <= prev["hi7"]):
        out.append(("breakout", "info", f"{name} broke above its 7-day high", f"${m['price']:,.4g} vs high ${m['hi7']:,.4g}"))
    if m["price"] < m["lo7"] and (prev is None or prev["price"] >= prev["lo7"]):
        out.append(("breakdown", "warn", f"{name} broke below its 7-day low", f"${m['price']:,.4g} vs low ${m['lo7']:,.4g}"))
    if m["vol_surge"] >= 3:
        out.append(("volume", "info", f"{name} volume {m['vol_surge']:.1f}x its weekly average", "last 4 hours vs 7-day mean"))
    if m["rsi"] >= 80:
        out.append(("rsi", "info", f"{name} overbought, RSI {m['rsi']:.0f}", "hourly RSI(14) at or above 80"))
    if m["rsi"] <= 20:
        out.append(("rsi", "info", f"{name} oversold, RSI {m['rsi']:.0f}", "hourly RSI(14) at or below 20"))
    if prev is not None and tier_rank(m["tier"]) > tier_rank(prev["tier"]) and m["tier"] in ("HIGH", "EXTREME"):
        out.append(("risk", "critical" if m["tier"] == "EXTREME" else "warn", f"{name} risk now {m['tier']} ({m['risk']:.0f})",
                    "drivers: " + ", ".join(m["risk_why"])))
    return out
