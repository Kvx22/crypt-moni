"""The opportunity scout: the five advisors look beyond the board's own list.

How they "seek", in plain terms:
  1. Every few hours (coins) and after each trading day's close (stocks) the scout pulls market
     data for a wider field of candidates, outside what the board already trades or watches.
  2. It measures each candidate with the same scanner arithmetic the board uses, standardises it
     against the board's own names, and scores it with the advisors' CURRENT learned weights. So
     what they have learned from being right and wrong decides what looks interesting.
  3. A candidate must clear the desk's BUY or WATCH bar, pass a liquidity floor, and not be rated
     EXTREME risk (the chair's veto). Each advisor who likes it says why, in numbers.
  4. The shortlist goes on the dashboard, where you can put a name on the board's watchlist
     (it then joins the universe and the normal approval-gated process), and where Claude can
     attach research notes (`run.py research`). The scout itself never trades.
"""
from __future__ import annotations

from .advisors import ADVISORS, describe_signal
from .feeds import HOUR, FeedError
from .scanner import MIN_BARS, SIGNALS, metrics, regime_tilt, signal_matrix
from .stocks import MIN_DAYS, daily_metrics, day_str
from .universe import AI_EXPOSED, CRYPTO_CANDIDATES, STOCK_CANDIDATES

KEEP = ("price", "ret_1h", "ret_24h", "ret_7d", "vol_24h", "rsi", "vol_surge", "dd", "beta", "usd_vol_24h",
        "risk", "tier", "risk_why", "range_pos", "spark", "z_1h")


class Scout:
    def __init__(self, cfg, store, feed, log):
        sc = cfg.get("scout") or {}
        self.on = sc.get("enabled", True)
        self.crypto = list(sc.get("crypto", CRYPTO_CANDIDATES))
        self.stocks = list(sc.get("stocks", STOCK_CANDIDATES))
        self.every = float(sc.get("crypto_hours", 3)) * HOUR
        self.top = int(sc.get("top", 8))
        self.min_usd = dict(crypto=float(sc.get("min_usd_volume_crypto", 5e6)), stock=float(sc.get("min_usd_volume_stock", 2e7)))
        self.ai = set(sc.get("ai_exposed", AI_EXPOSED))
        self.s, self.feed, self.log = store, feed, log
        self.proxy = cfg.get("market_proxy", "BTC-USD")

    # ------------------------------------------------------------------ helpers
    def known(self):
        """Symbols already on the board's watchlist (adopted into the universe)."""
        return {w["symbol"] for w in self.s.kv_get("watchlist", [])}

    def _rank(self, asset, uni_mets, cand_mets, desk, probs, regime, now, scanned):
        if len(uni_mets) < 4 or not cand_mets:
            self._save(asset, now, scanned, [])
            return []
        syms = list(uni_mets) + list(cand_mets)
        mets = {**uni_mets, **cand_mets}
        X = signal_matrix(mets, syms, regime_tilt(probs))
        S, coins = desk.decide(X, syms, [mets[s]["tier"] for s in syms])
        idx = {s: i for i, s in enumerate(syms)}
        items = []
        for c in coins:
            sym = c["symbol"]
            if sym not in cand_mets or c["verdict"] not in ("BUY", "WATCH"):
                continue
            m = cand_mets[sym]
            if m["usd_vol_24h"] < self.min_usd[asset]:
                continue
            i = idx[sym]
            why = []
            for a, adv in enumerate(ADVISORS):
                if c["votes"][adv.key] != "y":
                    continue
                sigs = desk.explain(a, X, i) or []
                if sigs:
                    # scores are relative to the field, so say where each number sits against the other names
                    rel = lambda g: " (above the field)" if X[i, SIGNALS.index(g)] > 0 else " (below the field)"
                    why.append(dict(key=adv.key, name=adv.name, text=" and ".join(describe_signal(g, m, regime) + rel(g) for g in sigs)))
            items.append(dict(symbol=sym, asset=asset, score=c["score"], verdict=c["verdict"], votes=c["votes"], why=why,
                              ai_exposed=sym in self.ai, **{k: m[k] for k in KEEP if k in m}))
        items.sort(key=lambda x: -x["score"])
        items = items[: self.top]
        self._save(asset, now, scanned, items)
        if items:
            lead = ", ".join(f"{x['symbol'].split('-')[0]} {x['score']:+.2f}" for x in items[:3])
            self.log(now, "scout", "info", f"Scout ({asset}): {len(items)} candidate(s) cleared the desk's bar",
                     f"Top: {lead}. Scanned {scanned} names outside the board's list, scored with the advisors' current learned weights.")
        return items

    def _save(self, asset, now, scanned, items):
        o = self.s.kv_get("opps", {})
        o[asset] = dict(ts=now, scanned=scanned, items=items)
        self.s.kv_set("opps", o)

    # ------------------------------------------------------------------ crypto (hourly bars)
    def scan_crypto(self, now, bars, uni_mets, desk, probs, regime):
        if not self.on or now - (self.s.kv_get("scout_crypto_ts") or 0) < self.every:
            return None
        self.s.kv_set("scout_crypto_ts", now)
        skip = set(uni_mets) | self.known()
        btc = bars.get(self.proxy)
        cm, scanned = {}, 0
        for sym in self.crypto:
            if sym in skip:
                continue
            try:
                b = self.feed.candles(sym, HOUR)
            except FeedError:
                continue
            scanned += 1
            if len(b) >= MIN_BARS:
                m = metrics(b, btc)
                if m:
                    cm[sym] = m
        return self._rank("crypto", uni_mets, cm, desk, probs, regime, now, scanned)

    # ------------------------------------------------------------------ stocks (daily bars)
    def scan_stocks(self, now, uni_mets, spy, desk, probs, regime, day):
        if not self.on or self.s.kv_get("scout_stock_day") == day:
            return None
        self.s.kv_set("scout_stock_day", day)
        skip = set(uni_mets) | self.known()
        cm, scanned = {}, 0
        for sym in self.stocks:
            if sym in skip:
                continue
            try:
                bars = self.feed.daily(sym, now, "1y")
            except FeedError:
                continue
            scanned += 1
            if len(bars) >= MIN_DAYS:
                m = daily_metrics([day_str(b.ts) for b in bars], [b.close for b in bars], spy, [b.volume for b in bars])
                if m:
                    cm[sym] = m
        return self._rank("stock", uni_mets, cm, desk, probs, regime, now, scanned)
