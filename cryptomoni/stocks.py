"""The stock book: US stocks and ETFs on the same advisors and the same safety layer,
run on a stock clock.

Stocks differ from coins in ways that matter, so they get their own pipeline instead of
being squeezed into the hourly crypto one:
  * they trade only 9:30-16:00 ET, Mon-Fri, so decisions use COMPLETED DAILY bars and
    orders are only proposed while the market is open;
  * their regime brain reads SPY (trading-day features), not bitcoin;
  * the five advisors keep a separate stock exposure board and a separate stock pick desk,
    so what they learn about coins never bends what they learn about stocks.
The crypto engine hands this class the clock, the feed and the store, and merges its target
weights with the crypto ones (the stock sleeve gets `share` of the account).
"""
from __future__ import annotations

import csv
import math
import os

import numpy as np

from .advisors import ADVISORS, ExposureBoard, PickDesk, pick_minutes
from .allocation import Allocator
from .feeds import DAY, FeedError
from .models import Bar
from .regime import REGIMES, RegimeTracker, features_now
from .scanner import ema, regime_tilt, risk_score, rsi, signal_matrix

MIN_DAYS = 130


def read_closes(path):
    if not os.path.exists(path):
        return {}
    with open(path, newline="") as f:
        return {r["date"]: float(r["close"]) for r in csv.DictReader(f) if r.get("close")}


def day_str(ts):
    import datetime as dt
    return dt.datetime.fromtimestamp(ts, dt.timezone.utc).date().isoformat()


def daily_metrics(dates, closes, spy, volumes=None, live=None):
    """Same keys as scanner.metrics, computed from daily bars. `closes` are completed closes
    (oldest first, aligned to `dates`); `spy` maps date -> SPY close for beta; `live` is the
    current price. Horizons: ret_24h = last completed day, ret_7d = 5 trading days,
    ret_1h = today's move since the last close (0 without a live price)."""
    if len(closes) < MIN_DAYS:
        return None
    c = np.asarray(closes, float)
    last = c[-1]
    p = float(live) if live else float(last)
    r = np.diff(np.log(c))
    sd60 = max(float(np.std(r[-60:], ddof=1)), 1e-6)
    sd20 = float(np.std(r[-20:], ddof=1))
    v = np.asarray(volumes, float) if volumes is not None and len(volumes) == len(c) else None
    have_vol = v is not None and v[-20:].sum() > 0
    usd = float((v[-20:] * c[-20:]).mean()) if have_vol else 5e9
    hi, lo = float(c[-21:].max()), float(c[-21:].min())
    beta = corr = 0.0
    common = [i for i, d in enumerate(dates) if d in spy][-61:]
    if len(common) > 30:
        rc = np.diff(np.log([closes[i] for i in common]))
        rb = np.diff(np.log([spy[dates[i]] for i in common]))
        vb = rb.var()
        if vb > 0:
            beta = float(np.cov(rc, rb, ddof=0)[0, 1] / vb)
            sc = rc.std()
            corr = float(np.cov(rc, rb, ddof=0)[0, 1] / (sc * math.sqrt(vb))) if sc > 0 else 0.0
    m = dict(
        price=p, ret_1h=float(p / last - 1) if live else 0.0, ret_24h=float(c[-1] / c[-2] - 1), ret_7d=float(c[-1] / c[-6] - 1),
        vol_24h=sd20 * math.sqrt(252), vol_7d=sd60 * math.sqrt(252), vol_ratio=sd20 / sd60, rsi=rsi(c[-60:]),
        vol_surge=float(v[-1] / max(v[-20:].mean(), 1e-9)) if have_vol else 1.0,
        trend=float(p / ema(c[-80:], 20)[-1] - 1), hi7=hi, lo7=lo,
        range_pos=float(np.clip((p - lo) / (hi - lo), -0.25, 1.25)) if hi > lo else 0.5,
        dd=float(p / max(c[-252:].max(), p) - 1), beta=beta, corr=corr, usd_vol_24h=usd,
        tail=float(np.abs(r[-5:]).max() / sd60), z_1h=float(math.log(p / last) / sd60) if live else 0.0,
        spark=[float(x) for x in c[-60:]], asset="stock",
    )
    m["risk"], m["tier"], m["risk_why"] = risk_score(m, vol_lo=0.12, vol_span=0.45, dd_span=0.25)
    return m


class StockBook:
    def __init__(self, cfg, store, feed, log):
        sc = cfg.get("stocks") or {}
        self.symbols = list(sc["symbols"])
        self.proxy = sc.get("proxy", "SPY")
        if self.proxy not in self.symbols:
            self.symbols.insert(0, self.proxy)
        self.share = float(sc.get("share", 0.4))
        self.csv_dir = sc.get("csv_dir", "data/stocks")
        self.horizon = int(sc.get("horizon_days", 2))
        self.warmup_days = int(sc.get("warmup_days", 250))
        self.propose_at = sc.get("propose_at", "10:00")
        self.conv_alpha = 1.0 / max(1.0, sc.get("conviction_days", 3))
        self.s, self.feed, self.log = store, feed, log
        tr = dict(refit_every=63, min_train=504, max_train=2500, filter_window=600, profile="stock")
        tr.update(sc.get("tracker", {}))
        self.tracker = RegimeTracker(**tr)
        if store.kv_get("stock_tracker"):
            self.tracker.load(store.kv_get("stock_tracker"))
        bd = dict(rf_annual=0.04 * 365 / 252)
        bd.update(sc.get("board", {}))
        self.board = ExposureBoard(**bd)
        if store.kv_get("stock_board"):
            self.board.load(store.kv_get("stock_board"))
        dk = dict(sc.get("desk", {}))
        dk.setdefault("decay", 0.98)      # days, not hours: roughly a 2-3 month memory
        dk.setdefault("ridge", 300.0)
        self.desk = PickDesk(**dk)
        if store.kv_get("stock_desk"):
            self.desk.load(store.kv_get("stock_desk"))
        al = dict(max_invested=0.95, neutral_exposure=0.55, target_vol=0.16, max_step=0.10, max_names=4,
                  core=self.proxy, caps={self.proxy: 0.5}, default_cap=0.25, stickiness=0.3)
        al.update(sc.get("allocator", {}))
        self.alloc = Allocator(**al)
        self.brain = store.kv_get("stock_brain") or {}
        self.errors = []
        self.closes, self.volumes, self.fetched = {}, {}, {}
        for s in self.symbols:
            d = read_closes(os.path.join(self.csv_dir, f"{s}.csv"))
            vol = {}
            for b in store.get_bars(s, DAY, 100000):
                d[day_str(b.ts)] = b.close
                vol[day_str(b.ts)] = b.volume
            self.closes[s] = dict(sorted(d.items()))
            self.volumes[s] = vol

    # ------------------------------------------------------------------ data
    def sync(self, now):
        """Pull completed daily bars; a failing symbol is skipped and retried, never fatal."""
        self.errors = []
        for s in self.symbols:
            try:   # a name added to the watchlist has no history yet: pull two years once
                bars = self.feed.daily(s, now, "2y" if len(self.closes[s]) < MIN_DAYS + 20 else "3mo")
            except FeedError as e:
                self.errors.append(f"{s} daily: {e}")
                continue
            last = next(reversed(self.closes[s])) if self.closes[s] else ""
            new = [b for b in bars if day_str(b.ts) > last]
            if new:
                self.s.put_bars(s, DAY, new)
                for b in new:
                    self.closes[s][day_str(b.ts)] = b.close
                    self.volumes[s][day_str(b.ts)] = b.volume

    def add(self, symbol):
        """Adopt a watchlisted stock into the book (its history arrives with the next sync)."""
        if symbol not in self.symbols:
            self.symbols.append(symbol)
            self.closes[symbol], self.volumes[symbol] = {}, {}

    def quotes(self, now):
        out = {}
        for s in self.symbols:
            try:
                out[s] = self.feed.ticker(s)
            except FeedError as e:
                self.errors.append(f"{s} quote: {e}")
                if self.closes[s]:   # last close with an old timestamp: shown, never traded on
                    out[s] = (next(reversed(self.closes[s].values())), now - 86400)
        return out

    def _series(self, s, upto=None):
        d = self.closes[s]
        dates = [x for x in d if upto is None or x <= upto]
        return dates, [d[x] for x in dates]

    # ------------------------------------------------------------------ brain
    def _brain_day(self, dates, closes, i, now, quiet=False):
        prev = self.brain
        if prev.get("close") and prev.get("date") != dates[i]:
            self.board.observe(np.array(prev["probs"]), closes[i] / prev["close"] - 1)
        st = self.tracker.update(np.array(closes[: i + 1]))
        if st is None:
            return False
        dec = self.board.decide(st["probs"], st["next_probs"], st["confidence"], self.s.kv_get("stock_e"))
        if not quiet:
            if prev.get("regime") is not None and st["regime"] != prev["regime"]:
                self.log(now, "regime", "warn" if st["regime"] < prev["regime"] else "info",
                         f"Stock regime {REGIMES[prev['regime']].upper()} -> {REGIMES[st['regime']].upper()}",
                         f"{self.proxy} close {dates[i]}: confidence {st['confidence']:.0%}.")
            self.log(now, "board", "info", f"Stock exposure board: {dec['board']:.0%} of the stock sleeve",
                     f"{REGIMES[st['regime']].upper()} at {st['confidence']:.0%} ({self.proxy} close {dates[i]}). Raw vote "
                     f"{dec['raw']:.0%}, chair ceiling {dec['chair_ceiling']:.0%}, dissent {dec['dissent']:.0%}.",
                     dict(lines=self.board.minutes(dec, st["regime"], st["confidence"])))
        hist = (prev.get("hist") or [])[-119:] + [[dates[i], int(st["regime"]), round(float(st["confidence"]), 3)]]
        self.brain = dict(date=dates[i], close=float(closes[i]), probs=st["probs"].tolist(), next_probs=st["next_probs"].tolist(),
                          regime=int(st["regime"]), conf=float(st["confidence"]), entropy=float(st["entropy"]), dec=dec,
                          hist=hist, features=features_now(np.array(closes[: i + 1]), "stock"))
        self.s.kv_set("stock_brain", self.brain, commit=False)
        self.s.kv_set("stock_tracker", self.tracker.state(), commit=False)
        self.s.kv_set("stock_board", self.board.state(), commit=False)
        return True

    def _warm(self, dates, closes, now):
        n = len(closes)
        w = max(0, min(self.warmup_days, n - self.tracker.warmup - self.tracker.min_train - 1))
        if w > 0:
            self.tracker.update(np.array(closes[: n - w]))
            for i in range(n - w, n):
                self._brain_day(dates, closes, i, now, quiet=True)
        else:
            self._brain_day(dates, closes, n - 1, now, quiet=True)
        self.log(now, "refit", "info", f"Stock brain warmed up on {n:,} days of {self.proxy}",
                 f"Replayed the last {w} days causally so the stock board learned before going live; {self.tracker.fit_count} HMM fits.")

    # ------------------------------------------------------------------ the daily step
    def step(self, now, breaker_cap, held, spy_vol_days=20):
        """Runs once per newly completed trading day. Returns the book's decisions, or None."""
        pdates, pcloses = self._series(self.proxy)
        if not pdates or not self.tracker.ready(len(pcloses)):
            return None
        day = pdates[-1]
        if self.s.kv_get("stock_day") == day:
            return None
        if self.tracker.model is None:
            self._warm(pdates, pcloses, now)
        else:
            last = self.s.kv_get("stock_day")
            i0 = pdates.index(last) + 1 if last in pdates else len(pdates) - 1
            for i in range(i0, len(pdates)):
                self._brain_day(pdates, pcloses, i, now)
        if not self.brain:
            return None

        spy = self.closes[self.proxy]
        mets, avail = {}, []
        for s in self.symbols:
            d, c = self._series(s)
            if len(c) >= MIN_DAYS and d[-1] == day:
                vol = [self.volumes[s].get(x, 0.0) for x in d]
                m = daily_metrics(d, c, spy, vol)
                if m:
                    mets[s] = m
                    avail.append(s)
        if len(avail) < 4:
            self.log(now, "info", "info", "Stock desk warming up", f"{len(avail)} stocks with {MIN_DAYS}+ days of history on {day}; need 4.")
            self.s.kv_set("stock_day", day)
            return None
        X = signal_matrix(mets, avail, regime_tilt(self.brain["next_probs"]))
        closes = {s: self.closes[s][day] for s in avail}

        # grade the calls from `horizon` trading days ago, and learn
        pending, keep, graded = self.s.kv_get("stock_pending", []), [], []
        for p in pending:
            age = (pdates.index(day) - pdates.index(p["day"])) if p["day"] in pdates else 99
            if age < self.horizon:
                keep.append(p)
                continue
            if age > self.horizon + 5:
                continue
            idx = [k for k, s in enumerate(p["syms"]) if s in closes]
            if len(idx) < 4:
                continue
            fwd = [math.log(closes[p["syms"][k]] / p["close"][k]) for k in idx]
            graded.append(self.desk.learn(np.array(p["X"])[idx], np.array(p["S"])[:, idx], fwd))
        S, coins = self.desk.decide(X, avail, [mets[s]["tier"] for s in avail])
        keep.append(dict(day=day, syms=avail, close=[closes[s] for s in avail], X=X.round(5).tolist(), S=S.round(6).tolist()))
        self.s.kv_set("stock_pending", keep[-(self.horizon + 3):], commit=False)
        self.s.kv_set("stock_desk", self.desk.state(), commit=False)
        conv = self.s.kv_get("stock_conv", {})
        for c in coins:
            k = c["symbol"]
            c["tier"], c["risk"], c["asset"] = mets[k]["tier"], mets[k]["risk"], "stock"
            conv[k] = (1 - self.conv_alpha) * conv.get(k, c["score"]) + self.conv_alpha * c["score"]
            c["conviction"] = conv[k]
        self.s.kv_set("stock_conv", conv, commit=False)
        if graded and self.desk.rounds % 20 == 0:
            ws = ", ".join(f"{a.name.split()[0]} {w:.0%} (IC {ic:+.2f})" for a, w, ic in zip(ADVISORS, self.desk.w, self.desk.ic))
            self.log(now, "learn", "info", f"Stock desk after {self.desk.rounds:,} graded days", ws)

        # target weights for the sleeve, then scaled to the account
        dec = self.brain["dec"]
        spy_c = np.array(pcloses[-(spy_vol_days + 1):], float)
        vol = float(np.std(np.diff(np.log(spy_c)), ddof=1) * math.sqrt(252)) if len(spy_c) > 5 else None
        prev_e = self.s.kv_get("stock_e")
        e = self.alloc.invested_fraction(dec["board"], self.brain["conf"], vol, breaker_cap, prev_e)
        notes = list(self.alloc.last_notes)
        if dec["chair_capped"]:
            notes.insert(0, f"chair ceiling {dec['chair_ceiling']:.0%} capped the board's {dec['raw']:.0%}")
        book = [dict(symbol=c["symbol"], score=c["conviction"], verdict=self.desk.verdict(c["conviction"], c["tier"])) for c in coins]
        w, tnotes = self.alloc.targets(e, book, held=held)
        notes += tnotes
        weights = {s: x * self.share for s, x in w.items()}
        self.s.kv_set("stock_e", e, commit=False)
        self.s.kv_set("stock_weights", weights, commit=False)
        self.s.kv_set("stock_day", day)
        regime = REGIMES[self.brain["regime"]].upper()
        picks = pick_minutes(self.desk, X, avail, mets, regime, S)
        out = dict(day=day, coins=coins, picks=picks, notes=notes, e=e, weights=weights, verdicts={c["symbol"]: c["verdict"] for c in book},
                   mets=mets, regime=regime)
        self.s.kv_set("stock_last", dict(coins=coins, picks=picks, notes=notes, e=e, day=day), commit=False)
        return out

    # ------------------------------------------------------------------ live read for the dashboard
    def live(self, now, quotes):
        spy = self.closes[self.proxy]
        out = []
        for s in self.symbols:
            d, c = self._series(s)
            q = quotes.get(s)
            vol = [self.volumes[s].get(x, 0.0) for x in d]
            fresh = q is not None and (now - q[1]) < 3 * 3600
            m = daily_metrics(d, c, spy, vol, q[0] if fresh else None)
            if m is None:
                out.append(dict(symbol=s, warming=True, bars=len(c), asset="stock"))
                continue
            m["symbol"], m["quote_age"] = s, (now - q[1]) if q else None
            out.append(m)
        return out

    def view(self):
        """Compact state for the dashboard snapshot."""
        b = self.brain
        if not b:
            return None
        return dict(symbols=self.symbols, proxy=self.proxy, share=self.share, regime=REGIMES[b["regime"]], confidence=b["conf"],
                    probs=b["probs"], next_probs=b["next_probs"], date=b["date"], hist=b.get("hist", [])[-60:],
                    features=b.get("features"), e_target=self.s.kv_get("stock_e"), board=b["dec"]["board"],
                    rounds=self.desk.rounds, horizon=self.horizon,
                    advisors=[dict(key=a.key, vote=b["dec"]["votes"][i], stance=b["dec"]["stances"][i],
                                   board_weight=b["dec"]["weights"][i], desk_weight=float(self.desk.w[i]), ic=float(self.desk.ic[i]),
                                   hit=float(self.desk.hit[i]), learned_share=self.desk.learned_share()[i], drift=self.desk.drift()[i])
                              for i, a in enumerate(ADVISORS)])

    def propose_due(self, now_et_time, day_iso):
        """True once per trading day, from `propose_at` ET, while a fresh decision is waiting."""
        hh, mm = map(int, self.propose_at.split(":"))
        return (now_et_time.hour, now_et_time.minute) >= (hh, mm) and self.s.kv_get("stock_proposed") != day_iso
