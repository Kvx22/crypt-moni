"""The 24/7 engine: scanner -> regime brain -> five learning advisors -> allocation
-> SAFETY GATE -> paper broker, or (live mode) an outbox of proposed orders that a
human approves before they go to Coinbase / Robinhood. Logs everything the
dashboard shows.

Three clocks, all UTC, all driven by one loop that never needs a market to open:
  every tick (default 60 s)   live prices, live risk board, alerts, heartbeat
  every closed hourly bar     advisors grade their calls from `horizon_hours`
                              ago, learn, and make new calls; paper rebalance
                              every `rebalance_every_hours`; dashboard snapshot
  every closed UTC day        the HMM brain reads BTC; the exposure board learns
"""
from __future__ import annotations

import csv
import datetime as dt
import math
import os
import time

import numpy as np

from .brokers import RobinhoodHandoff, make_broker
from .advisors import ADVISORS, ExposureBoard, PickDesk, learned_view, pick_minutes
from .allocation import Allocator
from .feeds import DAY, ET, HOUR, FeedError
from .models import Order
from .paper import PaperBroker
from .regime import REGIMES, WARMUP, RegimeTracker, features_now
from .notify import push
from .safety import CircuitBreaker, GateContext, KillSwitch, SafetyConfig, SafetyGate, live_armed
from .scanner import MIN_BARS, alerts as scan_alerts, metrics, regime_tilt, signal_matrix
from .scout import Scout
from .stocks import StockBook


def day_str(ts):
    return dt.datetime.fromtimestamp(ts, dt.timezone.utc).date().isoformat()


def day_ts(s):
    return int(dt.datetime.fromisoformat(s).replace(tzinfo=dt.timezone.utc).timestamp())


def read_daily_csv(path):
    if not path or not os.path.exists(path):
        return {}
    with open(path, newline="") as f:
        return {r["date"]: float(r["close"]) for r in csv.DictReader(f) if r.get("close")}


class Engine:
    def __init__(self, cfg, feed, store, broker=None):
        self.cfg, self.feed, self.s = cfg, feed, store
        self.universe = list(cfg["universe"])
        self.proxy = cfg.get("market_proxy", "BTC-USD")
        if self.proxy not in self.universe:
            self.universe.insert(0, self.proxy)
        self.sc = SafetyConfig.from_dict(cfg.get("safety", {}))
        self.stocks = StockBook(cfg, store, feed, self._ev) if cfg.get("stocks") else None
        self.stock_set = set(self.stocks.symbols) if self.stocks else set()
        self.scout = Scout(cfg, store, feed, self._ev) if (cfg.get("scout") or {}).get("enabled", True) else None
        if not self.sc.allowlist:
            self.sc.allowlist = list(self.universe) + sorted(self.stock_set)
        self.kill = KillSwitch(self.sc.kill_file)
        self.gate = SafetyGate(self.sc, self.kill)
        self.breaker = CircuitBreaker(self.sc, store.kv_get("breaker"))
        self.tracker = RegimeTracker(**cfg.get("tracker", {}))
        if store.kv_get("tracker"):
            self.tracker.load(store.kv_get("tracker"))
        self.board = ExposureBoard(**cfg.get("board", {}))
        if store.kv_get("board"):
            self.board.load(store.kv_get("board"))
        self.desk = PickDesk(**cfg.get("desk", {}))
        if store.kv_get("desk"):
            self.desk.load(store.kv_get("desk"))
        self.alloc = Allocator(**cfg.get("allocator", {}))
        live = broker if broker is not None else make_broker(cfg)
        if (live is None) != (self.sc.mode == "paper"):
            raise ValueError("config mismatch: safety.mode 'live' needs execution.broker coinbase|robinhood, "
                             "and a live broker needs safety.mode 'live'")
        if self.stocks and live is not None and live.name == "coinbase":
            raise ValueError("stocks can't trade on Coinbase: use config.robinhood.json (or paper mode) for the stock book")
        self.approve = live is not None          # live: every order waits in the outbox for a human
        if live is not None:
            self.broker = live
        else:
            self.broker = PaperBroker(**cfg.get("paper", {}))
            if store.kv_get("paper"):
                self.broker.load(store.kv_get("paper"))
        ex = cfg.get("execution", {})
        self.ttl = ex.get("approval_ttl_min", 120) * 60
        self.dash_url = ex.get("dashboard_url")
        self.acct = None
        self.horizon = int(cfg.get("horizon_hours", 4))
        self.rebalance_every = int(cfg.get("rebalance_every_hours", 8))
        self.conv_alpha = 1.0 / max(1.0, cfg.get("conviction_hours", 12))
        self.band = cfg.get("rebalance_band_pct", 0.02)
        self.min_trade = cfg.get("min_trade_usd", 25.0)
        self.warmup_days = int(cfg.get("warmup_days", 365))
        self.backfill_days = int(cfg.get("backfill_days", 2200))
        self.alert_cooldown = cfg.get("alert_cooldown_hours", 3) * HOUR
        self.slow_alerts = {"risk", "rsi", "volume"}  # these repeat while a condition lasts: 3x the cooldown
        self.daily_csv = cfg.get("daily_csv", "data/BTC-USD.csv")
        self.brain = store.kv_get("brain") or {}
        self.errors = 0
        self.feed_errors = []
        self.load_daily()
        self.adopt_watchlist()

    # ------------------------------------------------------------------ utils
    def _ev(self, ts, kind, sev, title, body="", payload=None):
        self.s.add_event(ts, kind, sev, title, body, payload)

    def load_daily(self):
        d = read_daily_csv(self.daily_csv)
        for b in self.s.get_bars(self.proxy, DAY, 100000):
            d[day_str(b.ts)] = b.close
        self.daily = dict(sorted(d.items()))

    # ------------------------------------------------------------------ data
    def sync_daily(self, now):
        today = int(now // DAY * DAY)
        last = day_ts(next(reversed(self.daily))) if self.daily else None
        if last is not None and last + DAY >= today:
            return 0
        start = last + DAY if last is not None else today - self.backfill_days * DAY
        added = []
        while start < today:
            bars = self.feed.candles(self.proxy, DAY, start, min(start + 299 * DAY, today - 1))
            bars = [b for b in bars if b.ts >= start and b.ts + DAY <= now]
            if not bars:
                break
            added += bars
            start = bars[-1].ts + DAY
        if added:
            self.s.put_bars(self.proxy, DAY, added)
            for b in added:
                self.daily[day_str(b.ts)] = b.close
        return len(added)

    def sync_hourly(self, sym, now):
        last = self.s.last_bar_ts(sym, HOUR)
        if last is not None and last + 2 * HOUR > now:
            return 0  # the next bar has not closed yet
        start = None if last is None or now - last > 290 * HOUR else last + HOUR
        bars = self.feed.candles(sym, HOUR, start, now if start is not None else None)
        bars = [b for b in bars if last is None or b.ts > last]
        if bars:
            self.s.put_bars(sym, HOUR, bars)
        return len(bars)

    def quotes(self, now, bars):
        out = {}
        for sym in self.universe:
            try:
                out[sym] = self.feed.ticker(sym)
            except FeedError as e:
                self.feed_errors.append(f"{sym} ticker: {e}")
                if bars.get(sym):  # fall back to the last close; the gate sees its age and refuses to trade on it
                    b = bars[sym][-1]
                    out[sym] = (b.close, b.ts + HOUR)
        return out

    # ------------------------------------------------------------------ brain (daily)
    def brain_step(self, now, live_btc=None):
        dates, closes = list(self.daily), np.array(list(self.daily.values()), float)
        if not self.tracker.ready(len(closes)):
            return False
        done = self.s.kv_get("brain_day")
        if done is None or self.tracker.model is None:
            self.warm_up(dates, closes, now)
            return True
        if done == dates[-1]:
            return False
        i0 = dates.index(done) + 1 if done in dates else len(dates) - 1
        for i in range(i0, len(dates)):
            self._brain_day(dates, closes, i, now)
        return True

    def warm_up(self, dates, closes, now):
        t0 = time.time()
        n = len(closes)
        w = max(0, min(self.warmup_days, n - WARMUP - self.tracker.min_train - 1))
        if w > 0:
            self.tracker.update(closes[: n - w])     # fit on the past only, then walk forward day by day
            for i in range(n - w, n):
                self._brain_day(dates, closes, i, now, quiet=True)
        else:
            self._brain_day(dates, closes, n - 1, now, quiet=True)
        self._ev(now, "refit", "info", f"Brain warmed up on {n:,} days of {self.proxy}",
                 f"Replayed the last {w} days causally (no look-ahead) so the board learned before going live; "
                 f"{self.tracker.fit_count} HMM fits, {time.time() - t0:.0f}s.", self.tracker.model.describe())

    def _brain_day(self, dates, closes, i, now, quiet=False):
        prev = self.brain
        if prev.get("close") and prev.get("date") != dates[i]:
            self.board.observe(np.array(prev["probs"]), closes[i] / prev["close"] - 1)
        st = self.tracker.update(closes[: i + 1])
        if st is None:
            return
        dec = self.board.decide(st["probs"], st["next_probs"], st["confidence"], self.s.kv_get("e"))
        if not quiet:
            if st["refit"]:
                self._ev(now, "refit", "info", f"HMM refit #{self.tracker.fit_count}",
                         "Regime model retrained on BTC history to date (warm start).", st["table"])
            if prev.get("regime") is not None and st["regime"] != prev["regime"]:
                self._ev(now, "regime", "warn" if st["regime"] < prev["regime"] else "info",
                         f"Regime {REGIMES[prev['regime']].upper()} -> {REGIMES[st['regime']].upper()}",
                         f"BTC daily close {dates[i]}: confidence {st['confidence']:.0%}.")
            mins = self.board.minutes(dec, st["regime"], st["confidence"])
            self._ev(now, "board", "info", f"Exposure board: {dec['board']:.0%} of the risk budget",
                     f"{REGIMES[st['regime']].upper()} at {st['confidence']:.0%} (BTC close {dates[i]}). Raw vote {dec['raw']:.0%}, "
                     f"chair ceiling {dec['chair_ceiling']:.0%}, dissent {dec['dissent']:.0%}.", dict(lines=mins))
        hist = (prev.get("hist") or [])[-119:] + [[dates[i], int(st["regime"]), round(float(st["confidence"]), 3)]]
        self.brain = dict(date=dates[i], close=float(closes[i]), probs=st["probs"].tolist(), next_probs=st["next_probs"].tolist(),
                          regime=int(st["regime"]), raw=int(st["raw_regime"]), conf=float(st["confidence"]),
                          entropy=float(st["entropy"]), dec=dec, hist=hist, features=features_now(closes[: i + 1]))
        self.s.kv_set("brain", self.brain, commit=False)
        self.s.kv_set("brain_day", dates[i], commit=False)
        self.s.kv_set("tracker", self.tracker.state(), commit=False)
        self.s.kv_set("board", self.board.state())

    def live_brain(self, live_btc):
        """Intraday read with today's live BTC price as a provisional close (state untouched)."""
        if self.tracker.model is None or not live_btc:
            return None
        closes = np.append(np.array(list(self.daily.values()), float), live_btc)
        st = self.tracker.peek(closes)
        return dict(probs=st["probs"].tolist(), next_probs=st["next_probs"].tolist(), raw=int(st["raw_regime"]),
                    conf=float(st["confidence"]))

    def btc_vol(self, days=30):
        c = np.array(list(self.daily.values())[-(days + 1):], float)
        return float(np.std(np.diff(np.log(c)), ddof=1) * math.sqrt(365)) if len(c) > 5 else None

    # ------------------------------------------------------------------ desk (hourly)
    def desk_step(self, now, bar_ts, bars, quotes, bs):
        syms = [s for s in self.universe if len(bars.get(s, [])) >= MIN_BARS and bars[s][-1].ts == bar_ts]
        if len(syms) < 4:
            self._ev(now, "info", "info", "Desk warming up", f"{len(syms)} coins with a full week of aligned hourly bars; need 4.")
            return None
        btc = bars.get(self.proxy)
        mets = {s: metrics(bars[s], btc) for s in syms}
        probs = self.brain.get("next_probs") or [0.2] * 5
        X = signal_matrix(mets, syms, regime_tilt(probs))
        closes = {s: bars[s][-1].close for s in syms}

        # 1) grade the calls made `horizon` hours ago, and learn from them
        pending, keep, graded = self.s.kv_get("pending", []), [], []
        for p in pending:
            if bar_ts - p["ts"] < self.horizon * HOUR:
                keep.append(p)
                continue
            if bar_ts - p["ts"] > (self.horizon + 6) * HOUR:
                continue  # too stale to grade fairly (the engine was down)
            idx = [k for k, s in enumerate(p["syms"]) if s in closes]
            if len(idx) < 4:
                continue
            fwd = [math.log(closes[p["syms"][k]] / p["close"][k]) for k in idx]
            res = self.desk.learn(np.array(p["X"])[idx], np.array(p["S"])[:, idx], fwd)
            graded.append(dict(ts=p["ts"], res=res))
        # 2) make new calls
        tiers = [mets[s]["tier"] for s in syms]
        S, coins = self.desk.decide(X, syms, tiers)
        keep.append(dict(ts=bar_ts, syms=syms, close=[closes[s] for s in syms], X=X.round(5).tolist(), S=S.round(6).tolist()))
        self.s.kv_set("pending", keep[-(self.horizon + 4):], commit=False)
        self.s.kv_set("desk", self.desk.state())
        conv = self.s.kv_get("conviction", {})
        for c in coins:
            c["tier"], c["risk"] = mets[c["symbol"]]["tier"], mets[c["symbol"]]["risk"]
            k = c["symbol"]
            conv[k] = (1 - self.conv_alpha) * conv.get(k, c["score"]) + self.conv_alpha * c["score"]
            c["conviction"] = conv[k]
        self.s.kv_set("conviction", conv, commit=False)
        if graded:
            self._learn_events(now, graded)

        # 3) allocation and the paper book
        dec = self.brain.get("dec")
        prev_e = self.s.kv_get("e")
        notes, fills = [], []
        if dec:
            e = self.alloc.invested_fraction(dec["board"], self.brain["conf"], self.btc_vol(), bs.exposure_cap, prev_e)
            notes = list(self.alloc.last_notes)
            if dec["chair_capped"]:
                notes.insert(0, f"chair ceiling {dec['chair_ceiling']:.0%} capped the board's {dec['raw']:.0%}")
            # the book follows the desk's smoothed conviction, not each hour's call, to keep churn and fees down
            book = [dict(symbol=c["symbol"], score=c["conviction"], verdict=self.desk.verdict(c["conviction"], c["tier"]))
                    for c in coins]
            weights, tnotes = self.alloc.targets(e, book, held=self.acct.positions if self.acct else {})
            notes += tnotes
            if self.stocks:   # the crypto book only gets its share of the account
                weights = {k: v * (1 - self.stocks.share) for k, v in weights.items()}
            self.s.kv_set("weights_crypto", weights, commit=False)
            self.s.kv_set("verdicts_crypto", {c["symbol"]: c["verdict"] for c in book}, commit=False)
            merged = self.merged_weights()
            self.s.kv_set("weights", merged, commit=False)
            due = (bar_ts // HOUR) % self.rebalance_every == 0 or prev_e is None or bs.exposure_cap < (prev_e or 0) - 0.01
            if due and self.acct is not None:
                fills = self.send_book(now, merged, quotes, bs)
            self.s.kv_set("e", e)
        else:
            e = 0.0
            notes.append("brain not ready: no paper allocation yet")
        regime_name = REGIMES[self.brain["regime"]].upper() if "regime" in self.brain else "WARMING UP"
        picks = pick_minutes(self.desk, X, syms, mets, regime_name, S)
        return dict(syms=syms, mets=mets, coins=coins, X=X, S=S, picks=picks, notes=notes, fills=fills, e=e, graded=graded)

    def adopt_watchlist(self):
        """Names you put on the watchlist from the scout join the board: scanned, scored, and tradable only
        through the same approval-gated process. (Removing a name takes effect at the next restart.)"""
        for w in self.s.kv_get("watchlist", []):
            sym = w["symbol"]
            if w["asset"] == "crypto" and sym not in self.universe:
                self.universe.append(sym)
            elif w["asset"] == "stock" and self.stocks and sym not in self.stock_set:
                self.stocks.add(sym)
                self.stock_set.add(sym)
            else:
                continue
            if sym not in self.sc.allowlist:
                self.sc.allowlist.append(sym)

    # ------------------------------------------------------------------ crypto + stock books together
    def merged_weights(self):
        w = dict(self.s.kv_get("weights_crypto", {}))
        w.update(self.s.kv_get("stock_weights", {}))
        return w

    def merged_verdicts(self):
        v = dict(self.s.kv_get("verdicts_crypto", {}))
        v.update(self.s.kv_get("stock_verdicts", {}))
        return v

    def send_book(self, now, weights, quotes, bs):
        """Paper: straight through the gate. Live: into the approval outbox."""
        if self.approve:
            return self.propose(now, weights, quotes, bs, self.merged_verdicts())
        return self.rebalance(now, weights, quotes, bs)

    def stock_tick(self, now, quotes, bs):
        """Daily: the stock brain and desk read the completed day. Each trading morning (from
        `propose_at` ET, market open): propose the combined book once."""
        res = self.stocks.step(now, bs.exposure_cap, self.acct.positions if self.acct else {})
        if res:
            self.s.kv_set("stock_verdicts", res["verdicts"], commit=False)
            self.s.kv_set("weights", self.merged_weights(), commit=False)
            if self.scout:
                try:
                    self.scout.scan_stocks(now, res["mets"], self.stocks.closes[self.stocks.proxy], self.stocks.desk,
                                           self.stocks.brain["next_probs"], res["regime"], res["day"])
                except FeedError as e:
                    self.feed_errors.append(f"scout stocks: {e}")
        et = dt.datetime.fromtimestamp(now, ET)
        if (self.acct is not None and self.feed.market_open(self.stocks.proxy) and self.s.kv_get("stock_weights")
                and self.stocks.propose_due(et.time(), et.date().isoformat())):
            self.s.kv_set("stock_proposed", et.date().isoformat())
            self.send_book(now, self.merged_weights(), quotes, bs)
        return res

    def _learn_events(self, now, graded):
        rounds = self.desk.rounds
        lead = int(np.argmax(self.desk.w))
        last_lead = self.s.kv_get("desk_lead")
        if last_lead is not None and last_lead != lead:
            self._ev(now, "learn", "info", f"{ADVISORS[lead].name} now leads the pick desk",
                     f"Desk weight {self.desk.w[lead]:.0%} (was {ADVISORS[last_lead].name}). Weights move with each "
                     f"advisor's realised, loss-averse pick payoff.")
        self.s.kv_set("desk_lead", lead, commit=False)
        if rounds % 24 == 0:
            ws = ", ".join(f"{a.name.split()[0]} {w:.0%} (IC {ic:+.2f})" for a, w, ic in zip(ADVISORS, self.desk.w, self.desk.ic))
            self._ev(now, "learn", "info", f"Pick desk after {rounds:,} graded hours", ws,
                     dict(drift=self.desk.drift(), learned_share=self.desk.learned_share()))

    # ------------------------------------------------------------------ orders
    def plan(self, acct, weights, quotes):
        orders = []
        for sym in sorted(set(weights) | set(acct.positions)):
            if sym not in quotes:
                continue
            if sym in self.stock_set and not self.feed.market_open(sym):
                continue   # stocks only trade in the session
            cur = acct.positions.get(sym, {}).get("value", 0.0)
            diff = weights.get(sym, 0.0) * acct.equity - cur
            if abs(diff) < max(self.min_trade, self.band * acct.equity):
                continue
            side = "buy" if diff > 0 else "sell"
            orders.append(Order(sym, side, abs(diff), "market", None,
                                f"rebalance to {weights.get(sym, 0):.1%} (now {cur / acct.equity:.1%})"))
        orders.sort(key=lambda o: (o.side != "sell", -o.notional_usd))
        return orders

    def _ctx(self, now, acct, quotes, bs):
        if self.approve:  # counters come from orders actually sent today
            n, t = self.s.executed_since(day_ts(day_str(now)))
            last = self.s.last_sent()
        else:
            day = self.s.kv_get("day", {})
            n, t = day.get("orders", 0), day.get("turnover", 0.0)
            last = {tuple(k.split("|")): v for k, v in self.s.kv_get("last_order", {}).items()}
        return GateContext(now=now, equity=acct.equity, cash=acct.cash,
                           positions={k: v["value"] for k, v in acct.positions.items()}, quotes=quotes, breaker=bs,
                           orders_today=n, turnover_today=t, last_order_ts=last)

    def _account(self, quotes, now):
        try:
            return self.broker.account(quotes)
        except Exception as ex:  # a broker outage never stops the scanner; it only pauses trading
            self.feed_errors.append(f"{self.broker.name} account: {str(ex)[:200]}")
            if now - (self.s.kv_get("acct_err_ts") or 0) > 3600:
                self._ev(now, "broker", "warn", f"Can't read the {self.broker.name} account", str(ex)[:300])
                self.s.kv_set("acct_err_ts", now)
            return None

    def rebalance(self, now, weights, quotes, bs):
        """Paper book: orders go straight through the gate to the paper broker."""
        acct = self.broker.account(quotes)
        if acct.equity <= 0:
            return []
        day = self.s.kv_get("day", {})
        orders = self.plan(acct, weights, quotes)
        ctx = self._ctx(now, acct, quotes, bs)
        out = []
        for o in orders:
            v = self.gate.check(o, ctx)
            if not v.ok:
                self._ev(now, "safety", "warn", f"BLOCKED {o.side} {o.symbol} ${o.notional_usd:,.0f}", "; ".join(v.reasons), o.d())
                continue
            if v.clamped:
                self._ev(now, "safety", "info", f"Clamped {o.side} {o.symbol}", "; ".join(v.reasons), o.d())
            SafetyGate.commit(v, ctx)
            f = self.broker.place(v.order, quotes, now)
            self.s.add_trade(f, o.reason)
            out.append(f.__dict__)
            if f.status != "filled":
                self._ev(now, "safety", "warn", f"Paper {f.status}: {o.symbol}", f.detail)
        day.update(orders=ctx.orders_today, turnover=ctx.turnover_today)
        self.s.kv_set("day", day, commit=False)
        self.s.kv_set("last_order", {f"{k[0]}|{k[1]}": v for k, v in ctx.last_order_ts.items()}, commit=False)
        self.s.kv_set("paper", self.broker.state())
        return out

    def propose(self, now, weights, quotes, bs, verdicts):
        """Live: the advisors' verdict becomes a proposal. Orders that pass the gate wait for approval."""
        if isinstance(self.broker, RobinhoodHandoff) and self.broker.stale():
            self._ev(now, "outbox", "warn", "No proposal: Robinhood balances are stale",
                     "Ask Claude to sync the Robinhood crypto board so data/robinhood_snapshot.json is fresh.")
            return []
        acct = self.broker.account(quotes)
        if acct.equity <= 0:
            return []
        self.s.supersede_pending()
        ctx = self._ctx(now, acct, quotes, bs)
        regime = REGIMES[self.brain["regime"]] if "regime" in self.brain else "n/a"
        made = []
        for o in self.plan(acct, weights, quotes):
            v = self.gate.check(o, ctx)
            if not v.ok:
                self._ev(now, "safety", "warn", f"Not proposed: {o.side} {o.symbol} ${o.notional_usd:,.0f}", "; ".join(v.reasons), o.d())
                continue
            SafetyGate.commit(v, ctx)
            oid = self.s.add_outbox(now, now + self.ttl, v.order, quotes[o.symbol][0], regime,
                                    verdicts.get(o.symbol, ""), self.broker.name)
            made.append(dict(id=oid, symbol=o.symbol, side=o.side, notional_usd=v.order.notional_usd, status="pending"))
        if made:
            lines = [f"#{m['id']} {m['side']} {m['symbol'].split('-')[0]} ${m['notional_usd']:,.0f}" for m in made]
            self._ev(now, "outbox", "info", f"{len(made)} order(s) await your approval ({self.broker.name})",
                     "; ".join(lines) + f". They expire in {self.ttl / 60:.0f} min.")
            push(self.cfg, f"Crypto Risk Board: {len(made)} order(s) to approve", "\n".join(lines), self.dash_url)
        return made

    def execute_approved(self, now, quotes, bs):
        """Send what a human approved, after fresh checks. Never anything else."""
        self.s.expire_outbox(now)
        if not self.broker.can_execute:
            return []  # Robinhood: a Claude session places approved orders (export-approved / ack)
        rows = sorted(self.s.outbox(200, "approved"), key=lambda r: (r["side"] != "sell", r["id"]))
        if not rows:
            return []
        acct = self.broker.account(quotes)
        ctx = self._ctx(now, acct, quotes, bs)
        out = []
        for r in rows:
            q = quotes.get(r["symbol"])
            if q is None:
                continue  # no fresh price this tick; try again next tick (until it expires)
            move = q[0] / r["ref_price"] - 1
            if abs(move) > self.sc.price_band_pct:
                self.s.outbox_set(r["id"], "skipped", now, detail=f"price moved {move:+.1%} since proposal (limit ±{self.sc.price_band_pct:.0%})")
                self._ev(now, "outbox", "warn", f"Skipped #{r['id']} {r['side']} {r['symbol']}", f"price moved {move:+.1%} since it was proposed")
                continue
            o = Order(r["symbol"], r["side"], r["notional"], "market", None, r["reason"])
            v = self.gate.check(o, ctx)
            if not v.ok:
                self.s.outbox_set(r["id"], "skipped", now, detail="; ".join(v.reasons)[:300])
                self._ev(now, "outbox", "warn", f"Skipped #{r['id']} {r['side']} {r['symbol']}", "; ".join(v.reasons))
                continue
            SafetyGate.commit(v, ctx)
            try:
                f = self.broker.place(v.order, quotes, now, r["client_order_id"])
            except Exception as ex:  # unknown outcome: never resend automatically
                self.s.outbox_set(r["id"], "check", now, detail=f"send failed or outcome unknown, check {self.broker.name} before retrying: {str(ex)[:200]}")
                self._ev(now, "outbox", "critical", f"Order #{r['id']} outcome unknown", str(ex)[:300])
                continue
            if f.status == "sent":
                self.s.outbox_set(r["id"], "sent", now, broker_order_id=f.detail)
                self.s.add_trade(f, f"#{r['id']} approved: {r['reason']}")
                self._ev(now, "outbox", "info", f"Sent #{r['id']} {f.side} {f.symbol} ${f.notional_usd:,.2f}", f"{self.broker.name} order {f.detail}")
            else:
                self.s.outbox_set(r["id"], "failed", now, detail=f.detail)
                self._ev(now, "outbox", "warn", f"{self.broker.name} rejected #{r['id']} {f.symbol}", f.detail)
            out.append(f.__dict__)
        return out

    # ------------------------------------------------------------------ the tick
    def tick(self):
        self.feed.tick()
        now = self.feed.now()
        self.feed_errors = []
        ok = 0
        for sym in self.universe:
            try:
                self.sync_hourly(sym, now)
                ok += 1
            except FeedError as e:
                self.feed_errors.append(f"{sym} candles: {e}")
        if ok == 0:
            raise FeedError("no market data from any source (first error: " + self.feed_errors[0] + ")")
        try:
            self.sync_daily(now)
        except FeedError as e:
            self.feed_errors.append(f"daily {self.proxy}: {e}")
        bars = {s: self.s.get_bars(s, HOUR, 400) for s in self.universe}
        quotes = self.quotes(now, bars)
        if self.stocks:
            self.stocks.sync(now)
            quotes.update(self.stocks.quotes(now))
            self.feed_errors += self.stocks.errors

        # breaker on book equity (paper or the real account), every tick
        acct = self.acct = self._account(quotes, now)
        date = day_str(now)
        day = self.s.kv_get("day", {})
        new_day = day.get("date") != date
        if new_day and acct is not None:
            day = dict(date=date, open=acct.equity, orders=0, turnover=0.0)
            self.s.kv_set("day", day, commit=False)
        old = self.breaker.level
        bs = self.breaker.update(acct.equity if acct else self.breaker.hwm, day.get("open", 0), new_day and acct is not None)
        self.s.kv_set("breaker", self.breaker.save(), commit=False)
        if bs.level != old:
            self._ev(now, "breaker", "critical" if bs.level >= 2 else "warn",
                     f"Circuit breaker {self.breaker.NAMES[old]} -> {bs.name}", "; ".join(bs.reasons), dict(drawdown=bs.drawdown))
        killed = self.kill.engaged()
        if killed and not self.s.kv_get("kill_noted"):
            self._ev(now, "safety", "critical", "KILL SWITCH engaged: no orders", self.kill.reason())
        self.s.kv_set("kill_noted", killed, commit=False)

        self.adopt_watchlist()
        self.brain_step(now)
        if self.stocks:
            self.stock_tick(now, quotes, bs)
        if self.approve and acct is not None:
            self.execute_approved(now, quotes, bs)
        live_btc = quotes.get(self.proxy, (None,))[0]

        # hourly: a new bar closed for the market proxy
        bar_ts = bars[self.proxy][-1].ts if bars.get(self.proxy) else None
        desk = None
        if bar_ts is not None and bar_ts != self.s.kv_get("last_bar"):
            desk = self.desk_step(now, bar_ts, bars, quotes, bs)
            self.s.kv_set("last_bar", bar_ts)
            if desk:
                self.snapshot(now, quotes, bs, desk)
                if self.scout:
                    probs = self.brain.get("next_probs") or [0.2] * 5
                    regime = REGIMES[self.brain["regime"]].upper() if "regime" in self.brain else "n/a"
                    self.scout.scan_crypto(now, bars, desk["mets"], self.desk, probs, regime)
        self.live_scan(now, bars, quotes, live_btc)
        self.s.heartbeat(time.time())
        return desk

    # ------------------------------------------------------------------ outputs
    def live_scan(self, now, bars, quotes, live_btc):
        btc = bars.get(self.proxy)
        prev = self.s.kv_get("live") or {}
        prev_m = {c["symbol"]: c for c in prev.get("coins", [])}
        last_alert = self.s.kv_get("alert_last", {})
        coins = []

        def alerts_for(sym, m):
            p = prev_m.get(sym)
            for kind, sev, title, body in scan_alerts(sym, m, p if p and not p.get("warming") else None):
                key = f"{sym}|{kind}"
                cool = self.alert_cooldown * (3 if kind in self.slow_alerts else 1)
                if now - last_alert.get(key, -1e18) >= cool:
                    self.s.add_alert(now, sym, kind, sev, title, body)
                    last_alert[key] = now

        for sym in self.universe:
            b = bars.get(sym, [])
            q = quotes.get(sym)
            m = metrics(b, btc, q[0] if q else None) if len(b) >= MIN_BARS else None
            if m is None:
                coins.append(dict(symbol=sym, warming=True, bars=len(b)))
                continue
            m["symbol"], m["quote_age"], m["asset"] = sym, (now - q[1]) if q else None, "crypto"
            alerts_for(sym, m)
            coins.append(m)
        if self.stocks:
            open_ = self.feed.market_open(self.stocks.proxy)
            for m in self.stocks.live(now, quotes):
                if not m.get("warming") and open_:
                    alerts_for(m["symbol"], m)   # alerts only while the market is open
                coins.append(m)
        self.s.kv_set("alert_last", last_alert, commit=False)
        lb = self.live_brain(live_btc)
        self.s.kv_set("live", dict(ts=now, coins=coins, brain=lb, feed=getattr(self.feed, "last_source", None) or self.feed.kind,
                                   simulated=self.feed.simulated, feed_errors=self.feed_errors[:6],
                                   stock_market_open=self.feed.market_open(self.stocks.proxy) if self.stocks else None))

    def snapshot(self, now, quotes, bs, desk):
        from .models import Account
        acct = self._account(quotes, now) or self.acct or Account(0.0, 0.0, {})
        b, dec = self.brain, self.brain.get("dec")
        weights = self.s.kv_get("weights", {})
        invested = sum(p["value"] for p in acct.positions.values())
        lead_lines = {l["key"]: l for l in desk["picks"]}
        exp_lines = {l["key"]: l for l in (self.board.minutes(dec, b["regime"], b["conf"]) if dec else [])}
        share, drift = self.desk.learned_share(), self.desk.drift()
        advisors = []
        for i, a in enumerate(ADVISORS):
            advisors.append(dict(
                key=a.key, name=a.name, role=a.role, color=a.color,
                vote=dec["votes"][i] if dec else 0.0, stance=dec["stances"][i] if dec else "hold",
                board_weight=dec["weights"][i] if dec else 0.2, desk_weight=float(self.desk.w[i]),
                ic=float(self.desk.ic[i]), hit=float(self.desk.hit[i]), excess=float(self.desk.excess[i]),
                learned_share=share[i], drift=drift[i], pick=lead_lines[a.key]["pick"], avoid=lead_lines[a.key]["avoid"],
                pick_line=lead_lines[a.key]["text"], board_line=exp_lines.get(a.key, {}).get("text", ""),
                learned=learned_view(self.desk, i)))
        sview = self.stocks.view() if self.stocks else None
        if sview:
            sw = {a["key"]: a for a in sview["advisors"]}
            for a in advisors:
                x = sw[a["key"]]
                a.update(stock_vote=x["vote"], stock_stance=x["stance"], stock_desk_weight=x["desk_weight"], stock_board_weight=x["board_weight"],
                         stock_ic=x["ic"], stock_hit=x["hit"], stock_learned_share=x["learned_share"], stock_drift=x["drift"],
                         stock_learned=learned_view(self.stocks.desk, ADVISORS.index(next(q for q in ADVISORS if q.key == a["key"]))))
        mins = []
        for a in advisors:
            if a["board_line"]:
                mins.append(dict(key=a["key"], name=a["name"], stance=a["stance"], text=a["board_line"]))
            mins.append(dict(key=a["key"], name=a["name"], stance="pick", text=a["pick_line"]))
        slast = self.s.kv_get("stock_last") if self.stocks else None
        if slast:
            for l in slast["picks"]:
                mins.append(dict(key=l["key"], name=l["name"], stance="pick", text="(stocks) " + l["text"]))
        self._ev(now, "minutes", "info", "Desk calls",
                 "; ".join(f"{a['name'].split()[0]}: {a['pick'].split('-')[0]}" for a in advisors), dict(lines=mins))
        g = self.s.kv_get("day", {})
        if self.approve:
            n, t = self.s.executed_since(day_ts(day_str(now)))
            g = dict(orders=n, turnover=t)
        snap = dict(
            ts=now, mode=self.sc.mode, broker=self.broker.name, feed=self.feed.kind, simulated=self.feed.simulated,
            live_armed=live_armed(self.sc), approval_ttl_min=self.ttl / 60,
            equity=acct.equity, cash=acct.cash, bench=quotes.get(self.proxy, (None,))[0], fees=self.broker.fees_paid,
            e_target=desk["e"] * (1 - self.stocks.share) + (self.s.kv_get("stock_e") or 0.0) * self.stocks.share if self.stocks else desk["e"],
            e_actual=invested / acct.equity if acct.equity else 0.0,
            regime=REGIMES[b["regime"]] if "regime" in b else "n/a", confidence=b.get("conf", 0.0),
            probs=b.get("probs", [0.2] * 5), next_probs=b.get("next_probs", [0.2] * 5), entropy=b.get("entropy"),
            brain_date=b.get("date"), regime_hist=b.get("hist", [])[-60:], features=b.get("features"),
            refits=self.tracker.fit_count, model=self.tracker.model.describe() if self.tracker.model else None,
            advisors=advisors, minutes=mins,
            board=dict(raw=dec["raw"], chair_ceiling=dec["chair_ceiling"], dissent=dec["dissent"], board=dec["board"],
                       chair_capped=dec["chair_capped"], learned=dec["learned"], trust=dec["learned_trust"]) if dec else {},
            desk=dict(rounds=self.desk.rounds, horizon=self.horizon, n_obs=self.desk.n_obs),
            coins=[dict(c, asset="crypto") for c in desk["coins"]] + (slast["coins"] if slast else []),
            notes=desk["notes"] + ([f"stocks: {n}" for n in slast["notes"]] if slast else []),
            stocks=sview, stock_market_open=self.feed.market_open(self.stocks.proxy) if self.stocks else None,
            positions={s: dict(value=acct.positions.get(s, {}).get("value", 0.0),
                               w=acct.positions.get(s, {}).get("value", 0.0) / acct.equity if acct.equity else 0.0,
                               target_w=weights.get(s, 0.0)) for s in sorted(set(acct.positions) | set(weights))},
            breaker=dict(name=bs.name, level=bs.level, cap=bs.exposure_cap, allow_buys=bs.allow_buys, halted=bs.halted,
                         drawdown=bs.drawdown, day_pnl=bs.day_pnl_pct, hwm=bs.hwm, reasons=bs.reasons),
            guards=dict(kill=self.kill.engaged(), cap_set=self.sc.max_per_trade_usd is not None,
                        max_per_trade=self.sc.max_per_trade_usd, orders_today=g.get("orders", 0),
                        max_orders=self.sc.max_orders_per_day,
                        turnover_today=g.get("turnover", 0.0) / acct.equity if acct.equity else 0.0,
                        max_turnover=self.sc.max_daily_turnover_pct, max_quote_age=self.sc.max_quote_age_s,
                        rebalance_every=self.rebalance_every),
            fills=desk["fills"],
        )
        self.s.add_snapshot(snap)
        return snap

    # ------------------------------------------------------------------ forever
    def run_forever(self, scan_seconds=60.0, max_backoff=600.0, prune_every=3600.0):
        wait, last_prune = scan_seconds, 0.0
        while True:
            t0 = time.time()
            try:
                self.tick()
                self.errors, wait = 0, scan_seconds
            except FeedError as ex:  # network trouble: back off and keep scanning, never trade on it
                wait = min(max(wait, scan_seconds) * 2, max_backoff)
                self._ev(time.time(), "feed", "warn", "Market data unavailable, retrying", f"{ex} (next try in {wait:.0f}s)")
                live = self.s.kv_get("live") or {}
                live.update(feed_errors=[str(ex)[:300]], feed=self.feed.kind, simulated=self.feed.simulated)
                self.s.kv_set("live", live)
                self.s.heartbeat(time.time())
            except Exception as ex:  # logic error: fail closed after 5 in a row
                self.errors += 1
                self._ev(time.time(), "error", "critical", f"Engine error #{self.errors}", repr(ex))
                if self.errors >= 5:
                    self.kill.trip("5 consecutive engine errors")
                wait = min(max(wait, scan_seconds) * 2, max_backoff)
            if time.time() - last_prune > prune_every:
                self.s.prune()
                last_prune = time.time()
            time.sleep(max(0.0, wait - (time.time() - t0)))
