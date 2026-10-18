"""python -m unittest tests.test_research_pretrain -v"""
import datetime as dt
import math
import os
import shutil
import sys
import tempfile
import time
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from cryptomoni.advisors import PickDesk  # noqa: E402
from cryptomoni.feeds import DAY, HOUR, FeedError, SimFeed  # noqa: E402
from cryptomoni.models import Bar, Order  # noqa: E402
from cryptomoni.research import pretrain as P  # noqa: E402
from cryptomoni.research.cadence import CadenceBooks, cadence_filter, max_drawdown  # noqa: E402
from cryptomoni.scanner import MIN_BARS, metrics, regime_tilt, signal_matrix  # noqa: E402
from cryptomoni.store import Store  # noqa: E402

SYMS = ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "DOGE-USD", "ADA-USD"]
T0 = 1_760_000_000 // HOUR * HOUR
DESK = dict(ridge=1000, decay=0.995, eta=0.08, top_k=3)
_SIM = {}


def sim_bars(history=900):
    if history not in _SIM:
        f = SimFeed(SYMS, start_ts=T0, seed=11, history=history)
        _SIM[history] = (f, {s: list(f.bars[s]) for s in SYMS})
    return _SIM[history][1]


def reference_desk(bars_by_sym, proxy, horizon, desk_kwargs, probs=None):
    """Engine.desk_step's learning loop transcribed literally, run hour by hour on the
    last 400 bars (slow on purpose: it is the yardstick)."""
    desk = PickDesk(**desk_kwargs)
    pending = []
    universe = list(bars_by_sym)
    for bar_ts in [b.ts for b in bars_by_sym[proxy]]:
        bars = {s: [b for b in v if b.ts <= bar_ts][-400:] for s, v in bars_by_sym.items()}
        syms = [s for s in universe if len(bars.get(s, [])) >= MIN_BARS and bars[s][-1].ts == bar_ts]
        if len(syms) < 4:
            continue
        btc = bars.get(proxy)
        mets = {s: metrics(bars[s], btc) for s in syms}
        X = signal_matrix(mets, syms, regime_tilt(probs or [0.2] * 5))
        closes = {s: bars[s][-1].close for s in syms}
        keep = []
        for p in pending:
            if bar_ts - p["ts"] < horizon * HOUR:
                keep.append(p)
                continue
            if bar_ts - p["ts"] > (horizon + 6) * HOUR:
                continue
            idx = [k for k, s in enumerate(p["syms"]) if s in closes]
            if len(idx) < 4:
                continue
            fwd = [math.log(closes[p["syms"][k]] / p["close"][k]) for k in idx]
            desk.learn(np.array(p["X"])[idx], np.array(p["S"])[:, idx], fwd)
        S, _ = desk.decide(X, syms, [mets[s]["tier"] for s in syms])
        keep.append(dict(ts=bar_ts, syms=syms, close=[closes[s] for s in syms], X=X.round(5).tolist(), S=S.round(6).tolist()))
        pending = keep[-(horizon + 4):]
    return desk


def same_state(a, b, tol=1e-9):
    for k in ("Sxx", "Sxy", "w", "ic", "hit", "excess", "score"):
        if not np.allclose(np.array(a[k]), np.array(b[k]), atol=tol, rtol=0):
            return False
    return a["rounds"] == b["rounds"] and abs(a["n_obs"] - b["n_obs"]) < tol


def stock_series(n_days=300, syms=("SPY", "QQQ", "GLD", "XLE", "XLV", "TLT"), seed=3):
    rng = np.random.default_rng(seed)
    d0 = dt.date(2025, 1, 1)
    dates = []
    d = d0
    while len(dates) < n_days:
        if d.weekday() < 5:
            dates.append(d.isoformat())
        d += dt.timedelta(days=1)
    closes, vols = {}, {}
    for i, s in enumerate(syms):
        r = rng.normal(0.0003, 0.01 + 0.002 * i, n_days)
        c = 100 * np.exp(np.cumsum(r))
        closes[s] = dict(zip(dates, c.tolist()))
        vols[s] = dict(zip(dates, rng.uniform(1e6, 5e6, n_days).tolist()))
    return closes, vols


# ---------------------------------------------------------------------------------- replay
class TestReplay(unittest.TestCase):
    def test_replay_learns(self):
        bars = sim_bars()
        out = P.replay_crypto(bars, "BTC-USD", DESK, 4)
        self.assertGreater(out["rounds"], 500)
        self.assertEqual(len(out["ic"]), out["rounds"])
        self.assertEqual(out["hours"], 900 - MIN_BARS + 1)
        fresh = PickDesk(**DESK)
        self.assertGreater(float(np.abs(out["desk"].W - fresh.W).max()), 1e-4)
        self.assertTrue(all(-1.0 <= x <= 1.0 for x in out["ic"]))

    def test_replay_matches_the_live_loop_exactly(self):
        bars = {s: v[:420] for s, v in sim_bars().items()}
        probs = [0.05, 0.1, 0.2, 0.4, 0.25]
        ref = reference_desk(bars, "BTC-USD", 4, DESK, probs)
        out = P.replay_crypto(bars, "BTC-USD", DESK, 4, probs_by_day={"2000-01-01": probs})
        self.assertGreater(ref.rounds, 100)
        self.assertTrue(same_state(ref.state(), out["desk"].state()))

    def test_replay_matches_engine_desk_step(self):
        """Drive the real Engine on a SimFeed and replay the bars it stored: same desk."""
        try:
            from cryptomoni.engine import Engine
        except Exception as e:  # noqa: BLE001
            self.skipTest(f"engine not importable: {e!r}")
        tmp = tempfile.mkdtemp()
        try:
            cfg = dict(universe=SYMS, market_proxy="BTC-USD", horizon_hours=4, rebalance_every_hours=8,
                       daily_csv=os.path.join(tmp, "none.csv"), desk=DESK, feed=dict(kind="sim"),
                       safety=dict(mode="paper", kill_file=os.path.join(tmp, "KILL")), scout=dict(enabled=False),
                       research=dict(enabled=False), pretrain=dict(enabled=False), paper=dict(start_cash=10000))
            feed = SimFeed(SYMS, start_ts=T0, seed=5, history=300)
            store = Store(os.path.join(tmp, "b.db"))
            try:
                eng = Engine(cfg, feed, store)
            except Exception as e:  # noqa: BLE001  (engine integration still in progress elsewhere)
                self.skipTest(f"engine not constructible here: {e!r}")
            eng.tick()
            first = store.kv_get("last_bar")      # the engine starts stepping at its first tick
            for _ in range(59):
                eng.tick()
            if eng.brain.get("next_probs"):
                self.skipTest("brain became ready; probs differ by design")
            bars = {s: store.get_bars(s, HOUR, 100000) for s in SYMS}
            out = P.crypto_schedule(bars, "BTC-USD", 4)
            steps = [st for st in out if st["ts"] >= first]
            r = P.run_schedule(P._grade_queue(steps, 4, 6, 4), DESK)
            self.assertGreater(eng.desk.rounds, 30)
            self.assertTrue(same_state(eng.desk.state(), r["desk"].state(), 1e-7))
            store.close()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_slices_by_timestamp_not_index(self):
        bars = {s: v[:500] for s, v in sim_bars().items()}
        # ADA lists 100 hours late and misses a stretch: the replay must align by timestamp
        bars["ADA-USD"] = [b for b in bars["ADA-USD"][100:] if not 300 <= (b.ts - T0) // HOUR + 900 < 310]
        ref = reference_desk(bars, "BTC-USD", 4, DESK)
        out = P.replay_crypto(bars, "BTC-USD", DESK, 4)
        self.assertTrue(same_state(ref.state(), out["desk"].state()))

    def test_too_few_symbols_or_bad_data_never_crash(self):
        bars = {s: v[:300] for s, v in list(sim_bars().items())[:3]}
        self.assertEqual(P.replay_crypto(bars, "BTC-USD", DESK, 4)["rounds"], 0)
        self.assertEqual(P.replay_crypto({}, "BTC-USD", DESK, 4)["rounds"], 0)
        bad = {s: list(v[:300]) for s, v in sim_bars().items()}
        bad["SOL-USD"][250] = Bar(bad["SOL-USD"][250].ts, 1, 1, 1, 0.0, 1)   # a zero close is skipped
        bad["XRP-USD"] = []
        out = P.replay_crypto(bad, "BTC-USD", DESK, 4)
        self.assertGreater(out["rounds"], 50)

    def test_stop_and_progress(self):
        bars = {s: v[:400] for s, v in sim_bars().items()}
        seen = []
        out = P.replay_crypto(bars, "BTC-USD", DESK, 4, progress=lambda i, n: seen.append((i, n)), stop=lambda: len(seen) > 2)
        self.assertLess(out["hours"], 100)
        self.assertTrue(seen)

    def test_probs_lookup(self):
        pb = P.probs_from_hist([["2025-01-01", 3, 0.8], ["2025-01-05", 0, 0.6], ["bad"], [None, 9, 0.5]])
        self.assertEqual(sorted(pb), ["2025-01-01", "2025-01-05"])
        self.assertAlmostEqual(sum(pb["2025-01-01"]), 1.0)
        self.assertEqual(pb["2025-01-01"][3], 0.8)
        look = P._Probs(pb)
        self.assertEqual(look("2024-12-31"), [0.2] * 5)
        self.assertEqual(look("2025-01-03"), pb["2025-01-01"])
        self.assertEqual(look("2025-02-01"), pb["2025-01-05"])

    def test_replay_stocks_learns(self):
        closes, vols = stock_series()
        out = P.replay_stocks(closes, vols, "SPY", dict(decay=0.98, ridge=300.0), 2)
        self.assertGreater(out["rounds"], 100)
        self.assertEqual(len(out["ic"]), out["rounds"])
        self.assertEqual(P.replay_stocks({}, {}, "SPY", {}, 2)["rounds"], 0)


# ---------------------------------------------------------------------------------- holdout + decision
class TestHoldout(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.steps = P.crypto_schedule(sim_bars(), "BTC-USD", 4)

    def test_holdout_report(self):
        champ = PickDesk(**DESK)
        res = P.holdout_compare(self.steps, DESK, champ.state())
        rep = res["report"]
        n = rep["rounds"]
        self.assertEqual(rep["holdout_rounds"], n - int(n * 0.8))
        for k in ("challenger", "prior", "champion"):
            self.assertEqual(rep[k]["n"], rep["holdout_rounds"])
            self.assertTrue(-1 <= rep[k]["ic"] <= 1)
        # a never-trained champion that learns through the holdout is not the frozen prior
        self.assertNotEqual(rep["prior"]["ic"], rep["challenger"]["ic"])
        self.assertLess(res["challenger_state"]["rounds"], res["final_state"]["rounds"])
        self.assertEqual(res["challenger_state"]["rounds"], int(n * 0.8))
        PickDesk(**DESK).load(res["final_state"])

    def test_prior_desk_scores_match_a_fresh_desk(self):
        """The prior-only desk's holdout IC equals scoring a frozen fresh desk on those calls."""
        res = P.holdout_compare(self.steps, DESK, None)
        self.assertIsNone(res["report"]["champion"])
        frozen = P._Runner(PickDesk(**DESK), learn=False)
        for j, st in enumerate(self.steps):
            for k, idx, fwd in st["grades"]:
                frozen.grade(k, idx, fwd, self.steps)
            frozen.predict(j, st)
        h = res["report"]["holdout_rounds"]
        self.assertAlmostEqual(res["report"]["prior"]["ic"], float(np.mean(frozen.ic[-h:])), places=12)

    def test_cancel_and_empty(self):
        self.assertIsNone(P.holdout_compare(self.steps, DESK, stop=lambda: True))
        res = P.holdout_compare([], DESK)
        self.assertEqual(res["report"]["rounds"], 0)
        self.assertFalse(P.decide(res, 0)[0])

    def test_decide_rules(self):
        def r(ch, pr, cp=None, n=50):
            blk = lambda ic: dict(ic=ic, top=0.0, n=n)  # noqa: E731
            return dict(final_state={"x": 1}, report=dict(challenger=blk(ch), prior=blk(pr),
                                                          champion=blk(cp) if cp is not None else None))
        # fresh desk: needs +0.005 over instincts
        self.assertEqual(P.decide(r(0.030, 0.024), 10)[:2], (True, "final"))
        self.assertEqual(P.decide(r(0.028, 0.024), 10)[:2], (False, "none"))
        # mature desk: needs +0.01 over both
        self.assertEqual(P.decide(r(0.05, 0.03, 0.035), 100)[:2], (True, "final"))
        self.assertFalse(P.decide(r(0.05, 0.03, 0.045), 100)[0])
        self.assertFalse(P.decide(r(0.05, 0.045, 0.02), 100)[0])
        self.assertFalse(P.decide(r(0.05, 0.0), 100)[0])           # no champion to beat
        self.assertEqual(P.decide(r(0.05, 0.0), 47)[0], True)       # threshold is configurable
        self.assertFalse(P.decide(r(0.05, 0.0), 47, min_rounds_fresh=40)[0])
        self.assertFalse(P.decide(r(0.5, 0.0, n=5), 0)[0])          # too small a holdout
        self.assertFalse(P.decide(None, 0)[0])
        self.assertFalse(P.decide({}, 0)[0])
        for x in (r(0.03, 0.0), r(0.0, 0.03)):
            self.assertIsInstance(P.decide(x, 0)[2], str)


# ---------------------------------------------------------------------------------- backfill
class FakeFeed:
    def __init__(self, bars, fail_after=None, delay=0.0):
        self.bars, self.calls, self.fail_after, self.delay = bars, [], fail_after, delay

    def candles(self, symbol, gran, start=None, end=None):
        self.calls.append((symbol, start, end))
        if self.delay:
            time.sleep(self.delay)
        if self.fail_after is not None and len(self.calls) > self.fail_after:
            raise FeedError("down")
        if symbol not in self.bars:
            raise FeedError("no such product")
        return [b for b in self.bars[symbol] if start <= b.ts <= end][-300:]


class TestBackfill(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.store = Store(os.path.join(self.tmp, "b.db"))

    def tearDown(self):
        self.store.close()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_pages_forward_and_skips_stored(self):
        bars = sim_bars()["ETH-USD"]
        now = bars[-1].ts + HOUR + 1800
        self.store.put_bars("ETH-USD", HOUR, bars[-50:])
        feed = FakeFeed({"ETH-USD": bars})
        n = P.backfill_hourly(feed, self.store, "ETH-USD", 700, now)
        self.assertEqual(n, 700 - 50)
        stored = self.store.get_bars("ETH-USD", HOUR, 5000)
        self.assertEqual(len(stored), 700)
        self.assertEqual(len({b.ts for b in stored}), 700)
        for _, s, e in feed.calls:
            self.assertLessEqual((e - s) // HOUR + 1, 290)
        self.assertEqual(feed.calls[0][1], int(now - 700 * HOUR) // HOUR * HOUR)
        self.assertEqual(P.backfill_hourly(feed, self.store, "ETH-USD", 700, now), 0)

    def test_feed_error_stops_early_and_unknown_symbol(self):
        bars = sim_bars()["ETH-USD"]
        now = bars[-1].ts + HOUR
        n = P.backfill_hourly(FakeFeed({"ETH-USD": bars}, fail_after=1), self.store, "ETH-USD", 800, now)
        self.assertEqual(n, 290)
        self.assertEqual(P.backfill_hourly(FakeFeed({}), self.store, "XYZ-USD", 800, now), 0)

    def test_never_more_than_hours_plus_48(self):
        class Greedy(FakeFeed):   # a feed that ignores the window and returns everything
            def candles(self, symbol, gran, start=None, end=None):
                self.calls.append((symbol, start, end))
                return list(self.bars[symbol])
        bars = sim_bars()["ETH-USD"]
        n = P.backfill_hourly(Greedy({"ETH-USD": bars}), self.store, "ETH-USD", 100, bars[-1].ts + HOUR)
        self.assertLessEqual(n, 148)
        self.assertTrue(all(b.ts >= bars[-1].ts - 100 * HOUR for b in self.store.get_bars("ETH-USD", HOUR, 5000)))

    def test_empty_stretch_before_listing(self):
        bars = sim_bars()["ETH-USD"][-200:]     # the coin only exists for the last 200 hours
        now = bars[-1].ts + HOUR
        self.assertEqual(P.backfill_hourly(FakeFeed({"ETH-USD": bars}), self.store, "ETH-USD", 800, now), 200)


# ---------------------------------------------------------------------------------- the job
class FakeStocks:
    def __init__(self, store):
        self.symbols = ["SPY", "QQQ", "GLD", "XLE", "XLV", "TLT"]
        self.closes, self.volumes = stock_series(syms=tuple(self.symbols))
        self.proxy, self.horizon = "SPY", 2
        self.desk = PickDesk(decay=0.98, ridge=300.0)
        self.brain = dict(hist=[["2025-03-03", 3, 0.7]])


class FakeEngine:
    def __init__(self, tmp, stocks=False, delay=0.0):
        self.s = Store(os.path.join(tmp, "board.db"))
        _f = SimFeed(SYMS, start_ts=T0, seed=11, history=600)
        self.feed = FakeFeed({s: list(_f.bars[s]) for s in SYMS}, delay=delay)
        self.now = _f.bars["BTC-USD"][-1].ts + HOUR + 60
        self.universe, self.proxy, self.horizon = list(SYMS), "BTC-USD", 4
        self.cfg = dict(desk=DESK)
        self.desk = PickDesk(**DESK)
        self.brain = dict(hist=[[P.day_str(T0 - 10 * DAY), 3, 0.6], [P.day_str(T0 - 5 * DAY), 1, 0.5]])
        self.stocks = FakeStocks(self.s) if stocks else None
        self.events = []

    def _ev(self, ts, kind, sev, title, body="", payload=None):
        self.events.append((kind, sev, title, body))


def wait(job, timeout=60.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        r = job.poll()
        if r is not None:
            return r
        time.sleep(0.05)
    raise AssertionError("pretrain job did not finish")


class TestPretrainJob(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_runs_in_a_thread_and_applies(self):
        eng = FakeEngine(self.tmp, stocks=True)
        job = P.PretrainJob(eng, hours=500, min_rounds_fresh=48)
        self.assertTrue(job.start(eng.now))
        self.assertFalse(job.start(eng.now))            # one at a time
        res = wait(job)
        self.assertIsNone(job.poll())                   # handed over once
        self.assertNotIn("error", res, res.get("error"))
        self.assertGreater(res["crypto"]["report"]["rounds"], 100)
        self.assertEqual(res["crypto"]["report"]["backfilled"], 6 * 500)
        self.assertGreater(res["stocks"]["report"]["rounds"], 50)
        self.assertIs(res["final_state"], res["crypto"]["final_state"])
        self.assertEqual(len(eng.s.get_bars("ETH-USD", HOUR, 5000)), 500)
        before = eng.desk.state()
        out = P.apply(eng, res, eng.now)
        self.assertIn("crypto", out)
        self.assertIn("stocks", out)
        kv = eng.s.kv_get("pretrain")
        self.assertEqual(kv["ts"], eng.now)
        self.assertIn("adopted", kv["crypto"])
        ev = [e for e in eng.events if e[0] == "learn"]
        self.assertEqual(len(ev), 1)
        self.assertIn("Advisors pretrained on", ev[0][2])
        self.assertIn("IC", ev[0][3])
        if out["crypto"][0]:
            self.assertEqual(eng.desk.rounds, res["crypto"]["final_state"]["rounds"])
            self.assertEqual(eng.s.kv_get("desk")["rounds"], eng.desk.rounds)
        else:
            self.assertTrue(same_state(before, eng.desk.state()))
        if out["stocks"][0]:
            self.assertEqual(eng.s.kv_get("stock_desk")["rounds"], eng.stocks.desk.rounds)
        eng.s.close()

    def test_forced_adopt_and_keep(self):
        eng = FakeEngine(self.tmp)
        final = P.replay_crypto({s: v[:400] for s, v in sim_bars().items()}, "BTC-USD", DESK, 4)["desk"].state()
        blk = lambda ic: dict(ic=ic, top=0.001, n=40)  # noqa: E731
        good = dict(crypto=dict(report=dict(rounds=900, hours=1000, challenger=blk(0.05), prior=blk(0.0), champion=None),
                                final_state=final))
        self.assertEqual(P.apply(eng, good, 1.0)["crypto"][0], True)
        self.assertEqual(eng.desk.rounds, final["rounds"])
        self.assertEqual(eng.s.kv_get("desk")["rounds"], final["rounds"])
        # now the live desk is mature and there is no champion score: keep it
        bad = dict(crypto=dict(report=dict(rounds=900, hours=1000, challenger=blk(0.05), prior=blk(0.0), champion=None),
                               final_state=PickDesk(**DESK).state()))
        self.assertFalse(P.apply(eng, bad, 2.0)["crypto"][0])
        self.assertEqual(eng.desk.rounds, final["rounds"])
        broken = dict(crypto=dict(report=good["crypto"]["report"], final_state={"Sxx": 1}))
        eng.desk = PickDesk(**DESK)
        self.assertFalse(P.apply(eng, broken, 3.0)["crypto"][0])
        wrong = dict(crypto=dict(report=good["crypto"]["report"], final_state=dict(final, Sxx=[[1.0]], Sxy=[1.0])))
        self.assertFalse(P.apply(eng, wrong, 4.0)["crypto"][0])     # loads half-way, then fails: restored
        self.assertTrue(same_state(eng.desk.state(), PickDesk(**DESK).state()))
        eng.desk.predict(np.zeros((4, len(final["Sxy"]))))
        eng.s.close()

    def test_cancel(self):
        eng = FakeEngine(self.tmp, delay=0.02)
        job = P.PretrainJob(eng, hours=500)
        job.start(eng.now)
        job.cancel()
        res = wait(job, 20)
        self.assertTrue(res.get("cancelled"))
        P.apply(eng, res, eng.now)
        self.assertEqual(eng.events[-1][1], "warn")
        self.assertIn("error", eng.s.kv_get("pretrain"))
        eng.s.close()

    def test_failure_is_reported_not_raised(self):
        eng = FakeEngine(self.tmp)
        eng.s.path = os.path.join(self.tmp, "no", "such", "dir", "x.db")   # the thread cannot open its store
        job = P.PretrainJob(eng, hours=200)
        job.start(eng.now)
        res = wait(job, 20)
        self.assertIn("error", res)
        self.assertEqual(job.status["phase"], "failed")
        eng.s.close()


# ---------------------------------------------------------------------------------- cadence
def O(sym, side, n):
    return Order(sym, side, float(n), "market", None, "test")


def plan(acct, weights, quotes, min_trade=25.0, band=0.02):
    """Engine.plan without the stock-session check."""
    orders = []
    for sym in sorted(set(weights) | set(acct.positions)):
        if sym not in quotes:
            continue
        cur = acct.positions.get(sym, {}).get("value", 0.0)
        diff = weights.get(sym, 0.0) * acct.equity - cur
        if abs(diff) < max(min_trade, band * acct.equity):
            continue
        orders.append(Order(sym, "buy" if diff > 0 else "sell", abs(diff), "market", None, "rebalance"))
    orders.sort(key=lambda o: (o.side != "sell", -o.notional_usd))
    return orders


class TestCadenceFilter(unittest.TestCase):
    RULES = dict(every_hours=2, min_conviction=0.35, cooldown_hours=6, max_trades_per_day=6)

    def test_conviction_cooldown_and_limit(self):
        now = 100 * HOUR
        orders = [O("A", "buy", 500), O("B", "buy", 900), O("C", "sell", 100), O("D", "buy", 300), O("E", "sell", 50),
                  O("F", "buy", 200)]
        conv = dict(A=0.5, B=0.36, C=-1.0, D=0.2, F=0.9)
        last = dict(F=now - 5 * HOUR, E=now - 7 * HOUR)
        keep, dropped = cadence_filter(orders, conv, last, 3, now, self.RULES)
        self.assertEqual([(o.symbol, o.side) for o in keep], [("C", "sell"), ("E", "sell"), ("B", "buy")])
        why = {o.symbol: r for o, r in dropped}
        self.assertIn("below", why["D"])
        self.assertIn("cooldown", why["F"])
        self.assertIn("trades a day", why["A"])
        self.assertEqual(len(keep) + len(dropped), len(orders))

    def test_sells_ignore_conviction_but_not_cooldown(self):
        now = 50 * HOUR
        keep, dropped = cadence_filter([O("A", "sell", 10)], {}, {"A": now - HOUR}, 0, now, self.RULES)
        self.assertEqual(keep, [])
        keep, _ = cadence_filter([O("A", "sell", 10), O("B", "buy", 10)], {}, {}, 0, now, self.RULES)
        self.assertEqual([o.symbol for o in keep], ["A"])

    def test_limit_already_used_and_defaults(self):
        keep, dropped = cadence_filter([O("A", "sell", 10)], {}, {}, 6, 0.0, self.RULES)
        self.assertEqual(keep, [])
        self.assertEqual(len(dropped), 1)
        keep, _ = cadence_filter([O("A", "buy", 10)], {"A": float("nan")}, {}, 0, 0.0)
        self.assertEqual(keep, [])
        keep, _ = cadence_filter([O("A", "buy", 10)], {"A": 0.35}, None, None, 0.0)
        self.assertEqual(len(keep), 1)

    def test_max_drawdown(self):
        self.assertEqual(max_drawdown([]), 0.0)
        self.assertAlmostEqual(max_drawdown([100, 120, 90, 130, 117]), -0.25)
        self.assertEqual(max_drawdown([None, 0, 5, 6]), 0.0)


class TestCadenceBooks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.store = Store(os.path.join(self.tmp, "b.db"))
        self.cfg = dict(paper=dict(start_cash=10000, fee_bps=40, slippage_bps=5), rebalance_every_hours=8)

    def tearDown(self):
        self.store.close()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_hours(self, books, h0, n, weights_fn, conv, price_fn):
        out = []
        for h in range(h0, h0 + n):
            bar_ts = h * HOUR
            q = {s: (price_fn(s, h), bar_ts + HOUR) for s in ("BTC-USD", "ETH-USD", "SOL-USD")}
            out.append((h, books.step(bar_ts + HOUR + 60, bar_ts, weights_fn(h), conv, q, plan)))
        return out

    def test_schedules_rules_and_stats(self):
        books = CadenceBooks(self.cfg, self.store)
        h0 = 480000   # divisible by 8
        w = lambda h: {"BTC-USD": 0.4, "ETH-USD": 0.3 if (h // 3) % 2 else 0.1, "SOL-USD": 0.2}  # noqa: E731
        px = lambda s, h: {"BTC-USD": 60000, "ETH-USD": 3000, "SOL-USD": 150}[s] * (1 + 0.01 * math.sin(h / 5))  # noqa: E731
        conv = {"BTC-USD": 0.6, "ETH-USD": 0.5, "SOL-USD": 0.1}
        res = self.run_hours(books, h0, 48, w, conv, px)
        for h, r in res:
            if h % 8:
                self.assertEqual(r["base"], [])
            if h % 2:
                self.assertEqual(r["cad"], [])
        self.assertTrue(res[0][1]["base"])               # the first 8-hour mark trades
        self.assertTrue(res[0][1]["cad"])
        self.assertTrue(any(s == "SOL-USD" and "below" in why for _, r in res for s, _, why in r["dropped"]))
        self.assertTrue(any("cooldown" in why for _, r in res for _, _, why in r["dropped"]))
        # cadence never trades a symbol within 6 hours of its last trade, nor more than 6 a day
        trades = [(h, f["symbol"]) for h, r in res for f in r["cad"]]
        last = {}
        for h, s in trades:
            if s in last:
                self.assertGreaterEqual(h - last[s], 6)
            last[s] = h
        per_day = {}
        for h, _ in trades:
            d = int((h * HOUR + HOUR + 60) // DAY)
            per_day[d] = per_day.get(d, 0) + 1
        self.assertLessEqual(max(per_day.values()), 6)
        st = books.stats(48 * HOUR + h0 * HOUR)
        self.assertAlmostEqual(st["days"], 2.0, delta=0.05)
        self.assertEqual(st["cad_trades"], len(trades))
        self.assertGreater(st["base_fees"], 0)
        self.assertLessEqual(st["base_dd"], 0)
        self.assertEqual(len(books.view()["history"]), 48)
        for k in ("days", "base_ret", "cad_ret", "base_dd", "cad_dd", "base_fees", "cad_fees", "base_trades", "cad_trades"):
            self.assertIn(k, st)
        # persisted: a restart picks up the same books
        again = CadenceBooks(self.cfg, self.store)
        self.assertEqual(again.stats(48 * HOUR + h0 * HOUR), st)
        self.assertIsNone(again.step(0, res[-1][0] * HOUR, {}, {}, {}, plan))   # same bar twice: ignored

    def test_defensive(self):
        books = CadenceBooks(self.cfg, self.store)
        q = {"BTC-USD": (60000.0, 1.0), "ETH-USD": (3000.0, 1.0)}
        r = books.step(10.0, 8 * HOUR, {"BTC-USD": 0.5, "ETH-USD": 0.6}, {"BTC-USD": 1, "ETH-USD": 1}, q, plan)
        self.assertGreater(len(r["base"]), 0)
        self.assertGreaterEqual(books.books["base"].cash, 0)        # an over-allocated plan is clamped to cash
        def boom(*a):
            raise RuntimeError("plan broke")
        r = books.step(20.0, 16 * HOUR, {"BTC-USD": 0.1}, {}, q, boom)
        self.assertEqual(r["base"], [])
        # a held symbol with no quote this hour is valued at its last price, never traded
        r = books.step(30.0, 24 * HOUR, {}, {}, {"ETH-USD": (3000.0, 1.0), "BAD": (None, 1), "NEG": (-1, 1)}, plan)
        self.assertNotIn("BTC-USD", [f["symbol"] for f in r["base"]])
        self.assertIn("ETH-USD", [f["symbol"] for f in r["base"]])
        self.assertGreater(books.hist[-1][1], 5000)
        self.assertNotIn("BAD", books.px)
        self.assertIsNone(books.step(40.0, None, {}, {}, q, plan))
        for bad in ({"base": {"cash": "x"}, "cad": 5}, {"base": {"cash": "x", "qty": {}}}, [1, 2],
                    {"hist": [[1, 2]]}, {"per_day": {"monday": 1}}):
            self.store.kv_set("cadence_books", bad)
            fresh = CadenceBooks(self.cfg, self.store)    # a damaged saved state starts the books over
            self.assertEqual(fresh.books["base"].cash, 10000)
            self.assertEqual(fresh.hist, [])
            fresh.step(50.0, 32 * HOUR, {"BTC-USD": 0.2}, {}, q, plan)

    def test_history_cap(self):
        books = CadenceBooks(self.cfg, self.store)
        books.hist = [[i, 1.0, 1.0] for i in range(24 * 120 + 5)]
        books.step(1.0, 3 * HOUR, {}, {}, {}, plan)
        self.assertEqual(len(books.hist), 24 * 120)


if __name__ == "__main__":
    unittest.main()
