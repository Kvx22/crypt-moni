"""python -m unittest discover -s tests"""
import ast
import json
import os
import shutil
import sys
import tempfile
import threading
import time
import unittest
import urllib.request

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from cryptomoni import advisors, scanner  # noqa: E402
from cryptomoni.advisors import ADVISORS, ExposureBoard, PickDesk  # noqa: E402
from cryptomoni.allocation import Allocator  # noqa: E402
from cryptomoni.engine import read_daily_csv  # noqa: E402
from cryptomoni.feeds import HOUR, CoinbaseAdvanced, CoinbaseExchange, Kraken, SimFeed, Yahoo, completed, stock_market_open  # noqa: E402
from cryptomoni.models import Bar, Order  # noqa: E402
from cryptomoni.paper import PaperBroker  # noqa: E402
from cryptomoni.regime import HMM, RegimeTracker, make_features  # noqa: E402
from cryptomoni.safety import BreakerState, CircuitBreaker, GateContext, KillSwitch, SafetyConfig, SafetyGate  # noqa: E402

BTC = read_daily_csv(os.path.join(ROOT, "data", "BTC-USD.csv"))
CLOSES = np.array(list(BTC.values()), float)


def bars_from(closes, vol=1000.0, t0=1_700_000_000):
    t0 = t0 // HOUR * HOUR
    out, prev = [], closes[0]
    for i, c in enumerate(closes):
        out.append(Bar(t0 + i * HOUR, prev, max(prev, c) * 1.001, min(prev, c) * 0.999, c, vol))
        prev = c
    return out


class TestRegime(unittest.TestCase):
    def test_features_are_causal(self):
        a = make_features(CLOSES[:1500])
        b = make_features(CLOSES[:1600])
        np.testing.assert_allclose(a[400:], b[400:1500])

    def test_filter_is_causal(self):
        X = make_features(CLOSES[:2000])[365:]
        m = HMM().fit(X, n_init=1, n_iter=20)
        p1 = m.filter(X[:1200])
        p2 = m.filter(X)
        np.testing.assert_allclose(p1, p2[:1200], atol=1e-12)

    def test_states_sorted_worst_to_best(self):
        X = make_features(CLOSES)[365:]
        m = HMM().fit(X, n_init=1, n_iter=30)
        health = m.mu[:, 2] + m.mu[:, 3]
        self.assertTrue(np.all(np.diff(health) > 0))

    def test_peek_does_not_move_state(self):
        tr = RegimeTracker(n_init=1, n_iter=20)
        st = tr.update(CLOSES)
        self.assertAlmostEqual(float(st["probs"].sum()), 1.0, places=9)
        before = json.dumps(tr.state())
        p = tr.peek(np.append(CLOSES, CLOSES[-1] * 0.8))
        self.assertEqual(before, json.dumps(tr.state()))
        self.assertAlmostEqual(float(np.sum(p["probs"])), 1.0, places=9)

    def test_state_round_trip(self):
        tr = RegimeTracker(n_init=1, n_iter=20)
        tr.update(CLOSES[:-5])
        tr2 = RegimeTracker()
        tr2.load(json.loads(json.dumps(tr.state())))
        np.testing.assert_allclose(tr.update(CLOSES)["probs"], tr2.update(CLOSES)["probs"])


class TestSafety(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.cfg = SafetyConfig(max_per_trade_usd=500, allowlist=["BTC-USD", "SOL-USD"], kill_file=os.path.join(self.dir, "KILL"))
        self.gate = SafetyGate(self.cfg)
        self.now = 1_800_000_000.0

    def tearDown(self):
        shutil.rmtree(self.dir)

    def ctx(self, **kw):
        base = dict(now=self.now, equity=10000, cash=5000, positions={"BTC-USD": 1000.0},
                    quotes={"BTC-USD": (80000.0, self.now - 10), "SOL-USD": (140.0, self.now - 10)}, breaker=BreakerState())
        base.update(kw)
        return GateContext(**base)

    def test_safety_imports_nothing_from_the_brain(self):
        src = open(os.path.join(ROOT, "cryptomoni", "safety.py")).read()
        mods = {n.module for n in ast.walk(ast.parse(src)) if isinstance(n, ast.ImportFrom)}
        self.assertFalse(mods & {"regime", "advisors", "scanner", "allocation", "engine", "feeds"})

    def test_unset_cap_refuses(self):
        self.cfg.max_per_trade_usd = None
        self.assertFalse(self.gate.check(Order("BTC-USD", "buy", 100), self.ctx()).ok)

    def test_live_mode_needs_both_arming_keys(self):
        self.cfg.mode = "live"
        os.environ.pop("CM_LIVE", None)
        self.cfg.live_armed = True
        v = self.gate.check(Order("BTC-USD", "buy", 100), self.ctx())
        self.assertFalse(v.ok)
        self.assertIn("not armed", v.reasons[0])
        os.environ["CM_LIVE"] = "1"
        try:
            self.cfg.live_armed = False
            self.assertFalse(self.gate.check(Order("BTC-USD", "buy", 100), self.ctx()).ok)
            self.cfg.live_armed = True
            self.assertTrue(self.gate.check(Order("BTC-USD", "buy", 100), self.ctx()).ok)
        finally:
            os.environ.pop("CM_LIVE")
        self.cfg.mode = "margin"
        self.assertFalse(self.gate.check(Order("BTC-USD", "buy", 100), self.ctx()).ok)

    def test_kill_switch_blocks(self):
        KillSwitch(self.cfg.kill_file).trip("test")
        self.assertIn("KILL", self.gate.check(Order("BTC-USD", "buy", 100), self.ctx()).reasons[0])

    def test_clamps_and_rules(self):
        v = self.gate.check(Order("SOL-USD", "buy", 5000), self.ctx())
        self.assertTrue(v.ok and v.clamped)
        self.assertLessEqual(v.order.notional_usd, 500)
        self.assertFalse(self.gate.check(Order("SOL-USD", "sell", 100), self.ctx()).ok)            # no shorting
        self.assertFalse(self.gate.check(Order("DOGE-USD", "buy", 100), self.ctx()).ok)           # allowlist
        stale = self.ctx(quotes={"BTC-USD": (80000.0, self.now - 3600)})
        self.assertFalse(self.gate.check(Order("BTC-USD", "buy", 100), stale).ok)                 # stale quote
        self.assertFalse(self.gate.check(Order("BTC-USD", "buy", 100, "limit", 90000), self.ctx()).ok)  # price band

    def test_breaker_latches_halt(self):
        br = CircuitBreaker(self.cfg)
        br.update(10000, 10000)
        self.assertEqual(br.update(8900, 10000).name, "YELLOW")
        s = br.update(7400, 10000)
        self.assertTrue(s.halted and not s.allow_buys and s.exposure_cap == 0.0)
        self.assertTrue(br.update(9900, 9900).halted)  # recovery does not clear it
        br.reset(9900)
        self.assertEqual(br.update(9900, 9900).name, "OK")


class TestScanner(unittest.TestCase):
    def test_rsi_extremes(self):
        self.assertGreater(scanner.rsi(np.linspace(1, 2, 60)), 99)
        self.assertLess(scanner.rsi(np.linspace(2, 1, 60)), 1)

    def test_metrics_numbers(self):
        c = 100 * np.exp(np.linspace(0, 0.1, 300))
        m = scanner.metrics(bars_from(c))
        self.assertAlmostEqual(m["ret_24h"], c[-1] / c[-25] - 1, places=9)
        self.assertAlmostEqual(m["ret_7d"], c[-1] / c[-169] - 1, places=9)
        self.assertGreater(m["range_pos"], 1.0)  # a fresh high
        self.assertIsNone(scanner.metrics(bars_from(c[:100])))

    def test_risk_orders_by_volatility(self):
        rng = np.random.default_rng(0)
        calm = 100 * np.exp(np.cumsum(rng.normal(0, 0.002, 300)))
        wild = 100 * np.exp(np.cumsum(rng.normal(0, 0.03, 300)))
        a, b = scanner.metrics(bars_from(calm, 1e6)), scanner.metrics(bars_from(wild, 1e6))
        self.assertLess(a["risk"], b["risk"])
        self.assertIn(b["tier"], ("HIGH", "EXTREME"))

    def test_alert_on_big_hourly_move(self):
        rng = np.random.default_rng(1)
        c = 100 * np.exp(np.cumsum(rng.normal(0, 0.003, 300)))
        m = scanner.metrics(bars_from(c), live_price=max(c[-170:]) * 1.05)
        kinds = {k for k, *_ in scanner.alerts("SOL-USD", m, None)}
        self.assertIn("move", kinds)
        self.assertIn("breakout", kinds)


class TestAdvisors(unittest.TestCase):
    def test_desk_learns_a_planted_signal(self):
        """When 24h momentum truly predicts: the momentum trader (Kai) leads at first,
        then everyone learns it; the contrarian (Vera) flips her momentum weight."""
        rng = np.random.default_rng(3)
        desk = PickDesk()
        j = scanner.SIGNALS.index("mom24")
        k = advisors.KEYS
        start = desk.W[:, j].copy()
        self.assertLess(start[k.index("vera")], 0)
        for t in range(400):
            X = np.clip(rng.standard_normal((12, len(scanner.SIGNALS))), -3, 3)
            S = desk.predict(X)
            fwd = 0.01 * X[:, j] + 0.01 * rng.standard_normal(12)
            desk.learn(X, S, fwd)
            if t == 15:
                self.assertEqual(int(np.argmax(desk.w)), k.index("kai"))
        self.assertTrue(np.all(desk.W[:, j] > start))
        self.assertGreater(desk.W[k.index("vera"), j], 0.1)      # she learned against her instinct
        self.assertTrue(np.all(desk.ic > 0.3))
        self.assertAlmostEqual(float(desk.w.sum()), 1.0, places=9)
        self.assertGreater(min(desk.w), 0.0)  # fixed share: nobody is silenced
        # stubborn Kai keeps more of his instinct than open-minded Vera
        self.assertLess(desk.learned_share()[k.index("kai")], desk.learned_share()[k.index("vera")])

    def test_chair_vetoes_extreme(self):
        desk = PickDesk()
        self.assertEqual(desk.verdict(0.9, "EXTREME"), "VETOED")
        self.assertEqual(desk.verdict(0.9, "HIGH"), "BUY")
        self.assertEqual(desk.verdict(-0.9, "LOW"), "AVOID")

    def test_exposure_board_chair_ceiling(self):
        bd = ExposureBoard(chair_slack=0.0)
        d = bd.decide([0, 0, 0, 0, 1.0], [0, 0, 0, 0, 1.0], 0.9)
        self.assertLessEqual(d["board"], d["votes"][advisors.KEYS.index("imani")] + 1e-12)
        self.assertEqual(len(bd.minutes(d, 4, 0.9)), len(ADVISORS))

    def test_desk_state_round_trip(self):
        rng = np.random.default_rng(4)
        d1 = PickDesk()
        X = rng.standard_normal((10, len(scanner.SIGNALS)))
        d1.learn(X, d1.predict(X), rng.standard_normal(10) * 0.01)
        d2 = PickDesk()
        d2.load(json.loads(json.dumps(d1.state())))
        np.testing.assert_allclose(d1.W, d2.W)


class TestAllocationAndPaper(unittest.TestCase):
    def test_targets_respect_caps(self):
        al = Allocator(caps={"BTC-USD": 0.5}, default_cap=0.15)
        coins = [dict(symbol="SOL-USD", score=2.0, verdict="BUY"), dict(symbol="DOGE-USD", score=1.0, verdict="BUY"),
                 dict(symbol="BTC-USD", score=0.0, verdict="NEUTRAL"), dict(symbol="ETH-USD", score=-1, verdict="AVOID")]
        w, notes = al.targets(0.8, coins)
        self.assertLessEqual(w["SOL-USD"], 0.15 + 1e-12)
        self.assertNotIn("ETH-USD", w)
        self.assertAlmostEqual(sum(w.values()), 0.8)          # overflow went to the BTC core
        coins[2]["verdict"] = "AVOID"
        w, _ = al.targets(0.8, coins)
        self.assertNotIn("BTC-USD", w)
        self.assertLess(sum(w.values()), 0.8)                  # ...unless the desk dislikes BTC: cash

    def test_doubt_only_lowers_exposure(self):
        al = Allocator()
        self.assertLessEqual(al.invested_fraction(0.9, 0.3, None, 1.0, None), 0.9)
        self.assertEqual(al.invested_fraction(0.2, 0.3, None, 1.0, None), 0.2)

    def test_paper_round_trip_pays_fees(self):
        pb = PaperBroker(1000, fee_bps=40, slippage_bps=5)
        q = {"BTC-USD": (100.0, 0)}
        pb.place(Order("BTC-USD", "buy", 500), q, 0)
        pb.place(Order("BTC-USD", "sell", 10_000), q, 0)
        self.assertNotIn("BTC-USD", pb.qty)
        self.assertLess(pb.cash, 1000)
        self.assertGreater(pb.cash, 990)


class TestFeeds(unittest.TestCase):
    def test_parsers(self):
        ex = CoinbaseExchange.parse_candles([[1792170000, 80693.69, 81709.97, 81334.01, 81437.88, 571.77]])[0]
        self.assertEqual((ex.ts, ex.open, ex.high, ex.low, ex.close), (1792170000, 81334.01, 81709.97, 80693.69, 81437.88))
        adv = CoinbaseAdvanced.parse_candles({"candles": [{"close": "81437.88", "high": "81709.97", "low": "80693.69",
                                                           "open": "81334.01", "start": "1792170000", "volume": "571.77"}]})[0]
        self.assertEqual(adv, ex)
        kr = Kraken.parse_ohlc([[1792170000, "81334.01", "81709.97", "80693.69", "81437.88", "81300", "571.77", 900]])[0]
        self.assertEqual(kr, ex)
        self.assertEqual(Kraken().pair("BTC-USD"), "XBTUSD")
        self.assertEqual(Kraken().pair("SOL-USD"), "SOLUSD")

    def test_completed_drops_open_bar(self):
        bars = [Bar(0, 1, 1, 1, 1, 1), Bar(3600, 1, 1, 1, 1, 1)]
        self.assertEqual(len(completed(bars, 3600, 7199)), 1)

    def test_sim_feed_moves_one_hour_per_tick(self):
        f = SimFeed(["BTC-USD", "ETH-USD"], start_ts=1_800_000_000, anchors={"BTC-USD": 80000.0})
        n = len(f.candles("BTC-USD", HOUR))
        f.tick()
        self.assertEqual(f.candles("BTC-USD", HOUR)[-1].ts, f.candles("ETH-USD", HOUR)[-1].ts)
        self.assertGreaterEqual(len(f.candles("BTC-USD", HOUR)), min(n + 1, 300))


class TestStocks(unittest.TestCase):
    def test_market_hours(self):
        import datetime as dt
        from zoneinfo import ZoneInfo
        et = ZoneInfo("America/New_York")
        ts = lambda y, m, d, h, mi: dt.datetime(y, m, d, h, mi, tzinfo=et).timestamp()
        self.assertTrue(stock_market_open(ts(2026, 10, 16, 9, 30)))      # Friday open
        self.assertFalse(stock_market_open(ts(2026, 10, 16, 16, 0)))     # closing bell
        self.assertFalse(stock_market_open(ts(2026, 10, 17, 11, 0)))     # Saturday
        self.assertFalse(stock_market_open(ts(2026, 10, 16, 8, 0)))

    def test_yahoo_parse(self):
        j = {"chart": {"error": None, "result": [{
            "meta": {"regularMarketPrice": 771.79, "regularMarketTime": 1792180000},
            "timestamp": [1792071000, 1792157400, 1792243800],
            "indicators": {"quote": [{"open": [768.0, 770.0, None], "high": [772.0, 773.0, None], "low": [766.0, 769.0, None],
                                      "close": [769.6, 771.79, None], "volume": [60000000, 55000000, None]}]}}]}}
        bars, px, qt = Yahoo.parse(j)
        self.assertEqual(len(bars), 2)                               # the unfinished session (null close) is dropped
        self.assertEqual(bars[1].close, 771.79)
        self.assertEqual(bars[0].ts % 86400, 0)                      # midnight UTC of the exchange date
        self.assertEqual((px, qt), (771.79, 1792180000.0))
        with self.assertRaises(Exception):
            Yahoo.parse({"chart": {"result": None, "error": {"description": "Not Found"}}})

    def test_daily_metrics_beta_and_horizons(self):
        from cryptomoni.stocks import daily_metrics
        rng = np.random.default_rng(5)
        spy_r = rng.normal(0.0004, 0.01, 300)
        spy = 400 * np.exp(np.cumsum(spy_r))
        stock = 50 * np.exp(np.cumsum(2 * spy_r))                      # a 2x-beta name
        dates = [f"d{i:04d}" for i in range(300)]
        m = daily_metrics(dates, list(stock), dict(zip(dates, spy)))
        self.assertAlmostEqual(m["beta"], 2.0, delta=0.1)
        self.assertAlmostEqual(m["ret_24h"], stock[-1] / stock[-2] - 1, places=9)
        self.assertAlmostEqual(m["ret_7d"], stock[-1] / stock[-6] - 1, places=9)
        self.assertEqual(m["ret_1h"], 0.0)                              # no live price: no intraday move
        live = daily_metrics(dates, list(stock), dict(zip(dates, spy)), live=stock[-1] * 1.03)
        self.assertAlmostEqual(live["ret_1h"], 0.03, places=9)
        self.assertEqual(m["asset"], "stock")
        self.assertIsNone(daily_metrics(dates[:50], list(stock[:50]), {}))

    def test_stock_risk_is_stock_sized(self):
        from cryptomoni.stocks import daily_metrics
        rng = np.random.default_rng(6)
        calm = 100 * np.exp(np.cumsum(rng.normal(0, 0.006, 300)))
        wild = 100 * np.exp(np.cumsum(rng.normal(0, 0.035, 300)))
        dates = [f"d{i:04d}" for i in range(300)]
        spy = dict(zip(dates, calm))
        a, b = daily_metrics(dates, list(calm), spy), daily_metrics(dates, list(wild), spy)
        self.assertEqual(a["tier"], "LOW")
        self.assertLess(a["risk"], b["risk"])
        self.assertNotEqual(b["tier"], "LOW")                          # a 55%-volatility stock is no longer "low risk"

    def test_coinbase_cannot_hold_stocks(self):
        from cryptomoni.brokers import CoinbaseBroker
        from cryptomoni.engine import Engine
        from cryptomoni.store import Store
        d = tempfile.mkdtemp()
        try:
            cfg = json.load(open(os.path.join(ROOT, "config.coinbase.json")))
            cfg.update(db=os.path.join(d, "x.db"), stocks=dict(symbols=["SPY"], csv_dir=os.path.join(ROOT, "data", "stocks")))
            cfg["safety"]["kill_file"] = os.path.join(d, "KILL")
            st = Store(cfg["db"])
            with self.assertRaises(ValueError):
                Engine(cfg, SimFeed(["BTC-USD"], start_ts=time.time()), st, broker=CoinbaseBroker(client=object(), cache_s=0))
            st.close()
        finally:
            shutil.rmtree(d)

    def test_robinhood_snapshot_counts_other_holdings(self):
        from cryptomoni.brokers import RobinhoodHandoff
        d = tempfile.mkdtemp()
        try:
            path = os.path.join(d, "rh.json")
            json.dump(dict(ts=time.time(), cash=100.0, positions={"SPY": 2.0, "BTC-USD": 0.001}, other_value=300.0), open(path, "w"))
            a = RobinhoodHandoff(path).account({"SPY": (700.0, 0), "BTC-USD": (80000.0, 0)})
            self.assertAlmostEqual(a.equity, 100 + 300 + 1400 + 80, places=6)
            self.assertEqual(set(a.positions), {"SPY", "BTC-USD"})
        finally:
            shutil.rmtree(d)


class TestEngineEndToEnd(unittest.TestCase):
    """Runs the whole loop on the simulator: scanner -> brain -> advisors -> paper -> store -> dashboard API."""

    @classmethod
    def setUpClass(cls):
        from cryptomoni.__main__ import build, load_cfg
        cls.dir = tempfile.mkdtemp()
        cfg = load_cfg(os.path.join(ROOT, "config.sim.json"))
        cfg.update(db=os.path.join(cls.dir, "t.db"), daily_csv=os.path.join(ROOT, "data", "BTC-USD.csv"), warmup_days=30)
        cfg["safety"]["kill_file"] = os.path.join(cls.dir, "KILL")
        cfg["tracker"] = dict(n_init=1, n_iter=30)
        cls.cfg = cfg
        cls.eng = build(cfg)
        for _ in range(60):
            cls.eng.tick()

    @classmethod
    def tearDownClass(cls):
        cls.eng.s.close()
        shutil.rmtree(cls.dir)

    def test_board_convened_and_learned(self):
        s = self.eng.s.latest()
        self.assertTrue(s["simulated"])
        self.assertIn(s["regime"], ("crash", "bear", "neutral", "bull", "euphoria"))
        self.assertGreater(s["desk"]["rounds"], 40)
        self.assertAlmostEqual(sum(a["desk_weight"] for a in s["advisors"]), 1.0, places=6)
        self.assertEqual(len(s["coins"]), len(self.cfg["universe"]) + len(self.cfg["stocks"]["symbols"]))
        self.assertTrue(all(c["verdict"] in ("BUY", "WATCH", "NEUTRAL", "AVOID", "VETOED") for c in s["coins"]))
        self.assertGreater(s["equity"], 0)
        self.assertLessEqual(s["e_target"], s["breaker"]["cap"] + 1e-9)
        self.assertTrue(s["minutes"])
        live = self.eng.s.kv_get("live")
        self.assertEqual(len(live["coins"]), len(self.cfg["universe"]) + len(self.cfg["stocks"]["symbols"]))
        self.assertFalse(os.path.exists(self.cfg["safety"]["kill_file"]))

    def test_stock_book_runs_beside_crypto(self):
        s = self.eng.s.latest()
        self.assertIsNotNone(s["stocks"])
        self.assertIn(s["stocks"]["regime"], ("crash", "bear", "neutral", "bull", "euphoria"))
        stock_coins = [c for c in s["coins"] if c.get("asset") == "stock"]
        self.assertEqual(len(stock_coins), len(self.cfg["stocks"]["symbols"]))
        self.assertTrue(all("stock_desk_weight" in a for a in s["advisors"]))
        self.assertAlmostEqual(sum(a["stock_desk_weight"] for a in s["advisors"]), 1.0, places=6)
        w = self.eng.merged_weights()
        stock_w = sum(v for k, v in w.items() if "-" not in k)
        crypto_w = sum(v for k, v in w.items() if "-" in k)
        self.assertLessEqual(stock_w, self.cfg["stocks"]["share"] + 1e-9)       # each book stays inside its share
        self.assertLessEqual(crypto_w, 1 - self.cfg["stocks"]["share"] + 1e-9)
        live = self.eng.s.kv_get("live")
        self.assertEqual(sum(1 for c in live["coins"] if c.get("asset") == "stock"), len(self.cfg["stocks"]["symbols"]))

    def test_plan_skips_stocks_when_market_is_closed(self):
        from cryptomoni.models import Account
        acct = Account(10000.0, 10000.0, {})
        quotes = {"SPY": (700.0, time.time()), "BTC-USD": (80000.0, time.time())}
        w = {"SPY": 0.2, "BTC-USD": 0.2}
        orig = self.eng.feed.market_open
        try:
            self.eng.feed.market_open = lambda sym=None: False
            self.assertEqual({o.symbol for o in self.eng.plan(acct, w, quotes)}, {"BTC-USD"})
            self.eng.feed.market_open = lambda sym=None: True
            self.assertEqual({o.symbol for o in self.eng.plan(acct, w, quotes)}, {"BTC-USD", "SPY"})
        finally:
            self.eng.feed.market_open = orig

    def test_kill_switch_stops_paper_orders(self):
        KillSwitch(self.cfg["safety"]["kill_file"]).trip("test")
        try:
            n = len(self.eng.s.trades(1000))
            for _ in range(10):
                self.eng.tick()
            self.assertEqual(len(self.eng.s.trades(1000)), n)
        finally:
            KillSwitch(self.cfg["safety"]["kill_file"]).reset()

    def test_dashboard_api(self):
        from http.server import ThreadingHTTPServer
        from cryptomoni.dashboard import make_handler
        srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(self.cfg["db"], self.cfg["safety"]["kill_file"]))
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            base = f"http://127.0.0.1:{srv.server_address[1]}"
            def get(path):
                with urllib.request.urlopen(base + path) as r:
                    return r.read()
            st = json.loads(get("/api/state"))
            self.assertIn("snapshot", st)
            self.assertLess(st["heartbeat_age"], 600)
            self.assertTrue(json.loads(get("/api/history?limit=50")))
            self.assertIn(b"Crypto Risk Board", get("/"))
            req = urllib.request.Request(base + "/api/kill", method="POST")  # no header: refused
            with self.assertRaises(urllib.error.HTTPError):
                urllib.request.urlopen(req)
        finally:
            srv.shutdown()

    def test_export_snapshot(self):
        from cryptomoni.__main__ import export
        out = os.path.join(self.dir, "board.html")
        export(self.cfg, out)
        with open(out, encoding="utf-8") as f:
            page = f.read()
        self.assertIn("window.CM_STATIC=", page)
        self.assertNotIn("<!doctype", page.lower())
        self.assertEqual(page.count("</script>"), 2)  # data never closes the script tag early

    def test_watchdog_trips_on_stale_heartbeat(self):
        from cryptomoni.store import Store
        from cryptomoni.watchdog import check
        kill = KillSwitch(os.path.join(self.dir, "KILL.wd"))
        st = Store(self.cfg["db"])
        try:
            self.assertEqual(check(st, kill, now=time.time() + 10_000, stale_s=900), "tripped: stale")
            self.assertTrue(kill.engaged())
        finally:
            st.close()


if __name__ == "__main__":
    unittest.main()


class FakeCoinbase:
    """Stands in for coinbase.rest.RESTClient: same method names and response shapes."""

    def __init__(self, feed, usd=1000.0):
        self.feed, self.bal, self.calls = feed, {"USD": usd}, []

    def get_accounts(self, limit=None):
        return {"accounts": [{"currency": k, "available_balance": {"value": str(v)}, "hold": {"value": "0"}} for k, v in self.bal.items()]}

    def get_product(self, pid):
        return {"product_id": pid, "base_increment": "0.00000001", "quote_increment": "0.01", "quote_min_size": "1", "base_min_size": "0.00000001"}

    def market_order_buy(self, client_order_id, product_id, quote_size):
        base = product_id.split("-")[0]
        px = self.feed.bars[base + "-USD"][-1].close
        self.bal["USD"] -= float(quote_size)
        self.bal[base] = self.bal.get(base, 0.0) + float(quote_size) * 0.994 / px
        self.calls.append(("buy", client_order_id, product_id, quote_size))
        return {"success": True, "success_response": {"order_id": f"cb-{len(self.calls)}"}}

    def market_order_sell(self, client_order_id, product_id, base_size):
        base = product_id.split("-")[0]
        px = self.feed.bars[base + "-USD"][-1].close
        self.bal[base] -= float(base_size)
        self.bal["USD"] += float(base_size) * px * 0.994
        self.calls.append(("sell", client_order_id, product_id, base_size))
        return {"success": True, "success_response": {"order_id": f"cb-{len(self.calls)}"}}


class TestLiveApproval(unittest.TestCase):
    """Live mode: the board proposes, nothing is sent until a human approves, then fresh checks."""

    def setUp(self):
        from cryptomoni.__main__ import load_cfg
        from cryptomoni.brokers import CoinbaseBroker
        from cryptomoni.engine import Engine
        from cryptomoni.store import Store
        self.dir = tempfile.mkdtemp()
        cfg = load_cfg(os.path.join(ROOT, "config.coinbase.json"))
        cfg.update(db=os.path.join(self.dir, "live.db"), daily_csv=os.path.join(ROOT, "data", "BTC-USD.csv"), warmup_days=30)
        cfg["safety"].update(kill_file=os.path.join(self.dir, "KILL"), live_armed=True, max_per_trade_usd=400)
        cfg["tracker"] = dict(n_init=1, n_iter=30)
        cfg["execution"]["snapshot"] = os.path.join(self.dir, "rh.json")
        self.cfg = cfg
        os.environ["CM_LIVE"] = "1"
        last = list(BTC)[-1]
        # the simulator's clock starts at the real time, so wall-clock approval expiry lines up
        self.feed = SimFeed(cfg["universe"], start_ts=time.time() - 1800, anchors={"BTC-USD": BTC[last]}, step=60)
        self.cb = FakeCoinbase(self.feed)
        self.store = Store(cfg["db"])
        self.eng = Engine(cfg, self.feed, self.store, broker=CoinbaseBroker(client=self.cb, cache_s=0))
        for _ in range(62):   # one-minute ticks: the first hourly bar closes inside this
            self.eng.tick()

    def tearDown(self):
        os.environ.pop("CM_LIVE", None)
        self.store.close()
        shutil.rmtree(self.dir)

    def post(self, path, headers=None):
        from http.server import ThreadingHTTPServer
        from cryptomoni.dashboard import make_handler
        srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(self.cfg["db"], self.cfg["safety"]["kill_file"], "s3cret"))
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{srv.server_address[1]}{path}", method="POST", headers=headers or {})
            try:
                with urllib.request.urlopen(req) as r:
                    return r.status
            except urllib.error.HTTPError as e:
                return e.code
        finally:
            srv.shutdown()

    def test_nothing_sent_without_approval(self):
        pend = self.store.outbox(100, "pending")
        self.assertTrue(pend, "the board should have proposed orders")
        for _ in range(30):
            self.eng.tick()
        self.assertEqual(self.cb.calls, [])

    def test_approve_then_send_with_fresh_checks(self):
        pend = self.store.outbox(100, "pending")
        ok = {"X-CM": "1", "X-CM-Token": "s3cret"}
        self.assertEqual(self.post(f"/api/outbox/{pend[0]['id']}/approve", {"X-CM": "1"}), 403)   # token required
        self.assertEqual(self.post("/api/outbox/all/approve", ok), 200)
        # one order's price "moved" 5% since it was proposed: it must be skipped, not sent
        moved = pend[-1]["id"]
        self.store.c.execute("UPDATE outbox SET ref_price=ref_price*1.05 WHERE id=?", (moved,)); self.store.c.commit()
        self.eng.tick()
        sent = [r for r in self.store.outbox(100) if r["status"] == "sent"]
        self.assertTrue(sent)
        self.assertEqual(len(self.cb.calls), len(sent))
        self.assertEqual({c[1] for c in self.cb.calls}, {r["client_order_id"] for r in sent})
        self.assertEqual(self.store.outbox_get(moved)["status"], "skipped")
        self.assertTrue(all(float(c[3]) <= 400 for c in self.cb.calls if c[0] == "buy"))  # per-trade cap held

    def test_kill_switch_blocks_approval_and_sending(self):
        pend = self.store.outbox(100, "pending")
        ok = {"X-CM": "1", "X-CM-Token": "s3cret"}
        self.assertEqual(self.post(f"/api/outbox/{pend[0]['id']}/approve", ok), 200)
        KillSwitch(self.cfg["safety"]["kill_file"]).trip("test")
        self.assertEqual(self.post(f"/api/outbox/{pend[1]['id']}/approve", ok), 409)
        self.eng.tick()
        self.assertEqual(self.cb.calls, [])
        self.assertEqual(self.store.outbox_get(pend[0]["id"])["status"], "skipped")

    def test_expired_orders_never_send(self):
        self.store.c.execute("UPDATE outbox SET status='approved', expires_ts=0"); self.store.c.commit()
        self.eng.tick()
        self.assertEqual(self.cb.calls, [])

    def test_robinhood_handoff(self):
        from cryptomoni.__main__ import main
        from cryptomoni.brokers import RobinhoodHandoff
        from cryptomoni.engine import Engine
        from cryptomoni.store import Store
        cfg = json.loads(json.dumps(self.cfg))
        cfg.update(db=os.path.join(self.dir, "rh.db"))
        cfg["execution"].update(broker="robinhood")
        rh = RobinhoodHandoff(cfg["execution"]["snapshot"])
        st = Store(cfg["db"])
        try:
            eng = Engine(cfg, self.feed, st, broker=rh)
            for _ in range(65):
                eng.tick()
            self.assertFalse(st.outbox(10))                       # no synced balances: no proposal
            self.assertIn("stale", st.events(20, "outbox")[0]["title"])
            with open(cfg["execution"]["snapshot"], "w") as f:
                json.dump(dict(ts=time.time(), cash=500.0, positions={}), f)
            st.kv_set("e", None)                                  # next bar proposes
            for _ in range(61):
                eng.tick()
            pend = st.outbox(100, "pending")
            self.assertTrue(pend)
            st.outbox_set(pend[0]["id"], "approved", time.time(), decided_by="test")
            eng.tick()
            self.assertEqual(st.outbox_get(pend[0]["id"])["status"], "approved")   # Python never places Robinhood orders
            cfgp = os.path.join(self.dir, "rh.json.cfg")
            with open(cfgp, "w") as f:
                json.dump(cfg, f)
            main(["ack", str(pend[0]["id"]), "--config", cfgp, "--status", "sent", "--order-id", "rh-1"])
            self.assertEqual(st.outbox_get(pend[0]["id"])["status"], "sent")
        finally:
            st.close()
