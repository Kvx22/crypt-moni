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
from cryptomoni.feeds import HOUR, CoinbaseAdvanced, CoinbaseExchange, Kraken, SimFeed, completed  # noqa: E402
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

    def test_only_paper_mode(self):
        self.cfg.mode = "live"
        v = self.gate.check(Order("BTC-USD", "buy", 100), self.ctx())
        self.assertFalse(v.ok)
        self.assertIn("paper-only", v.reasons[0])

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
        self.assertEqual(len(s["coins"]), len(self.cfg["universe"]))
        self.assertTrue(all(c["verdict"] in ("BUY", "WATCH", "NEUTRAL", "AVOID", "VETOED") for c in s["coins"]))
        self.assertGreater(s["equity"], 0)
        self.assertLessEqual(s["e_target"], s["breaker"]["cap"] + 1e-9)
        self.assertTrue(s["minutes"])
        live = self.eng.s.kv_get("live")
        self.assertEqual(len(live["coins"]), len(self.cfg["universe"]))
        self.assertFalse(os.path.exists(self.cfg["safety"]["kill_file"]))

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
