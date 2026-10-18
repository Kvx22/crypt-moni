"""End to end on the simulator: the research service (in-process, simulated sources) feeds the
board; research stays in shadow until its gates pass; nothing research does can place an order.
python -m unittest tests.test_research_integration"""
import json
import os
import shutil
import sys
import tempfile
import threading
import unittest
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


class TestResearchOnTheBoard(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from cryptomoni.__main__ import build, load_cfg
        cls.dir = tempfile.mkdtemp()
        cfg = load_cfg(os.path.join(ROOT, "config.sim.json"))
        cfg.update(db=os.path.join(cls.dir, "s.db"), daily_csv=os.path.join(ROOT, "data", "BTC-USD.csv"), warmup_days=30,
                   scout=dict(crypto_hours=1, top=5, crypto=["ATOM-USD", "NEAR-USD", "SUI-USD"], stocks=["QQQ", "NVDA", "TLT"]))
        cfg["stocks"]["warmup_days"] = 30
        cfg["safety"]["kill_file"] = os.path.join(cls.dir, "KILL")
        cfg["tracker"] = dict(n_init=1, n_iter=30)
        cfg["pretrain"] = dict(enabled=False)
        cls.cfg = cfg
        cls.eng = build(cfg)
        assert cls.eng.research_service is not None
        for _ in range(80):
            cls.eng.tick()

    @classmethod
    def tearDownClass(cls):
        cls.eng.s.close()
        shutil.rmtree(cls.dir, ignore_errors=True)

    def test_service_fills_the_research_database(self):
        r = self.eng.research.db
        self.assertTrue(r.subscribed(now=self.eng.feed.now()))
        self.assertGreater(len(r.notes(1000)), 10)
        self.assertTrue(r.view())
        self.assertTrue(r.workers())
        self.assertTrue(r.kv_get("health_now"))

    def test_research_reaches_the_board_in_shadow(self):
        s = self.eng.s.latest()
        self.assertTrue(s["research"]["enabled"])
        self.assertTrue(all(v == "shadow" for v in s["research"]["stages"].values()))
        coins = [c for c in s["coins"] if c.get("asset", "crypto") == "crypto"]
        self.assertTrue(all("research" in c for c in coins))
        self.assertTrue(all(not c["research"]["applied"]["signals"] for c in coins))
        shadow = self.eng.research.db.shadow("score")
        self.assertGreater(len(shadow), 50)
        self.assertTrue(all(r["detail"].get("price0") for r in shadow[:20]))

    def test_research_desk_learns_from_graded_rounds(self):
        self.assertGreater(self.eng.rx.rdesk.rounds, 20)
        self.assertTrue(self.eng.s.kv_get("rdesk"))

    def test_outcomes_and_cadence_books(self):
        r = self.eng.research.db
        n = r.q1("SELECT COUNT(*) n FROM outcomes")["n"]
        self.assertGreater(n, 0)
        st = self.eng.cadence.stats(self.eng.feed.now())
        self.assertIn("cad_ret", st)

    def test_dashboard_research_route(self):
        from http.server import ThreadingHTTPServer
        from cryptomoni.dashboard import make_handler
        from cryptomoni.research.db import research_path
        srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(self.cfg["db"], self.cfg["safety"]["kill_file"], "tok",
                                                                 research_path(self.cfg), self.eng.research.board))
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{srv.server_address[1]}"
        try:
            with urllib.request.urlopen(base + "/api/research") as resp:
                d = json.loads(resp.read())
            self.assertTrue(d["available"])
            self.assertTrue(d["notes"])
            nid = d["notes"][0]["id"]
            body = json.dumps(dict(catalyst="earnings", direction=1, flags=[])).encode()
            req = urllib.request.Request(base + f"/api/research/label/{nid}", data=body, method="POST",
                                         headers={"X-CM": "1", "X-CM-Token": "tok", "Content-Type": "application/json"})
            with urllib.request.urlopen(req) as resp:
                self.assertTrue(json.loads(resp.read())["ok"])
            self.assertEqual(self.eng.research.db.labels()[-1]["catalyst"], "earnings")
            bad = urllib.request.Request(base + f"/api/research/label/{nid}", data=body, method="POST", headers={"X-CM": "1"})
            with self.assertRaises(urllib.error.HTTPError):
                urllib.request.urlopen(bad)
        finally:
            srv.shutdown()

    def test_research_never_trades_and_switches_apply_one_at_a_time(self):
        from cryptomoni.research.stages import set_manual
        db = self.eng.research.db
        set_manual(db, "penalties", "active")
        self.eng.research._stages_ts = 0
        for _ in range(3):
            self.eng.tick()
        s = self.eng.s.latest()
        coins = [c for c in s["coins"] if c.get("asset", "crypto") == "crypto"]
        self.assertTrue(all(c["research"]["applied"]["penalties"] for c in coins))
        self.assertTrue(all(not c["research"]["applied"]["signals"] for c in coins))
        set_manual(db, "penalties", "auto")
        self.eng.research._stages_ts = 0
        self.eng.tick()   # leave the board's latest snapshot back in shadow for the tests that follow

    def test_broken_research_database_does_not_stop_the_board(self):
        link = self.eng.research
        real = link.db
        try:
            class Broken:
                def __getattr__(self, k):
                    raise RuntimeError("disk on fire")
            link.db = Broken()
            link._stages_ts = 0
            self.eng.tick()
            self.eng.tick()
            self.assertIsNotNone(self.eng.s.latest())
        finally:
            link.db = real
            link._stages_ts = 0


if __name__ == "__main__":
    unittest.main()
