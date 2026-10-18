"""python -m unittest tests.test_research_rdesk"""
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from cryptomoni.advisors import ADVISORS, KEYS  # noqa: E402
from cryptomoni.research import rdesk, stages  # noqa: E402
from cryptomoni.research.db import ResearchDB  # noqa: E402
from cryptomoni.research.rdesk import (RESEARCH_STYLE, ResearchDesk, combine, overlay, research_matrix,  # noqa: E402
                                       spearman, thin_requests)
from cryptomoni.research.taxonomy import RESEARCH_SIGNALS  # noqa: E402

DAY = 86400.0
NOW = 1_800_000_000.0


class TestResearchDesk(unittest.TestCase):
    def test_prior_shapes(self):
        d = ResearchDesk()
        self.assertEqual(set(RESEARCH_STYLE), set(KEYS))
        self.assertEqual(d.prior.shape, (len(ADVISORS), len(RESEARCH_SIGNALS)))
        np.testing.assert_allclose(np.linalg.norm(d.prior, axis=1), 0.15)
        i = KEYS.index("imani")
        self.assertEqual(int(np.argmax(d.prior[i])), RESEARCH_SIGNALS.index("r_risk"))
        self.assertLess(d.prior[KEYS.index("vera"), RESEARCH_SIGNALS.index("r_sentiment")], 0)
        # stubborn Kai (open-mindedness 0.2) is pulled harder than Vera (0.5)
        self.assertGreater(d.alpha[KEYS.index("kai")], d.alpha[KEYS.index("vera")])
        self.assertAlmostEqual(d.alpha[KEYS.index("vera")], 500.0)
        np.testing.assert_allclose(d.W, d.prior)
        self.assertEqual(d.learned_share(), [0.0] * 5)
        self.assertEqual(d.drift(), [0.0] * 5)

    def test_research_matrix(self):
        views = {"BTC-USD": {"features": {"r_catalyst": 0.5, "r_risk": -0.4, "r_sentiment": float("nan"), "r_structure": "x"}},
                 "SOL-USD": {"features": {"r_coverage": 7}}, "BAD": "junk"}
        R = research_matrix(views, ["BTC-USD", "SOL-USD", "ETH-USD", "BAD"])
        self.assertEqual(R.shape, (4, 6))
        self.assertEqual(R[0, 0], 0.5); self.assertEqual(R[0, 4], -0.4); self.assertEqual(R[0, 1], 0.0); self.assertEqual(R[0, 3], 0.0)
        self.assertEqual(R[1, 5], 1.0)                                   # clipped
        self.assertTrue((R[2:] == 0).all())
        self.assertEqual(research_matrix(None, ["A"]).shape, (1, 6))
        self.assertEqual(research_matrix({}, []).shape, (0, 6))

    def test_zr_zero_without_research(self):
        d = ResearchDesk()
        self.assertTrue((d.zr(np.zeros((8, 6))) == 0).all())
        self.assertTrue((d.zr(np.full((8, 6), 0.3)) == 0).all())          # same for every coin: no information
        self.assertTrue((d.zr(np.zeros((1, 6))) == 0).all())
        self.assertEqual(d.zr(np.zeros((0, 6))).shape, (5, 0))
        R = np.zeros((8, 6)); R[2, 0] = 0.9
        z = d.zr(R)
        self.assertEqual(z.shape, (5, 8))
        np.testing.assert_allclose(z.mean(axis=1), 0, atol=1e-12)
        self.assertTrue((z[:, 2] > 0).all())                              # every prior likes catalysts
        self.assertTrue(np.isfinite(d.zr(np.full((4, 6), np.nan))).all())

    def test_learning_planted_relation(self):
        rng = np.random.default_rng(1)
        d = ResearchDesk()
        w0 = d.W[:, 0].copy()
        j_vera_sent = RESEARCH_SIGNALS.index("r_sentiment")
        for _ in range(300):
            R = rng.uniform(-1, 1, size=(20, 6))
            fwd = 0.02 * R[:, 0] + 0.003 * rng.standard_normal(20)
            out = d.learn(R, fwd)
        self.assertEqual(len(out), 5)
        self.assertEqual(d.rounds, 300)
        share = d.learned_share()
        self.assertTrue(all(s > 0.5 for s in share), share)
        self.assertGreater(share[KEYS.index("vera")], share[KEYS.index("kai")])
        self.assertTrue((d.W[:, 0] > w0).all())                            # every advisor moved toward r_catalyst
        for a in range(5):
            self.assertEqual(int(np.argmax(np.abs(d.W[a]))), 0)
        self.assertGreater(d.W[KEYS.index("vera"), j_vera_sent], d.prior[KEYS.index("vera"), j_vera_sent])
        self.assertTrue(all(ic > 0.5 for ic in d.ic), d.ic)
        self.assertTrue(all(x > 0 for x in d.drift()))
        R = np.zeros((3, 6)); R[1, 0] = 1.0
        self.assertEqual(d.explain(0, R, 1)[0], "r_catalyst")
        self.assertEqual(d.explain(0, R, 0), [])

    def test_learn_skips_useless_rounds(self):
        d = ResearchDesk()
        self.assertIsNone(d.learn(np.zeros((10, 6)), np.linspace(-0.01, 0.01, 10)))       # no research yet
        self.assertIsNone(d.learn(np.eye(6)[:2], [0.01, -0.01]))                          # < 3 coins
        self.assertIsNone(d.learn(np.eye(6), [0.01] * 5))                                 # length mismatch
        self.assertEqual(d.rounds, 0); self.assertEqual(d.n_obs, 0.0)
        R = np.eye(6)[:4]
        fwd = np.array([0.02, np.nan, -0.01, 0.0])
        self.assertIsNotNone(d.learn(R, fwd))                                             # NaN coin dropped, 3 left
        self.assertEqual(d.n_obs, 3.0)
        self.assertTrue(np.isfinite(d.W).all())
        # constant returns: y = 0, no blow-up
        self.assertIsNotNone(d.learn(np.eye(6)[:5], [0.01] * 5))
        self.assertTrue(np.isfinite(d.W).all())

    def test_ic_uses_weights_before_update(self):
        d = ResearchDesk()
        R = np.zeros((6, 6)); R[:, 0] = np.linspace(-1, 1, 6)
        fwd = -R[:, 0]                                                    # opposite to every prior on r_catalyst
        out = d.learn(R, fwd)
        self.assertTrue(all(o["ic"] < -0.99 for o in out))

    def test_state_round_trip(self):
        rng = np.random.default_rng(2)
        d = ResearchDesk()
        for _ in range(20):
            R = rng.uniform(-1, 1, (10, 6)); d.learn(R, 0.01 * R[:, 2] + 0.001 * rng.standard_normal(10))
        s = d.state()
        import json
        s = json.loads(json.dumps(s))
        d2 = ResearchDesk()
        self.assertTrue(d2.load(s))
        np.testing.assert_allclose(d2.W, d.W); np.testing.assert_allclose(d2.ic, d.ic)
        self.assertEqual(d2.rounds, 20); self.assertAlmostEqual(d2.n_obs, d.n_obs)
        d3 = ResearchDesk()
        for bad in ({}, {"Sxx": [[1]], "Sxy": [1]}, None, {"Sxx": "x", "Sxy": []}):
            self.assertFalse(d3.load(bad))
        np.testing.assert_allclose(d3.W, d3.prior)
        sing = dict(s, Sxx=(-np.eye(6) * d3.alpha[0]).tolist())          # singular for advisor 0
        self.assertFalse(d3.load(sing))
        np.testing.assert_allclose(d3.W, d3.prior); self.assertEqual(d3.n_obs, 0.0)


class TestOverlayAndRequests(unittest.TestCase):
    def test_overlay(self):
        Z = np.array([[0.8, 0.1, -0.6], [0.4, 0.6, 0.0]])
        zr = np.array([[1.0, -1.0, 0.0], [0.5, 0.5, 0.5]])
        Z2 = overlay(Z, zr, weight=0.35)
        np.testing.assert_allclose(Z2, Z + 0.35 * zr)
        self.assertIsNot(Z2, Z)
        thin = [True, True, False]
        Z3 = overlay(Z, zr, thin=thin)
        self.assertAlmostEqual(Z3[0, 0], (0.8 + 0.35) * 0.85)        # liked and thin: discounted
        self.assertAlmostEqual(Z3[1, 1], (0.6 + 0.175) * 0.85)
        self.assertAlmostEqual(Z3[0, 1], 0.1 - 0.35)                  # thin but not liked: untouched
        self.assertAlmostEqual(Z3[1, 0], 0.4 + 0.175)                 # below buy: untouched
        np.testing.assert_allclose(overlay(Z, np.zeros_like(Z)), Z)  # no research: unchanged
        np.testing.assert_allclose(overlay(Z, None), Z)
        np.testing.assert_allclose(overlay(Z, np.ones((3, 3))), Z)   # wrong shape: ignored
        np.testing.assert_allclose(overlay(Z, np.full_like(Z, np.nan)), Z)
        Z0 = Z.copy(); overlay(Z, zr, thin=thin)
        np.testing.assert_array_equal(Z, Z0)                          # input not mutated

    def test_combine(self):
        w = np.array([0.5, 0.5]); Z = np.array([[1.0, 0.0], [0.0, 1.0]])
        np.testing.assert_allclose(combine(w, Z), [0.5, 0.5])
        np.testing.assert_allclose(combine(w, Z, [0.2, np.nan]), [0.3, 0.5])

    def test_thin_requests(self):
        Z = np.zeros((5, 3)); kai = KEYS.index("kai")
        Z[kai, 0] = 0.8; Z[kai, 1] = 0.9; Z[0, 2] = 1.2; Z[2, 0] = 0.5
        views = {"SOL-USD": {"n_independent": 1, "thin": True}, "BTC-USD": {"n_independent": 5}}
        reqs = thin_requests(Z, ["SOL-USD", "BTC-USD", "NVDA"], views)
        self.assertEqual([(r["symbol"], r["advisor_index"]) for r in reqs], [("NVDA", 0), ("SOL-USD", kai), ("SOL-USD", 2)])
        sol = reqs[1]
        self.assertEqual(sol["question"],
                         "Kai likes SOL (z +0.8) but only 1 independent source covers it: check for news, filings and risk")
        self.assertIn("no independent source covers it yet", reqs[0]["question"])
        self.assertIn("Vera likes NVDA", reqs[0]["question"])
        self.assertEqual(sol["advisor"], "kai")
        views["SOL-USD"]["n_independent"] = 2
        self.assertEqual([r["symbol"] for r in thin_requests(Z, ["SOL-USD", "BTC-USD", "NVDA"], views)], ["NVDA"])
        self.assertEqual(len(thin_requests(Z[:, :2], ["SOL-USD", "BTC-USD", "NVDA"], {})), 3)  # extra symbol ignored
        self.assertIn("only 2 independent sources cover it",
                      thin_requests(Z, ["SOL-USD"], views, thin_sources=3)[0]["question"])

    def test_spearman(self):
        self.assertAlmostEqual(spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)
        self.assertAlmostEqual(spearman([1, 2, 3, 4], [4, 3, 2, 1]), -1.0)
        self.assertEqual(spearman([1, 1, 1], [1, 2, 3]), 0.0)
        self.assertEqual(spearman([1, 2], [1, 2]), 0.0)
        self.assertAlmostEqual(spearman([1, 2, np.nan, 3], [1, 2, 9, 3]), 1.0)
        # ties share their average rank: [0,0,1] vs [1,2,3]
        self.assertAlmostEqual(spearman([0, 0, 1], [1, 2, 3]), np.corrcoef([0.5, 0.5, 2], [0, 1, 2])[0, 1])


# ---------------------------------------------------------------------------
class StageBase(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.db = ResearchDB(os.path.join(self.dir, "r.db"))

    def tearDown(self):
        self.db.close()
        shutil.rmtree(self.dir, ignore_errors=True)

    def penalty_rows(self, n, days, excess_fn, end=NOW, horizon="1d", start=None):
        start = end - days * DAY if start is None else start
        rows = [(start + (end - start) * k / max(n - 1, 1), "crypto", f"C{k}-USD", "crypto", "penalty", 0.8, 0.5,
                 {"horizon": horizon, "price0": 1, "bench0": 1}) for k in range(n)]
        self.db.add_shadow(rows)
        ids = [r["id"] for r in self.db.shadow(kind="penalty", matured=False)]
        self.db.mature_shadow([(excess_fn(k), i) for k, i in enumerate(ids)])

    def score_rows(self, hours, per_hour, good, end=NOW, horizon="4h"):
        rng = np.random.default_rng(3)
        rows = []
        for h in range(hours):
            ts = end - (hours - h) * 3600 - 60
            for k in range(per_hour):
                ex = rng.standard_normal() * 0.01
                base = rng.standard_normal()
                adj = (ex * 100 + 0.3 * rng.standard_normal()) if good else (-ex * 100)
                rows.append((ts, "crypto", f"C{k}-USD", "crypto", "score", base, adj, {"horizon": horizon, "ex": ex}))
        self.db.add_shadow(rows)
        sh = self.db.shadow(kind="score", matured=False)
        self.db.mature_shadow([(r["detail"]["ex"], r["id"]) for r in sh])

    def state(self, s):
        return self.db.stage(s)


class TestStages(StageBase):
    def test_defaults_and_overrides(self):
        self.assertEqual(stages.STAGES, ("penalties", "signals", "watchlist", "cadence"))
        g = stages.gates_from({"research": {"gates": {"penalties": {"min_events": 5}, "min_gap_days": 1}}})
        self.assertEqual(g["penalties"]["min_events"], 5); self.assertEqual(g["penalties"]["min_days"], 7)
        self.assertEqual(g["common"]["min_gap_days"], 1)
        self.assertEqual(stages.DEFAULT_GATES["penalties"]["min_events"], 30)   # defaults untouched
        self.assertEqual(stages.gates_from(None), stages.DEFAULT_GATES)
        self.assertEqual(stages.gates_from({"signals": {"min_obs": 9}})["signals"]["min_obs"], 9)

    def test_empty_db(self):
        self.assertEqual(stages.evaluate(self.db, NOW), [])
        ev = self.db.kv_get("stage_evidence")
        self.assertEqual(set(ev["stages"]), set(stages.STAGES))
        self.assertFalse(any(e["pass"] for e in ev["stages"].values()))
        self.assertTrue(all(self.state(s) == "shadow" for s in stages.STAGES))

    def test_good_penalty_record_promotes(self):
        self.penalty_rows(40, 9, lambda k: -0.01 if k % 4 else 0.004)   # 75% lag, mean -0.65%
        ch = stages.evaluate(self.db, NOW)
        self.assertEqual([(c["stage"], c["from"], c["to"]) for c in ch], [("penalties", "shadow", "active")])
        self.assertEqual(self.state("penalties"), "active")
        self.assertEqual(self.db.stages()["penalties"]["by"], "gate")
        e = ch[0]["evidence"]
        self.assertEqual(e["n"], 40); self.assertAlmostEqual(e["hit"], 0.75); self.assertAlmostEqual(e["mean_excess"], -0.0065)
        ev = self.db.events()
        self.assertEqual(ev[0]["title"], "Research penalties switched on")
        self.assertIn("40 penalised names", ev[0]["body"])
        # nothing else to promote, and a second run changes nothing
        self.assertEqual(stages.evaluate(self.db, NOW + 3600), [])

    def test_penalty_gate_thresholds(self):
        self.penalty_rows(40, 5, lambda k: -0.01)               # too few days
        self.assertEqual(stages.evaluate(self.db, NOW), [])
        self.assertIn("5.0 days", self.db.kv_get("stage_evidence")["stages"]["penalties"]["reason"])

    def test_penalty_wrong_horizon_and_unpenalised_ignored(self):
        self.penalty_rows(40, 9, lambda k: -0.01, horizon="4h")
        self.db.add_shadow([(NOW - 8 * DAY + k, "crypto", "X-USD", "crypto", "penalty", 0.5, 0.5, {}) for k in range(40)])
        ids = [r["id"] for r in self.db.shadow(kind="penalty", matured=False)]
        self.db.mature_shadow([(-0.01, i) for i in ids])
        self.assertEqual(stages.evaluate(self.db, NOW), [])
        self.assertEqual(self.db.kv_get("stage_evidence")["stages"]["penalties"]["n"], 0)

    def test_penalty_noise_does_not_promote(self):
        self.penalty_rows(60, 10, lambda k: 0.01 if k % 2 else -0.009)   # half lag, mean slightly positive
        self.assertEqual(stages.evaluate(self.db, NOW), [])

    def test_ladder_signals_wait_for_penalties(self):
        self.score_rows(hours=200, per_hour=10, good=True)
        ch = stages.evaluate(self.db, NOW)
        self.assertEqual(ch, [])
        ev = self.db.kv_get("stage_evidence")["stages"]["signals"]
        self.assertTrue(ev["pass"], ev["reason"])                # the evidence is good...
        self.assertIn("penalties", ev["waiting"])                # ...but it waits its turn
        self.assertEqual(self.state("signals"), "shadow")

    def test_min_gap_and_one_at_a_time(self):
        self.penalty_rows(40, 9, lambda k: -0.01)
        self.score_rows(hours=200, per_hour=10, good=True)       # ~8.3 days of good signal evidence
        ch = stages.evaluate(self.db, NOW)
        self.assertEqual([c["stage"] for c in ch], ["penalties"])           # one promotion per run
        self.assertEqual(stages.evaluate(self.db, NOW + 2 * DAY), [])       # gap not met
        ev = self.db.kv_get("stage_evidence")["stages"]["signals"]
        self.assertIn("days ago", ev["waiting"])
        ch = stages.evaluate(self.db, NOW + 3 * DAY + 60)
        self.assertEqual([(c["stage"], c["to"]) for c in ch], [("signals", "active")])
        self.assertGreater(ch[0]["evidence"]["ic_gain"], 0.5)
        self.assertEqual(self.db.events()[0]["title"], "Research signals switched on")

    def test_off_sticks_and_unblocks_ladder(self):
        self.penalty_rows(40, 9, lambda k: -0.01)
        self.assertIn("switched off", stages.set_manual(self.db, "penalties", "off"))
        for k in range(3):
            stages.evaluate(self.db, NOW + k * DAY)
        self.assertEqual(self.state("penalties"), "off")
        self.score_rows(hours=200, per_hour=10, good=True)
        ch = stages.evaluate(self.db, NOW + 4 * DAY)
        self.assertEqual([c["stage"] for c in ch], ["signals"])           # "off" counts as done for the ladder
        self.assertEqual(self.state("penalties"), "off")

    def test_human_shadow_holds_auto_releases(self):
        self.penalty_rows(40, 9, lambda k: -0.01)
        stages.set_manual(self.db, "penalties", "shadow")
        self.assertEqual(stages.evaluate(self.db, NOW), [])
        self.assertIn("by hand", self.db.kv_get("stage_evidence")["stages"]["penalties"]["waiting"])
        self.assertIn("back under the gate", stages.set_manual(self.db, "penalties", "auto"))
        self.assertEqual(self.db.stages()["penalties"]["by"], "auto")
        self.assertEqual([c["stage"] for c in stages.evaluate(self.db, NOW)], ["penalties"])

    def test_set_manual_validation(self):
        self.assertIn("Unknown stage", stages.set_manual(self.db, "nope", "active"))
        self.assertIn("Unknown state", stages.set_manual(self.db, "signals", "maybe"))
        self.assertEqual(self.state("signals"), "shadow")
        self.assertIn("switched on by hand", stages.set_manual(self.db, "signals", "active"))
        self.assertEqual(self.state("signals"), "active")

    def test_auto_promote_off(self):
        self.penalty_rows(40, 9, lambda k: -0.01)
        self.assertEqual(stages.evaluate(self.db, NOW, {"research": {"gates": {"auto_promote": False}}}), [])

    def test_demotion_on_bad_evidence(self):
        self.penalty_rows(40, 30, lambda k: -0.01, end=NOW - 20 * DAY)    # an old, good record
        self.assertEqual([c["stage"] for c in stages.evaluate(self.db, NOW - 20 * DAY)], ["penalties"])
        self.penalty_rows(35, 12, lambda k: 0.012, end=NOW - 3600)               # the recent record turns bad
        ch = stages.evaluate(self.db, NOW)
        self.assertEqual([(c["stage"], c["from"], c["to"]) for c in ch], [("penalties", "active", "shadow")])
        self.assertEqual(self.state("penalties"), "shadow")
        ev = self.db.events()[0]
        self.assertEqual((ev["title"], ev["severity"]), ("Research penalties switched back to shadow", "warn"))
        self.assertIn("35 penalised names", ev["body"])
        # it starts counting afresh: the good old record cannot switch it straight back on
        self.assertEqual(stages.evaluate(self.db, NOW + 4 * DAY), [])
        self.assertEqual(self.db.kv_get("stage_evidence")["stages"]["penalties"]["n"], 0)

    def test_signals_demotion(self):
        stages.set_manual(self.db, "penalties", "active")
        stages.set_manual(self.db, "signals", "active")
        self.score_rows(hours=40, per_hour=10, good=False)
        ch = stages.evaluate(self.db, NOW)
        self.assertEqual([(c["stage"], c["to"]) for c in ch], [("signals", "shadow")])
        self.assertEqual(self.state("penalties"), "active")

    def test_watchlist_gate(self):
        self.db.set_stage("penalties", "active", by="gate", now=NOW - 30 * DAY)
        self.db.set_stage("signals", "active", by="gate", now=NOW - 20 * DAY)
        for k in range(12):
            pid = self.db.add_proposal(NOW - 25 * DAY + k * DAY, "crypto", f"P{k}-USD", "crypto", "kai", 0.8, "r", "shadow", 1, 1)
            self.db.set_proposal(pid, "shadow", 0.02 if k % 3 else -0.01)
        ch = stages.evaluate(self.db, NOW)
        self.assertEqual([c["stage"] for c in ch], ["watchlist"])
        e = ch[0]["evidence"]
        self.assertEqual(e["n"], 12); self.assertAlmostEqual(e["mean_excess"], (8 * 0.02 - 4 * 0.01) / 12)
        # then a run of bad proposals demotes it
        for k in range(12):
            pid = self.db.add_proposal(NOW + k * 3600, "crypto", f"Q{k}-USD", "crypto", "kai", 0.8, "r", "shadow", 1, 1)
            self.db.set_proposal(pid, "shadow", -0.08)
        ch = stages.evaluate(self.db, NOW + 8 * DAY)
        self.assertEqual([(c["stage"], c["to"]) for c in ch], [("watchlist", "shadow")])

    def test_cadence_gate(self):
        for s in ("penalties", "signals", "watchlist"):
            self.db.set_stage(s, "active", by="gate", now=NOW - 30 * DAY)
        self.db.kv_set("cadence:crypto", dict(days=15, base_ret=0.02, cad_ret=0.03, base_dd=-0.05, cad_dd=-0.06))
        self.db.kv_set("cadence:stock", dict(days=15, base_ret=0.01, cad_ret=0.012, base_dd=0.04, cad_dd=0.04))
        self.assertEqual(stages.evaluate(self.db, NOW), [])      # the stock board gained only 0.2%
        self.db.kv_set("cadence:stock", dict(days=15, base_ret=0.01, cad_ret=0.02, base_dd=0.04, cad_dd=0.05))
        ch = stages.evaluate(self.db, NOW)
        self.assertEqual([c["stage"] for c in ch], ["cadence"])
        self.db.kv_set("cadence:crypto", dict(days=20, base_ret=0.02, cad_ret=0.0, base_dd=-0.05, cad_dd=-0.06))
        ch = stages.evaluate(self.db, NOW + DAY)
        self.assertEqual([(c["stage"], c["to"]) for c in ch], [("cadence", "shadow")])

    def test_cadence_deep_drawdown_blocks(self):
        for s in ("penalties", "signals", "watchlist"):
            self.db.set_stage(s, "active", by="human", now=NOW - 30 * DAY)
        self.db.kv_set("cadence:crypto", dict(days=15, base_ret=0.02, cad_ret=0.05, base_dd=0.05, cad_dd=0.09))
        self.assertEqual(stages.evaluate(self.db, NOW), [])
        self.db.kv_set("cadence:crypto", dict(days=15, base_ret=None, cad_ret=0.05, base_dd=0.05, cad_dd=0.05))
        self.assertEqual(stages.evaluate(self.db, NOW), [])       # incomplete stats are ignored

    def test_malformed_rows_never_raise(self):
        self.db.add_shadow([(NOW - DAY, "crypto", "X", "crypto", "penalty", None, None, {})])
        self.db.x("UPDATE shadow SET matured=1, excess=NULL")
        self.db.x("INSERT INTO kv(key,value) VALUES('cadence:x','not json')")
        self.db.x("INSERT INTO stages(name,state,since,evidence,by) VALUES('signals','weird',NULL,'',NULL)")
        self.assertEqual(stages.evaluate(self.db, NOW), [])


if __name__ == "__main__":
    unittest.main()
