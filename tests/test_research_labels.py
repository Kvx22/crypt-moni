"""python -m unittest tests.test_research_labels -v"""
import os
import subprocess
import sys
import unittest
from collections import Counter

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from cryptomoni.research.classifier import (TextModel, challenger_wins, evaluate, features,  # noqa: E402
                                            split_by_time)
from cryptomoni.research.labels import LABEL_HELP, SEED, rule_label  # noqa: E402
from cryptomoni.research.taxonomy import CATALYSTS, DIRECTIONS, RISK_FLAGS  # noqa: E402

HOLD = [e for i, e in enumerate(SEED) if i % 4 == 0]
TRAIN = [e for i, e in enumerate(SEED) if i % 4]


class TestSeed(unittest.TestCase):
    def test_coverage(self):
        self.assertGreaterEqual(len(SEED), 300)
        cats = Counter(e["catalyst"] for e in SEED)
        flags = Counter(f for e in SEED for f in e["risk_flags"])
        for c in CATALYSTS:
            self.assertGreaterEqual(cats[c], 8, c)
        for f in RISK_FLAGS:
            self.assertGreaterEqual(flags[f], 6, f)
        self.assertEqual(set(Counter(e["direction"] for e in SEED)), set(DIRECTIONS))
        for e in SEED:
            self.assertEqual(set(e), {"title", "text", "catalyst", "direction", "risk_flags"})
            self.assertIn(e["catalyst"], CATALYSTS)
            self.assertTrue(set(e["risk_flags"]) <= set(RISK_FLAGS))
        self.assertEqual(len({e["title"] for e in SEED}), len(SEED))
        self.assertEqual(set(LABEL_HELP), set(CATALYSTS))

    def test_rules_agree_with_seed(self):
        r = [rule_label(e["title"], e["text"]) for e in SEED]
        cat = sum(a["catalyst"] == e["catalyst"] for a, e in zip(r, SEED)) / len(SEED)
        flg = sum(sorted(a["risk_flags"]) == sorted(e["risk_flags"]) for a, e in zip(r, SEED)) / len(SEED)
        self.assertGreater(cat, 0.85)
        self.assertGreater(flg, 0.95)


class TestRules(unittest.TestCase):
    def check(self, title, cat=None, d=None, flags=None, text=""):
        r = rule_label(title, text)
        if cat is not None:
            self.assertEqual(r["catalyst"], cat, (title, r))
        if d is not None:
            self.assertEqual(r["direction"], d, (title, r))
        if flags is not None:
            self.assertEqual(sorted(r["risk_flags"]), sorted(flags), (title, r))
        self.assertTrue(0.3 <= r["confidence"] <= 0.95)
        return r

    def test_spec_cues(self):
        self.check("Acme Robotics beats estimates for the third quarter", "earnings", 1)
        self.check("Acme Robotics misses estimates", "earnings", -1)
        self.check("Novacoin listing on Kestrel Exchange confirmed", "listing", 1)
        self.check("Kestrel Exchange to delist Novacoin", "delisting", -1, ["delisting"])
        self.check("Meridian downgrades Halcyon Motors to sell", "analyst", -1)
        self.check("Redwood upgrades Acme Robotics to buy", "analyst", 1)
        self.check("Pinewood Retail raises guidance for the year", "guidance", 1)
        self.check("SEC sues Kestrel Exchange", "regulatory", -1, ["regulatory", "lawsuit"])
        self.check("Spot Novacoin ETF approved", "etf", 1)
        self.check("Brightline Foods files for bankruptcy", "legal", -1, ["bankruptcy"])
        self.check("Zephyr protocol exploited, $40m drained", "security", -1, ["exploit"])

    def test_negation(self):
        self.check("Novacoin ETF not approved", "etf", -1)
        self.check("Orchid Pharma fails to win approval for its drug", d=-1)
        self.check("Kestrel Exchange denies hack", "security", 0, [])
        self.check("Kestrel resumes trading after brief halt", "outage", 1, [])

    def test_flags_need_a_victim(self):
        self.check("How to protect your wallet against hacks", "other", 0, [])
        self.check("The 10 biggest exploits of all time", "other", 0, [])
        self.check("Bridge exploits explained", "other", 0, [])
        self.check("Hacks could hit more bridges, researchers warn", flags=[])
        self.check("Harbor Bridge drained of $120m", "security", -1, ["exploit"])

    def test_nothing_and_junk(self):
        for junk in ("", None, 12, b"\xff\xfe", "zzz qqq"):
            r = rule_label(junk, junk)
            self.assertEqual((r["catalyst"], r["direction"], r["risk_flags"], r["confidence"]), ("other", 0, [], 0.3))
        r = rule_label("x" * 100000, "y " * 100000)
        self.assertEqual(r["catalyst"], "other")

    def test_every_catalyst_and_flag_reachable(self):
        cats = {rule_label(e["title"], e["text"])["catalyst"] for e in SEED}
        flags = {f for e in SEED for f in rule_label(e["title"], e["text"])["risk_flags"]}
        self.assertEqual(cats, set(CATALYSTS))
        self.assertEqual(flags, set(RISK_FLAGS))


class TestClassifier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = TextModel().fit(TRAIN)

    def test_beats_baseline_on_holdout(self):
        m = evaluate(self.model, HOLD)
        base = Counter(e["catalyst"] for e in HOLD).most_common(1)[0][1] / len(HOLD)
        text_only = evaluate(TextModel(rules=False).fit(TRAIN), HOLD)
        self.assertEqual(m["n"], len(HOLD))
        self.assertGreater(m["cat_acc"], 0.5)
        self.assertGreater(m["cat_acc"], base)
        self.assertGreater(m["cat_acc"], text_only["cat_acc"])
        self.assertGreater(m["dir_acc"], 0.6)
        self.assertGreater(m["flag_f1"], 0.6)
        for k in ("cat_acc", "cat_macro_f1", "dir_acc", "flag_f1"):
            self.assertTrue(0.0 <= m[k] <= 1.0)

    def test_learns_corrections(self):
        # the rules call this a lawsuit story; a human says it is noise. The model should learn that.
        fix = [dict(title=f"Glint Wallet lawsuit chatter number {i} on forums", text="", catalyst="sentiment", direction=0,
                    risk_flags=[], weight=3.0) for i in range(30)]
        m = TextModel().fit(TRAIN + fix)
        p = m.predict("Glint Wallet lawsuit chatter number 99 on forums")
        self.assertEqual(p["catalyst"], "sentiment")

    def test_predict_shape(self):
        p = self.model.predict("Harbor Bridge drained of $50m in exploit", "")
        self.assertEqual(set(p), {"catalyst", "cat_p", "direction", "dir_p", "risk_flags", "flag_p"})
        self.assertIn(p["catalyst"], CATALYSTS)
        self.assertIn(p["direction"], DIRECTIONS)
        self.assertEqual(set(p["flag_p"]), set(RISK_FLAGS))
        self.assertIn("exploit", p["risk_flags"])
        self.assertEqual(self.model.predict_many([]), [])
        self.assertEqual(len(self.model.predict_many([("a", ""), ("b", "c")])), 2)
        self.assertEqual(self.model.predict("", "")["catalyst"], "other")

    def test_blob_round_trip(self):
        blob = self.model.to_blob()
        self.assertIsInstance(blob, bytes)
        m2 = TextModel.from_blob(blob)
        self.assertEqual((m2.n_train, m2.version), (self.model.n_train, self.model.version))
        for e in HOLD[:40]:
            self.assertEqual(self.model.predict(e["title"], e["text"]), m2.predict(e["title"], e["text"]))
        with self.assertRaises(Exception):
            TextModel.from_blob(b"not a model")

    def test_fit_is_deterministic(self):
        a, b = TextModel().fit(TRAIN[:80], epochs=2), TextModel().fit(TRAIN[:80], epochs=2)
        self.assertEqual(a.version, b.version)
        self.assertTrue(np.array_equal(a.W, b.W))

    def test_bad_examples_are_skipped(self):
        m = TextModel().fit([dict(title="x", catalyst="nope", direction=1), dict(title="Acme beats estimates", catalyst="earnings",
                                                                                 direction="junk", risk_flags=["bogus"])])
        self.assertEqual(m.n_train, 1)
        self.assertEqual(TextModel().fit([]).n_train, 0)


class TestFeatures(unittest.TestCase):
    def test_shape_and_norm(self):
        idx, val = features("Novacoin ETF approved", "Trading begins Tuesday.")
        self.assertEqual((idx.dtype, val.dtype), (np.int32, np.float32))
        self.assertAlmostEqual(float((val * val).sum()), 1.0, places=5)
        self.assertTrue(((idx >= 0) & (idx < 2 ** 15)).all())
        self.assertEqual(len(features("", "")[0]), 0)
        # title words count double
        i1, v1 = features("alpha", "beta")
        w = dict(zip(i1.tolist(), v1.tolist()))
        a, b = features("alpha")[0][0], features("beta")[0][0]
        self.assertGreater(w[int(a)], w[int(b)])

    def test_stable_across_processes(self):
        code = ("import sys; sys.path.insert(0, %r); from cryptomoni.research.classifier import features; "
                "i, v = features('Zephyr protocol exploited for $40m', 'Attackers drained the pool.'); "
                "print(i.tolist(), [round(float(x), 6) for x in v])") % ROOT
        outs = {subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                               env=dict(os.environ, PYTHONHASHSEED=str(s))).stdout for s in (1, 2)}
        self.assertEqual(len(outs), 1)
        i, v = features("Zephyr protocol exploited for $40m", "Attackers drained the pool.")
        self.assertEqual(outs.pop().strip(), f"{i.tolist()} {[round(float(x), 6) for x in v]}")


class TestChampionChallenger(unittest.TestCase):
    def test_rules(self):
        good = dict(n=80, cat_acc=0.8, cat_macro_f1=0.8, dir_acc=0.7, flag_f1=0.8)
        self.assertTrue(challenger_wins(None, good)[0])
        self.assertTrue(challenger_wins(dict(good, cat_macro_f1=0.7), good)[0])
        self.assertFalse(challenger_wins(dict(good, cat_macro_f1=0.795), good)[0])       # not by the margin
        self.assertFalse(challenger_wins(dict(good, cat_macro_f1=0.7, dir_acc=0.75), good)[0])   # direction got worse
        self.assertFalse(challenger_wins(dict(good, cat_macro_f1=0.5), dict(good, n=30))[0])     # too few examples
        ok, why = challenger_wins(dict(good, cat_macro_f1=0.7), good)
        self.assertIsInstance(why, str)

    def test_split_by_time(self):
        ex = [dict(title=str(i), ts=float(i)) for i in (5, 1, 9, 3, 7, 2, 8, 4)]
        train, hold = split_by_time(ex, 0.25)
        self.assertEqual([e["ts"] for e in hold], [8.0, 9.0])
        self.assertEqual(len(train), 6)
        self.assertTrue(max(e["ts"] for e in train) < min(e["ts"] for e in hold))
        self.assertEqual(split_by_time([], 0.25), ([], []))

    def test_evaluate_edge_cases(self):
        self.assertEqual(evaluate(None, HOLD)["cat_acc"], 0.0)
        self.assertEqual(evaluate(TextModel(), [])["n"], 0)


if __name__ == "__main__":
    unittest.main()
