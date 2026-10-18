"""python -m unittest tests.test_research_aggregate -v"""
import math
import os
import shutil
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from cryptomoni.research import text  # noqa: E402
from cryptomoni.research.aggregate import build_view, team_coverage  # noqa: E402
from cryptomoni.research.aliases import Aliases  # noqa: E402
from cryptomoni.research.db import ResearchDB  # noqa: E402
from cryptomoni.research.pipeline import label_items, relabel_recent  # noqa: E402
from cryptomoni.research.taxonomy import FLAG_FRESH_S, FRESH_S, RESEARCH_SIGNALS, severity  # noqa: E402
from cryptomoni.research.types import Item, Note  # noqa: E402

H = 3600.0
NOW = 1_790_000_000.0
SUBS = {"NOVA-USD": "crypto", "ACME": "stock"}
NAMES = {"NOVA-USD": ["Novacoin"], "ACME": ["Acme Robotics"]}

STORIES = [
    "Novacoin wins payments partnership with regional bank network",
    "Developers ship Novacoin wallet with built-in staking for all users",
    "Exchange volume for Novacoin climbs as new markets open in Asia",
    "Novacoin foundation reports steady growth in active addresses this month",
    "Analysts lift their view of Novacoin after strong quarterly network fees",
    "Merchants in three countries start accepting Novacoin at checkout",
]


class DBCase(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.db = ResearchDB(os.path.join(self.dir, "r.db"))

    def tearDown(self):
        self.db.close()
        shutil.rmtree(self.dir, ignore_errors=True)

    def note(self, title, publisher, direction=1, evidence="news", catalyst="partnership", conf=0.8, age_h=1.0, flags=(),
             quarantine=False, official=False, source=None, symbol="NOVA-USD"):
        ts = NOW - age_h * H
        return Note(symbol=symbol, asset=SUBS.get(symbol, "crypto"), ts=ts, fetched_ts=ts, source=source or publisher, team="catalysts",
                    publisher=publisher, evidence=evidence, catalyst=catalyst, direction=direction, confidence=conf,
                    risk_flags=list(flags), severity=severity(flags), label_by="rules", title=title, url=f"https://{publisher}/{len(title)}",
                    cluster=text.fingerprint(title), quarantine=quarantine, official=official, expires_ts=NOW + 86400)

    def view(self, notes, **cfg):
        self.db.add_notes(notes)
        return build_view(self.db, NOW, SUBS, cfg or None)["NOVA-USD"]


class TestCorroboration(DBCase):
    def test_five_publishers_two_kinds_boost(self):
        notes = [self.note(t, f"pub{i}.com", evidence="news" if i < 3 else "filing") for i, t in enumerate(STORIES[:5])]
        v = self.view(notes)
        self.assertEqual((v["n_notes"], v["n_stories"], v["n_independent"]), (5, 5, 5))
        self.assertEqual(v["kinds"], ["filing", "news"])
        self.assertTrue(v["boost_ok"])
        self.assertFalse(v["thin"])
        self.assertGreater(v["score"], 0)
        self.assertEqual(len(v["top"]), 5)

    def test_syndicated_copies_are_one_story(self):
        notes = [self.note(STORIES[0], f"pub{i}.com", evidence="news" if i < 3 else "filing") for i in range(5)]
        v = self.view(notes)
        self.assertEqual((v["n_notes"], v["n_stories"], v["n_independent"]), (5, 1, 1))
        self.assertFalse(v["boost_ok"])
        self.assertTrue(v["thin"])
        w = 0.8 * 0.5 * 0.5 ** (1 / 48)          # the story's best copy: conf x default reliability x 1 h of filing decay
        self.assertAlmostEqual(v["features"]["r_catalyst"], math.tanh(0.25 * w / 1.5), places=4)   # thin: heard faintly
        self.assertEqual(len(v["top"]), 1)

    def test_one_kind_is_not_enough(self):
        v = self.view([self.note(t, f"pub{i}.com") for i, t in enumerate(STORIES[:5])])
        self.assertEqual(v["n_independent"], 5)
        self.assertFalse(v["boost_ok"])

    def test_same_publisher_counts_once(self):
        v = self.view([self.note(t, "samepub.com") for t in STORIES[:4]])
        self.assertEqual((v["n_stories"], v["n_independent"]), (4, 1))

    def test_bad_news_is_never_scaled_down(self):
        v = self.view([self.note(STORIES[0], "pub.com", direction=-1)])
        w = 0.8 * 0.5 * 0.5 ** (1 / 12)
        self.assertAlmostEqual(v["features"]["r_catalyst"], math.tanh(-w / 1.5), places=4)
        self.assertLess(v["score"], 0)

    def test_quarantine_only_does_not_count(self):
        v = self.view([self.note(t, f"q{i}.net", quarantine=True) for i, t in enumerate(STORIES[:5])])
        self.assertEqual(v["n_notes"], 5)
        self.assertEqual(v["n_independent"], 0)
        self.assertEqual(v["score"], 0.0)
        self.assertEqual(v["top"], [])
        self.assertTrue(v["thin"])

    def test_quarantine_counts_once_corroborated(self):
        notes = [self.note(t, f"q{i}.net", quarantine=True) for i, t in enumerate(STORIES[:3])]
        notes.append(self.note(STORIES[5], "real.com"))
        v = self.view(notes)
        self.assertEqual(v["n_independent"], 4)
        self.assertEqual(v["kinds"], ["news"])      # quarantine stories never add an evidence kind
        v2 = build_view(self.db, NOW, SUBS, dict(quarantine_needs=2))["NOVA-USD"]
        self.assertEqual(v2["n_independent"], 1)

    def test_decay_by_evidence_kind(self):
        fresh = self.view([self.note(STORIES[0], "a.com", direction=-1, evidence="market", catalyst="market_structure", age_h=0)])
        self.db.x("DELETE FROM notes")
        old = self.view([self.note(STORIES[0], "a.com", direction=-1, evidence="market", catalyst="market_structure", age_h=3)])
        self.assertAlmostEqual(math.atanh(old["features"]["r_structure"]) / math.atanh(fresh["features"]["r_structure"]), 0.5, places=3)

    def test_market_other_feeds_fundamental(self):
        v = self.view([self.note(STORIES[0], "a.com", direction=-1, evidence="market", catalyst="other")])
        self.assertLess(v["features"]["r_fundamental"], 0)
        self.assertEqual(v["features"]["r_catalyst"], 0.0)

    def test_reliability_from_sources_table(self):
        self.db.source_ok("good_feed", "catalysts", "news", 1, 600, now=NOW)
        self.db.set_reliability([(9.0, 1.0, 0.9, "good_feed")])
        n = self.note(STORIES[0], "a.com", direction=-1, source="good_feed")
        v = self.view([n])
        w = 0.8 * 0.9 * 0.5 ** (1 / 12)
        self.assertAlmostEqual(v["features"]["r_catalyst"], math.tanh(-w / 1.5), places=4)
        v2 = build_view(self.db, NOW, SUBS, reliability={"good_feed": 0.5})["NOVA-USD"]
        self.assertAlmostEqual(v2["features"]["r_catalyst"], math.tanh(-0.8 * 0.5 * 0.5 ** (1 / 12) / 1.5), places=4)


class TestPenalty(DBCase):
    def test_cap(self):
        notes = [self.note(t, f"pub{i}.com", direction=-1, catalyst="security", flags=["exploit"], conf=0.95, official=True)
                 for i, t in enumerate(STORIES)]
        v = self.view(notes)
        self.assertAlmostEqual(v["penalty"], 0.35)
        self.assertAlmostEqual(v["features"]["r_risk"], -1.0)
        self.assertEqual(v["flags"][0]["flag"], "exploit")
        self.assertTrue(v["flags"][0]["credible"])
        self.assertEqual(v["flags"][0]["sources"], 6)
        self.assertFalse(v["halted"])

    def test_decay_and_credibility(self):
        official = self.note(STORIES[0], "status.example", direction=-1, catalyst="outage", flags=["outage"], conf=1.0,
                             official=True, age_h=0)
        v = self.view([official])
        self.assertAlmostEqual(v["penalty"], 0.3, places=4)                  # credible: full severity
        self.db.x("DELETE FROM notes")
        v = self.view([self.note(STORIES[0], "status.example", direction=-1, catalyst="outage", flags=["outage"], conf=1.0,
                                 official=True, age_h=24)])
        self.assertAlmostEqual(v["penalty"], 0.15, places=4)                 # 24 h half-life
        self.db.x("DELETE FROM notes")
        v = self.view([self.note(STORIES[0], "blog.example", direction=-1, catalyst="outage", flags=["outage"], conf=1.0, age_h=0)])
        self.assertAlmostEqual(v["penalty"], 0.25 * 0.3, places=4)           # one unconfirmed source: a quarter
        self.assertFalse(v["flags"][0]["credible"])

    def test_two_independent_stories_make_a_flag_credible(self):
        notes = [self.note(STORIES[i], f"blog{i}.example", direction=-1, catalyst="legal", flags=["lawsuit"], conf=1.0, age_h=0)
                 for i in range(2)]
        v = self.view(notes)
        self.assertTrue(v["flags"][0]["credible"])
        self.assertAlmostEqual(v["penalty"], 0.35)       # 2 x 0.4, capped

    def test_halted_fast_path_and_resume(self):
        halt = self.note("Exchange halts Novacoin trading", "exchange.example", direction=-1, catalyst="outage", flags=["halt"],
                         official=True, age_h=0.5, evidence="official")
        v = self.view([halt])
        self.assertTrue(v["halted"])
        resumed = self.note("Exchange resumes Novacoin trading", "exchange.example", direction=1, catalyst="outage",
                            official=True, age_h=0.1, evidence="official")
        v = self.view([resumed])
        self.assertFalse(v["halted"])
        self.db.x("DELETE FROM notes")
        v = self.view([self.note("Exchange halts Novacoin trading", "exchange.example", direction=-1, catalyst="outage",
                                 flags=["halt"], official=True, age_h=3, evidence="official")])
        self.assertFalse(v["halted"])
        self.db.x("DELETE FROM notes")
        v = self.view([self.note("Novacoin halted, a blog says", "blog.example", direction=-1, catalyst="outage", flags=["halt"],
                                 age_h=0.2)])
        self.assertFalse(v["halted"])                   # only official notes take the fast path


class TestShape(DBCase):
    def test_features_ranges_and_empty_views(self):
        notes = [self.note(t, f"pub{i}.com", direction=(-1) ** i, evidence=("news", "filing", "sentiment")[i % 3],
                           catalyst=("earnings", "sentiment", "etf", "listing")[i % 4], flags=["dilution"] if i == 2 else [])
                 for i, t in enumerate(STORIES)]
        self.db.add_notes(notes)
        views = build_view(self.db, NOW, SUBS)
        self.assertEqual(set(views), set(SUBS))
        v = views["NOVA-USD"]
        self.assertEqual(set(v["features"]), set(RESEARCH_SIGNALS))
        for k, x in v["features"].items():
            self.assertTrue(-1.0 <= x <= 1.0, k)
        self.assertTrue(-1.0 <= v["features"]["r_risk"] <= 0.0)
        self.assertTrue(0.0 <= v["features"]["r_coverage"] <= 1.0)
        self.assertTrue(-1.0 <= v["score"] <= 1.0)
        self.assertLessEqual(len(v["top"]), 6)
        self.assertEqual(v["last_note_ts"], max(n.ts for n in notes))
        for k in ("symbol", "asset", "ts", "n_notes", "n_stories", "n_independent", "kinds", "score", "boost_ok", "penalty",
                  "halted", "thin", "flags", "features", "top", "last_note_ts"):
            self.assertIn(k, v)
        e = views["ACME"]
        self.assertEqual((e["n_notes"], e["thin"], e["asset"]), (0, True, "stock"))
        self.assertTrue(all(x == 0 for x in e["features"].values()))
        self.assertEqual(build_view(self.db, NOW, {}), {})

    def test_team_coverage(self):
        self.db.add_notes([self.note(STORIES[0], "a.com"), self.note(STORIES[1], "b.com", symbol="ACME"),
                           self.note(STORIES[2], "c.com", age_h=30)])
        cov = team_coverage(self.db, NOW, SUBS)
        c = cov["catalysts"]
        self.assertEqual((c["symbols_covered"], c["symbols_total"], c["notes"], c["sources_active"]), (2, 2, 2, 2))
        self.assertEqual(cov["risk"]["symbols_covered"], 0)
        self.assertEqual(cov["risk"]["symbols_total"], 2)


class FakeModel:
    def __init__(self, cat="earnings", cat_p=0.9, direction=1, dir_p=0.81, flag_p=None, boom=False):
        self.p = dict(catalyst=cat, cat_p=cat_p, direction=direction, dir_p=dir_p, risk_flags=[], flag_p=flag_p or {})
        self.boom = boom

    def predict(self, title, text=""):
        if self.boom:
            raise RuntimeError("model broke")
        return dict(self.p)


class TestPipeline(DBCase):
    def setUp(self):
        super().setUp()
        self.aliases = Aliases(SUBS, NAMES)

    def add(self, *items, now=NOW):
        for it in items:
            self.db.source_ok(it.source, "catalysts", it.kind, 1, 600, now=now)
        return self.db.add_items(list(items), "catalysts", now=now)

    def test_text_item_to_notes(self):
        self.add(Item(source="feed", url="https://news.example/1", title="Novacoin network halted after consensus bug",
                      text="Acme Robotics, which accepts Novacoin, said payments are paused.", published_ts=NOW - 600,
                      publisher="news.example"))
        r = label_items(self.db, self.aliases, None, NOW, {})
        self.assertEqual((r["items"], r["notes"]), (1, 2))
        notes = {n.symbol: n for n in self.db.notes(10)}
        n = notes["NOVA-USD"]
        self.assertEqual((n.catalyst, n.direction, n.risk_flags, n.label_by), ("outage", -1, ["halt"], "rules"))
        self.assertEqual((n.asset, notes["ACME"].asset), ("crypto", "stock"))
        self.assertEqual(n.ts, NOW - 600)
        self.assertEqual(n.expires_ts, n.ts + FLAG_FRESH_S)
        self.assertEqual(n.severity, 1.0)
        self.assertEqual(n.cluster, text.fingerprint(n.title, "Acme Robotics, which accepts Novacoin, said payments are paused."))
        self.assertEqual(self.db.sources()["feed"]["notes"], 2)
        self.assertEqual(label_items(self.db, self.aliases, None, NOW, {})["items"], 0)   # consumed

    def test_structured_item_keeps_its_labels(self):
        self.add(Item(source="futures", url="data://futures/NOVA-USD/1", title="Novacoin funding rate turns negative",
                      symbols=["NOVA-USD"], kind="market", catalyst="market_structure", direction=-1, confidence=0.65,
                      published_ts=NOW))
        label_items(self.db, self.aliases, FakeModel(), NOW, {})
        n = self.db.notes(1)[0]
        self.assertEqual((n.catalyst, n.direction, n.confidence, n.label_by, n.evidence), ("market_structure", -1, 0.65, "source", "market"))
        self.assertEqual(n.publisher, "futures")       # text.domain of a data:// url
        self.assertEqual(n.expires_ts, NOW + FRESH_S["market"])

    def test_model_when_confident(self):
        self.add(Item(source="feed", url="https://x.example/1", title="Acme Robotics shares in focus", published_ts=NOW))
        label_items(self.db, self.aliases, FakeModel(flag_p={"lawsuit": 0.75, "fraud": 0.6}), NOW, {})
        n = self.db.notes(1)[0]
        self.assertEqual((n.catalyst, n.direction, n.label_by), ("earnings", 1, "model"))
        self.assertAlmostEqual(n.confidence, 0.9 * 0.9, places=4)
        self.assertEqual(n.risk_flags, ["lawsuit"])     # the model adds flags only when very sure

    def test_rules_when_model_unsure_or_broken(self):
        self.add(Item(source="feed", url="https://x.example/2", title="Acme Robotics raises full-year guidance", published_ts=NOW),
                 Item(source="feed", url="https://x.example/3", title="Acme Robotics cuts full-year guidance", published_ts=NOW))
        label_items(self.db, self.aliases, FakeModel(cat_p=0.4), NOW, dict(model_min_conf=0.6), batch=1)
        label_items(self.db, self.aliases, FakeModel(boom=True), NOW, {}, batch=1)
        got = sorted((n.direction, n.catalyst, n.label_by) for n in self.db.notes(5))
        self.assertEqual(got, [(-1, "guidance", "rules"), (1, "guidance", "rules")])

    def test_timestamps(self):
        self.add(Item(source="feed", url="https://x.example/f", title="Acme Robotics news one", published_ts=NOW + 86400),
                 Item(source="feed", url="https://x.example/o", title="Acme Robotics news two", published_ts=NOW - 60 * 86400),
                 Item(source="feed", url="https://x.example/n", title="Acme Robotics news three"), now=NOW)
        label_items(self.db, self.aliases, None, NOW + 100, {})
        self.assertEqual({n.ts for n in self.db.notes(5)}, {NOW})

    def test_market_wide_skipped_and_discovery(self):
        self.add(Item(source="status", url="data://status/1", title="Exchange maintenance window", kind="market",
                      facts={"scope": "exchange"}),
                 Item(source="feed", url="https://x.example/d", title="Bittensor token mentioned in weekly roundup"),
                 Item(source="gecko", url="data://gecko/ZORP", title="ZORP listed on two venues", kind="discovery",
                      symbols=["ZORP-USD"]))
        r = label_items(self.db, self.aliases, None, NOW, {})
        self.assertEqual((r["items"], r["notes"], r["skipped"], r["unmapped"]), (3, 0, 1, 2))
        found = {d["symbol"]: d for d in self.db.discovered()}
        self.assertIn("TAO-USD", found)
        self.assertIn("ZORP-USD", found)
        self.assertEqual(found["ZORP-USD"]["sources"], ["gecko"])
        self.assertEqual(self.db.q1("SELECT COUNT(*) n FROM items WHERE labelled=0")["n"], 0)

    def test_presymbolled_item_only_scans_text_when_news(self):
        self.add(Item(source="filings", url="https://sec.example/1", title="Acme Robotics 8-K mentions Novacoin treasury",
                      symbols=["ACME"], kind="filing"),
                 Item(source="q", url="https://news.example/q", title="Acme Robotics and Novacoin in the news",
                      symbols=["ACME"], kind="search"))
        label_items(self.db, self.aliases, None, NOW, {})
        got = sorted((n.source, n.symbol) for n in self.db.notes(10))
        self.assertEqual(got, [("filings", "ACME"), ("q", "ACME"), ("q", "NOVA-USD")])

    def test_relabel_recent(self):
        self.add(Item(source="feed", url="https://x.example/r1", title="Acme Robotics raises full-year guidance", published_ts=NOW),
                 Item(source="feed", url="https://x.example/r2", title="Acme Robotics cuts full-year guidance", published_ts=NOW),
                 Item(source="feed", url="https://x.example/r3", title="Acme Robotics old story", published_ts=NOW - 5 * 86400))
        label_items(self.db, self.aliases, None, NOW, {})
        ids = sorted(n.id for n in self.db.notes(5))
        self.db.set_label(ids[1], "other", 0, [], by="human")
        changed = relabel_recent(self.db, FakeModel(cat="capital_return", cat_p=0.95, dir_p=1.0), self.aliases, NOW)
        self.assertEqual(changed, 1)
        by_id = {n.id: n for n in self.db.notes(5)}
        self.assertEqual((by_id[ids[0]].catalyst, by_id[ids[0]].label_by), ("capital_return", "model"))
        self.assertEqual((by_id[ids[1]].catalyst, by_id[ids[1]].label_by), ("other", "human"))
        self.assertEqual(by_id[ids[2]].label_by, "rules")       # too old to relabel
        self.assertEqual(relabel_recent(self.db, None, self.aliases, NOW), 0)


if __name__ == "__main__":
    unittest.main()
