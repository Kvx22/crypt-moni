"""python -m unittest tests.test_research_outcome -v"""
import datetime as dt
import json
import math
import os
import random
import shutil
import sys
import tempfile
import unittest
import zlib

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from cryptomoni.research import outcome  # noqa: E402
from cryptomoni.research.db import ResearchDB  # noqa: E402
from cryptomoni.research.outcome import OutcomeModel, PriceKeeper, mature, reliability, train_outcome_model, update_reliability  # noqa: E402
from cryptomoni.research.taxonomy import HORIZON_BARS_STOCK, HORIZON_HOURS_CRYPTO, HORIZONS  # noqa: E402
from cryptomoni.research.types import Note  # noqa: E402

H, D = 3600, 86400
T0 = int(dt.datetime(2026, 10, 5, tzinfo=dt.timezone.utc).timestamp())   # a Monday, 00:00 UTC


def utc(y, m, d, hh=0, mm=0):
    return int(dt.datetime(y, m, d, hh, mm, tzinfo=dt.timezone.utc).timestamp())


# crypto: hourly bars from T0, log-linear so every excess is easy to predict
CRYPTO = {"BTC-USD": [(T0 + k * H, 100 * math.exp(0.001 * k)) for k in range(400)],
          "SOL-USD": [(T0 + k * H, 10 * math.exp(0.003 * k)) for k in range(400)]}
# stocks: 7 regular-session bars a day (13:30 .. 19:30 UTC in EDT), weekdays only: three weeks with two weekends
SESSION = [T0 + day * D + 13 * H + 1800 + b * H for day in range(19) if (day % 7) < 5 for b in range(7)]
STOCK = {"NVDA": [(t, 50 * math.exp(0.004 * m)) for m, t in enumerate(SESSION)],
         "SPY": [(t, 400 * math.exp(0.001 * m)) for m, t in enumerate(SESSION)]}
SERIES = {**CRYPTO, **STOCK}


def note(symbol, asset, ts, source="src", direction=1, catalyst="other", title="a headline", cluster="", **kw):
    return Note(symbol=symbol, asset=asset, ts=ts, fetched_ts=ts, source=source, team="catalysts", publisher="x",
                evidence=kw.pop("evidence", "news"), catalyst=catalyst, direction=direction, confidence=kw.pop("confidence", 0.6),
                risk_flags=[], severity=0.0, label_by=kw.pop("label_by", "rules"), title=title, url=f"u://{symbol}/{ts}/{random.random()}",
                cluster=cluster, expires_ts=ts + 2 * D, **kw)


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db = ResearchDB(os.path.join(self.tmp, "r.db"))

    def tearDown(self):
        self.db.close()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def note_id(self, symbol, ts):
        return self.db.q1("SELECT id FROM notes WHERE symbol=? AND ts=?", (symbol, ts))["id"]

    def outcomes(self, nid):
        return {o["horizon"]: o for o in self.db.q("SELECT * FROM outcomes WHERE note_id=?", (nid,))}


class PricesTest(Base):
    def test_price_at_bar_known_at_its_end(self):
        self.db.put_prices("SOL-USD", CRYPTO["SOL-USD"][:20])
        pk = PriceKeeper(self.db)
        close = dict(CRYPTO["SOL-USD"])
        self.assertEqual(pk.price_at("SOL-USD", T0 + 10 * H, "crypto"), close[T0 + 9 * H])      # bar 9 ends exactly at 10h
        self.assertEqual(pk.price_at("SOL-USD", T0 + 10 * H - 1, "crypto"), close[T0 + 8 * H])  # one second earlier: not yet
        self.assertIsNone(pk.price_at("SOL-USD", T0 + 30 * 60, "crypto"))                      # no bar has ended yet
        self.assertEqual(pk.price_at("SOL-USD", T0 + 23 * H, "crypto"), close[T0 + 19 * H])    # 3h stale: still ok
        self.assertIsNone(pk.price_at("SOL-USD", T0 + 23 * H + 1, "crypto"))                   # beyond 3h: unknown

    def test_stock_weekend_and_staleness(self):
        self.db.put_prices("NVDA", STOCK["NVDA"])
        pk = PriceKeeper(self.db)
        fri_last = dict(STOCK["NVDA"])[utc(2026, 10, 9, 19, 30)]
        self.assertEqual(pk.price_at("NVDA", utc(2026, 10, 10, 12), "stock"), fri_last)        # Saturday: Friday's last bar
        self.assertEqual(pk.price_at("NVDA", utc(2026, 10, 12, 14, 0), "stock"), fri_last)     # Monday before first bar ends
        last_end = SESSION[-1] + H
        self.assertIsNotNone(pk.price_at("NVDA", last_end + 4 * D, "stock"))
        self.assertIsNone(pk.price_at("NVDA", last_end + 4 * D + 1, "stock"))

    def test_refresh_throttles_adds_benchmarks_and_collects_errors(self):
        calls, clock = [], [T0 + 10 * H + 600]

        def crypto(sym, since):
            calls.append((sym, since))
            if sym == "BAD-USD":
                raise RuntimeError("boom")
            # includes the still-forming bar (start 10h) and junk rows, which must be dropped
            return [b for b in CRYPTO.get(sym, []) if b[0] <= clock[0] and (since is None or b[0] >= since)] + [(T0, float("nan")), ("x",)]

        pk = PriceKeeper(self.db, {"crypto": crypto})
        r = pk.refresh({"SOL-USD": "crypto", "BAD-USD": "crypto", "NVDA": "stock"}, now=clock[0])
        self.assertEqual(set(r["fetched"]), {"SOL-USD", "BAD-USD", "BTC-USD", "NVDA", "SPY"})
        self.assertIn("boom", r["errors"]["BAD-USD"])
        self.assertIn("no price fetcher", r["errors"]["NVDA"])
        self.assertIn("SPY", r["errors"])
        self.assertEqual(self.db.last_price_ts("SOL-USD"), T0 + 9 * H)                        # forming bar dropped
        self.assertEqual(len(self.db.price_series("SOL-USD")), 10)
        n = len(calls)
        self.assertEqual(pk.refresh({"SOL-USD": "crypto"}, now=clock[0] + 19 * 60)["fetched"], [])  # throttled (20 min)
        self.assertEqual(len(calls), n)
        clock[0] += 3600
        r = pk.refresh({"SOL-USD": "crypto"}, now=clock[0])
        self.assertEqual(set(r["fetched"]), {"SOL-USD", "BTC-USD"})                            # no SPY without stocks
        self.assertIn(("SOL-USD", T0 + 9 * H), calls)                                         # asked from the last stored bar
        self.assertEqual(self.db.last_price_ts("SOL-USD"), T0 + 10 * H)
        self.assertEqual(pk.price_at("SOL-USD", clock[0], "crypto"), dict(CRYPTO["SOL-USD"])[T0 + 10 * H])  # cache invalidated

    def test_price0_waits_for_a_lagging_fetch(self):
        self.db.put_prices("BTC-USD", CRYPTO["BTC-USD"][:30])
        self.db.put_prices("SOL-USD", CRYPTO["SOL-USD"][:21])                                 # last bar ends 21h
        self.db.add_notes([note("SOL-USD", "crypto", T0 + 22 * H + 1800)])
        self.assertEqual(mature(self.db, T0 + 22 * H + 1900)["price0"], 0)                    # 22h bar not here yet
        self.db.put_prices("SOL-USD", CRYPTO["SOL-USD"][21:22])
        self.assertEqual(mature(self.db, T0 + 22 * H + 2000)["price0"], 1)
        n = self.db.q1("SELECT price0, bench0 FROM notes")
        self.assertAlmostEqual(n["price0"], CRYPTO["SOL-USD"][21][1])
        self.assertAlmostEqual(n["bench0"], CRYPTO["BTC-USD"][21][1])


class MatureWalkForwardTest(Base):
    """Prices arrive hour by hour through PriceKeeper; notes, shadow rows and proposals are
    written when they happen; mature() runs every hour. Then every number is checked."""

    def test_walk_forward(self):
        clock = [0]

        def fetch(sym, since):
            return [b for b in SERIES[sym] if b[0] <= clock[0] and (since is None or b[0] >= since)]

        pk = PriceKeeper(self.db, {"crypto": fetch, "stock": fetch})
        sol_ts, sol_edge, btc_ts = T0 + 10 * H + 1800, T0 + 10 * H, T0 + 11 * H + 900
        nv_ts, nv_sat = utc(2026, 10, 8, 15), utc(2026, 10, 10, 12)
        late_ts = T0 + 300 * H + 1800
        pending = [
            (sol_ts, lambda: self.db.add_notes([note("SOL-USD", "crypto", sol_ts)])),
            (sol_edge, lambda: self.db.add_notes([note("SOL-USD", "crypto", sol_edge, direction=-1)])),
            (btc_ts, lambda: self.db.add_notes([note("BTC-USD", "crypto", btc_ts)])),
            (nv_ts, lambda: self.db.add_notes([note("NVDA", "stock", nv_ts)])),
            (nv_sat, lambda: self.db.add_notes([note("NVDA", "stock", nv_sat)])),
            (late_ts, lambda: self.db.add_notes([note("SOL-USD", "crypto", late_ts)])),
            (sol_ts, lambda: self.db.add_shadow([
                (sol_ts, "crypto", "SOL-USD", "crypto", "score", 0.1, 0.2,
                 dict(horizon="4h", price0=10 * math.exp(0.027), bench0=100 * math.exp(0.009))),
                (sol_ts, "crypto", "SOL-USD", "crypto", "score", 0.1, 0.2, dict(horizon="4h")),     # no prices: never matures
                (nv_ts, "stocks", "NVDA", "stock", "penalty", 0.1, 0.0,
                 dict(price0=50 * math.exp(0.084), bench0=400 * math.exp(0.021)))])),             # default horizon 1d
            (sol_ts, lambda: [self.db.add_proposal(sol_ts, "crypto", "SOL-USD", "crypto", "kai", 2.0, "r", "shadow",
                                                   10 * math.exp(0.027), 100 * math.exp(0.009)),
                              self.db.add_proposal(sol_ts, "crypto", "BTC-USD", "crypto", "kai", 2.0, "r", "dismissed", 1.0, 1.0)]),
            (nv_ts, lambda: self.db.add_proposal(nv_ts, "stocks", "NVDA", "stock", "kai", 2.0, "r", "added",
                                                 50 * math.exp(0.084), 400 * math.exp(0.021))),
            (late_ts, lambda: self.db.add_proposal(late_ts, "crypto", "BTC-USD", "crypto", "kai", 2.0, "r", "added", 1.0, 1.0)),
        ]
        pending.sort(key=lambda p: p[0])
        totals = dict(price0=0, outcomes=0, shadow=0, proposals=0)
        end = SESSION[-1] + 2 * H
        for now in range(T0, end + 1, 1800):
            clock[0] = now
            while pending and pending[0][0] <= now:
                pending.pop(0)[1]()
            self.assertEqual(pk.refresh({"SOL-USD": "crypto", "NVDA": "stock"}, now)["errors"], {})
            for k, v in mature(self.db, now).items():
                totals[k] += v
            # nothing is ever written before its horizon bar has closed
            bad = self.db.q("SELECT * FROM outcomes WHERE ts>?", (now,))
            self.assertEqual(bad, [])
        self.assertEqual(totals["price0"], 6)

        # crypto: clock hours from the end of the price0 bar (bar 9 ends at 10h)
        for ts, sign in ((sol_ts, 1), (sol_edge, -1)):
            oc = self.outcomes(self.note_id("SOL-USD", ts))
            self.assertEqual(set(oc), set(HORIZONS))
            for h in HORIZONS:
                hrs = HORIZON_HOURS_CRYPTO[h]
                self.assertAlmostEqual(oc[h]["ret"], 0.003 * hrs, places=10)
                self.assertAlmostEqual(oc[h]["bench_ret"], 0.001 * hrs, places=10)
                self.assertAlmostEqual(oc[h]["excess"], 0.002 * hrs, places=10)
                self.assertEqual(oc[h]["ts"], T0 + (10 + hrs) * H)
        n = self.db.q1("SELECT price0, bench0 FROM notes WHERE ts=?", (sol_ts,))
        self.assertAlmostEqual(n["price0"], 10 * math.exp(0.027))
        self.assertAlmostEqual(n["bench0"], 100 * math.exp(0.009))
        for h, o in self.outcomes(self.note_id("BTC-USD", btc_ts)).items():
            self.assertAlmostEqual(o["excess"], 0.0, places=12)                               # BTC against itself

        # stocks: bar counts in the stock's own series, through the weekend
        idx = {t: m for m, t in enumerate(SESSION)}
        oc = self.outcomes(self.note_id("NVDA", nv_ts))
        m0 = idx[utc(2026, 10, 8, 13, 30)]                                                    # Thursday's first bar
        for h in HORIZONS:
            b = HORIZON_BARS_STOCK[h]
            self.assertAlmostEqual(oc[h]["excess"], 0.003 * b, places=10)
            self.assertEqual(oc[h]["ts"], SESSION[m0 + b] + H)
        self.assertEqual(oc["1d"]["ts"], utc(2026, 10, 9, 14, 30))                            # same hour next session
        self.assertEqual(oc["7d"]["ts"], utc(2026, 10, 15, 14, 30))                           # five sessions later
        oc = self.outcomes(self.note_id("NVDA", nv_sat))                                      # Saturday note: Friday's close
        self.assertAlmostEqual(oc["1h"]["excess"], 0.003)
        self.assertEqual(oc["1h"]["ts"], utc(2026, 10, 12, 14, 30))                           # Monday's first bar
        self.assertEqual(oc["1d"]["ts"], utc(2026, 10, 12, 20, 30))                           # 7 bars on: Monday's last

        # the late note only has the horizons the data reached (crypto data stops at 400h)
        self.assertEqual(set(self.outcomes(self.note_id("SOL-USD", late_ts))), {"1h", "4h", "8h", "12h", "1d"})

        sh = {(r["kind"], bool(r["detail"].get("price0"))): r for r in self.db.shadow()}
        self.assertEqual(sh[("score", True)]["matured"], 1)
        self.assertAlmostEqual(sh[("score", True)]["excess"], 0.008, places=10)
        self.assertEqual(sh[("score", False)]["matured"], 0)
        self.assertAlmostEqual(sh[("penalty", True)]["excess"], 0.003 * 7, places=10)
        self.assertEqual(totals["shadow"], 2)

        props = {(p["symbol"], p["status"], p["ts"]): p for p in self.db.proposals()}
        self.assertAlmostEqual(props[("SOL-USD", "shadow", sol_ts)]["excess_7d"], 0.002 * 168, places=10)
        self.assertAlmostEqual(props[("NVDA", "added", nv_ts)]["excess_7d"], 0.003 * 35, places=10)
        self.assertIsNone(props[("BTC-USD", "dismissed", sol_ts)]["excess_7d"])
        self.assertIsNone(props[("BTC-USD", "added", late_ts)]["excess_7d"])                  # not a week old
        self.assertEqual(totals["proposals"], 2)
        self.assertEqual(mature(self.db, end), dict(price0=0, outcomes=0, shadow=0, proposals=0))  # idempotent


class ReliabilityTest(Base):
    def add(self, source, ts, direction, excess):
        self.db.add_notes([note("SOL-USD", "crypto", ts, source=source, direction=direction)])
        nid = self.db.q1("SELECT MAX(id) i FROM notes")["i"]
        self.db.add_outcomes([(nid, "1d", excess, 0.0, excess, ts + D)])

    def test_good_bad_and_ignored(self):
        now = T0 + 200 * D
        for i in range(40):
            ts = now - (i % 10 + 1) * D
            self.add("good", ts, 1 if i % 2 else -1, 0.01 if i % 2 else -0.01)
            self.add("bad", ts, 1, -0.02)
            self.add("tiny", ts, 1, 0.001)                     # moves below min_move are not graded
            self.add("old", now - 130 * D, 1, 0.05)            # outside the 120-day window
        self.add("neutral", now - D, 0, 0.05)                  # direction 0 is never graded
        self.add("one", now - 30 * D, 1, 0.01)                 # exactly one half-life old
        self.db.source_ok("old", "catalysts", "news", 1, 900, now=now)
        rows = {r["source"]: r for r in update_reliability(self.db, now)}
        self.assertGreater(rows["good"]["mean"], 0.85)
        self.assertEqual(rows["good"]["usefulness"], 1.0)
        self.assertLess(rows["bad"]["mean"], 0.15)
        self.assertEqual(rows["bad"]["usefulness"], 0.05)
        self.assertNotIn("neutral", rows)                      # never graded and never registered
        for s in ("tiny", "old"):
            self.assertEqual((rows[s]["rel_a"], rows[s]["rel_b"], rows[s]["usefulness"]), (2.0, 2.0, 0.5))
        self.assertAlmostEqual(rows["one"]["rel_a"], 2.5)
        self.assertAlmostEqual(rows["one"]["usefulness"], (2.5 / 4.5 - 0.5) * 2 * (0.5 / 30) + 0.5)
        rel = reliability(self.db)
        self.assertAlmostEqual(rel["good"], rows["good"]["mean"])
        self.assertEqual(rel["never-seen"], 0.5)
        self.assertEqual(self.db.sources()["bad"]["usefulness"], 0.05)


def synth_notes(db, planted, n_sym=30, days=90, seed=7, start=T0):
    """One directional note per symbol per day. planted=True: earnings +1 notes are right 80%
    of the time, everything else is a coin flip. About 40% of the notes are earnings +1."""
    rnd = random.Random(seed)
    words = [f"w{i}" for i in range(300)]
    notes, truth = [], []
    for d in range(days):
        for s in range(n_sym):
            ts = start + d * D + rnd.randint(0, 20 * H)
            cat = rnd.choice(["earnings", "earnings", "analyst", "partnership"])
            direction = 1 if (cat == "earnings" and rnd.random() < 0.85) else rnd.choice([-1, 1])
            title = " ".join(rnd.sample(words, 6)) + (" quarterly results" if cat == "earnings" else "")
            p_right = 0.8 if (planted and cat == "earnings" and direction == 1) else 0.5
            right = rnd.random() < p_right
            mag = rnd.uniform(0.005, 0.05)
            notes.append(note(f"S{s}-USD", "crypto", ts, source=f"src{s % 4}", direction=direction, catalyst=cat,
                              title=title, cluster=f"c{d}-{s}"))
            truth.append(direction * mag * (1 if right else -1))
    db.add_notes(notes)
    rows = db.q("SELECT id, ts FROM notes ORDER BY id")
    db.add_outcomes([(r["id"], "1d", e, 0.0, e, r["ts"] + D) for r, e in zip(rows, truth)])
    return start + days * D + 2 * D


class OutcomeModelTest(Base):
    def test_learns_planted_pattern_and_promotes_once(self):
        now = synth_notes(self.db, planted=True)
        rep = train_outcome_model(self.db, now)
        self.assertTrue(rep["promoted"], rep)
        self.assertLess(rep["logloss_challenger"], rep["logloss_baseline"] - 0.002)
        self.assertEqual(rep["n_holdout"], 675)                                              # newest 25% of 2700
        self.assertLess(rep["n_train"], 2025)                                                # purge removed the last day
        champ = self.db.champion("outcome")
        self.assertEqual(champ["version"], rep["version"])
        m = OutcomeModel.from_blob(champ["blob"])
        good = dict(catalyst="earnings", direction=1, confidence=0.6, evidence="news", label_by="rules",
                    title="w1 w2 quarterly results", reliability=0.5)
        other = dict(good, catalyst="analyst", direction=-1, title="w1 w2 w3")
        e = m.predict([good, other])
        self.assertGreater(e[0], 0.3)
        self.assertLess(abs(e[1]), 0.25)
        # edges for the last week's notes were written
        self.assertGreater(rep["edges"], 0)
        recent = self.db.q("SELECT edge FROM notes WHERE ts>?", (now - 7 * D,))
        self.assertTrue(all(r["edge"] is not None and -1 <= r["edge"] <= 1 for r in recent))
        # same data again: the challenger cannot beat the champion by the margin
        rep2 = train_outcome_model(self.db, now)
        self.assertFalse(rep2["promoted"], rep2)
        self.assertIn("champion", rep2["reason"])
        self.assertEqual(self.db.champion("outcome")["version"], rep["version"])

    def test_no_look_ahead_on_a_replay_clock(self):
        synth_notes(self.db, planted=True)
        mid = T0 + 45 * D + 7 * H                       # replaying history: outcomes after `mid` exist but are unknown yet
        expect = self.db.q1("SELECT COUNT(*) n FROM notes WHERE ts + ? <= ?", (D, mid))["n"]
        self.assertEqual(train_outcome_model(self.db, mid)["n_rows"], expect)
        rows = {r["source"]: r for r in update_reliability(self.db, mid)}
        expect = self.db.q1("SELECT COUNT(*) n FROM notes WHERE source='src0' AND ts + ? <= ?", (D, mid))["n"]
        self.assertEqual(rows["src0"]["n"], expect)

    def test_neutral_notes_get_zero_edge(self):
        now = synth_notes(self.db, planted=True)
        self.db.add_notes([note("S0-USD", "crypto", now - D, direction=0, title="market colour")])
        rep = train_outcome_model(self.db, now)
        self.assertTrue(rep["promoted"])
        self.assertEqual(self.db.q1("SELECT edge FROM notes WHERE direction=0")["edge"], 0.0)
        self.assertEqual(train_outcome_model(self.db, now, horizon="4h")["edges"], 0)   # only the 1d model writes edges

    def test_no_promotion_on_noise(self):
        now = synth_notes(self.db, planted=False, seed=11)
        rep = train_outcome_model(self.db, now)
        self.assertFalse(rep["promoted"], rep)
        self.assertIsNone(self.db.champion("outcome"))
        self.assertEqual(rep["edges"], 0)                                                    # no champion: no edges
        self.assertEqual(len(self.db.models("outcome")), 1)                                  # challenger kept for the record

    def test_too_few_rows(self):
        now = synth_notes(self.db, planted=True, n_sym=2, days=30)
        rep = train_outcome_model(self.db, now)
        self.assertFalse(rep["promoted"])
        self.assertIn("need 300", rep["reason"])

    def test_dedupe_keeps_first_per_cluster_and_three_per_day(self):
        rows = [dict(symbol="A", cluster="x", ts=T0 + i) for i in range(3)]
        rows += [dict(symbol="A", cluster=f"y{i}", ts=T0 + 100 + i) for i in range(5)]
        rows += [dict(symbol="A", cluster="", ts=T0 + D + i) for i in range(5)]
        out = outcome._dedupe(rows)
        self.assertEqual([r["ts"] for r in out], [T0, T0 + 100, T0 + 101, T0 + D, T0 + D + 1, T0 + D + 2])

    def test_point_in_time_reliability_has_no_look_ahead(self):
        rows = [dict(id=i, ts=T0 + i * D, source="s", direction=1, outcomes={"1d": 0.05}) for i in range(5)]
        known = {i: T0 + i * D + D for i in range(5)}                                         # each known a day later
        outcome._pit_reliability(rows, known, "1d")
        self.assertEqual(rows[0]["reliability"], 0.5)                                       # nothing known yet
        self.assertAlmostEqual(rows[1]["reliability"], 3 / 5)                               # only note 0 known
        self.assertGreater(rows[4]["reliability"], rows[2]["reliability"])

    def test_fit_skips_nan_and_blob_round_trip(self):
        rows = [dict(catalyst="earnings", direction=1, confidence=float("nan"), title="beat", outcomes={"1d": 0.05}),
                dict(catalyst="legal", direction=-1, title="sued", outcomes={"1d": float("nan")}),
                dict(catalyst="legal", direction=-1, title="sued", outcomes={}),
                dict(catalyst="nonsense", direction=1, severity=None, title="", outcomes={"1d": -0.05})]
        m = OutcomeModel().fit(rows)
        self.assertEqual(m.metrics["n"], 2)
        e = m.predict(rows)
        self.assertTrue(np.all(np.isfinite(e)) and np.all(np.abs(e) <= 1))
        m2 = OutcomeModel.from_blob(m.to_blob())
        np.testing.assert_allclose(m2.predict(rows), e)
        self.assertEqual(m2.metrics["n"], 2)
        with self.assertRaises(ValueError):
            OutcomeModel.from_blob(b"not a model")
        meta = json.loads(zlib.decompress(m.to_blob()))
        meta["tok"] = "something-else"                                                    # hashed words would not line up
        with self.assertRaises(ValueError):
            OutcomeModel.from_blob(zlib.compress(json.dumps(meta).encode()))
        self.assertTrue(np.all(OutcomeModel().predict(rows) == 0))                          # unfitted: no idea


if __name__ == "__main__":
    unittest.main()
