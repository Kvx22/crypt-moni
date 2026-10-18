"""python -m unittest tests.test_research_workers"""
import importlib
import os
import shutil
import sys
import tempfile
import threading
import time
import types as pytypes
import unittest
import zlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from cryptomoni.research import sources, workers  # noqa: E402
from cryptomoni.research.db import ResearchDB  # noqa: E402
from cryptomoni.research.http import RateLimited, SourceError  # noqa: E402
from cryptomoni.research.sources.base import Adapter  # noqa: E402
from cryptomoni.research.types import Item  # noqa: E402
from cryptomoni.research.workers import Slot, WorkerPool, in_shard, usefulness_mult  # noqa: E402

T0 = 1_800_000_000.0


class Clock:
    def __init__(self, t=T0):
        self.t = t

    def __call__(self):
        return self.t


def make(name, team="catalysts", role="primary", standby_for=None, per_symbol=False, interval=900.0,
         assets=("crypto", "stock"), cap=8, behaviour="ok"):
    """A fake adapter. behaviour: 'ok' | 'fail' | 'rate' | 'garbage'; change .behaviour at will."""
    class Fake(Adapter):
        def fetch(self, ctx):
            self.calls.append(dict(symbols=dict(ctx.symbols), requested=list(ctx.requested), names=dict(ctx.names)))
            ctx.state["runs"] = ctx.state.get("runs", 0) + 1
            if self.behaviour == "fail":
                raise SourceError(f"{self.name}: HTTP 500")
            if self.behaviour == "rate":
                raise RateLimited(f"{self.name}: 429", retry_after=self.retry_after)
            if self.behaviour == "garbage":
                return 42
            if self.behaviour == "boom":
                raise KeyError("price")
            n = len(self.calls)
            return [Item(source=self.name, url=f"https://x.test/{self.name}/{n}/{s}", title=f"{s} story {n}", symbols=[s])
                    for s in (ctx.symbols or {"": ""}) if s] or [Item(source=self.name, url=f"https://x.test/{self.name}/{n}",
                                                                      title="general story")]
    a = Fake()
    a.name, a.team, a.role, a.standby_for, a.per_symbol = name, team, role, standby_for, per_symbol
    a.interval_s, a.assets, a.max_symbols_per_run, a.behaviour = interval, assets, cap, behaviour
    a.calls, a.retry_after = [], 120.0
    return a


class Base(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.db = ResearchDB(os.path.join(self.dir, "r.db"))
        self.clock = Clock()

    def tearDown(self):
        self.db.close()
        shutil.rmtree(self.dir, ignore_errors=True)

    def pool(self, adapters, **cfg):
        cfg.setdefault("gap_warmup_s", 0)
        p = WorkerPool(self.db, http=None, adapters=adapters, cfg=cfg, clock=self.clock)
        p.plan()
        return p

    def subscribe(self, symbols):
        self.db.subscribe("crypto", symbols, now=self.clock())

    def slot(self, p, name):
        return [s for s in p.slots if s.name == name][0]


class TestPlan(Base):
    def test_plan_counts_and_rows(self):
        ads = [make("a"), make("b", role="standby", standby_for="a"), make("c", team="risk"),
               make("g", per_symbol=True), make("d", role="dormant"), make("o", role="standby", standby_for="gone")]
        p = self.pool(ads, shards={"g": 2})
        ids = sorted(s.id for s in p.slots)
        self.assertEqual(ids, sorted(["catalysts/a#0of1", "catalysts/b#0of1", "risk/c#0of1", "catalysts/g#0of2",
                                      "catalysts/g#1of2", "catalysts/o#0of1"]))
        b, o = self.slot(p, "b"), self.slot(p, "o")
        self.assertEqual((b.state, b.standby_of), ("standby", "catalysts/a#0of1"))
        self.assertEqual((o.state, o.standby_of), ("active", None))   # its primary is missing: it runs instead
        self.assertEqual(sorted(r["slot"] for r in self.db.workers()), ids)
        h = p.health()
        self.assertEqual(h["counts"], dict(active=5, standby=1, failed=0, grown=0, total=6))
        self.assertEqual(h["teams"]["risk"], dict(active=1, standby=0, failed=0, last_ok=None))
        self.assertEqual(sources.count_roles(ads), dict(primary=3, standby=2, dormant=1))

    def test_plan_drops_stale_rows_and_respects_max_workers(self):
        self.db.worker_set("old/slot#0of1", "old", "risk", "primary", 0, 1, "active")
        p = self.pool([make(f"a{i}") for i in range(5)], max_workers=3)
        self.assertEqual(len(p.slots), 3)
        self.assertNotIn("old/slot#0of1", [r["slot"] for r in self.db.workers()])


class TestRuns(Base):
    def test_success_stores_items_state_and_schedule(self):
        a = make("a")
        p = self.pool([a])
        self.subscribe({"BTC-USD": "crypto", "SPY": "stock"})
        s = p.slots[0]
        self.assertEqual(p.due(), [s])
        self.assertEqual(p.run_slot(s, T0), 2)
        self.assertEqual((s.runs, s.items, s.last_ok, s.next_due), (1, 2, T0, T0 + 900))
        self.assertEqual(self.db.kv_get("adapter_state:a"), {"runs": 1})
        self.assertEqual(self.db.sources()["a"]["items"], 2)
        self.assertEqual(p.due(T0 + 899), [])
        self.assertEqual(p.due(T0 + 900), [s])

    def test_backoff_doubles_and_caps(self):
        a = make("a", behaviour="fail")
        p = self.pool([a])
        s = p.slots[0]
        for k, now in enumerate([T0, T0 + 10, T0 + 20], start=1):
            p.run_slot(s, now)
            self.assertEqual(s.next_due, now + 900 * 2 ** k)
        self.assertEqual(s.consecutive_err, 3)
        self.assertEqual(s.err, "a: HTTP 500")
        s.consecutive_err = 9
        p.run_slot(s, T0 + 100)
        self.assertEqual(s.next_due, T0 + 100 + 6 * 3600)
        self.assertEqual(self.db.sources()["a"]["consecutive_err"], 4)

    def test_rate_limit_counts_only_after_three(self):
        a = make("a", behaviour="rate")
        a.retry_after = 300
        p = self.pool([a])
        s = p.slots[0]
        p.run_slot(s, T0)
        p.run_slot(s, T0 + 300)
        self.assertEqual((s.consecutive_err, s.next_due), (0, T0 + 600))
        p.run_slot(s, T0 + 600)
        self.assertEqual(s.consecutive_err, 1)
        self.assertEqual(s.next_due, T0 + 600 + 1800)   # max(retry_after, interval*2)
        a.behaviour = "ok"
        p.run_slot(s, s.next_due)
        self.assertEqual((s.consecutive_err, s.rate_limited), (0, 0))

    def test_bad_adapters_never_raise(self):
        for b in ("garbage", "boom"):
            a = make("a" + b, behaviour=b)
            p = self.pool([a])
            s = p.slots[0]
            self.assertEqual(p.run_slot(s, T0), 0)
            self.assertEqual(s.consecutive_err, 1)
        self.assertIn("KeyError", s.err)

    def test_dicts_and_junk_in_results(self):
        class Mixed(Adapter):
            name = "mixed"

            def fetch(self, ctx):
                return [Item(source="mixed", url="https://x.test/1", title="one"), "junk", None,
                        dict(source="mixed", url="https://x.test/2", title="two", bogus=1), dict(nope=1)]
        p = self.pool([Mixed()])
        self.assertEqual(p.run_slot(p.slots[0], T0), 2)

    def test_dead_database_never_raises(self):
        p = self.pool([make("a")])
        self.db.close()
        self.assertEqual(p.run_slot(p.slots[0], T0), 0)
        self.assertEqual(p.supervise(T0), [])
        self.db = ResearchDB(os.path.join(self.dir, "r.db"))

    def test_usefulness_changes_interval(self):
        p = self.pool([make("a"), make("b", interval=100)])
        self.db.source_ok("a", "catalysts", "news", 0, 900, now=T0)
        self.db.source_ok("b", "catalysts", "news", 0, 900, now=T0)
        self.db.set_reliability([(1, 9, 0.05, "a"), (9, 1, 1.0, "b")])
        p.refresh_usefulness()
        self.assertAlmostEqual(self.slot(p, "a").interval, 900 * 2 ** 0.9)
        self.assertEqual(self.slot(p, "b").interval, 120.0)   # 50 s, clamped
        self.assertEqual(usefulness_mult(0.5), 1.0)
        self.assertEqual(usefulness_mult(0.0), 2.0)
        self.assertEqual(usefulness_mult(None), 1.0)


class TestSharding(Base):
    SYMS = {f"C{i}-USD": "crypto" for i in range(30)} | {f"S{i}": "stock" for i in range(20)}

    def test_assignment_stable_and_complete(self):
        for n in (1, 2, 3, 4):
            parts = [{s for s in self.SYMS if in_shard(s, k, n)} for k in range(n)]
            self.assertEqual(set().union(*parts), set(self.SYMS))
            self.assertEqual(sum(map(len, parts)), len(self.SYMS))
        self.assertTrue(in_shard("BTC-USD", zlib.crc32(b"BTC-USD") % 3, 3))

    def test_rotation_covers_shard_and_requested_first(self):
        a = make("g", per_symbol=True, cap=4, assets=("crypto",))
        p = self.pool([a], shards={"g": 2})
        self.subscribe(self.SYMS)
        s0 = self.slot(p, "g")
        mine = {s for s, x in self.SYMS.items() if x == "crypto" and in_shard(s, s0.shard, 2)}
        req = sorted(mine)[-1]
        self.db.request("crypto", req, "crypto", "kai", "why?", 1.0, now=T0)
        self.db.request("crypto", "S1", "stock", "kai", "why?", 0.5, now=T0)   # not covered by this adapter
        seen = set()
        for i in range(len(mine)):
            p.run_slot(s0, T0 + i)
            got = list(a.calls[-1]["symbols"])
            self.assertEqual(got[0], req)
            self.assertLessEqual(len(got), 4)
            self.assertTrue(set(got) <= mine)
            seen |= set(got)
        self.assertEqual(seen, mine)
        self.assertIn("S1", a.calls[-1]["requested"])
        self.assertIn("_cursor_0of2", self.db.kv_get("adapter_state:g"))

    def test_requests_make_per_symbol_slots_due_sooner(self):
        a = make("g", per_symbol=True)
        p = self.pool([a, make("plain")])
        self.subscribe({"BTC-USD": "crypto"})
        g, plain = p.slots
        p.run_slot(g, T0)
        p.run_slot(plain, T0)
        self.assertEqual(p.due(T0 + 10), [])
        self.db.request("crypto", "BTC-USD", "crypto", "kai", "?", 1.0, now=T0)
        self.assertEqual(p.due(T0 + 10), [])
        self.assertEqual(p.due(T0 + 310), [g])

    def test_empty_shard_skips_fetch(self):
        a = make("g", per_symbol=True)
        p = self.pool([a])
        self.assertEqual(p.run_slot(p.slots[0], T0), 0)
        self.assertEqual(a.calls, [])
        self.assertEqual(p.slots[0].next_due, T0 + 900)

    def test_state_merge_between_shards(self):
        p = self.pool([make("g", per_symbol=True)], shards={"g": 2})
        self.db.kv_set("adapter_state:g", {"keep": 1})
        p._persist_state("g", {"keep": 1}, {"keep": 1, "_cursor_0of2": 3})
        p._persist_state("g", {"keep": 1}, {"keep": 1, "_cursor_1of2": 5})   # same stale base: no clobbering
        self.assertEqual(self.db.kv_get("adapter_state:g"), {"keep": 1, "_cursor_0of2": 3, "_cursor_1of2": 5})


class TestFailover(Base):
    def setUp(self):
        super().setUp()
        self.a = make("coindesk_rss", behaviour="fail")
        self.b = make("cointelegraph_rss", role="standby", standby_for="coindesk_rss")
        self.p = self.pool([self.a, self.b])
        self.pa, self.pb = self.p.slots

    def fail3(self, slot, t=T0):
        for i in range(3):
            self.p.run_slot(slot, t + i)

    def test_failover_and_recovery(self):
        self.fail3(self.pa)
        self.assertNotIn(self.pb, self.p.due(T0 + 5))
        ev = self.p.supervise(T0 + 5)
        self.assertEqual(ev[0]["title"], "Standby cointelegraph_rss took over from coindesk_rss: coindesk_rss: HTTP 500")
        self.assertEqual((self.pa.state, self.pb.state), ("failed", "active"))
        self.assertEqual(self.p.due(T0 + 5), [self.pb])
        self.assertEqual(self.p.run_slot(self.pb, T0 + 5), 1)
        self.assertEqual(self.p.supervise(T0 + 6), [])            # no repeat events
        self.assertNotIn(self.pa, self.p.due(T0 + 5 + 1799))
        self.assertIn(self.pa, self.p.due(T0 + 5 + 1800))           # one probe run per 30 min
        self.p.run_slot(self.pa, T0 + 1805)                        # still broken
        self.assertEqual((self.pa.state, self.pa.next_due), ("failed", T0 + 1805 + 1800))
        self.a.behaviour = "ok"
        self.p.run_slot(self.pa, T0 + 3605)
        self.assertEqual((self.pa.state, self.pb.state), ("active", "standby"))
        ev = self.p.supervise(T0 + 3606)
        self.assertEqual([e["kind"] for e in ev], ["recovered"])
        self.assertIn("went back on standby", ev[0]["body"])
        states = {r["slot"]: r["state"] for r in self.db.workers()}
        self.assertEqual(states, {self.pa.id: "active", self.pb.id: "standby"})
        kinds = [e["kind"] for e in self.db.events()]
        self.assertEqual(sorted(kinds), ["failover", "recovered"])

    def test_degraded_when_standby_fails_too(self):
        self.fail3(self.pa)
        self.p.supervise(T0 + 5)
        self.b.behaviour = "fail"
        self.fail3(self.pb, T0 + 10)
        ev = self.p.supervise(T0 + 20)
        self.assertEqual(len(ev), 1)
        self.assertEqual(ev[0]["severity"], "warn")
        self.assertIn("degraded", ev[0]["title"])
        self.assertEqual(self.p.health()["counts"]["failed"], 2)
        # the standby recovers first: it keeps covering while the primary is still down
        self.b.behaviour = "ok"
        self.p.run_slot(self.pb, T0 + 20 + 1800)
        self.assertEqual(self.pb.state, "active")

    def test_late_failover_gives_standby_a_fresh_heartbeat(self):
        later = T0 + 10 * 86400
        self.fail3(self.pa, later)
        self.p.supervise(later + 5)
        self.assertEqual(self.p.supervise(later + 6), [])
        self.assertEqual(self.pb.state, "active")

    def test_stale_heartbeat_fails_over(self):
        self.a.behaviour = "ok"
        self.assertEqual(self.p.supervise(T0 + 3600), [])           # 4 x 900 s: not yet older
        ev = self.p.supervise(T0 + 3601)
        self.assertEqual(self.pa.state, "failed")
        self.assertIn("no heartbeat", ev[0]["title"])

    def test_rate_limits_alone_do_not_fail_over(self):
        self.a.behaviour = "rate"
        self.p.run_slot(self.pa, T0)
        self.p.run_slot(self.pa, T0 + 120)
        self.assertEqual(self.p.supervise(T0 + 130), [])
        self.assertEqual(self.pa.state, "active")


class TestGrowth(Base):
    def cov(self, covered, total=10, team="catalysts"):
        return {team: dict(symbols_covered=covered, symbols_total=total, notes=covered, sources_active=1)}

    def test_grow_dormant_then_shard_then_shortfall(self):
        ads = [make("a"), make("g", per_symbol=True), make("d", role="dormant"), make("r", team="risk"),
               make("x", role="dormant", team="risk")]
        p = self.pool(ads, max_workers=60)
        ev = p.gap_check(T0, self.cov(3))
        self.assertEqual(len(ev), 1)
        d = self.slot(p, "d")
        self.assertEqual((d.role, d.state, d.grown), ("grown", "active", True))
        self.assertNotIn("x", [s.name for s in p.slots])           # other team: not grown
        ev = p.gap_check(T0 + 1, self.cov(3))
        self.assertIn("Split g into 2 shards", ev[0]["title"])
        self.assertEqual(sorted(s.id for s in p.slots if s.name == "g"), ["catalysts/g#0of2", "catalysts/g#1of2"])
        for n in (3, 4):
            p.gap_check(T0 + n, self.cov(3))
        self.assertEqual(len([s for s in p.slots if s.name == "g"]), 4)
        ev = p.gap_check(T0 + 10, self.cov(3))
        self.assertEqual(ev[0]["kind"], "diversity")
        self.assertEqual(p.gap_check(T0 + 20, self.cov(3)), [])     # once a day
        self.assertEqual(p.gap_check(T0 + 10 + 86400, self.cov(3))[0]["kind"], "diversity")
        self.assertEqual(sorted(r["slot"] for r in self.db.workers()), sorted(s.id for s in p.slots))

    def test_no_gap_no_growth_and_zero_active_is_a_gap(self):
        p = self.pool([make("a"), make("d", role="dormant"), make("r", team="risk", behaviour="ok"),
                       make("rd", team="risk", role="dormant")])
        self.assertEqual(p.gap_check(T0, self.cov(6)), [])
        self.slot(p, "r").state = "failed"
        ev = p.gap_check(T0 + 1, self.cov(6))
        self.assertEqual(len(ev), 1)
        self.assertEqual(self.slot(p, "rd").state, "active")

    def test_never_past_max_workers(self):
        ads = [make("a"), make("g", per_symbol=True)] + [make(f"d{i}", role="dormant") for i in range(5)]
        p = self.pool(ads, max_workers=4)
        for i in range(10):
            p.gap_check(T0 + i, self.cov(0))
        self.assertEqual(p.health()["counts"]["total"], 4)
        self.assertTrue(any(e["kind"] == "diversity" for e in self.db.events()))

    def test_warmup_ignores_coverage_but_not_zero_slots(self):
        p = self.pool([make("a"), make("d", role="dormant")], gap_warmup_s=3600)
        self.assertEqual(p.gap_check(T0 + 10, self.cov(0)), [])
        self.assertEqual(len(p.gap_check(T0 + 3600, self.cov(0))), 1)

    def test_shrink_after_48h_without_gap(self):
        ads = [make("a"), make("g", per_symbol=True), make("d", role="dormant")]
        p = self.pool(ads)
        p.gap_check(T0, self.cov(1))       # grows d
        p.gap_check(T0 + 1, self.cov(1))   # splits g
        self.db.source_ok("d", "catalysts", "news", 0, 900, now=T0)
        self.db.source_ok("g", "catalysts", "news", 0, 900, now=T0)
        self.db.set_reliability([(1, 3, 0.3, "d"), (1, 3, 0.3, "g")])
        self.assertEqual(p.gap_check(T0 + 1 + 47 * 3600, self.cov(9)), [])
        ev = p.gap_check(T0 + 2 + 48 * 3600, self.cov(9))
        self.assertEqual(sorted(e["kind"] for e in ev), ["shrink", "shrink"])
        self.assertEqual(self.slot(p, "d").state, "stopped")
        self.assertEqual([s.id for s in p.slots if s.name == "g"], ["catalysts/g#0of1"])
        self.assertEqual(p.health()["counts"], dict(active=2, standby=0, failed=0, grown=0, total=2))
        # a useful grown source stays; a new gap regrows the stopped one
        ev = p.gap_check(T0 + 3 + 48 * 3600, self.cov(1))
        self.assertEqual(self.slot(p, "d").state, "active")


class TestLoop(Base):
    def test_run_forever_runs_due_slots_and_stops(self):
        a, slow = make("a"), make("slow")
        started = threading.Event()
        orig = slow.fetch

        def slow_fetch(ctx):
            started.set()
            time.sleep(0.3)
            return orig(ctx)
        slow.fetch = slow_fetch
        p = WorkerPool(self.db, None, [a, slow], cfg={}, clock=time.time)
        p.plan()
        stop = threading.Event()
        th = threading.Thread(target=p.run_forever, args=(stop,), kwargs=dict(threads=2), daemon=True)
        th.start()
        self.assertTrue(started.wait(2))
        self.assertNotIn(self.slot(p, "slow"), p.due())          # running: never submitted twice
        deadline = time.time() + 2
        while time.time() < deadline and (not a.calls or self.slot(p, "slow").running):
            time.sleep(0.02)
        stop.set()
        th.join(2)
        self.assertFalse(th.is_alive())
        self.assertEqual((len(a.calls), len(slow.calls)), (1, 1))
        self.assertEqual(p.health()["counts"]["active"], 2)


class TestCatalog(unittest.TestCase):
    def test_import_errors_and_injected_sim(self):
        name = "cryptomoni.research.sources.sim"
        saved = sys.modules.get(name)
        a1, a2 = make("sim_a"), make("sim_b", role="standby", standby_for="sim_a")

        class Broken(Adapter):
            name = "broken"

            def __init__(self):
                raise RuntimeError("nope")
        fake = pytypes.ModuleType(name)
        fake.SIM_ADAPTERS = [type(a1), a2, type(a1), Broken, object]
        type(a1).name = "sim_a"
        sys.modules[name] = fake
        try:
            got = sources.catalog({"disable": ["sim_b"]}, sim=True)
        finally:
            if saved is not None:
                sys.modules[name] = saved
            else:
                sys.modules.pop(name, None)
        self.assertEqual([x.name for x in got], ["sim_a"])
        self.assertIn("Broken", sources.IMPORT_ERRORS)
        self.assertIsInstance(sources.catalog(None), list)


def _have_all():
    for m in ("crypto", "stocks", "web"):
        try:
            importlib.import_module(f"cryptomoni.research.sources.{m}")
        except Exception:
            return False
    return True


@unittest.skipIf(not _have_all(), "source modules crypto/stocks/web not all present yet")
class TestRealCatalog(unittest.TestCase):
    def test_forty_workers(self):
        ads = sources.catalog({})
        self.assertGreaterEqual(len(ads), 40)
        roles = sources.count_roles(ads)
        self.assertEqual((roles["primary"], roles["standby"]), (20, 20))
        prim = {a.name: a.team for a in ads if sources.role_of(a) == "primary"}
        for a in ads:
            if sources.role_of(a) == "standby":
                self.assertEqual(prim.get(a.standby_for), a.team, a.name)
        self.assertEqual(len({a.name for a in ads}), len(ads))


if __name__ == "__main__":
    unittest.main()
