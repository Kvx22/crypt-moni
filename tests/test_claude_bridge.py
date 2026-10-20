"""The Claude file bridge: offers only approved orders, applies only validated results.
python -m unittest tests.test_claude_bridge"""
import json
import os
import shutil
import sys
import tempfile
import time
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import claude_bridge as cb  # noqa: E402
from cryptomoni.models import Order  # noqa: E402
from cryptomoni.store import Store  # noqa: E402


class TestBridge(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.cfg = dict(db=os.path.join(self.d, "b.db"), safety=dict(kill_file=os.path.join(self.d, "KILL"), max_per_trade_usd=20),
                        execution=dict(broker="robinhood", snapshot=os.path.join(self.d, "snap.json")))
        self.st = Store(self.cfg["db"])
        self.now = time.time()

    def tearDown(self):
        self.st.c.close()
        shutil.rmtree(self.d, ignore_errors=True)

    def propose(self, sym="XRP-USD", side="buy", notional=50.0, approve=True):
        oid = self.st.add_outbox(self.now, self.now + 7200, Order(sym, side, notional, "market", None, "t"), 1.27, "bull", "buy", "robinhood")
        if approve:
            self.st.outbox_set(oid, "approved", self.now, decided_by="test")
        return oid

    def test_inbox_lists_only_approved_orders_capped_with_a_stable_ref_id(self):
        a = self.propose()
        self.propose("ETH-USD", approve=False)
        from cryptomoni.safety import KillSwitch
        inbox = cb.build_inbox(self.cfg, self.st, KillSwitch(self.cfg["safety"]["kill_file"]), self.now, self.cfg["execution"]["snapshot"])
        self.assertEqual([o["outbox_id"] for o in inbox["orders"]], [a])
        o = inbox["orders"][0]
        self.assertEqual(o["notional_usd"], 20.0)
        self.assertEqual(o["ref_id"], cb.ref_id_for(o["client_order_id"]))
        self.assertTrue(inbox["snapshot_needed"])

    def test_kill_switch_empties_the_inbox(self):
        self.propose()
        open(self.cfg["safety"]["kill_file"], "w").write("x")
        from cryptomoni.safety import KillSwitch
        inbox = cb.build_inbox(self.cfg, self.st, KillSwitch(self.cfg["safety"]["kill_file"]), self.now, self.cfg["execution"]["snapshot"])
        self.assertTrue(inbox["kill_switch"])
        self.assertEqual(inbox["orders"], [])

    def results(self, oid, **kw):
        r = dict(run_id="r1", written_at=self.now, snapshot=dict(cash=43.57, positions={"xrp-usd": 5}, other_value=0),
                 orders=[dict(outbox_id=oid, status="sent", order_id="abc", detail="d")])
        r.update(kw)
        return r

    def test_applies_an_offered_approved_order_once(self):
        a = self.propose()
        done = []
        msgs, acc, snap = cb.apply_results(self.cfg, self.st, self.results(a), self.now, {a}, done, self.cfg["execution"]["snapshot"])
        self.assertEqual(acc, [a])
        self.assertTrue(snap)
        self.assertEqual(self.st.outbox_get(a)["status"], "sent")
        saved = json.load(open(self.cfg["execution"]["snapshot"]))
        self.assertEqual(saved["positions"], {"XRP-USD": 5.0})
        msgs, acc, _ = cb.apply_results(self.cfg, self.st, self.results(a), self.now, {a}, done, self.cfg["execution"]["snapshot"])
        self.assertEqual(acc, [])   # same run_id: nothing happens twice

    def test_refuses_orders_it_never_offered_or_that_are_not_approved(self):
        a = self.propose()
        pending = self.propose("ETH-USD", approve=False)
        for oid, offered in ((a, set()), (pending, {pending})):
            _, acc, _ = cb.apply_results(self.cfg, self.st, self.results(oid, run_id=f"r{oid}"), self.now, offered, [], self.cfg["execution"]["snapshot"])
            self.assertEqual(acc, [])
        self.assertEqual(self.st.outbox_get(a)["status"], "approved")
        self.assertEqual(self.st.outbox_get(pending)["status"], "pending")

    def test_rejects_bad_snapshots_and_stale_results(self):
        a = self.propose()
        for snap in (dict(cash=-1, positions={}), dict(cash=1, positions={"bad name!": 1}), dict(cash="x", positions={}),
                     dict(cash=1, positions={"BTC-USD": float("nan")})):
            _, _, applied = cb.apply_results(self.cfg, self.st, self.results(a, snapshot=snap, orders=[]), self.now, {a}, [], self.cfg["execution"]["snapshot"])
            self.assertFalse(applied)
        self.assertFalse(os.path.exists(self.cfg["execution"]["snapshot"]))
        _, acc, _ = cb.apply_results(self.cfg, self.st, self.results(a, written_at=self.now - 7200), self.now, {a}, [], self.cfg["execution"]["snapshot"])
        self.assertEqual(acc, [])

    def test_sent_needs_an_order_id_and_failed_is_recorded(self):
        a, b = self.propose(), self.propose("SOL-USD")
        res = self.results(a, orders=[dict(outbox_id=a, status="sent", order_id=""), dict(outbox_id=b, status="failed", detail="boom")])
        _, acc, _ = cb.apply_results(self.cfg, self.st, res, self.now, {a, b}, [], self.cfg["execution"]["snapshot"])
        self.assertEqual(acc, [b])
        self.assertEqual(self.st.outbox_get(b)["status"], "failed")
        self.assertEqual(self.st.outbox_get(a)["status"], "approved")

    def test_full_cycle_files_and_half_written_results(self):
        a = self.propose()
        self.st.c.commit()
        inbox = cb.cycle(self.cfg, now=self.now, log=lambda *_: None)
        p = cb.paths(self.cfg)
        self.assertEqual([o["outbox_id"] for o in inbox["orders"]], [a])
        self.assertEqual(json.load(open(p["inbox"]))["orders"][0]["outbox_id"], a)
        open(p["results"], "w").write('{"run_id": "x", ')            # Claude is still writing
        cb.cycle(self.cfg, now=self.now, log=lambda *_: None)
        self.assertTrue(os.path.exists(p["results"]))                # left alone for the next cycle
        json.dump(self.results(a, written_at=time.time()), open(p["results"], "w"))
        cb.cycle(self.cfg, now=time.time(), log=lambda *_: None)
        self.assertFalse(os.path.exists(p["results"]))
        self.assertTrue(os.path.exists(p["results"].replace(".json", ".done.json")))
        self.assertEqual(Store(self.cfg["db"]).outbox_get(a)["status"], "sent")
        self.assertEqual(json.load(open(p["inbox"]))["orders"], [])


if __name__ == "__main__":
    unittest.main()
