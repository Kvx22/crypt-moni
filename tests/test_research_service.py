"""The research service's process plumbing: one service per research database (lease), the
board-side supervisor (restart with back-off, crash-loop report, clean stop through stdin).
python -m unittest tests.test_research_service"""
import os
import shutil
import sys
import tempfile
import time
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from cryptomoni.research.db import ResearchDB  # noqa: E402
from cryptomoni.research.service import ServiceSupervisor  # noqa: E402


class TestLease(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.db = ResearchDB(os.path.join(self.dir, "r.db"))

    def tearDown(self):
        self.db.close()
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_one_holder_and_takeover_when_stale(self):
        self.assertTrue(self.db.lease_try("a", 1, now=1000))
        self.assertFalse(self.db.lease_try("b", 2, now=1010))     # a is fresh: b stands by
        self.assertTrue(self.db.lease_try("a", 1, now=1020))      # a renews
        self.assertFalse(self.db.lease_try("b", 2, now=1050))
        self.assertTrue(self.db.lease_try("b", 2, now=1100))      # a went quiet > 45 s: b takes over
        self.assertFalse(self.db.lease_try("a", 1, now=1101))     # and a must stand down
        self.db.lease_release("b")
        self.assertTrue(self.db.lease_try("a", 1, now=1102))


class TestSupervisor(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.events = []

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def _sup(self, script):
        path = os.path.join(self.dir, "child.py")
        with open(path, "w") as f:
            f.write(script)
        cfg = dict(db=os.path.join(self.dir, "b.db"), research=dict(db=os.path.join(self.dir, "r.db")))
        return ServiceSupervisor("cfg.json", cfg, lambda *a: self.events.append(a), run_py=path)

    def _wait_exit(self, sup, timeout=20):
        t0 = time.time()
        while sup.proc is not None and sup.proc.poll() is None and time.time() - t0 < timeout:
            time.sleep(0.05)

    def test_restart_backoff_and_crash_loop(self):
        sup = self._sup("import sys; sys.exit(3)\n")
        now = 10_000.0
        self.assertEqual(sup.tick(now), "started")
        for k in range(5):
            self._wait_exit(sup)
            r = sup.tick(now)
            self.assertEqual(r, "waiting")
            self.assertGreaterEqual(sup.next_start - now, ServiceSupervisor.BACKOFF[min(k, 3)] - 1e-6)
            now = sup.next_start
            self.assertEqual(sup.tick(now), "started")
        self._wait_exit(sup)
        sup.tick(now)
        self.assertTrue(sup.degraded)
        self.assertTrue(any(e[2] == "critical" and "crashing" in e[3] for e in self.events))
        self.assertTrue(os.path.exists(sup.log_path))

    def test_stop_closes_stdin_and_the_child_exits(self):
        sup = self._sup("import sys\nwhile sys.stdin.buffer.read(1):\n    pass\nsys.exit(0)\n")
        self.assertEqual(sup.tick(), "started")
        time.sleep(0.3)
        self.assertIsNone(sup.proc.poll())          # alive while the parent holds its stdin
        p = sup.proc
        sup.stop()
        self.assertEqual(p.wait(10), 0)


if __name__ == "__main__":
    unittest.main()
