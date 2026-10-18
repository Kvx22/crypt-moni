"""python -m unittest tests.test_research_forever"""
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from cryptomoni.research import forever, thesis  # noqa: E402
from cryptomoni.research.db import ResearchDB  # noqa: E402
from cryptomoni.research.forever import (ForeverSupervisor, QuarantineDB, SupervisorPair, Worker,  # noqa: E402
                                         build_plan, ingest, safe_env)
from cryptomoni.research.http import Blocked, Http, SourceError  # noqa: E402


class Tmp(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="cm-forever-")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def path(self, name):
        return os.path.join(self.dir, name)

    def qdb(self, name="q.db"):
        q = QuarantineDB(self.path(name))
        self.addCleanup(q.close)
        return q

    def closing(self, *sups):
        for s in sups:
            self.addCleanup(lambda s=s: s.qdb is not None and s.qdb.close())

    def plan(self, name="plan.json", **kw):
        p = dict(queries=["alpha", "beta"], backends=["fake"], searxng_url=None, per_hour=360000,
                 page_fetch=True, max_page_bytes=1_500_000)
        p.update(kw)
        with open(self.path(name), "w", encoding="utf-8") as f:
            json.dump(p, f)
        return self.path(name)


# ---------------------------------------------------------------------------- sandbox environment

class SafeEnvTest(unittest.TestCase):
    def test_strips_secrets_keeps_essentials(self):
        base = {"PATH": "/usr/bin", "SYSTEMROOT": r"C:\Windows", "COMSPEC": "cmd.exe", "https_proxy": "http://proxy:8080",
                "SSL_CERT_FILE": "/etc/ca.pem", "LANG": "en_US.UTF-8",
                "COINBASE_API_KEY": "k", "ROBINHOOD_PASSWORD": "p", "GH_TOKEN": "t", "MY_SECRET": "s",
                "CM_CONFIG": "c", "RANDOM_VAR": "x", "PYTHONPATH": "/elsewhere", "HOME": "/home/u"}
        env = safe_env(base)
        for k in ("PATH", "SYSTEMROOT", "COMSPEC", "https_proxy", "SSL_CERT_FILE", "LANG", "HOME"):
            self.assertEqual(env[k], base[k])
        for k in ("COINBASE_API_KEY", "ROBINHOOD_PASSWORD", "GH_TOKEN", "MY_SECRET", "CM_CONFIG", "RANDOM_VAR", "PYTHONPATH"):
            self.assertNotIn(k, env)

    def test_extra_pythonpath_but_never_secrets(self):
        env = safe_env({"PATH": "/bin"}, extra={"PYTHONPATH": ROOT, "API_KEY": "x", "CM_FOREVER_FAKE": "1"})
        self.assertEqual(env["PYTHONPATH"], ROOT)
        self.assertNotIn("API_KEY", env)
        self.assertNotIn("CM_FOREVER_FAKE", env)

    def test_supervisor_env(self):
        with mock.patch.dict(os.environ, {"BINANCE_SECRET": "s", "CM_DASH_PASS": "p", "PATH": os.environ.get("PATH", "")}):
            env = ForeverSupervisor("q.db", "plan.json").env()
        self.assertEqual(env["PYTHONPATH"], ROOT)
        self.assertFalse([k for k in env if any(w in k.upper() for w in ("KEY", "SECRET", "TOKEN", "PASS", "CM_"))])


# ---------------------------------------------------------------------------- quarantine database + ingest

class QuarantineTest(Tmp):
    def test_round_trip(self):
        q = self.qdb()
        n = q.add([dict(url="https://a.example/1", title="One", domain="a.example", query="q", backend="bing_news",
                        published_ts=1.8e9, text="x" * 5000),
                   dict(url="https://a.example/1", title="dup"), dict(url="", title="no url"), "junk",
                   dict(url="https://b.example/2", title="Two")])
        self.assertEqual(n, 2)
        rows = q.take(10)
        self.assertEqual([r["title"] for r in rows], ["One", "Two"])
        self.assertEqual(len(rows[0]["text"]), 4000)
        q.mark([rows[0]["id"]])
        self.assertEqual([r["title"] for r in q.take(10)], ["Two"])
        q.beat("w", 123, {"stored": 2}, ts=1000.0)
        q.beat("w", 124, {"stored": 3}, ts=1010.0)
        h = q.hearts()["w"]
        self.assertEqual((h["pid"], h["heartbeat"], h["stats"]["stored"]), (124, 1010.0, 3))
        q.close()
        q2 = self.qdb()   # persisted
        self.assertEqual(q2.counts(), dict(n=2, pending=1))

    def test_ingest_marks_quarantine(self):
        q = self.qdb()
        rdb = ResearchDB(self.path("research.db"))
        q.add([dict(url="https://a.example/1", title="Solana upgrade ships", domain="a.example", backend="bing_news",
                    published_ts=1.8e9, text="body"),
               dict(url="https://news.google.com/rss/articles/x", title="SEC charges exchange", domain="reuters.com",
                    backend="google_news")])
        self.assertEqual(ingest(q, rdb, 1.8e9 + 60), 2)
        self.assertEqual(q.take(10), [])
        self.assertEqual(ingest(q, rdb, 1.8e9 + 120), 0)
        rows = rdb.q("SELECT source, team, kind, publisher, data FROM items ORDER BY id")
        self.assertEqual([r["source"] for r in rows], ["forever_bing_news", "forever_google_news"])
        self.assertEqual({r["team"] for r in rows}, {"discovery"})
        self.assertEqual({r["kind"] for r in rows}, {"search"})
        self.assertEqual(rows[1]["publisher"], "reuters.com")
        self.assertTrue(all(json.loads(r["data"])["quarantine"] for r in rows))
        items = [it for _, _, _, it in rdb.unlabelled_items(10)]
        self.assertEqual(len(items), 2)
        self.assertTrue(all(it.quarantine and it.symbols == [] for it in items))
        rdb.close()


# ---------------------------------------------------------------------------- plan

class PlanTest(unittest.TestCase):
    def test_build_plan(self):
        subs = {"SOL-USD": "crypto", "NVDA": "stock", "BTC-USD": "crypto"}
        p = build_plan(subs, lambda syms: {"SOL-USD": ["SOL", "Solana"], "NVDA": ["Nvidia"]}, {})
        self.assertEqual(p["backends"], ["google_news", "bing_news", "searxng"])
        self.assertIn("token unlock", p["queries"])
        self.assertIn("guidance cut", p["queries"])
        self.assertIn('"Solana" crypto', p["queries"])
        self.assertIn('"Nvidia" stock', p["queries"])
        self.assertIn('"BTC" crypto', p["queries"])       # no names: the ticker
        self.assertEqual(p["per_hour"], 240)
        self.assertTrue(p["page_fetch"])
        self.assertEqual(len(p["queries"]), len(set(q.lower() for q in p["queries"])))

    def test_overrides_and_bad_names_fn(self):
        def boom(_):
            raise RuntimeError("x")
        p = build_plan({"SOL-USD": "crypto"}, boom, dict(backends=["bing_news", "nope"], searxng_url="http://127.0.0.1:8888",
                                                          per_hour=60))
        self.assertEqual(p["backends"], ["bing_news"])
        self.assertEqual((p["searxng_url"], p["per_hour"]), ("http://127.0.0.1:8888", 60))
        self.assertIn('"SOL" crypto', p["queries"])
        self.assertEqual(build_plan(None, None, None)["backends"], ["google_news", "bing_news", "searxng"])


# ---------------------------------------------------------------------------- parsers + guard

GNEWS = """<?xml version="1.0" encoding="UTF-8"?><rss version="2.0" xmlns:media="http://search.yahoo.com/mrss/"><channel>
<title>"Solana" crypto - Google News</title><link>https://news.google.com/search?q=solana</link>
<item><title>Solana upgrade ships after delay - Decrypt</title>
<link>https://news.google.com/rss/articles/CBMiYWh0dHBzOi8vZGVjcnlwdC5jbw?oc=5</link>
<guid isPermaLink="false">CBMiYWh0dHBzOi8vZGVjcnlwdC5jbw</guid><pubDate>Sat, 17 Oct 2026 14:05:00 GMT</pubDate>
<description>&lt;a href="https://news.google.com/rss/articles/CBMi?oc=5"&gt;Solana upgrade ships after delay&lt;/a&gt;&amp;nbsp;&amp;nbsp;&lt;font color="#6f6f6f"&gt;Decrypt&lt;/font&gt;</description>
<source url="https://decrypt.co">Decrypt</source></item>
</channel></rss>"""

BING = """<?xml version="1.0" encoding="utf-8" ?><rss version="2.0" xmlns:News="https://www.bing.com:443/news/search?q=nvidia&amp;format=rss"><channel>
<title>nvidia - BingNews</title><link>https://www.bing.com:443/news/search?q=nvidia&amp;format=rss</link>
<item><title>Nvidia guidance beats estimates</title>
<link>http://www.bing.com/news/apiclick.aspx?ref=FexRss&amp;aid=&amp;tid=6713&amp;url=https%3a%2f%2fwww.reuters.com%2ftechnology%2fnvidia-guidance-2026-10-17%2f&amp;c=1234&amp;mkt=en-us</link>
<description>Nvidia forecast quarterly revenue above estimates on Thursday.</description>
<pubDate>Fri, 16 Oct 2026 21:10:00 GMT</pubDate><News:Source>Reuters</News:Source></item>
</channel></rss>"""


class ParseGuardTest(unittest.TestCase):
    def test_google_news(self):
        r = forever.parse_google_news(GNEWS)
        self.assertEqual(len(r), 1)
        self.assertEqual(r[0]["title"], "Solana upgrade ships after delay")
        self.assertEqual(r[0]["publisher"], "decrypt.co")
        self.assertTrue(r[0]["link"].startswith("https://news.google.com/"))
        self.assertGreater(r[0]["published"], 1.79e9)

    def test_bing_news_unwrapped(self):
        r = forever.parse_bing_news(BING)
        self.assertEqual(r[0]["link"], "https://www.reuters.com/technology/nvidia-guidance-2026-10-17/")
        self.assertEqual(r[0]["publisher"], "reuters.com")
        self.assertEqual(forever.unwrap_bing("https://x.example/a?url=https://y"), "https://x.example/a?url=https://y")

    def test_searxng(self):
        r = forever.parse_searxng({"results": [{"url": "https://a.example/n", "title": "T", "content": "c",
                                                "publishedDate": "2026-10-17T10:00:00"}, {"url": "ftp://x", "title": "x"}, 5]})
        self.assertEqual([x["link"] for x in r], ["https://a.example/n"])
        with self.assertRaises(SourceError):
            forever.parse_searxng([])

    def test_private_guard(self):
        h = Http(guard_private=True, respect_robots=True)
        for url in ("http://127.0.0.1/", "http://10.0.0.5/admin", "http://localhost:8080/", "http://192.168.1.1/"):
            with self.assertRaises(Blocked):
                h.get(url)
        with self.assertRaises(Blocked):
            h.get("file:///etc/passwd")


# ---------------------------------------------------------------------------- the worker, in process

class WorkerStepTest(Tmp):
    def test_one_step_survives_bad_results(self):
        q = self.qdb()
        plan = self.plan(backends=["google_news"], queries=["guidance cut"])
        fetched = []
        results = [dict(title="A - Reuters", link="https://news.google.com/rss/articles/abc", snippet="snip a",
                        publisher="reuters.com", published=1.8e9),
                   None, {"title": "no link"},
                   dict(title="B", link="https://blocked.example/b", snippet="snip b"),
                   dict(title="C", link="https://ok.example/c", snippet="snip c"),
                   dict(title="C again", link="https://ok.example/c"),
                   dict(title="D", link="https://boom.example/d")]

        def fetch(url):
            fetched.append(url)
            if "blocked" in url:
                raise Blocked("robots.txt disallows /b")
            if "boom" in url:
                raise RuntimeError("parser exploded")
            return "C", "A long enough article body. " * 10

        w = Worker(q, "t", plan, search=lambda b, qq: list(results), fetch_page=fetch, log=lambda m: None)
        w.reload_plan(force=True)
        self.assertTrue(w.step())
        rows = {r["title"]: r for r in q.take(20)}
        self.assertEqual(set(rows), {"A - Reuters", "B", "C"})
        self.assertNotIn("https://news.google.com/rss/articles/abc", fetched)   # redirect stubs are never fetched
        self.assertEqual(rows["A - Reuters"]["text"], "snip a")
        self.assertEqual(rows["A - Reuters"]["domain"], "reuters.com")
        self.assertEqual(rows["B"]["text"], "snip b")
        self.assertTrue(rows["C"]["text"].startswith("A long enough article body."))
        self.assertEqual((w.stats["dup"], w.stats["blocked"], w.stats["queries_done"]), (1, 1, 1))
        self.assertGreaterEqual(w.stats["errors"], 1)
        # the next run of the same query stores nothing new
        self.assertTrue(w.step())
        self.assertEqual(q.counts()["n"], 3)

    def test_fake_backend_needs_env(self):
        q = self.qdb()
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CM_FOREVER_FAKE", None)
            w = Worker(q, "t", self.plan(), log=lambda m: None)
        w.reload_plan(force=True)
        self.assertFalse(w.step())
        self.assertEqual(q.counts()["n"], 0)

    def test_budget(self):
        t = [0.0]
        b = forever.Budget(360, clock=lambda: t[0])     # 0.1 per second
        spent = 0
        while b.wait_s() == 0:
            spent += 1
        self.assertLessEqual(spent, 12)
        t[0] += 3600
        n = 0
        while b.wait_s() == 0:
            n += 1
        self.assertLessEqual(n, 12)                     # the burst is capped, not an hour's worth at once
        self.assertGreater(b.wait_s(), 0)


# ---------------------------------------------------------------------------- the worker as a process

def _worker_env():
    env = safe_env(extra={"PYTHONPATH": ROOT})
    env["CM_FOREVER_FAKE"] = "1"
    return env


class WorkerProcessTest(Tmp):
    def _start(self, *extra):
        self.log = open(self.path("worker.log"), "wb")
        self.addCleanup(self.log.close)
        argv = [sys.executable, "-m", "cryptomoni.research.forever", "--db", self.path("q.db"), "--name", "test",
                "--plan", self.plan(), *extra]
        p = subprocess.Popen(argv, cwd=self.dir, env=_worker_env(), stdin=subprocess.PIPE, stdout=self.log,
                             stderr=subprocess.STDOUT)
        self.addCleanup(p.stdin.close)
        self.addCleanup(lambda: p.poll() is None and (p.kill(), p.wait()))
        return p

    def _logtext(self):
        with open(self.path("worker.log"), "rb") as f:
            return f.read().decode("utf-8", "replace")

    def test_runs_two_seconds_and_stores(self):
        t0 = time.monotonic()
        p = self._start("--max-seconds", "2")
        rc = p.wait(timeout=30)          # stdin stays open: only --max-seconds ends it
        took = time.monotonic() - t0
        self.assertEqual(rc, 0, self._logtext())
        self.assertGreaterEqual(took, 1.8)
        q = self.qdb()
        rows = q.take(100)
        self.assertGreaterEqual(len(rows), 3, self._logtext())
        self.assertTrue(all(r["url"].startswith("https://fake.example/") for r in rows))   # bing links unwrapped
        self.assertTrue(all(r["backend"] == "fake" and r["text"] for r in rows))
        self.assertTrue(any("canned article body" in r["text"] for r in rows))
        h = q.hearts()["test"]
        self.assertEqual(h["pid"], p.pid)
        for k in ("fetched", "stored", "errors", "blocked", "dup", "queries_done"):
            self.assertIn(k, h["stats"])
        self.assertGreater(h["stats"]["queries_done"], 0)
        self.assertNotIn("Traceback", self._logtext())

    def test_exits_on_stdin_eof(self):
        p = self._start()
        time.sleep(1.0)
        self.assertIsNone(p.poll(), self._logtext())
        p.stdin.close()
        self.assertEqual(p.wait(timeout=15), 0, self._logtext())

    def test_imports_only_the_sandbox_modules(self):
        code = ("import sys, cryptomoni.research.forever as f; "
                "print(sorted(m for m in sys.modules if m.startswith('cryptomoni') or m in ('run', 'numpy')))")
        out = subprocess.run([sys.executable, "-c", code], cwd=self.dir, env=safe_env(extra={"PYTHONPATH": ROOT}),
                             capture_output=True, text=True, timeout=30)
        mods = set(eval(out.stdout.strip()))  # noqa: S307  (our own repr)
        self.assertEqual(mods, {"cryptomoni", "cryptomoni.research", "cryptomoni.research.forever",
                                "cryptomoni.research.http", "cryptomoni.research.text"})


# ---------------------------------------------------------------------------- supervisors

class FakeProc:
    n = 0

    def __init__(self):
        FakeProc.n += 1
        self.pid = 50_000 + FakeProc.n
        self.code = None
        self.stdin = self

    def close(self):           # stdin.close(): the worker exits on EOF
        self.code = 0 if self.code is None else self.code

    def poll(self):
        return self.code

    def wait(self, timeout=None):
        if self.code is None:
            raise subprocess.TimeoutExpired("fake", timeout)
        return self.code

    def terminate(self):
        self.code = -15

    kill = terminate


class Spawner:
    def __init__(self):
        self.calls, self.procs = [], []

    def __call__(self, argv, cwd=None, env=None, log_path=None):
        p = FakeProc()
        self.calls.append(dict(argv=argv, cwd=cwd, env=env, log_path=log_path))
        self.procs.append(p)
        return p


class KV:
    def __init__(self):
        self.d = {}

    def kv_get(self, k, default=None):
        return json.loads(self.d[k]) if k in self.d else default

    def kv_set(self, k, v):
        self.d[k] = json.dumps(v)


class SupervisorTest(Tmp):
    def test_backoff_and_standby_switch(self):
        sp = Spawner()
        s1, s2 = self.plan("s1.json"), self.plan("s2.json")
        sup = ForeverSupervisor(self.path("q.db"), self.plan(), cwd=self.dir, log_path=self.path("f.log"), name="w",
                                standby_plans=(s1, s2), clock=lambda: 0.0, spawn=sp)
        self.closing(sup)
        self.assertEqual(sup.ensure(0), "started")
        argv = sp.calls[0]["argv"]
        self.assertEqual(argv[1:3], ["-m", "cryptomoni.research.forever"])
        self.assertEqual(argv[argv.index("--plan") + 1], os.path.abspath(self.path("plan.json")))
        self.assertEqual(sp.calls[0]["cwd"], self.dir)
        self.assertEqual(sup.ensure(1), "running")
        sp.procs[-1].code = 1
        self.assertEqual(sup.ensure(2), "waiting")         # crash 1: back off 10 s
        self.assertEqual(sup.ensure(11), "waiting")
        self.assertEqual(sup.ensure(12), "started")
        sp.procs[-1].code = 1
        self.assertEqual(sup.ensure(13), "waiting")        # crash 2: 30 s
        self.assertEqual(sup.ensure(42), "waiting")
        self.assertEqual(sup.ensure(43), "started")
        sp.procs[-1].code = 1
        self.assertEqual(sup.ensure(44), "waiting")        # crash 3 within 10 min: standby plan 1, 2 min back-off
        self.assertEqual(sup.plan_index, 1)
        self.assertEqual(sup.ensure(163), "waiting")
        self.assertEqual(sup.ensure(164), "started")
        self.assertEqual(sp.calls[-1]["argv"][-1], os.path.abspath(s1))
        st = sup.status(164)
        self.assertEqual(st["plan"], "s1.json")
        self.assertTrue(any("standby" in e["title"] for e in st["events"]))
        self.assertEqual(sup.healthy(164), False)          # alive but no heartbeat yet
        self.qdb().beat("w", sp.procs[-1].pid, {}, ts=170)
        self.assertTrue(sup.healthy(200))
        self.assertFalse(sup.healthy(231))
        sup.stop()
        self.assertEqual(sp.procs[-1].code, 0)              # stdin closed -> exited

    def test_hung_worker_restarted(self):
        sp = Spawner()
        sup = ForeverSupervisor(self.path("q.db"), self.plan(), cwd=self.dir, name="w", spawn=sp, clock=lambda: 0.0)
        self.closing(sup)
        sup.ensure(0)
        self.assertEqual(sup.ensure(100), "running")
        self.assertEqual(sup.ensure(200), "waiting")       # no heartbeat for 200 s: killed, restart after back-off
        self.assertIsNotNone(sp.procs[0].code)
        self.assertEqual(sup.ensure(210), "started")

    def test_pair_takeover_and_return(self):
        sp, kv, t = Spawner(), KV(), [0.0]
        q = self.qdb()
        pair = SupervisorPair(self.path("q.db"), self.plan(), cwd=self.dir, log_path=self.path("forever.log"),
                              standby_plans=(self.plan("s1.json"),), kv=kv, spawn=sp, clock=lambda: t[0])
        self.closing(pair.a, pair.b)
        self.assertEqual(pair.tick_a(0), "started")
        a_proc = sp.procs[0]
        self.assertIn("forever-A", sp.calls[0]["argv"])
        self.assertEqual(sp.calls[0]["env"]["PYTHONPATH"], ROOT)
        self.assertEqual(kv.kv_get("forever_supervisor:A")["ts"], 0)
        q.beat("forever-A", a_proc.pid, {"stored": 1}, ts=5)
        self.assertEqual(pair.tick_b(10), "watching")
        self.assertEqual(pair.tick_a(10), "running")
        self.assertEqual(pair.tick_b(60), "watching")      # A beat 50 s ago: fine
        # A's thread stops ticking; its worker keeps beating
        q.beat("forever-A", a_proc.pid, {"stored": 2}, ts=95)
        self.assertEqual(pair.tick_b(101), "adopted")      # A stale 91 s: B takes over, A's worker is healthy
        self.assertEqual((pair.holder, pair.takeovers), ("B", 1))
        self.assertEqual(len(sp.calls), 1)
        self.assertEqual(pair.tick_b(111), "adopted")
        self.assertEqual(pair.takeovers, 1)
        # A's worker goes quiet too: B stops it and starts its own
        self.assertEqual(pair.tick_b(200), "started")
        self.assertIsNotNone(a_proc.code)
        self.assertIn("forever-B", sp.calls[1]["argv"])
        self.assertTrue(sp.calls[1]["log_path"].endswith("forever-b.log"))
        st = pair.status(200)
        self.assertEqual((st["holder"], st["takeovers"], st["state_b"]), ("B", 1, "started"))
        self.assertTrue(any("took over" in e["title"] for e in st["events"]))
        # A comes back: B's worker is stopped, A runs its own again
        b_proc = sp.procs[1]
        self.assertEqual(pair.tick_a(210), "started")
        self.assertIsNotNone(b_proc.code)
        self.assertIn("forever-A", sp.calls[2]["argv"])
        self.assertEqual(pair.tick_b(215), "watching")
        self.assertEqual(pair.status(215)["holder"], "A")
        pair.stop()
        self.assertIsNotNone(sp.procs[2].code)

    def test_pair_b_takes_over_when_a_never_beats(self):
        sp = Spawner()
        pair = SupervisorPair(self.path("q.db"), self.plan(), cwd=self.dir, log_path=None, kv=None, spawn=sp,
                              clock=lambda: 0.0)
        self.closing(pair.a, pair.b)
        self.assertEqual(pair.tick_b(30), "watching")
        self.assertEqual(pair.tick_b(91), "started")
        self.assertEqual(pair.holder, "B")


# ---------------------------------------------------------------------------- thesis

VIEW = dict(symbol="SOL-USD", n_notes=9, n_stories=6, n_independent=5, kinds=["news", "official"], score=0.42,
            boost_ok=True, penalty=0.0, halted=False, thin=False, flags=[],
            top=[dict(title="Coinbase status normal", publisher="coinbase.com", quarantine=False),
                 dict(title="Solana upgrade will ship", publisher="hype.example", quarantine=False),
                 dict(title="Unverified rumour", publisher="x.example", quarantine=True),
                 dict(title="Solana upgrade ships", publisher="decrypt.co", quarantine=False)])
ADV = [("vera", "Vera Holt"), ("kai", "Kai Arden"), ("imani", "Imani Okoro")]
BANNED = (" will ", "guarantee", "Will ")


class ThesisTest(unittest.TestCase):
    def check(self, s):
        self.assertLessEqual(len(s), 600)
        for w in BANNED:
            self.assertNotIn(w, s)

    def test_boosted(self):
        s = thesis.write("SOL-USD", "buy", "watch", {"kai": "y", "imani": "y", "vera": "a"}, VIEW, "NEUTRAL", ADV, 20)
        self.check(s)
        self.assertTrue(s.startswith("Buy SOL ($20): the desk rates it WATCH in a NEUTRAL market."), s)
        self.assertIn("Kai and Imani like it (momentum, clean risk)", s)
        self.assertIn("5 independent sources over 2 kinds, leaning positive", s)
        self.assertIn("'Solana upgrade ships' – decrypt.co", s)
        self.assertNotIn("Unverified rumour", s)          # quarantined items are not quoted
        self.assertIn("Risk: none flagged.", s)

    def test_thin(self):
        for v in (None, dict(VIEW, n_independent=1, thin=True, top=[])):
            s = thesis.write("NVDA", "buy", "buy", {"kai": "y"}, v, "bull", ADV)
            self.check(s)
            self.assertIn("thin", s)
            self.assertTrue(s.startswith("Buy NVDA: "))

    def test_flagged(self):
        v = dict(VIEW, penalty=0.21, flags=[dict(flag="exploit", severity=0.8, credible=True, sources=2)])
        s = thesis.write("SOL-USD", "buy", "hold", {"kai": "y"}, v, "neutral", ADV, 25)
        self.check(s)
        self.assertIn("exploit flagged", s)
        self.assertIn("capped penalty of 0.21", s)

    def test_sell(self):
        s = thesis.write("ETH-USD", "sell", "sell", {"vera": "n", "imani": "n", "kai": "y"}, dict(VIEW, score=-0.4),
                         "risk_off", ADV, 40)
        self.check(s)
        self.assertTrue(s.startswith("Sell ETH ($40): the desk rates it SELL in a RISK OFF market."), s)
        self.assertIn("Vera and Imani would sell", s)
        self.assertIn("leaning negative", s)

    def test_length_cap(self):
        long = dict(VIEW, top=[dict(title="x" * 300, publisher="p" * 100)] * 6,
                    flags=[dict(flag="flag_" + "y" * 80, credible=False)] * 5, penalty=0.3, halted=True)
        s = thesis.write("A" * 50 + "-USD", "buy", "w" * 80, {k: "y" for k, _ in ADV}, long, "r" * 60, ADV, 1e9)
        self.check(s)

    def test_default_advisors(self):
        s = thesis.write("SOL-USD", "buy", "buy", {"kai": "y"}, VIEW, "bull")
        self.assertIn("Kai likes it", s)


class LlmThesisTest(unittest.TestCase):
    def test_none_without_config(self):
        with mock.patch.object(thesis, "_post", side_effect=AssertionError("no call")):
            self.assertIsNone(thesis.llm_thesis({}, {"symbol": "SOL-USD"}))
            self.assertIsNone(thesis.llm_thesis({"research": {}}, {"symbol": "SOL-USD"}))
            self.assertIsNone(thesis.llm_thesis(None, {}))

    def test_refuses_non_local(self):
        with mock.patch.object(thesis, "_post", side_effect=AssertionError("no call")):
            for url in ("http://example.com/api/generate", "https://127.0.0.1/api/generate",
                        "http://10.0.0.5:11434/api/generate", "http://127.0.0.1.evil.com/api/generate"):
                cfg = {"research": {"thesis_llm": {"url": url, "model": "llama3"}}}
                self.assertIsNone(thesis.llm_thesis(cfg, {"symbol": "SOL-USD"}), url)

    def test_local_draft(self):
        seen = {}

        def post(url, body, timeout):
            seen.update(url=url, body=body, timeout=timeout)
            return {"response": "  SOL is proposed as a small buy.\nFive sources lean positive. " + "z" * 900}

        cfg = {"research": {"thesis_llm": {"url": "http://localhost:11434/api/generate", "model": "llama3"}}}
        with mock.patch.object(thesis, "_post", side_effect=post):
            s = thesis.llm_thesis(cfg, {"symbol": "SOL-USD", "score": 0.42, "titles": ["Solana upgrade ships"]})
        self.assertTrue(s.startswith("(AI draft, display only) SOL is proposed"))
        self.assertLessEqual(len(s), 600)
        self.assertEqual((seen["body"]["model"], seen["body"]["stream"], seen["timeout"]), ("llama3", False, 20.0))
        self.assertIn("Solana upgrade ships", seen["body"]["prompt"])
        self.assertIn("3", seen["body"]["prompt"])
        with mock.patch.object(thesis, "_post", side_effect=OSError("down")):
            self.assertIsNone(thesis.llm_thesis(cfg, {}))


if __name__ == "__main__":
    unittest.main()
