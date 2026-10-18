"""SQLite (WAL) store shared by the scanner engine, the watchdog and the dashboard."""
from __future__ import annotations

import json
import sqlite3
import time

from .models import Bar

SCHEMA = """
CREATE TABLE IF NOT EXISTS bars(symbol TEXT, gran INTEGER, ts INTEGER, o REAL, h REAL, l REAL, c REAL, v REAL,
  PRIMARY KEY(symbol, gran, ts));
CREATE TABLE IF NOT EXISTS snapshots(id INTEGER PRIMARY KEY, ts REAL, equity REAL, bench REAL,
  e_target REAL, e_actual REAL, regime TEXT, conf REAL, lite TEXT, data TEXT);
CREATE TABLE IF NOT EXISTS trades(id INTEGER PRIMARY KEY, ts REAL, symbol TEXT, side TEXT, notional REAL,
  qty REAL, price REAL, fee REAL, status TEXT, detail TEXT, reason TEXT);
CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY, ts REAL, kind TEXT, severity TEXT,
  title TEXT, body TEXT, payload TEXT);
CREATE TABLE IF NOT EXISTS alerts(id INTEGER PRIMARY KEY, ts REAL, symbol TEXT, kind TEXT, severity TEXT,
  title TEXT, body TEXT);
CREATE TABLE IF NOT EXISTS kv(key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS outbox(id INTEGER PRIMARY KEY, created_ts REAL, expires_ts REAL, symbol TEXT, side TEXT,
  notional REAL, ref_price REAL, reason TEXT, regime TEXT, verdict TEXT, broker TEXT, status TEXT,
  decided_ts REAL, decided_by TEXT, executed_ts REAL, client_order_id TEXT, broker_order_id TEXT, detail TEXT);
CREATE INDEX IF NOT EXISTS ev_kind ON events(kind, id);
"""


class Store:
    def __init__(self, path, readonly=False):
        self.path = path
        self.c = sqlite3.connect(path, timeout=10, check_same_thread=False)
        self.c.row_factory = sqlite3.Row
        if not readonly:
            self.c.execute("PRAGMA journal_mode=WAL")
            self.c.executescript(SCHEMA)
            cols = {r[1] for r in self.c.execute("PRAGMA table_info(outbox)")}
            for col in ("thesis", "research"):   # added with the research service; older databases get them here
                if col not in cols:
                    self.c.execute(f"ALTER TABLE outbox ADD COLUMN {col} TEXT")
            self.c.commit()

    def close(self):
        self.c.close()

    # ---- key/value ----------------------------------------------------------
    def kv_get(self, k, default=None):
        r = self.c.execute("SELECT value FROM kv WHERE key=?", (k,)).fetchone()
        return json.loads(r[0]) if r else default

    def kv_set(self, k, v, commit=True):
        self.c.execute("INSERT INTO kv(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                       (k, json.dumps(v)))
        if commit:
            self.c.commit()

    def heartbeat(self, ts=None):
        self.kv_set("heartbeat", time.time() if ts is None else ts)

    # ---- bars ---------------------------------------------------------------
    def put_bars(self, symbol, gran, bars):
        self.c.executemany("INSERT OR REPLACE INTO bars(symbol,gran,ts,o,h,l,c,v) VALUES(?,?,?,?,?,?,?,?)",
                           [(symbol, gran, b.ts, b.open, b.high, b.low, b.close, b.volume) for b in bars])
        self.c.commit()

    def get_bars(self, symbol, gran, limit=400):
        rows = self.c.execute("SELECT ts,o,h,l,c,v FROM bars WHERE symbol=? AND gran=? ORDER BY ts DESC LIMIT ?",
                              (symbol, gran, limit)).fetchall()
        return [Bar(r[0], r[1], r[2], r[3], r[4], r[5]) for r in reversed(rows)]

    def last_bar_ts(self, symbol, gran):
        r = self.c.execute("SELECT MAX(ts) FROM bars WHERE symbol=? AND gran=?", (symbol, gran)).fetchone()
        return r[0]

    # ---- history the dashboard draws ----------------------------------------
    def add_snapshot(self, s):
        lite = dict(probs=s["probs"], stock_w=[a.get("stock_desk_weight") for a in s["advisors"]] if s.get("stocks") else None,
                    desk_w=[a["desk_weight"] for a in s["advisors"]],
                    board_w=[a["board_weight"] for a in s["advisors"]], ic=[a["ic"] for a in s["advisors"]],
                    votes=[a["vote"] for a in s["advisors"]])
        self.c.execute("INSERT INTO snapshots(ts,equity,bench,e_target,e_actual,regime,conf,lite,data) VALUES(?,?,?,?,?,?,?,?,?)",
                       (s["ts"], s["equity"], s.get("bench"), s["e_target"], s["e_actual"],
                        s["regime"], s["confidence"], json.dumps(lite), json.dumps(s)))
        self.c.commit()

    def latest(self):
        r = self.c.execute("SELECT data FROM snapshots ORDER BY id DESC LIMIT 1").fetchone()
        return json.loads(r[0]) if r else None

    def history(self, limit=3000):
        rows = self.c.execute("SELECT id,ts,equity,bench,e_target,e_actual,regime,conf,lite FROM snapshots ORDER BY id DESC LIMIT ?",
                              (limit,)).fetchall()
        out = []
        for r in reversed(rows):
            d = dict(r)
            d.update(json.loads(d.pop("lite")))
            out.append(d)
        return out

    def add_trade(self, f, reason=""):
        self.c.execute("INSERT INTO trades(ts,symbol,side,notional,qty,price,fee,status,detail,reason) VALUES(?,?,?,?,?,?,?,?,?,?)",
                       (f.ts, f.symbol, f.side, f.notional_usd, f.qty, f.price, f.fee_usd, f.status, f.detail, reason))
        self.c.commit()

    def trades(self, limit=100):
        return [dict(r) for r in self.c.execute("SELECT * FROM trades ORDER BY id DESC LIMIT ?", (limit,))]

    def add_event(self, ts, kind, severity, title, body="", payload=None):
        self.c.execute("INSERT INTO events(ts,kind,severity,title,body,payload) VALUES(?,?,?,?,?,?)",
                       (ts, kind, severity, title, body, json.dumps(payload) if payload is not None else None))
        self.c.commit()

    def events(self, limit=100, kind=None):
        q = "SELECT * FROM events " + ("WHERE kind=? " if kind else "") + "ORDER BY id DESC LIMIT ?"
        out = []
        for r in self.c.execute(q, (kind, limit) if kind else (limit,)):
            d = dict(r)
            d["payload"] = json.loads(d["payload"]) if d["payload"] else None
            out.append(d)
        return out

    def add_alert(self, ts, symbol, kind, severity, title, body=""):
        self.c.execute("INSERT INTO alerts(ts,symbol,kind,severity,title,body) VALUES(?,?,?,?,?,?)",
                       (ts, symbol, kind, severity, title, body))
        self.c.commit()

    def alerts(self, limit=100):
        return [dict(r) for r in self.c.execute("SELECT * FROM alerts ORDER BY id DESC LIMIT ?", (limit,))]

    def counts(self):
        q = "SELECT (SELECT COUNT(*) FROM snapshots),(SELECT COALESCE(MAX(id),0) FROM events),(SELECT COALESCE(MAX(id),0) FROM alerts),(SELECT COALESCE(MAX(id),0) FROM trades)"
        out = list(self.c.execute(q).fetchone())
        out.append(hash(tuple((r[0], r[1]) for r in self.c.execute("SELECT id,status FROM outbox ORDER BY id DESC LIMIT 100"))))
        return out

    # ---- outbox: proposed real orders that wait for a human --------------------------
    def add_outbox(self, ts, expires, o, ref_price, regime, verdict, broker, thesis=None, research=None):
        """`thesis` is a display-only written case; `research` a small dict (flags, sources) shown with the order."""
        cur = self.c.execute("""INSERT INTO outbox(created_ts,expires_ts,symbol,side,notional,ref_price,reason,regime,verdict,broker,status,
                                thesis,research) VALUES(?,?,?,?,?,?,?,?,?,?,'pending',?,?)""",
                             (ts, expires, o.symbol, o.side, o.notional_usd, ref_price, o.reason, regime, verdict, broker,
                              thesis, json.dumps(research) if research is not None else None))
        oid = cur.lastrowid
        self.c.execute("UPDATE outbox SET client_order_id=? WHERE id=?", (f"cm-{int(ts)}-{oid}", oid))
        self.c.commit()
        return oid

    def outbox(self, limit=100, status=None):
        q = "SELECT * FROM outbox " + ("WHERE status=? " if status else "") + "ORDER BY id DESC LIMIT ?"
        return [dict(r) for r in self.c.execute(q, (status, limit) if status else (limit,))]

    def outbox_get(self, oid):
        r = self.c.execute("SELECT * FROM outbox WHERE id=?", (oid,)).fetchone()
        return dict(r) if r else None

    def outbox_set(self, oid, status, ts=None, **kw):
        sets, vals = ["status=?"], [status]
        if status in ("approved", "rejected"):
            sets += ["decided_ts=?"]; vals += [ts]
        if status in ("sent", "failed", "check", "skipped"):
            sets += ["executed_ts=?"]; vals += [ts]
        for k in ("decided_by", "broker_order_id", "detail"):
            if k in kw:
                sets.append(k + "=?"); vals.append(kw[k])
        self.c.execute(f"UPDATE outbox SET {', '.join(sets)} WHERE id=?", (*vals, oid))
        self.c.commit()

    def supersede_pending(self):
        """A new proposal replaces every older one that hasn't been approved yet."""
        self.c.execute("UPDATE outbox SET status='superseded', detail='replaced by a newer proposal' WHERE status='pending'")
        self.c.commit()

    def expire_outbox(self, now):
        n = self.c.execute("UPDATE outbox SET status='expired' WHERE status IN ('pending','approved') AND expires_ts<?", (now,)).rowcount
        self.c.commit()
        return n

    def executed_since(self, ts):
        r = self.c.execute("SELECT COUNT(*) n, COALESCE(SUM(notional),0) t FROM outbox WHERE status='sent' AND executed_ts>=?", (ts,)).fetchone()
        return r["n"], r["t"]

    def last_sent(self):
        out = {}
        for r in self.c.execute("SELECT symbol, side, MAX(executed_ts) t FROM outbox WHERE status='sent' GROUP BY symbol, side"):
            out[(r["symbol"], r["side"])] = r["t"]
        return out

    def prune(self, keep_snapshots=20000, keep_rows=20000, keep_bars=3000):
        """Bounded growth for a process that never stops."""
        for table, keep in (("snapshots", keep_snapshots), ("events", keep_rows), ("alerts", keep_rows), ("trades", keep_rows)):
            self.c.execute(f"DELETE FROM {table} WHERE id <= (SELECT COALESCE(MAX(id),0) FROM {table}) - ?", (keep,))
        # charts only need the light columns; keep full snapshots for the last two days
        self.c.execute("UPDATE snapshots SET data=NULL WHERE data IS NOT NULL AND id <= (SELECT MAX(id) FROM snapshots) - 48")
        self.c.execute("""DELETE FROM bars WHERE gran=3600 AND ts < (SELECT MAX(ts) FROM bars b2 WHERE b2.symbol=bars.symbol
                          AND b2.gran=3600) - ? * 3600""", (keep_bars,))
        self.c.commit()
