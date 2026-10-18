"""The research database (SQLite, WAL). One file shared by the research service, both
boards' engines and the dashboards. Every writer keeps its transactions short; every
reader tolerates a missing or half-filled database.

Who writes what
  service   items, notes, sources, workers, lease, view, prices, outcomes, models,
            stages, discovered, health, events
  engines   subscriptions, requests, shadow, proposals (and read view/stages/discovered)
  dashboard labels (the "fix this note" button), proposals.status when you promote one
"""
from __future__ import annotations

import os
import sqlite3
import threading
import time

from .types import Item, Note, dumps, loads

SCHEMA = """
CREATE TABLE IF NOT EXISTS items(id INTEGER PRIMARY KEY, url_hash TEXT UNIQUE, url TEXT, source TEXT, team TEXT, kind TEXT,
  publisher TEXT, published_ts REAL, fetched_ts REAL, title TEXT, text TEXT, symbols TEXT, data TEXT, labelled INTEGER DEFAULT 0);
CREATE INDEX IF NOT EXISTS items_lab ON items(labelled, id);
CREATE TABLE IF NOT EXISTS notes(id INTEGER PRIMARY KEY, item_id INTEGER, symbol TEXT, asset TEXT, ts REAL, fetched_ts REAL,
  source TEXT, team TEXT, publisher TEXT, evidence TEXT, catalyst TEXT, direction INTEGER, confidence REAL, risk_flags TEXT,
  severity REAL, label_by TEXT, title TEXT, url TEXT, facts TEXT, cluster TEXT, quarantine INTEGER, official INTEGER,
  expires_ts REAL, price0 REAL, bench0 REAL, edge REAL);
CREATE INDEX IF NOT EXISTS notes_sym ON notes(symbol, ts);
CREATE INDEX IF NOT EXISTS notes_ts ON notes(ts);
CREATE INDEX IF NOT EXISTS notes_item ON notes(item_id);
CREATE TABLE IF NOT EXISTS outcomes(note_id INTEGER, horizon TEXT, ret REAL, bench_ret REAL, excess REAL, ts REAL,
  PRIMARY KEY(note_id, horizon));
CREATE TABLE IF NOT EXISTS labels(note_id INTEGER PRIMARY KEY, catalyst TEXT, direction INTEGER, risk_flags TEXT, by TEXT, ts REAL,
  title TEXT, text TEXT);
CREATE TABLE IF NOT EXISTS sources(name TEXT PRIMARY KEY, team TEXT, kind TEXT, last_ok REAL, last_err REAL, err TEXT,
  consecutive_err INTEGER DEFAULT 0, runs INTEGER DEFAULT 0, items INTEGER DEFAULT 0, notes INTEGER DEFAULT 0,
  interval_s REAL, rel_a REAL DEFAULT 2, rel_b REAL DEFAULT 2, usefulness REAL DEFAULT 0.5);
CREATE TABLE IF NOT EXISTS workers(slot TEXT PRIMARY KEY, adapter TEXT, team TEXT, role TEXT, shard INTEGER, nshards INTEGER,
  state TEXT, heartbeat REAL, started REAL, detail TEXT, grown INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS lease(name TEXT PRIMARY KEY, owner TEXT, pid INTEGER, heartbeat REAL, started REAL);
CREATE TABLE IF NOT EXISTS subscriptions(board TEXT PRIMARY KEY, symbols TEXT, ts REAL);
CREATE TABLE IF NOT EXISTS requests(id INTEGER PRIMARY KEY, ts REAL, board TEXT, symbol TEXT, asset TEXT, advisor TEXT,
  question TEXT, priority REAL, status TEXT, done_ts REAL, notes_found INTEGER DEFAULT 0);
CREATE INDEX IF NOT EXISTS req_open ON requests(status, symbol);
CREATE TABLE IF NOT EXISTS view(symbol TEXT PRIMARY KEY, asset TEXT, ts REAL, data TEXT);
CREATE TABLE IF NOT EXISTS prices(symbol TEXT, ts INTEGER, close REAL, PRIMARY KEY(symbol, ts));
CREATE TABLE IF NOT EXISTS models(name TEXT, version INTEGER, ts REAL, role TEXT, metrics TEXT, blob BLOB,
  PRIMARY KEY(name, version));
CREATE TABLE IF NOT EXISTS shadow(id INTEGER PRIMARY KEY, ts REAL, board TEXT, symbol TEXT, asset TEXT, kind TEXT,
  base REAL, adj REAL, detail TEXT, matured INTEGER DEFAULT 0, excess REAL);
CREATE INDEX IF NOT EXISTS shadow_k ON shadow(kind, ts);
CREATE TABLE IF NOT EXISTS stages(name TEXT PRIMARY KEY, state TEXT, since REAL, evidence TEXT, by TEXT);
CREATE TABLE IF NOT EXISTS discovered(symbol TEXT PRIMARY KEY, asset TEXT, first_seen REAL, last_seen REAL, sources TEXT,
  reason TEXT, mentions INTEGER DEFAULT 1, status TEXT, detail TEXT);
CREATE TABLE IF NOT EXISTS proposals(id INTEGER PRIMARY KEY, ts REAL, board TEXT, symbol TEXT, asset TEXT, advisor TEXT,
  z REAL, reason TEXT, status TEXT, price0 REAL, bench0 REAL, excess_7d REAL);
CREATE TABLE IF NOT EXISTS health(id INTEGER PRIMARY KEY, ts REAL, data TEXT);
CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY, ts REAL, kind TEXT, severity TEXT, title TEXT, body TEXT);
CREATE TABLE IF NOT EXISTS kv(key TEXT PRIMARY KEY, value TEXT);
"""

STAGE_ORDER = ("penalties", "signals", "watchlist", "cadence")


def url_hash(url):
    import hashlib
    return hashlib.sha1((url or "").strip().encode("utf-8", "ignore")).hexdigest()


class ResearchDB:
    """Thread-safe handle (one connection, one lock). Open one per process."""

    def __init__(self, path, readonly=False):
        self.path = path
        d = os.path.dirname(path)
        if d and not readonly:
            os.makedirs(d, exist_ok=True)
        self.c = sqlite3.connect(path, timeout=30, check_same_thread=False, isolation_level=None)
        self.c.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        self.readonly = readonly
        self.c.execute("PRAGMA busy_timeout=30000")
        if not readonly:
            self.c.execute("PRAGMA journal_mode=WAL")
            self.c.execute("PRAGMA synchronous=NORMAL")
            self.c.executescript(SCHEMA)

    def close(self):
        with self.lock:
            self.c.close()

    # ---- low level -------------------------------------------------------------
    def x(self, sql, args=()):
        with self.lock:
            return self.c.execute(sql, args)

    def q(self, sql, args=()):
        with self.lock:
            return [dict(r) for r in self.c.execute(sql, args).fetchall()]

    def q1(self, sql, args=()):
        with self.lock:
            r = self.c.execute(sql, args).fetchone()
            return dict(r) if r else None

    def many(self, sql, rows):
        with self.lock:
            self.c.execute("BEGIN IMMEDIATE")
            try:
                self.c.executemany(sql, rows)
                self.c.execute("COMMIT")
            except Exception:
                self.c.execute("ROLLBACK")
                raise

    # ---- kv ----------------------------------------------------------------------
    def kv_get(self, k, default=None):
        r = self.q1("SELECT value FROM kv WHERE key=?", (k,))
        return loads(r["value"], default) if r else default

    def kv_set(self, k, v):
        self.x("INSERT INTO kv(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (k, dumps(v)))

    # ---- events --------------------------------------------------------------------
    def event(self, kind, severity, title, body="", ts=None):
        self.x("INSERT INTO events(ts,kind,severity,title,body) VALUES(?,?,?,?,?)",
               (time.time() if ts is None else ts, kind, severity, title[:300], (body or "")[:2000]))

    def events(self, limit=100):
        return self.q("SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,))

    # ---- items ---------------------------------------------------------------------
    def add_items(self, items, team, now=None):
        """Insert new items (dedupe on URL). Returns the number actually added."""
        now = time.time() if now is None else now
        rows = []
        for it in items:
            if not it.url or not it.title:
                continue
            extra = dict(facts=it.facts, catalyst=it.catalyst, direction=it.direction, risk_flags=it.risk_flags,
                         confidence=it.confidence, official=it.official, quarantine=it.quarantine)
            rows.append((url_hash(it.url), it.url[:1000], it.source, team, it.kind, (it.publisher or "")[:120],
                         it.published_ts, now, it.title[:500], (it.text or "")[:4000], dumps(list(it.symbols or [])), dumps(extra)))
        if not rows:
            return 0
        with self.lock:
            before = self.c.total_changes
            self.c.execute("BEGIN IMMEDIATE")
            try:
                self.c.executemany("""INSERT OR IGNORE INTO items(url_hash,url,source,team,kind,publisher,published_ts,fetched_ts,
                                      title,text,symbols,data) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""", rows)
                self.c.execute("COMMIT")
            except Exception:
                self.c.execute("ROLLBACK")
                raise
            return self.c.total_changes - before

    def unlabelled_items(self, limit=200):
        out = []
        for r in self.q("SELECT * FROM items WHERE labelled=0 ORDER BY id LIMIT ?", (limit,)):
            extra = loads(r["data"], {}) or {}
            it = Item(source=r["source"], url=r["url"], title=r["title"], text=r["text"] or "", published_ts=r["published_ts"],
                      symbols=loads(r["symbols"], []) or [], kind=r["kind"], publisher=r["publisher"] or "",
                      facts=extra.get("facts") or {}, catalyst=extra.get("catalyst"), direction=extra.get("direction"),
                      risk_flags=extra.get("risk_flags") or [], confidence=extra.get("confidence"),
                      official=bool(extra.get("official")), quarantine=bool(extra.get("quarantine")))
            out.append((r["id"], r["team"], r["fetched_ts"], it))
        return out

    def mark_labelled(self, ids, state=1):
        if ids:
            self.many("UPDATE items SET labelled=? WHERE id=?", [(state, i) for i in ids])

    # ---- notes ---------------------------------------------------------------------
    def add_notes(self, notes):
        rows = [(n.item_id, n.symbol, n.asset, n.ts, n.fetched_ts, n.source, n.team, n.publisher, n.evidence, n.catalyst,
                 int(n.direction), float(n.confidence), dumps(n.risk_flags), float(n.severity), n.label_by, n.title[:500],
                 n.url[:1000], dumps(n.facts), n.cluster, int(bool(n.quarantine)), int(bool(n.official)), n.expires_ts)
                for n in notes]
        if rows:
            self.many("""INSERT INTO notes(item_id,symbol,asset,ts,fetched_ts,source,team,publisher,evidence,catalyst,direction,
                         confidence,risk_flags,severity,label_by,title,url,facts,cluster,quarantine,official,expires_ts)
                         VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", rows)
        return len(rows)

    @staticmethod
    def _note(r):
        n = Note(symbol=r["symbol"], asset=r["asset"], ts=r["ts"], fetched_ts=r["fetched_ts"], source=r["source"], team=r["team"],
                 publisher=r["publisher"] or "", evidence=r["evidence"], catalyst=r["catalyst"], direction=int(r["direction"] or 0),
                 confidence=float(r["confidence"] or 0), risk_flags=loads(r["risk_flags"], []) or [], severity=float(r["severity"] or 0),
                 label_by=r["label_by"], title=r["title"] or "", url=r["url"] or "", facts=loads(r["facts"], {}) or {},
                 cluster=r["cluster"] or "", quarantine=bool(r["quarantine"]), official=bool(r["official"]),
                 expires_ts=r["expires_ts"] or 0.0, item_id=r["item_id"], id=r["id"])
        return n

    def fresh_notes(self, now=None, symbols=None, since=None):
        """Notes not yet expired (optionally for some symbols), newest first."""
        now = time.time() if now is None else now
        if symbols is not None:
            symbols = list(symbols)
            if not symbols:
                return []
            marks = ",".join("?" * len(symbols))
            rows = self.q(f"SELECT * FROM notes WHERE expires_ts>? AND symbol IN ({marks}) ORDER BY ts DESC", (now, *symbols))
        else:
            rows = self.q("SELECT * FROM notes WHERE expires_ts>? ORDER BY ts DESC", (now,))
        out = [self._note(r) for r in rows]
        return [n for n in out if since is None or n.ts >= since]

    def notes(self, limit=100, symbol=None):
        if symbol:
            rows = self.q("SELECT * FROM notes WHERE symbol=? ORDER BY id DESC LIMIT ?", (symbol, limit))
        else:
            rows = self.q("SELECT * FROM notes ORDER BY id DESC LIMIT ?", (limit,))
        return [self._note(r) for r in rows]

    def note_rows(self, sql_where="1", args=(), limit=100000):
        """Raw note rows plus outcome columns for training (excess at every horizon)."""
        rows = self.q(f"SELECT * FROM notes WHERE {sql_where} ORDER BY id LIMIT ?", (*args, limit))
        if not rows:
            return rows
        ids = [r["id"] for r in rows]
        out = {}
        for i in range(0, len(ids), 900):
            chunk = ids[i:i + 900]
            for o in self.q(f"SELECT note_id,horizon,excess FROM outcomes WHERE note_id IN ({','.join('?' * len(chunk))})", chunk):
                out.setdefault(o["note_id"], {})[o["horizon"]] = o["excess"]
        for r in rows:
            r["outcomes"] = out.get(r["id"], {})
        return rows

    def set_price0(self, rows):
        """rows: [(price0, bench0, note_id)]"""
        if rows:
            self.many("UPDATE notes SET price0=?, bench0=? WHERE id=?", rows)

    def set_edge(self, rows):
        """rows: [(edge, note_id)]: the outcome model's predicted edge for a note."""
        if rows:
            self.many("UPDATE notes SET edge=? WHERE id=?", rows)

    # ---- outcomes ------------------------------------------------------------------
    def add_outcomes(self, rows):
        """rows: [(note_id, horizon, ret, bench_ret, excess, ts)]"""
        if rows:
            self.many("INSERT OR REPLACE INTO outcomes(note_id,horizon,ret,bench_ret,excess,ts) VALUES(?,?,?,?,?,?)", rows)

    def pending_outcomes(self, horizon, before_ts, limit=5000):
        """Notes old enough for `horizon` that have a price0 and no outcome yet."""
        return self.q("""SELECT n.id,n.symbol,n.asset,n.ts,n.price0,n.bench0 FROM notes n WHERE n.price0 IS NOT NULL AND n.ts<=?
                         AND NOT EXISTS(SELECT 1 FROM outcomes o WHERE o.note_id=n.id AND o.horizon=?) ORDER BY n.id LIMIT ?""",
                      (before_ts, horizon, limit))

    # ---- labels (human corrections and seeds) ----------------------------------------
    def set_label(self, note_id, catalyst, direction, risk_flags, by="human", ts=None):
        n = self.q1("SELECT title, url, item_id FROM notes WHERE id=?", (note_id,))
        if not n:
            return False
        it = self.q1("SELECT text FROM items WHERE id=?", (n["item_id"],)) if n["item_id"] else None
        self.x("""INSERT INTO labels(note_id,catalyst,direction,risk_flags,by,ts,title,text) VALUES(?,?,?,?,?,?,?,?)
                  ON CONFLICT(note_id) DO UPDATE SET catalyst=excluded.catalyst, direction=excluded.direction,
                  risk_flags=excluded.risk_flags, by=excluded.by, ts=excluded.ts""",
               (note_id, catalyst, int(direction), dumps(list(risk_flags or [])), by, time.time() if ts is None else ts,
                n["title"], (it or {}).get("text", "")))
        from .taxonomy import severity
        self.x("UPDATE notes SET catalyst=?, direction=?, risk_flags=?, severity=?, label_by=? WHERE id=?",
               (catalyst, int(direction), dumps(list(risk_flags or [])), severity(risk_flags), by, note_id))
        return True

    def labels(self, limit=100000):
        return self.q("SELECT * FROM labels ORDER BY ts LIMIT ?", (limit,))

    # ---- sources and workers (health) --------------------------------------------------
    def source_ok(self, name, team, kind, n_items, interval_s, now=None):
        now = time.time() if now is None else now
        self.x("""INSERT INTO sources(name,team,kind,last_ok,consecutive_err,runs,items,interval_s) VALUES(?,?,?,?,0,1,?,?)
                  ON CONFLICT(name) DO UPDATE SET last_ok=excluded.last_ok, consecutive_err=0, runs=runs+1,
                  items=items+excluded.items, interval_s=excluded.interval_s, team=excluded.team, kind=excluded.kind""",
               (name, team, kind, now, n_items, interval_s))

    def source_err(self, name, team, kind, err, interval_s, now=None):
        now = time.time() if now is None else now
        self.x("""INSERT INTO sources(name,team,kind,last_err,err,consecutive_err,runs,interval_s) VALUES(?,?,?,?,?,1,1,?)
                  ON CONFLICT(name) DO UPDATE SET last_err=excluded.last_err, err=excluded.err,
                  consecutive_err=consecutive_err+1, runs=runs+1, interval_s=excluded.interval_s""",
               (name, team, kind, now, str(err)[:300], interval_s))

    def sources(self):
        return {r["name"]: r for r in self.q("SELECT * FROM sources")}

    def set_reliability(self, rows):
        """rows: [(rel_a, rel_b, usefulness, name)]"""
        if rows:
            self.many("UPDATE sources SET rel_a=?, rel_b=?, usefulness=? WHERE name=?", rows)

    def bump_source_notes(self, counts):
        if counts:
            self.many("UPDATE sources SET notes=notes+? WHERE name=?", [(v, k) for k, v in counts.items()])

    def worker_set(self, slot, adapter, team, role, shard, nshards, state, detail="", grown=0, now=None):
        now = time.time() if now is None else now
        self.x("""INSERT INTO workers(slot,adapter,team,role,shard,nshards,state,heartbeat,started,detail,grown)
                  VALUES(?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(slot) DO UPDATE SET adapter=excluded.adapter, team=excluded.team,
                  role=excluded.role, shard=excluded.shard, nshards=excluded.nshards, state=excluded.state,
                  heartbeat=excluded.heartbeat, detail=excluded.detail, grown=excluded.grown""",
               (slot, adapter, team, role, shard, nshards, state, now, now, detail[:300], grown))

    def worker_beat(self, slot, state=None, detail=None, now=None):
        now = time.time() if now is None else now
        if state is None and detail is None:
            self.x("UPDATE workers SET heartbeat=? WHERE slot=?", (now, slot))
        else:
            self.x("UPDATE workers SET heartbeat=?, state=COALESCE(?,state), detail=COALESCE(?,detail) WHERE slot=?",
                   (now, state, None if detail is None else detail[:300], slot))

    def workers(self):
        return self.q("SELECT * FROM workers ORDER BY team, role, slot")

    def drop_workers(self, keep_slots):
        keep = list(keep_slots)
        if keep:
            self.x(f"DELETE FROM workers WHERE slot NOT IN ({','.join('?' * len(keep))})", keep)
        else:
            self.x("DELETE FROM workers")

    # ---- the single-service lease --------------------------------------------------------
    def lease_try(self, owner, pid, stale_s=45.0, now=None, name="service"):
        """Take or renew the service lease. True if `owner` holds it afterwards."""
        now = time.time() if now is None else now
        with self.lock:
            self.c.execute("BEGIN IMMEDIATE")
            try:
                r = self.c.execute("SELECT owner, heartbeat FROM lease WHERE name=?", (name,)).fetchone()
                if r is None:
                    self.c.execute("INSERT INTO lease(name,owner,pid,heartbeat,started) VALUES(?,?,?,?,?)", (name, owner, pid, now, now))
                    ok = True
                elif r["owner"] == owner:
                    self.c.execute("UPDATE lease SET heartbeat=?, pid=? WHERE name=?", (now, pid, name))
                    ok = True
                elif now - (r["heartbeat"] or 0) > stale_s:
                    self.c.execute("UPDATE lease SET owner=?, pid=?, heartbeat=?, started=? WHERE name=?", (owner, pid, now, now, name))
                    ok = True
                else:
                    ok = False
                self.c.execute("COMMIT")
                return ok
            except Exception:
                self.c.execute("ROLLBACK")
                raise

    def lease_release(self, owner, name="service"):
        self.x("DELETE FROM lease WHERE name=? AND owner=?", (name, owner))

    def lease(self, name="service"):
        return self.q1("SELECT * FROM lease WHERE name=?", (name,))

    # ---- subscriptions: what each board wants researched ------------------------------------
    def subscribe(self, board, symbols, now=None):
        """symbols: {symbol: asset}"""
        self.x("INSERT INTO subscriptions(board,symbols,ts) VALUES(?,?,?) ON CONFLICT(board) DO UPDATE SET symbols=excluded.symbols, ts=excluded.ts",
               (board, dumps(symbols), time.time() if now is None else now))

    def subscribed(self, max_age_s=6 * 3600, now=None):
        """Union of fresh subscriptions: {symbol: asset}."""
        now = time.time() if now is None else now
        out = {}
        for r in self.q("SELECT * FROM subscriptions WHERE ts>?", (now - max_age_s,)):
            out.update(loads(r["symbols"], {}) or {})
        return out

    # ---- requests: advisors asking for more evidence ------------------------------------------
    def request(self, board, symbol, asset, advisor, question, priority, now=None):
        """Open a request unless one is already open for this symbol (then raise its priority)."""
        now = time.time() if now is None else now
        r = self.q1("SELECT id, priority FROM requests WHERE status='open' AND symbol=?", (symbol,))
        if r:
            if priority > (r["priority"] or 0):
                self.x("UPDATE requests SET priority=? WHERE id=?", (priority, r["id"]))
            return r["id"]
        cur = self.x("INSERT INTO requests(ts,board,symbol,asset,advisor,question,priority,status) VALUES(?,?,?,?,?,?,?,'open')",
                     (now, board, symbol, asset, advisor, question[:300], priority))
        return cur.lastrowid

    def open_requests(self, limit=100):
        return self.q("SELECT * FROM requests WHERE status='open' ORDER BY priority DESC, ts LIMIT ?", (limit,))

    def close_requests(self, symbol, found, now=None):
        self.x("UPDATE requests SET status='done', done_ts=?, notes_found=? WHERE status='open' AND symbol=?",
               (time.time() if now is None else now, found, symbol))

    def expire_requests(self, max_age_s=48 * 3600, now=None):
        now = time.time() if now is None else now
        self.x("UPDATE requests SET status='expired', done_ts=? WHERE status='open' AND ts<?", (now, now - max_age_s))

    def requests(self, limit=100):
        return self.q("SELECT * FROM requests ORDER BY id DESC LIMIT ?", (limit,))

    # ---- the research view -----------------------------------------------------------------
    def set_view(self, views, now=None):
        """views: {symbol: dict}. Replaces the whole view atomically."""
        now = time.time() if now is None else now
        with self.lock:
            self.c.execute("BEGIN IMMEDIATE")
            try:
                self.c.execute("DELETE FROM view")
                self.c.executemany("INSERT INTO view(symbol,asset,ts,data) VALUES(?,?,?,?)",
                                   [(s, v.get("asset", ""), now, dumps(v)) for s, v in views.items()])
                self.c.execute("COMMIT")
            except Exception:
                self.c.execute("ROLLBACK")
                raise

    def view(self, symbols=None):
        if symbols is None:
            rows = self.q("SELECT * FROM view")
        else:
            symbols = list(symbols)
            if not symbols:
                return {}
            rows = self.q(f"SELECT * FROM view WHERE symbol IN ({','.join('?' * len(symbols))})", symbols)
        out = {}
        for r in rows:
            v = loads(r["data"], {}) or {}
            v["view_ts"] = r["ts"]
            out[r["symbol"]] = v
        return out

    # ---- prices (hourly closes for outcomes) ---------------------------------------------------
    def put_prices(self, symbol, bars):
        """bars: [(ts, close)] hourly bar START times (crypto) or session hour starts (stocks)."""
        if bars:
            self.many("INSERT OR REPLACE INTO prices(symbol,ts,close) VALUES(?,?,?)", [(symbol, int(t), float(c)) for t, c in bars])

    def price_series(self, symbol, since=None):
        if since is None:
            return [(r["ts"], r["close"]) for r in self.q("SELECT ts,close FROM prices WHERE symbol=? ORDER BY ts", (symbol,))]
        return [(r["ts"], r["close"]) for r in self.q("SELECT ts,close FROM prices WHERE symbol=? AND ts>=? ORDER BY ts", (symbol, since))]

    def last_price_ts(self, symbol):
        r = self.q1("SELECT MAX(ts) t FROM prices WHERE symbol=?", (symbol,))
        return r["t"] if r else None

    # ---- models ----------------------------------------------------------------------------
    def save_model(self, name, blob, metrics, role="challenger", now=None):
        r = self.q1("SELECT COALESCE(MAX(version),0) v FROM models WHERE name=?", (name,))
        v = (r["v"] if r else 0) + 1
        self.x("INSERT INTO models(name,version,ts,role,metrics,blob) VALUES(?,?,?,?,?,?)",
               (name, v, time.time() if now is None else now, role, dumps(metrics), blob))
        return v

    def promote_model(self, name, version):
        """Make `version` the champion; the old champion is retired. Keeps the last 3 versions' blobs."""
        with self.lock:
            self.c.execute("BEGIN IMMEDIATE")
            try:
                self.c.execute("UPDATE models SET role='retired' WHERE name=? AND role='champion'", (name,))
                self.c.execute("UPDATE models SET role='champion' WHERE name=? AND version=?", (name, version))
                self.c.execute("""UPDATE models SET blob=NULL WHERE name=? AND role!='champion' AND version NOT IN
                                  (SELECT version FROM models WHERE name=? ORDER BY version DESC LIMIT 3)""", (name, name))
                self.c.execute("COMMIT")
            except Exception:
                self.c.execute("ROLLBACK")
                raise

    def set_model_role(self, name, version, role):
        self.x("UPDATE models SET role=? WHERE name=? AND version=?", (role, name, version))

    def champion(self, name):
        r = self.q1("SELECT * FROM models WHERE name=? AND role='champion' ORDER BY version DESC LIMIT 1", (name,))
        if r:
            r["metrics"] = loads(r["metrics"], {})
        return r

    def models(self, name=None):
        rows = self.q("SELECT name,version,ts,role,metrics FROM models " + ("WHERE name=? " if name else "") + "ORDER BY name, version DESC",
                      (name,) if name else ())
        for r in rows:
            r["metrics"] = loads(r["metrics"], {})
        return rows

    # ---- shadow decisions --------------------------------------------------------------------
    def add_shadow(self, rows):
        """rows: [(ts, board, symbol, asset, kind, base, adj, detail_dict)]"""
        if rows:
            self.many("INSERT INTO shadow(ts,board,symbol,asset,kind,base,adj,detail) VALUES(?,?,?,?,?,?,?,?)",
                      [(r[0], r[1], r[2], r[3], r[4], r[5], r[6], dumps(r[7] if len(r) > 7 else {})) for r in rows])

    def shadow(self, kind=None, since=None, matured=None, limit=200000):
        w, a = ["1"], []
        if kind:
            w.append("kind=?"); a.append(kind)
        if since is not None:
            w.append("ts>=?"); a.append(since)
        if matured is not None:
            w.append("matured=?"); a.append(int(matured))
        rows = self.q(f"SELECT * FROM shadow WHERE {' AND '.join(w)} ORDER BY id LIMIT ?", (*a, limit))
        for r in rows:
            r["detail"] = loads(r["detail"], {})
        return rows

    def mature_shadow(self, rows):
        """rows: [(excess, id)]"""
        if rows:
            self.many("UPDATE shadow SET matured=1, excess=? WHERE id=?", rows)

    # ---- stages ---------------------------------------------------------------------------
    def stage(self, name):
        r = self.q1("SELECT * FROM stages WHERE name=?", (name,))
        return r["state"] if r else "shadow"

    def stages(self):
        out = {n: dict(name=n, state="shadow", since=None, evidence={}, by="default") for n in STAGE_ORDER}
        for r in self.q("SELECT * FROM stages"):
            r["evidence"] = loads(r["evidence"], {})
            out[r["name"]] = r
        return out

    def set_stage(self, name, state, evidence=None, by="gate", now=None):
        self.x("""INSERT INTO stages(name,state,since,evidence,by) VALUES(?,?,?,?,?) ON CONFLICT(name) DO UPDATE SET
                  state=excluded.state, since=excluded.since, evidence=excluded.evidence, by=excluded.by""",
               (name, state, time.time() if now is None else now, dumps(evidence or {}), by))

    # ---- discovery and advisor watch-only proposals ----------------------------------------------
    def discover(self, symbol, asset, source, reason, now=None):
        now = time.time() if now is None else now
        r = self.q1("SELECT sources, mentions FROM discovered WHERE symbol=?", (symbol,))
        if r:
            src = set(loads(r["sources"], []) or []) | {source}
            self.x("UPDATE discovered SET last_seen=?, sources=?, mentions=mentions+1, reason=COALESCE(reason,?) WHERE symbol=?",
                   (now, dumps(sorted(src)), reason, symbol))
        else:
            self.x("INSERT INTO discovered(symbol,asset,first_seen,last_seen,sources,reason,mentions,status) VALUES(?,?,?,?,?,?,1,'new')",
                   (symbol, asset, now, now, dumps([source]), reason[:300]))

    def discovered(self, status=None, asset=None, limit=500):
        w, a = ["1"], []
        if status:
            w.append("status=?"); a.append(status)
        if asset:
            w.append("asset=?"); a.append(asset)
        rows = self.q(f"SELECT * FROM discovered WHERE {' AND '.join(w)} ORDER BY last_seen DESC LIMIT ?", (*a, limit))
        for r in rows:
            r["sources"] = loads(r["sources"], [])
        return rows

    def set_discovered(self, symbol, status, detail=""):
        self.x("UPDATE discovered SET status=?, detail=? WHERE symbol=?", (status, detail[:300], symbol))

    def add_proposal(self, ts, board, symbol, asset, advisor, z, reason, status, price0=None, bench0=None):
        """An advisor wants a name on the watch-only list. Deduped per board+symbol while open."""
        r = self.q1("SELECT id FROM proposals WHERE board=? AND symbol=? AND status IN ('shadow','added')", (board, symbol))
        if r:
            return None
        cur = self.x("INSERT INTO proposals(ts,board,symbol,asset,advisor,z,reason,status,price0,bench0) VALUES(?,?,?,?,?,?,?,?,?,?)",
                     (ts, board, symbol, asset, advisor, z, reason[:300], status, price0, bench0))
        return cur.lastrowid

    def proposals(self, board=None, status=None, limit=200):
        w, a = ["1"], []
        if board:
            w.append("board=?"); a.append(board)
        if status:
            w.append("status=?"); a.append(status)
        return self.q(f"SELECT * FROM proposals WHERE {' AND '.join(w)} ORDER BY id DESC LIMIT ?", (*a, limit))

    def proposals_today(self, board, since):
        r = self.q1("SELECT COUNT(*) n FROM proposals WHERE board=? AND ts>=?", (board, since))
        return r["n"] if r else 0

    def set_proposal(self, pid, status, excess_7d=None):
        if excess_7d is None:
            self.x("UPDATE proposals SET status=? WHERE id=?", (status, pid))
        else:
            self.x("UPDATE proposals SET status=?, excess_7d=? WHERE id=?", (status, excess_7d, pid))

    # ---- health history ----------------------------------------------------------------------
    def add_health(self, data, now=None):
        self.x("INSERT INTO health(ts,data) VALUES(?,?)", (time.time() if now is None else now, dumps(data)))

    def last_health(self):
        r = self.q1("SELECT * FROM health ORDER BY id DESC LIMIT 1")
        if r:
            r["data"] = loads(r["data"], {})
        return r

    # ---- pruning ------------------------------------------------------------------------------
    def prune(self, keep_days=90, now=None):
        now = time.time() if now is None else now
        cut = now - keep_days * 86400
        with self.lock:
            self.c.execute("BEGIN IMMEDIATE")
            try:
                self.c.execute("DELETE FROM outcomes WHERE note_id IN (SELECT id FROM notes WHERE ts<? AND id NOT IN (SELECT note_id FROM labels))", (cut,))
                self.c.execute("DELETE FROM notes WHERE ts<? AND id NOT IN (SELECT note_id FROM labels)", (cut,))
                self.c.execute("DELETE FROM items WHERE fetched_ts<? AND labelled!=0", (cut,))
                self.c.execute("UPDATE items SET text='' WHERE fetched_ts<? AND labelled!=0", (now - 14 * 86400,))
                self.c.execute("DELETE FROM shadow WHERE ts<?", (cut,))
                self.c.execute("DELETE FROM prices WHERE ts<?", (now - 400 * 86400,))
                self.c.execute("DELETE FROM health WHERE id <= (SELECT COALESCE(MAX(id),0) FROM health) - 3000")
                self.c.execute("DELETE FROM events WHERE id <= (SELECT COALESCE(MAX(id),0) FROM events) - 5000")
                self.c.execute("DELETE FROM requests WHERE ts<?", (cut,))
                self.c.execute("COMMIT")
            except Exception:
                self.c.execute("ROLLBACK")
                raise


def open_db(cfg, readonly=False):
    """The research database named in config (research.db, default data/research.db)."""
    return ResearchDB(research_path(cfg), readonly=readonly)


def research_path(cfg):
    """research.db from config; the simulator gets its own file so invented data never mixes with real research."""
    r = ((cfg or {}).get("research") or {}).get("db")
    if r:
        return r
    sim = ((cfg or {}).get("feed") or {}).get("kind") == "sim"
    base = os.path.dirname((cfg or {}).get("db", os.path.join("data", "x.db"))) or "data"
    return os.path.join(base, "research-sim.db" if sim else "research.db")
