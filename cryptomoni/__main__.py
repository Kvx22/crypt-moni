"""CLI:  python run.py <command>   (Windows: py run.py <command>)

  up        [--config F] [--port N]      scanner + dashboard in one process (the usual way to run it 24/7)
  scan      [--config F]                 the 24/7 scanner loop only
  dashboard [--config F] [--port N] [--host H]
  watchdog  [--config F]                 separate process: trips the kill switch on a stale scanner
  demo      [--hours N]                  offline simulation (invented data), N simulated hours, fast
  once      [--config F]                 one scan tick, then print the board
  status    [--config F]                 print the latest board from the database
  export    [--config F] [--out FILE]    the dashboard as one self-contained, read-only HTML snapshot
  doctor    [--config F]                 check config, data, feed and safety settings
  backtest  [--days N]                   walk-forward test of the brain + exposure board on BTC daily closes

Opportunity scout (the advisors look beyond the board's own list):
  scout-export                           the shortlist as JSON, with what Claude should research for each name
  research SYMBOL --verdict favorable|mixed|unfavorable|unverified --note TEXT [--source URL]... [--catalyst TEXT]
                                         attach a research note (what Claude does after researching a candidate)
  watch SYMBOL | unwatch SYMBOL          put a shortlisted name on the board's watchlist (or take it off)

Live mode (config.coinbase.json / config.robinhood.json): the board proposes, you approve.
  outbox                                 list proposed / approved / sent orders
  approve ID|all, reject ID              same rules as the dashboard buttons
  export-approved                        Robinhood: approved orders as JSON for Claude to place
  ack ID --status sent|failed [--order-id X] [--detail TEXT]   Robinhood: record what Claude did
  rh-snapshot --cash N --pos BTC-USD=0.01,SPY=1.5 [--other N] Robinhood: write the synced balances (crypto + stocks)
  kill | reset-kill | reset-breaker      safety controls (only a human resets)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time


def load_cfg(path):
    with open(path) as f:
        return json.load(f)


def build(cfg, fresh=False):
    from .engine import Engine, day_ts, read_daily_csv
    from .feeds import DAY, make_feed
    from .store import Store
    db = cfg["db"]
    os.makedirs(os.path.dirname(db) or ".", exist_ok=True)
    if fresh or cfg.get("feed", {}).get("kind") == "sim":  # a simulation always starts from scratch
        for ext in ("", "-wal", "-shm"):
            if os.path.exists(db + ext):
                os.remove(db + ext)
    anchors, f = None, cfg.setdefault("feed", {})
    if f.get("kind") == "sim":  # continue the real BTC history where it ends
        daily = read_daily_csv(cfg.get("daily_csv", "data/BTC-USD.csv"))
        if daily:
            last = next(reversed(daily))
            anchors = {cfg.get("market_proxy", "BTC-USD"): daily[last]}
            f.setdefault("start_ts", day_ts(last) + DAY)
            from .stocks import read_closes
            sd = (cfg.get("stocks") or {}).get("csv_dir", "data/stocks")
            for sym in (cfg.get("stocks") or {}).get("symbols", []):   # simulated stocks start from their real last close
                cs = read_closes(os.path.join(sd, f"{sym}.csv"))
                if cs:
                    anchors[sym] = cs[next(reversed(cs))]
    return Engine(cfg, make_feed(cfg, anchors), Store(db))


def print_board(st):
    s, live = st.latest(), st.kv_get("live")
    if not s:
        return print("No board yet: the scanner needs a closed hourly bar and a week of history. Run it, or try `demo`.")
    from datetime import datetime, timezone
    when = datetime.fromtimestamp(s["ts"], timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print(f"{'SIMULATION · ' if s['simulated'] else ''}{when} · BTC regime {s['regime'].upper()} {s['confidence']:.0%}"
          f" (brain on daily close {s['brain_date']})")
    b = s.get("board") or {}
    if b:
        print(f"Exposure board {b['raw']:.0%} -> chair ceiling {b['chair_ceiling']:.0%} -> target invested {s['e_target']:.0%}"
              f" (actual {s['e_actual']:.0%}) · breaker {s['breaker']['name']}")
    print(f"Paper equity ${s['equity']:,.2f} · cash ${s['cash']:,.2f} · fees ${s['fees']:,.2f}")
    print("\nAdvisor          board vote  board w  desk w   IC     hit   pick")
    for a in s["advisors"]:
        print(f"  {a['name']:<16}{a['vote']:>8.0%}{a['board_weight']:>9.1%}{a['desk_weight']:>8.1%}{a['ic']:>+7.2f}{a['hit']:>7.0%}   {a['pick']}")
    print("\nCoin        price        24h      7d   risk         desk    verdict")
    lm = {c["symbol"]: c for c in (live or {}).get("coins", []) if not c.get("warming")}
    for c in sorted(s["coins"], key=lambda c: -c["score"]):
        m = lm.get(c["symbol"], {})
        print(f"  {c['symbol']:<9}{m.get('price', 0):>11,.4g}{m.get('ret_24h', 0):>8.1%}{m.get('ret_7d', 0):>8.1%}"
              f"   {c['risk']:>4.0f} {c['tier']:<9}{c['score']:>+6.2f}   {c['verdict']}")
    for n in s["notes"]:
        print("  note:", n)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="run.py", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["up", "scan", "dashboard", "watchdog", "demo", "once", "status", "export", "doctor", "backtest",
                                    "kill", "reset-kill", "reset-breaker", "outbox", "approve", "reject",
                                    "export-approved", "ack", "rh-snapshot", "scout-export", "research", "watch", "unwatch"])
    ap.add_argument("--verdict", choices=["favorable", "mixed", "unfavorable", "unverified"], default="unverified")
    ap.add_argument("--note", default="")
    ap.add_argument("--source", action="append", default=[])
    ap.add_argument("--catalyst", default="")
    ap.add_argument("id", nargs="?", default=None)
    ap.add_argument("--status", choices=["sent", "failed"], default=None)
    ap.add_argument("--order-id", default="")
    ap.add_argument("--detail", default="")
    ap.add_argument("--cash", type=float, default=None)
    ap.add_argument("--pos", default="")
    ap.add_argument("--other", type=float, default=0.0)
    ap.add_argument("--config", default=None)
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--host", default=None)
    ap.add_argument("--hours", type=int, default=24 * 21)
    ap.add_argument("--days", type=int, default=1095)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--out", default="crypto-risk-board.html")
    a = ap.parse_args(argv)
    cfg_path = a.config or ("config.sim.json" if a.cmd == "demo" else "config.json")
    cfg = load_cfg(cfg_path)
    kill_file = cfg.get("safety", {}).get("kill_file", "KILL")
    dash = cfg.get("dashboard", {})
    port, host = a.port or dash.get("port", 8765), a.host or dash.get("host", "127.0.0.1")

    from .safety import KillSwitch
    if a.cmd == "kill":
        KillSwitch(kill_file).trip("manual (cli)")
        return print(f"KILL switch engaged ({kill_file}). No paper order passes until `reset-kill`.")
    if a.cmd == "reset-kill":
        KillSwitch(kill_file).reset()
        return print("Kill switch cleared.")
    if a.cmd == "reset-breaker":
        from .safety import CircuitBreaker, SafetyConfig
        from .store import Store
        st = Store(cfg["db"])
        s = st.latest()
        br = CircuitBreaker(SafetyConfig.from_dict(cfg.get("safety", {})), st.kv_get("breaker"))
        br.reset(s["equity"] if s else 0.0)
        st.kv_set("breaker", br.save())
        return print("Breaker reset; high-water mark re-based to current paper equity.")

    token = os.environ.get("CM_TOKEN") or cfg.get("execution", {}).get("approve_token")
    if a.cmd in ("outbox", "approve", "reject", "export-approved", "ack", "rh-snapshot"):
        return live_cmds(a, cfg, kill_file)
    if a.cmd in ("scout-export", "research", "watch", "unwatch"):
        return scout_cmds(a, cfg)
    if a.cmd == "dashboard":
        from .dashboard import serve
        return serve(cfg["db"], kill_file, port, host, token)
    if a.cmd == "watchdog":
        from .watchdog import run
        w = cfg.get("watchdog", {})
        return run(cfg["db"], kill_file, poll=w.get("poll", 60), stale_s=w.get("stale_s", 900), hard_dd=w.get("hard_dd", 0.30))
    if a.cmd == "status":
        from .store import Store
        return print_board(Store(cfg["db"]))
    if a.cmd == "export":
        return export(cfg, a.out)
    if a.cmd == "doctor":
        return doctor(cfg, cfg_path, kill_file)
    if a.cmd == "backtest":
        return backtest(cfg, a.days)

    eng = build(cfg, fresh=a.fresh)
    if a.cmd == "once":
        from .feeds import FeedError
        try:
            eng.tick()
        except FeedError as e:
            return print(f"No market data: {e}\nCheck your internet connection (python run.py doctor).")
        return print_board(eng.s)
    if a.cmd == "demo":
        print(f"SIMULATION: {a.hours} simulated hours of invented market data into {cfg['db']} (real BTC history seeds the brain).")
        t0 = time.time()
        for h in range(a.hours):
            eng.tick()
            if h % 24 == 23:
                s = eng.s.latest()
                if s:
                    lead = max(s["advisors"], key=lambda x: x["desk_weight"])
                    print(f"  day {h // 24 + 1:>3}: regime {s['regime']:<8} equity ${s['equity']:>10,.0f}  "
                          f"desk leader {lead['name']:<16} {lead['desk_weight']:.0%}  ({time.time() - t0:.0f}s)", flush=True)
        eng.s.prune()
        print_board(eng.s)
        return print(f"\nOpen the board: python run.py dashboard --config {cfg_path}")
    scan_s = cfg.get("scan_seconds", 60)
    if a.cmd == "up":
        from .dashboard import serve
        threading.Thread(target=serve, args=(cfg["db"], kill_file, port, host, token), daemon=True).start()
    how = (f"LIVE on {eng.broker.name}: every order waits for your approval" if eng.approve else "paper only")
    print(f"Scanning {len(eng.universe)} coins every {scan_s}s, 24/7 ({eng.feed.kind} feed, {how}). Ctrl+C to stop.", flush=True)
    try:
        eng.run_forever(scan_s)
    except KeyboardInterrupt:
        print("stopped")


def scout_cmds(a, cfg):
    from .store import Store
    st = Store(cfg["db"])
    now = time.time()
    opps = st.kv_get("opps") or {}
    items = [i for v in opps.values() for i in v.get("items", [])]
    if a.cmd == "scout-export":
        research = st.kv_get("research", {})
        out = []
        for i in items:
            out.append(dict(symbol=i["symbol"], asset=i["asset"], desk_score=round(i["score"], 2), verdict=i["verdict"], risk=f"{i['tier']} {i['risk']:.0f}",
                            price=i["price"], ret_24h=round(i["ret_24h"], 4), ret_7d=round(i["ret_7d"], 4), annual_vol=round(i["vol_24h"], 2),
                            ai_exposed=i["ai_exposed"], advisors_who_like_it=[w["name"] for w in i["why"]],
                            already_researched=i["symbol"] in research))
        return print(json.dumps(dict(
            asof=time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(max([v["ts"] for v in opps.values()] or [now]))),
            what_this_is="candidates the five advisors found outside the board's own list by scoring market data with their learned weights; they are leads, not recommendations",
            how_to_research=["for stocks: fundamentals, analyst ratings, the next earnings date and recent SEC filings (Robinhood tools), plus a current news search",
                             "for coins: a current news search (listings, unlocks, exploits, regulation), what the project does, and who holds the supply",
                             "open pages rather than trusting search snippets; say which sources are aggregators; mark any date you could not confirm; check the weekday of every catalyst date",
                             "flag anything that depends on AI valuations (ai_exposed marks the obvious ones); the owner wants to avoid those",
                             "give a verdict: favorable, mixed, unfavorable or unverified, with the reasons and what would change it; never place an order"],
            record_with="python run.py research SYMBOL --config <config> --verdict ... --note \"...\" --source URL --catalyst \"date: event\"",
            candidates=out), indent=2))
    if a.cmd == "research":
        sym = (a.id or "").upper()
        if not sym or not a.note:
            return print("usage: research SYMBOL --verdict favorable|mixed|unfavorable|unverified --note TEXT [--source URL]... [--catalyst TEXT]")
        r = st.kv_get("research", {})
        r[sym] = dict(ts=now, verdict=a.verdict, note=a.note[:1500], sources=a.source[:8], catalyst=a.catalyst[:200], by="Claude")
        st.kv_set("research", r)
        st.add_event(now, "scout", "info", f"Research on {sym}: {a.verdict}", a.note[:200])
        return print(f"saved research on {sym} ({a.verdict})")
    sym = (a.id or "").upper()
    wl = st.kv_get("watchlist", [])
    if a.cmd == "watch":
        found = next((i for i in items if i["symbol"] == sym), None)
        if not found:
            return print(f"{sym} is not on the scout's shortlist (run `scout-export` to see it)")
        if not any(w["symbol"] == sym for w in wl):
            wl.append(dict(symbol=sym, asset=found["asset"], added=now))
        st.kv_set("watchlist", wl)
        return print(f"{sym} is on the watchlist: the running scanner adopts it on its next tick.")
    st.kv_set("watchlist", [w for w in wl if w["symbol"] != sym])
    return print(f"{sym} removed from the watchlist (it leaves the board at the next restart).")


def live_cmds(a, cfg, kill_file):
    from .safety import KillSwitch, SafetyConfig
    from .store import Store
    ex = cfg.get("execution", {})
    if a.cmd == "rh-snapshot":
        if a.cash is None:
            return print("usage: rh-snapshot --cash 123.45 --pos BTC-USD=0.0012,SPY=1.5 [--other VALUE_OF_OTHER_HOLDINGS]")
        pos = {}
        for kv in filter(None, a.pos.split(",")):
            k, v = kv.split("=")
            pos[k.strip().upper()] = float(v)
        path = ex.get("snapshot", "data/robinhood_snapshot.json")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(dict(ts=time.time(), cash=a.cash, positions=pos, other_value=a.other, source="Claude Robinhood connector"), f, indent=2)
        return print(f"wrote {path}: cash ${a.cash:,.2f}, {len(pos)} position(s), other holdings ${a.other:,.2f}")
    st = Store(cfg["db"])
    now = time.time()
    st.expire_outbox(now)
    if a.cmd == "outbox":
        rows = st.outbox(60)
        for r in reversed(rows):
            print(f"#{r['id']:>4} {r['status']:<10} {r['side']:<4} {r['symbol']:<9} ${r['notional']:>9,.2f}  {r['verdict'] or '':<7} {r['detail'] or ''}")
        return print("" if rows else "No proposals yet.")
    if a.cmd in ("approve", "reject"):
        if a.id is None:
            return print(f"usage: {a.cmd} ID" + (" | all" if a.cmd == "approve" else ""))
        if a.cmd == "approve" and KillSwitch(kill_file).engaged():
            return print("refused: kill switch engaged")
        rows = st.outbox(200, "pending") if a.id == "all" else [st.outbox_get(int(a.id))]
        for r in rows:
            if r is None:
                print("no such order"); continue
            if a.cmd == "approve" and (r["status"] != "pending" or now > r["expires_ts"]):
                print(f"#{r['id']}: cannot approve ({r['status']})"); continue
            if a.cmd == "reject" and r["status"] not in ("pending", "approved"):
                print(f"#{r['id']}: cannot reject ({r['status']})"); continue
            st.outbox_set(r["id"], "approved" if a.cmd == "approve" else "rejected", now, decided_by="cli")
            print(f"#{r['id']} {a.cmd}d: {r['side']} {r['symbol']} ${r['notional']:,.2f}")
        return
    if a.cmd == "export-approved":
        sc = SafetyConfig.from_dict(cfg.get("safety", {}))
        killed = KillSwitch(kill_file).engaged()
        rows = [] if killed else st.outbox(200, "approved")
        out = [dict(outbox_id=r["id"], symbol=r["symbol"], asset="crypto" if "-" in r["symbol"] else "stock", side=r["side"],
                    notional_usd=round(min(r["notional"], sc.max_per_trade_usd or 0), 2),
                    ref_price=r["ref_price"], max_move=sc.price_band_pct, client_order_id=r["client_order_id"],
                    expires_in_min=round((r["expires_ts"] - now) / 60)) for r in reversed(rows)]
        return print(json.dumps(dict(broker=ex.get("broker"), kill_switch=killed, max_per_trade_usd=sc.max_per_trade_usd,
                                     rule="place each only if the live price is within max_move of ref_price; stocks only while the US market is open; then ack it",
                                     orders=out), indent=2))
    if a.cmd == "ack":
        r = st.outbox_get(int(a.id)) if a.id else None
        if not r or not a.status:
            return print("usage: ack ID --status sent|failed [--order-id X] [--detail TEXT]")
        if r["status"] != "approved":
            return print(f"refused: #{r['id']} is {r['status']}, not approved")
        st.outbox_set(r["id"], a.status, now, broker_order_id=a.order_id, detail=a.detail)
        if a.status == "sent":
            from .models import Fill
            st.add_trade(Fill(r["symbol"], r["side"], r["notional"], 0, r["ref_price"], now, "sent", a.order_id), f"#{r['id']} approved, placed by Claude")
        st.add_event(now, "outbox", "info" if a.status == "sent" else "warn", f"#{r['id']} {a.status} via Claude ({r['broker']})", a.detail or a.order_id)
        return print(f"#{r['id']} marked {a.status}")


def export(cfg, out):
    """Bake the latest board into one HTML page (no server, nothing it can change)."""
    import re
    from .dashboard import STATIC
    from .store import Store
    if not os.path.exists(cfg["db"]):
        return print(f"No database at {cfg['db']}: run the scanner (or `demo`) first.")
    st = Store(cfg["db"], readonly=True)
    if not st.latest():
        return print("No board yet: the scanner needs a closed hourly bar and a week of history.")
    events = st.events(150)
    for e in events:
        e["payload"] = None
    data = dict(state=dict(snapshot=st.latest(), live=st.kv_get("live"), opps=st.kv_get("opps"), research=st.kv_get("research", {}),
                           watchlist=st.kv_get("watchlist", []), heartbeat_age=None, kill=False,
                           kill_reason="", version=st.counts()),
                history=st.history(2000), trades=st.trades(80), events=events, alerts=st.alerts(60))
    with open(os.path.join(STATIC, "index.html"), encoding="utf-8") as f:
        html = f.read()
    style = re.search(r"<style>.*?</style>", html, re.S).group(0)
    body = re.search(r"<body>(.*)</body>", html, re.S).group(1)
    blob = json.dumps(data, separators=(",", ":"), default=float).replace("</", "<\\/")
    page = ('<title>Crypto Risk Board</title>\n<meta name="claude:viewport-fit-sides" content="cover">\n'
            + style + "\n<script>window.CM_STATIC=" + blob + ";</script>\n" + body)
    with open(out, "w", encoding="utf-8") as f:
        f.write(page)
    print(f"wrote {out} ({len(page) / 1024:,.0f} KB): a read-only snapshot of the board")


def doctor(cfg, cfg_path, kill_file):
    from .feeds import FeedError, make_feed
    from .safety import KillSwitch
    ok = True

    def line(good, what):
        nonlocal ok
        ok &= bool(good)
        print(("  OK    " if good else "  FIX   ") + what)
    print(f"config {cfg_path}")
    sc = cfg.get("safety", {})
    print(f"  INFO  mode {sc.get('mode', 'paper')}" + (" (simulated fills)" if sc.get("mode", "paper") == "paper" else " (real orders, each approved by you)"))
    line(sc.get("max_per_trade_usd"), f"max_per_trade_usd = {sc.get('max_per_trade_usd')} (unset refuses every order; yours to set)")
    line(not KillSwitch(kill_file).engaged(), f"kill switch {'ENGAGED: ' + KillSwitch(kill_file).reason() if KillSwitch(kill_file).engaged() else 'clear'}")
    csvp = cfg.get("daily_csv", "data/BTC-USD.csv")
    if os.path.exists(csvp):
        with open(csvp) as f:
            rows = f.read().strip().splitlines()
        line(len(rows) > 1500, f"{csvp}: {len(rows) - 1:,} daily closes through {rows[-1].split(',')[0]} (the scanner appends new days)")
    else:
        print(f"  NOTE  {csvp} missing: the scanner will backfill BTC daily history from the feed on first start")
    ex = cfg.get("execution", {})
    if sc.get("mode") == "live":
        from .safety import SafetyConfig, live_armed
        line(sc.get("live_armed") is True, "live_armed = true in config (you set this)")
        line(os.environ.get("CM_LIVE") == "1", "environment variable CM_LIVE=1 (second arming key)")
        armed = live_armed(SafetyConfig.from_dict(sc))
        print(f"  INFO  live orders {'ARMED: every order still needs your approval' if armed else 'not armed: the board proposes nothing'}")
        if ex.get("broker") == "coinbase":
            try:
                from .brokers import CoinbaseBroker
                b = CoinbaseBroker(ex.get("key_file", "secrets/cdp_api_key.json"), ex.get("quote_currency", "USD"))
                bal, _ = b.balances()
                held = {k: v for k, v in bal.items() if v > 0}
                line(True, f"Coinbase key works (read-only check): {len(held)} non-zero balance(s), "
                           f"{ex.get('quote_currency', 'USD')} {bal.get(ex.get('quote_currency', 'USD'), 0):,.2f}")
            except Exception as e:
                line(False, f"Coinbase API: {e}")
        elif ex.get("broker") == "robinhood":
            from .brokers import RobinhoodHandoff
            rh = RobinhoodHandoff(ex.get("snapshot", "data/robinhood_snapshot.json"), ex.get("snapshot_max_age_h", 24))
            age = rh.age()
            line(age is not None and not rh.stale(), "Robinhood balances synced " + ("never: ask Claude to sync" if age is None else f"{age / 3600:.1f}h ago"))
        else:
            line(False, "execution.broker must be coinbase or robinhood in live mode")
        print("  INFO  approval token " + ("set" if (os.environ.get("CM_TOKEN") or ex.get("approve_token")) else "not set (fine on localhost; set one before reaching the dashboard over a network)"))
        print("  INFO  phone push " + ("on (ntfy)" if ex.get("ntfy_topic") else "off (set execution.ntfy_topic)"))
    if cfg.get("stocks"):
        from .feeds import StockFeed, stock_market_open
        sp = cfg["stocks"].get("proxy", "SPY")
        print(f"  INFO  stock book: {len(cfg['stocks']['symbols'])} symbols, {cfg['stocks'].get('share', 0.4):.0%} of the account, "
              f"US market {'open' if stock_market_open(time.time()) else 'closed'} now")
        try:
            p, ts = StockFeed().ticker(sp)
            line(True, f"Yahoo Finance: {sp} ${p:,.2f} (last trade {(time.time() - ts) / 60:.0f} min ago)")
        except FeedError as e:
            line(False, f"stock data unreachable (Yahoo Finance): {e}")
    feed = make_feed(cfg)
    try:
        p, ts = feed.ticker(cfg.get("market_proxy", "BTC-USD"))
        line(True, f"feed {getattr(feed, 'last_source', feed.kind)}: BTC-USD ${p:,.2f}")
    except FeedError as e:
        line(False, f"feed unreachable: {e}")
    if os.path.exists(cfg["db"]):
        from .store import Store
        st = Store(cfg["db"], readonly=True)
        hb = st.kv_get("heartbeat")
        print(f"  INFO  database {cfg['db']}: last heartbeat {'never' if hb is None else f'{time.time() - hb:.0f}s ago'}")
    print("READY" if ok else "NOT READY")


def backtest(cfg, days):
    """Walk-forward: the brain refits on past data only, the exposure board learns online, and its
    vote is applied to BTC the NEXT day (1-day lag). Cash earns 4%/yr. Indicative, not a forecast."""
    import numpy as np
    from .advisors import ExposureBoard
    from .engine import read_daily_csv
    from .regime import RegimeTracker
    d = read_daily_csv(cfg.get("daily_csv", "data/BTC-USD.csv"))
    dates, c = list(d), np.array(list(d.values()), float)
    tr, bd = RegimeTracker(**cfg.get("tracker", {})), ExposureBoard(**cfg.get("board", {}))
    n = len(c)
    start = max(n - days, 365 + tr.min_train + 1)
    tr.update(c[:start])
    e_prev, prev_probs, eq, bh, ex = None, None, [1.0], [1.0], []
    t0 = time.time()
    for t in range(start, n):
        r = c[t] / c[t - 1] - 1
        if e_prev is not None:
            eq.append(eq[-1] * (1 + e_prev * r + (1 - e_prev) * 0.04 / 365))
            bh.append(bh[-1] * (1 + r))
            bd.observe(prev_probs, r)
        s = tr.update(c[: t + 1])
        dec = bd.decide(s["probs"], s["next_probs"], s["confidence"])
        e_prev, prev_probs = dec["board"], s["probs"]
        ex.append(e_prev)

    def stats(x):
        x = np.array(x); r = np.diff(np.log(x)); yrs = len(r) / 365
        dd = (x / np.maximum.accumulate(x) - 1).min()
        return x[-1] ** (1 / yrs) - 1, r.std() * np.sqrt(365), (r.mean() * 365 - 0.04) / (r.std() * np.sqrt(365)), dd
    a, b = stats(eq), stats(bh)
    print(f"BTC-USD {dates[start]} -> {dates[-1]} ({n - start} days, {tr.fit_count} HMM fits, {time.time() - t0:.0f}s)")
    print(f"{'':16}{'board':>10}{'buy&hold':>10}")
    for k, i, f in (("CAGR", 0, "{:+.1%}"), ("volatility", 1, "{:.0%}"), ("Sharpe (4% cash)", 2, "{:.2f}"), ("max drawdown", 3, "{:.0%}")):
        print(f"{k:16}{f.format(a[i]):>10}{f.format(b[i]):>10}")
    print(f"average exposure {np.mean(ex):.0%}. History, not a forecast; one asset, settings chosen once.")


if __name__ == "__main__":
    main()
