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
                                    "kill", "reset-kill", "reset-breaker"])
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

    if a.cmd == "dashboard":
        from .dashboard import serve
        return serve(cfg["db"], kill_file, port, host)
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
        threading.Thread(target=serve, args=(cfg["db"], kill_file, port, host), daemon=True).start()
    print(f"Scanning {len(eng.universe)} coins every {scan_s}s, 24/7 ({eng.feed.kind} feed, paper only). Ctrl+C to stop.", flush=True)
    try:
        eng.run_forever(scan_s)
    except KeyboardInterrupt:
        print("stopped")


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
    data = dict(state=dict(snapshot=st.latest(), live=st.kv_get("live"), heartbeat_age=None, kill=False,
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
    line(sc.get("mode", "paper") == "paper", f"mode {sc.get('mode', 'paper')} (this build is paper-only)")
    line(sc.get("max_per_trade_usd"), f"max_per_trade_usd = {sc.get('max_per_trade_usd')} (unset refuses every paper order)")
    line(not KillSwitch(kill_file).engaged(), f"kill switch {'ENGAGED: ' + KillSwitch(kill_file).reason() if KillSwitch(kill_file).engaged() else 'clear'}")
    csvp = cfg.get("daily_csv", "data/BTC-USD.csv")
    if os.path.exists(csvp):
        with open(csvp) as f:
            rows = f.read().strip().splitlines()
        line(len(rows) > 1500, f"{csvp}: {len(rows) - 1:,} daily closes through {rows[-1].split(',')[0]} (the scanner appends new days)")
    else:
        print(f"  NOTE  {csvp} missing: the scanner will backfill BTC daily history from the feed on first start")
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
