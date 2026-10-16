"""Separate-process watchdog. Reads only the database; imports nothing from the
brain. Crypto never closes, so there is no market-hours exemption: if the
scanner goes silent, or paper equity falls past a hard-stop drawdown, it trips
the kill switch even if the engine itself is wedged."""
from __future__ import annotations

import time

from .safety import KillSwitch
from .store import Store


def check(store: Store, kill: KillSwitch, now=None, stale_s=900, hard_dd=0.30):
    now = now or time.time()
    if kill.engaged():
        return "already killed"
    hb = store.kv_get("heartbeat")
    if hb is not None and now - hb > stale_s:
        kill.trip(f"watchdog: scanner heartbeat stale {int(now - hb)}s")
        store.add_event(now, "safety", "critical", "WATCHDOG tripped kill switch", f"heartbeat stale {int(now - hb)}s")
        return "tripped: stale"
    row = store.c.execute("SELECT MAX(equity) m, (SELECT equity FROM snapshots ORDER BY id DESC LIMIT 1) l FROM snapshots").fetchone()
    if row and row["m"] and row["l"] and 1 - row["l"] / row["m"] >= hard_dd:
        kill.trip(f"watchdog: drawdown {1 - row['l'] / row['m']:.1%} >= {hard_dd:.0%}")
        store.add_event(now, "safety", "critical", "WATCHDOG tripped kill switch", f"hard-stop drawdown {hard_dd:.0%}")
        return "tripped: drawdown"
    return "ok"


def run(db, kill_file, poll=60, **kw):
    st, kill = Store(db), KillSwitch(kill_file)
    while True:
        print(time.strftime("%Y-%m-%d %H:%M:%S"), check(st, kill, **kw), flush=True)
        time.sleep(poll)
