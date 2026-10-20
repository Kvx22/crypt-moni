"""File bridge between the Risk Board and a Claude task that has no shell (for example Cowork).

Runs on the same machine as the board, as plain Python. It never talks to a broker and never approves
anything: it only (1) writes the orders YOU already approved in the dashboard to data/claude_inbox.json,
and (2) applies the results Claude writes to data/claude_results.json (acks and the balance snapshot),
after checking them.

  python claude_bridge.py --config config.robinhood.json once     one cycle, then exit (Task Scheduler)
  python claude_bridge.py --config config.robinhood.json loop     a cycle every 30 s until closed

Claude's side: read data/claude_inbox.json, place the orders through the Robinhood connector, write
data/claude_results.json (its format is RESULTS_EXAMPLE below).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MAX_FILE = 64 * 1024
SYMBOL = re.compile(r"^[A-Z0-9][A-Z0-9.\-]{0,14}$")
RESULTS_MAX_AGE_S = 3600
SNAPSHOT_REFRESH_H = 12.0
STALE_AFTER_MIN = 10
RESULTS_EXAMPLE = {
    "run_id": "any unique text for this run",
    "written_at": 1790000000,
    "snapshot": {"cash": 43.57, "positions": {"BTC-USD": 0.00018116, "XRP-USD": 5}, "other_value": 0},
    "orders": [{"outbox_id": 12, "status": "sent", "order_id": "approval or order id", "detail": "awaiting Robinhood app approval"}],
}


def _num(x):
    if isinstance(x, bool) or not isinstance(x, (int, float)) or x != x or x in (float("inf"), float("-inf")):
        return None
    return float(x)


def paths(cfg):
    base = os.path.dirname(cfg["db"]) or "data"
    ex = cfg.get("execution", {})
    return dict(inbox=os.path.join(base, "claude_inbox.json"), results=os.path.join(base, "claude_results.json"),
                state=os.path.join(base, "claude_bridge_state.json"), log=os.path.join(base, "claude_bridge.log"),
                snapshot=ex.get("snapshot", os.path.join(base, "robinhood_snapshot.json")))


def _read_json(path, default=None):
    try:
        if os.path.getsize(path) > MAX_FILE:
            return default
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def _write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)   # readers never see half a file


def ref_id_for(client_order_id):
    """Same order -> same UUID, so a retry can never create a second order at the broker."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, "riskboard:" + str(client_order_id)))


def build_inbox(cfg, st, kill, now, snapshot_path, offered_before=()):
    from cryptomoni.safety import SafetyConfig
    sc = SafetyConfig.from_dict(cfg.get("safety", {}))
    st.expire_outbox(now)
    killed = kill.engaged()
    rows = [] if killed else st.outbox(200, "approved")
    orders = []
    for r in reversed(rows):
        cap = sc.max_per_trade_usd or 0
        orders.append(dict(outbox_id=r["id"], symbol=r["symbol"], asset="crypto" if "-" in r["symbol"] else "stock",
                           side=r["side"], notional_usd=round(min(r["notional"], cap), 2), ref_price=r["ref_price"],
                           max_move=sc.price_band_pct, client_order_id=r["client_order_id"],
                           ref_id=ref_id_for(r["client_order_id"]), expires_in_min=round((r["expires_ts"] - now) / 60)))
    snap = _read_json(snapshot_path) or {}
    age_h = (now - snap["ts"]) / 3600 if isinstance(snap.get("ts"), (int, float)) else None
    return dict(generated_at=now, stale_after_min=STALE_AFTER_MIN, broker=cfg.get("execution", {}).get("broker"),
                kill_switch=killed, max_per_trade_usd=sc.max_per_trade_usd,
                snapshot_age_h=None if age_h is None else round(age_h, 2),
                snapshot_needed=age_h is None or age_h > SNAPSHOT_REFRESH_H or bool(orders),
                rule="Place each order only if the live price is within max_move of ref_price; stocks only while the US market "
                     "is open; send ref_id exactly as given; then write claude_results.json. Never place anything not listed here.",
                orders=orders)


def apply_results(cfg, st, res, now, offered, done_runs, snapshot_path):
    """Validate and apply one results document. Returns (messages, accepted_order_ids, applied_snapshot)."""
    from cryptomoni.models import Fill
    msgs, accepted, applied_snap = [], [], False
    if not isinstance(res, dict):
        return ["results: not a JSON object, ignored"], accepted, applied_snap
    run_id = str(res.get("run_id") or "")[:100]
    if not run_id:
        return ["results: no run_id, ignored"], accepted, applied_snap
    if run_id in done_runs:
        return [f"results {run_id}: already applied"], accepted, applied_snap
    ts = _num(res.get("written_at"))
    if ts is None or now - ts > RESULTS_MAX_AGE_S or ts - now > 300:
        return [f"results {run_id}: written_at missing, stale or in the future, ignored"], accepted, applied_snap

    snap = res.get("snapshot")
    if isinstance(snap, dict):
        cash, other = _num(snap.get("cash")), _num(snap.get("other_value", 0.0))
        pos_in = snap.get("positions") if isinstance(snap.get("positions"), dict) else None
        ok = cash is not None and 0 <= cash <= 1e7 and other is not None and 0 <= other <= 1e8 and pos_in is not None
        pos = {}
        for k, v in (pos_in or {}).items():
            q = _num(v)
            if not isinstance(k, str) or not SYMBOL.match(k.upper()) or q is None or not (0 <= q <= 1e9):
                ok = False
                break
            pos[k.upper()] = q
        if ok:
            _write_json(snapshot_path, dict(ts=now, cash=cash, positions=pos, other_value=other, source="Claude Robinhood connector (bridge)"))
            applied_snap = True
            msgs.append(f"snapshot written: cash ${cash:,.2f}, {len(pos)} position(s)")
        else:
            msgs.append("snapshot rejected: values missing or out of range")

    for o in res.get("orders") or []:
        if not isinstance(o, dict):
            continue
        oid = o.get("outbox_id")
        status = o.get("status")
        if isinstance(oid, bool) or not isinstance(oid, int) or status not in ("sent", "failed"):
            msgs.append(f"order {oid!r}: bad entry, ignored")
            continue
        if oid not in offered:
            msgs.append(f"#{oid}: was never offered to Claude, ignored")
            continue
        r = st.outbox_get(oid)
        if not r or r["status"] != "approved":
            msgs.append(f"#{oid}: not in the approved state ({r['status'] if r else 'missing'}), ignored")
            continue
        order_id = str(o.get("order_id") or "")[:100]
        detail = str(o.get("detail") or "")[:300]
        if status == "sent" and not order_id:
            msgs.append(f"#{oid}: 'sent' needs an order_id, ignored")
            continue
        st.outbox_set(oid, status, now, broker_order_id=order_id, detail=detail)
        if status == "sent":
            st.add_trade(Fill(r["symbol"], r["side"], r["notional"], 0, r["ref_price"], now, "sent", order_id),
                         f"#{oid} approved, placed by Claude (bridge)")
        st.add_event(now, "outbox", "info" if status == "sent" else "warn", f"#{oid} {status} via Claude bridge ({r['broker']})",
                     detail or order_id)
        accepted.append(oid)
        msgs.append(f"#{oid} marked {status}")
    done_runs.append(run_id)
    return msgs, accepted, applied_snap


def cycle(cfg, now=None, log=print):
    from cryptomoni.safety import KillSwitch
    from cryptomoni.store import Store
    now = time.time() if now is None else now
    p = paths(cfg)
    kill = KillSwitch(cfg.get("safety", {}).get("kill_file", "KILL"))
    st = Store(cfg["db"])
    try:
        state = _read_json(p["state"], {}) or {}
        offered = set(int(x) for x in state.get("offered", []) if isinstance(x, int))
        done_runs = list(state.get("done_runs", []))[-50:]
        if os.path.exists(p["results"]):
            res = _read_json(p["results"])
            if res is None:   # half-written (or unreadable): try again next cycle, give up after ten minutes
                try:
                    if time.time() - os.path.getmtime(p["results"]) > 600:
                        os.replace(p["results"], p["results"].replace(".json", ".bad.json"))
                        log("[bridge] results file unreadable for 10 minutes, set aside as .bad.json")
                except OSError:
                    pass
            else:
                msgs, _acc, _snap = apply_results(cfg, st, res, now, offered, done_runs, p["snapshot"])
                for m in msgs:
                    log(f"[bridge] {m}")
                try:
                    os.replace(p["results"], p["results"].replace(".json", ".done.json"))
                except OSError:
                    pass
        inbox = build_inbox(cfg, st, kill, now, p["snapshot"])
        offered |= {o["outbox_id"] for o in inbox["orders"]}
        _write_json(p["inbox"], inbox)
        keep = [r["id"] for r in st.outbox(200, "approved")]
        _write_json(p["state"], dict(offered=sorted(set(keep) | {o["outbox_id"] for o in inbox["orders"]}), done_runs=done_runs[-50:]))
        return inbox
    finally:
        st.c.close()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["once", "loop"])
    ap.add_argument("--config", default="config.robinhood.json")
    ap.add_argument("--every", type=float, default=30.0)
    a = ap.parse_args(argv)
    with open(a.config) as f:
        cfg = json.load(f)
    if a.mode == "once":
        inbox = cycle(cfg)
        print(f"inbox written: {len(inbox['orders'])} approved order(s), kill switch {'ON' if inbox['kill_switch'] else 'off'}")
        return 0
    print(f"bridge running every {a.every:.0f}s for {a.config} (Ctrl+C to stop)")
    while True:
        try:
            cycle(cfg)
        except Exception as ex:  # noqa: BLE001  (keep running; the inbox goes stale and Claude stops by itself)
            print(f"[bridge] cycle failed: {ex!r}")
        time.sleep(max(5.0, a.every))


if __name__ == "__main__":
    sys.exit(main() or 0)
