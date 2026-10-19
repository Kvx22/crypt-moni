"""Research commands for `python run.py ...` (see the CLI help)."""
from __future__ import annotations

import json
import time

from .db import STAGE_ORDER, ResearchDB, research_path


def _age(ts, now):
    if not ts:
        return "never"
    s = now - ts
    return f"{s:.0f}s ago" if s < 90 else f"{s / 60:.0f}m ago" if s < 5400 else f"{s / 3600:.1f}h ago" if s < 172800 else f"{s / 86400:.0f}d ago"


def research_cmds(a, cfg, cfg_path):
    if a.cmd == "research-service":
        from .service import run_service
        return run_service(cfg)
    if a.cmd == "research-probe":
        return probe(cfg, a.team)
    if a.cmd == "pretrain":
        return pretrain_now(cfg)
    db = ResearchDB(research_path(cfg))
    now = time.time()
    if a.cmd == "research-status":
        return status(db, now)
    if a.cmd == "research-stage":
        from .stages import set_manual
        if not a.id:
            for name, st in db.stages().items():
                ev = ((db.kv_get("stage_evidence") or {}).get("stages") or {}).get(name) or {}
                print(f"  {name:<10} {st['state']:<7} since {_age(st.get('since'), now):<10} by {st.get('by') or 'default':<8} "
                      f"{ev.get('waiting') or ev.get('reason', '')}")
            return print("Switch one: research-stage NAME on|off|shadow|auto   (on = active now, auto = back under the evidence gates)")
        if a.id not in STAGE_ORDER or a.state not in ("on", "off", "shadow", "auto", "active"):
            return print(f"usage: research-stage {'|'.join(STAGE_ORDER)} on|off|shadow|auto")
        return print(set_manual(db, a.id, "active" if a.state == "on" else a.state))
    if a.cmd == "research-label":
        from .taxonomy import CATALYSTS, RISK_FLAGS
        flags = [f.strip() for f in (a.flags or "").split(",") if f.strip()]
        if not a.id or not str(a.id).isdigit() or a.catalyst not in CATALYSTS or a.direction not in (-1, 0, 1) \
                or any(f not in RISK_FLAGS for f in flags):
            return print(f"usage: research-label NOTE_ID --catalyst {{{','.join(CATALYSTS)}}} --direction -1|0|1 "
                         f"[--flags {','.join(RISK_FLAGS)}]")
        ok = db.set_label(int(a.id), a.catalyst, a.direction, flags, by="human")
        return print(f"note #{a.id} relabelled: {a.catalyst}, direction {a.direction:+d}, flags {flags or 'none'}" if ok else "no such note")
    if a.cmd == "research-train":
        from .outcome import train_outcome_model
        from .service import Service
        svc = Service(cfg, db=db, plan=False, log=lambda *x: None)
        r1 = svc.train_extractor(now)
        print(f"extractor v{r1['version']}: {'PROMOTED' if r1['promoted'] else 'kept the champion'} ({r1['reason']})")
        r2 = train_outcome_model(db, now)
        return print("outcome model:", json.dumps({k: v for k, v in (r2 or {}).items() if not isinstance(v, (list, dict))}, default=str))


def status(db, now):
    h = db.kv_get("health_now") or {}
    lease = db.lease()
    print(f"Research service: {'running, last beat ' + _age(lease['heartbeat'], now) if lease else 'not running'}"
          f"{' (owner ' + lease['owner'] + ')' if lease else ''}")
    c = h.get("counts") or {}
    print(f"Workers: {c.get('active', 0)} active, {c.get('standby', 0)} standby, {c.get('failed', 0)} failed, "
          f"{c.get('grown', 0)} grown, {c.get('total', 0)} total")
    for team, t in sorted((h.get("teams") or {}).items()):
        print(f"  {team:<10} active {t.get('active', 0)}  standby {t.get('standby', 0)}  failed {t.get('failed', 0)}  "
              f"last ok {_age(t.get('last_ok'), now)}")
    failed = [s for s in h.get("slots", []) if s.get("state") == "failed"]
    for s in failed[:10]:
        print(f"  FAILED {s.get('adapter')}: {s.get('err', '')[:120]}")
    c24 = h.get("counts24") or {}
    print(f"Last 24h: {c24.get('items', 0)} items, {c24.get('notes', 0)} notes; {c24.get('open_requests', 0)} open requests; "
          f"{c24.get('discovered', 0)} discovered names")
    print("Stages: " + ", ".join(f"{k} {v['state']}" for k, v in db.stages().items()))
    for m in db.models():
        if m["role"] == "champion":
            mm = m["metrics"] or {}
            print(f"Model {m['name']} v{m['version']} (champion): " + ", ".join(f"{k} {v:.3f}" for k, v in mm.items() if isinstance(v, float))[:200])
    print("\nLatest notes:")
    for n in db.notes(12):
        fl = f" [{', '.join(n.risk_flags)}]" if n.risk_flags else ""
        print(f"  #{n.id:<6} {n.symbol:<9} {n.catalyst:<16} {n.direction:+d} {n.confidence:.2f} {n.label_by:<6} {n.publisher[:20]:<20} "
              f"{n.title[:70]}{fl}")
    ev = db.events(8)
    if ev:
        print("\nRecent events:")
        for e in ev:
            print(f"  {time.strftime('%m-%d %H:%M', time.gmtime(e['ts']))} {e['severity']:<8} {e['title'][:100]}")


def probe(cfg, team=None):
    """Fetch every live source once (no database writes) and say what answered. Run it on the PC
    that runs the boards: it is the only way to see which sites your network and location allow."""
    from ..feeds import stock_symbols
    from . import sources
    from .aliases import Aliases
    from .http import DEFAULT_UA, Http
    from .service import service_cfg
    from .sources.base import FetchContext
    rc = service_cfg(cfg)
    subs = {s: "crypto" for s in cfg.get("universe", [])}
    subs.update({s: "stock" for s in stock_symbols(cfg)})
    if not any(a == "stock" for a in subs.values()):
        subs.update({"SPY": "stock", "NVDA": "stock"})   # so the stock sources get probed too
    al = Aliases(subs)
    http = Http(user_agent=DEFAULT_UA + (f" contact:{rc['contact']}" if rc.get("contact") else ""),
                sec_contact=rc.get("contact"))
    ok = bad = 0
    for ad in sources.catalog(rc):
        if team and ad.team != team:
            continue
        syms = {s: x for s, x in subs.items() if ad.covers(x)}
        if ad.per_symbol:
            pick = [s for s, x in syms.items() if x == "stock"][:1] + [s for s, x in syms.items() if x == "crypto"][:1]
            syms = {s: syms[s] for s in pick}
        ctx = FetchContext(http=http, now=time.time(), symbols=syms, state={}, cfg=rc, names={s: al.names(s) for s in syms})
        t0 = time.time()
        try:
            items = ad.fetch(ctx)
            ok += 1
            print(f"  OK    {ad.team:<10} {ad.role:<8} {ad.name:<20} {len(items):>3} items  {time.time() - t0:4.1f}s"
                  + (f"  e.g. {items[0].title[:60]!r}" if items else ""))
        except Exception as e:  # noqa: BLE001
            bad += 1
            print(f"  FAIL  {ad.team:<10} {ad.role:<8} {ad.name:<20} {type(e).__name__}: {str(e)[:90]}")
    if sources.IMPORT_ERRORS:
        print("  modules that did not load:", sources.IMPORT_ERRORS)
    print(f"{ok} sources answered, {bad} failed. Failed primaries are covered by their standbys; a failing standby too leaves a gap "
          f"the health panel will show. SEC asks for a contact in the User-Agent: set research.contact in the config.")


def pretrain_now(cfg):
    from ..__main__ import build
    from ..store import Store
    from .pretrain import PretrainJob, apply
    st = Store(cfg["db"], readonly=True)
    hb = st.kv_get("heartbeat")
    if hb and time.time() - hb < 600:
        return print("This board is running: it pretrains on its own (fresh start, then weekly) and would overwrite a "
                     "result from here. Stop it first, or let it do the job.")
    eng = build(cfg)
    hours = (cfg.get("pretrain") or {}).get("hours", 2160)
    print(f"Pretraining the advisors on up to {hours} hours of history (this can take a few minutes)...", flush=True)
    job = PretrainJob(eng, hours=hours)
    job.start(time.time())
    while True:
        res = job.poll()
        if res is not None:
            break
        time.sleep(2)
    apply(eng, res, time.time())
    eng.s.kv_set("pretrain_ts", time.time())
    print(json.dumps(eng.s.kv_get("pretrain"), indent=2, default=str)[:3000])
