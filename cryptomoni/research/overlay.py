"""How research reaches a desk's decision (crypto hourly, stocks daily), in one place.

Every decision round:
  1. read the research view for the desk's names and turn it into the research matrix R;
  2. the advisors' research desk scores R (each advisor with their own learned weights);
  3. compute what research WOULD do: the adjusted scores (signals), the capped penalties,
     halts, and which advisors like a name nobody has researched yet (thin evidence);
  4. apply only the pieces that are switched on (stages), log the rest as shadow rows;
  5. ask the research service for more evidence where an advisor likes a thin name.
Learning: R is stored with the round's pending calls, so the research desk learns from
the same graded outcomes as the pick desk, in shadow mode too.
Nothing here can block a sell, and the only hard stop is a halted market (no buys).
"""
from __future__ import annotations

import numpy as np

from ..advisors import ADVISORS, zscore
from .rdesk import ResearchDesk, overlay, research_matrix, thin_requests


class BoardResearch:
    def __init__(self, link, store, key="rdesk", asset="crypto", horizon="4h", desk_kwargs=None):
        self.link, self.s, self.key, self.asset, self.horizon = link, store, key, asset, horizon
        self.rc = link.rc
        self.rdesk = ResearchDesk(**(desk_kwargs or {}))
        st = store.kv_get(key)
        if st:
            try:
                self.rdesk.load(st)
            except (KeyError, ValueError, TypeError):
                pass   # an incompatible saved state starts the research desk afresh
        self.last = {}            # symbol -> compact research dict (for the board and the outbox)
        self.halted = set()

    def stages(self, now):
        return self.link.stages(now)

    # ---- learning from graded rounds --------------------------------------------------------
    def learn(self, pending, idx, fwd):
        R = pending.get("R")
        if R is None or len(idx) < 4:
            return
        try:
            self.rdesk.learn(np.asarray(R, float)[idx], np.asarray(fwd, float))
        except (ValueError, IndexError):
            return
        self.s.kv_set(self.key, self.rdesk.state(), commit=False)

    # ---- one decision round -------------------------------------------------------------------
    def apply(self, now, desk, S, syms, tiers, closes, bench_close, entry):
        """Returns the coins list (like PickDesk.decide). `entry` is the pending dict to which R is added."""
        views = self.link.views(syms, now) if self.rc.get("enabled", True) else {}
        R = research_matrix(views, syms)
        entry["R"] = R.round(4).tolist()
        Z = np.array([zscore(s) for s in S])
        zr = self.rdesk.zr(R)
        thin = np.array([bool(views.get(s, {}).get("thin", True)) for s in syms])
        pen = np.array([min(float(views.get(s, {}).get("penalty", 0.0) or 0.0), self.rc["penalty_cap"]) for s in syms])
        halted = {s for s in syms if views.get(s, {}).get("halted")}
        Z2 = overlay(Z, zr, self.rc["weight"], thin, self.rc["thin_discount"], desk.buy)
        st = self.stages(now)
        sig_on, pen_on = st.get("signals") == "active", st.get("penalties") == "active"
        coins = desk.decide_z(Z2 if sig_on else Z, syms, tiers, pen if pen_on else None)
        base = desk.w @ Z
        adj = desk.w @ Z2          # what the research signals alone would make of it (the penalty is logged apart)
        self.halted = halted if pen_on else set()
        rows, self.last = [], {}
        for i, (c, sym) in enumerate(zip(coins, syms)):
            v = views.get(sym) or {}
            if pen_on and sym in halted and c["verdict"] in ("BUY", "WATCH"):
                c["verdict"] = "HALTED"
            r = dict(n=int(v.get("n_independent", 0) or 0), kinds=v.get("kinds", []), score=round(float(v.get("score", 0.0) or 0.0), 3),
                     penalty=round(float(pen[i]), 3), halted=sym in halted, thin=bool(thin[i]),
                     flags=[f.get("flag") for f in (v.get("flags") or [])][:3], adj=round(float(adj[i]), 3), base=round(float(base[i]), 3),
                     applied=dict(signals=sig_on, penalties=pen_on),
                     top=[dict(title=t.get("title", "")[:140], url=t.get("url", ""), publisher=t.get("publisher", ""))
                          for t in (v.get("top") or [])[:3]])
            c["research"] = r
            self.last[sym] = dict(r, view=v)
            detail = dict(price0=closes.get(sym), bench0=bench_close, horizon=self.horizon,
                          verdict=c["verdict"], verdict_adj=desk.verdict(float(adj[i] - pen[i]), tiers[i]), penalty=r["penalty"],
                          thin=r["thin"], n=r["n"], zr=round(float(zr[:, i].mean()), 3))
            rows.append((now, sym, self.asset, "score", float(base[i]), float(adj[i]), detail))
            if pen[i] > 0.01:
                rows.append((now, sym, self.asset, "penalty", float(base[i]), float(base[i] - pen[i]),
                             dict(detail, horizon="1d", flags=r["flags"])))
        self.link.shadow(rows)
        reqs = thin_requests(Z, syms, views, buy=desk.buy, thin_sources=self.rc["thin_sources"])
        if reqs:
            self.link.request_many([dict(symbol=q["symbol"], asset=self.asset, advisor=ADVISORS[q["advisor_index"]].key,
                                         question=q["question"], priority=float(q["z"])) for q in reqs], now)
        return coins

    def view(self):
        """For the dashboard: what each advisor has learned about research."""
        d = self.rdesk
        try:
            share, drift = d.learned_share(), d.drift()
        except Exception:  # noqa: BLE001
            share, drift = [0.0] * len(ADVISORS), [0.0] * len(ADVISORS)
        return dict(rounds=getattr(d, "rounds", 0), advisors=[dict(key=a.key, learned_share=float(share[i]), drift=float(drift[i]),
                                                                   ic=float(getattr(d, "ic", np.zeros(len(ADVISORS)))[i]))
                                                              for i, a in enumerate(ADVISORS)])
