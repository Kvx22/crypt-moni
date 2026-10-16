"""Allocation for the paper book: how much is invested, and in which coins.

E (invested fraction) = exposure-board vote, shrunk toward neutral when the
regime is uncertain (doubt only ever lowers exposure), vol-targeted on BTC,
capped by max_invested and the safety breaker, then rate-limited (de-risking
twice as fast as re-risking) with a deadband against churn.

Which coins: the pick desk's BUY and WATCH names, weighted by board score,
each under its position cap. Whatever the picks can't absorb goes to the BTC
core (unless the desk rates BTC AVOID or the chair vetoed it), the rest stays cash.
"""
from __future__ import annotations

import numpy as np


class Allocator:
    def __init__(self, max_invested=0.90, neutral_exposure=0.50, target_vol=0.55, max_step=0.15,
                 min_conf_shrink=0.35, deadband=0.04, max_names=5, core="BTC-USD",
                 caps=None, default_cap=0.15, stickiness=0.3):
        self.max_invested, self.neutral, self.target_vol = max_invested, neutral_exposure, target_vol
        self.max_step, self.min_conf_shrink, self.deadband = max_step, min_conf_shrink, deadband
        self.max_names, self.core = max_names, core
        self.caps = caps or {"BTC-USD": 0.50, "ETH-USD": 0.35}
        self.default_cap = default_cap
        self.stickiness = stickiness
        self.last_notes = []

    def cap(self, sym):
        return self.caps.get(sym, self.default_cap)

    def invested_fraction(self, board_vote, confidence, realised_vol_ann, safety_cap, prev_e):
        notes = []
        k = float(np.clip((confidence - 0.2) / 0.6, self.min_conf_shrink, 1.0))
        e0 = e = k * board_vote + (1 - k) * min(board_vote, self.neutral)   # doubt pulls DOWN, never up
        if e0 < board_vote - 0.01:
            notes.append(f"doubt (confidence {confidence:.0%}) trimmed {board_vote:.0%} -> {e0:.0%}")
        if realised_vol_ann and realised_vol_ann > self.target_vol:
            ev = self.target_vol / realised_vol_ann
            if ev < e - 0.01:
                notes.append(f"vol target: BTC vol {realised_vol_ann:.0%} > {self.target_vol:.0%} -> cap {ev:.0%}")
            e = min(e, ev)
        if safety_cap < e - 0.01:
            notes.append(f"SAFETY breaker cap {safety_cap:.0%} binds")
        e = min(e, self.max_invested, safety_cap)
        pre = e
        if prev_e is not None:
            e = float(np.clip(e, prev_e - 2 * self.max_step, prev_e + self.max_step))
            e = min(e, safety_cap)
            if abs(e - prev_e) < self.deadband and prev_e <= safety_cap:
                e = prev_e  # ignore small changes (never blocks a de-risk to the cap)
            if abs(e - pre) > 0.01 and abs(e - prev_e) > 0.005:
                notes.append(f"rate limit: moving {prev_e:.0%} -> {e:.0%} instead of {pre:.0%}")
        self.last_notes = notes
        return float(np.clip(e, 0.0, 1.0))

    def targets(self, e, coins, held=()):
        """coins: desk decisions [{symbol, score, verdict}]; held: symbols in the book now.
        Returns (weights of total equity, notes). Held coins get a small score bonus
        (stickiness) so the book doesn't churn every time two picks swap places."""
        notes = []
        held = set(held)
        sc = {c["symbol"]: c["score"] + (self.stickiness if c["symbol"] in held else 0.0) for c in coins}
        picks = sorted([c for c in coins if c["verdict"] in ("BUY", "WATCH")
                        or (c["symbol"] in held and c["verdict"] == "NEUTRAL" and sc[c["symbol"]] > 0)],
                       key=lambda c: -sc[c["symbol"]])[: self.max_names]
        weights = {}
        tot = sum(sc[c["symbol"]] for c in picks)
        left = e
        for c in picks:
            w = min(e * sc[c["symbol"]] / tot, self.cap(c["symbol"]))
            weights[c["symbol"]] = w
            left -= w
        core = next((c for c in coins if c["symbol"] == self.core), None)
        core_ok = core is not None and core["verdict"] not in ("AVOID", "VETOED")
        if left > 0.005 and core_ok:
            add = min(left, self.cap(self.core) - weights.get(self.core, 0.0))
            if add > 0.005:
                weights[self.core] = weights.get(self.core, 0.0) + add
                left -= add
                notes.append(f"{add:.0%} the picks could not absorb went to the {self.core.split('-')[0]} core")
        if not picks:
            notes.append("no coin rated BUY or WATCH by the desk")
        if left > 0.005:
            notes.append(f"{left:.0%} of the {e:.0%} target stays in cash (caps or no takers)")
        return {k: v for k, v in weights.items() if v > 1e-4}, notes
