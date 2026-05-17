"""Per-match deterministic comparison of v4a-flat-mild + full vs
v4a-flat-mild + elo_form_nt on the phase-3 16-tournament set, under
current constants (post-HA-halving and T-in-sampler). No MC needed
for the layer-default decision: layers differ only by host bonus.

Usage: uv run python scripts/compare_layers_full_vs_nt.py
"""
from __future__ import annotations
import time
from datetime import timedelta

import numpy as np

from worldcup.comprehensive_validation import TOURNAMENTS, actual_outcome
from worldcup.db import connect
from worldcup.elo import HOME_ADVANTAGE, run as elo_run
from worldcup.form import compute_form_bumps
from worldcup.goals import expected_goals, wld_probs
from worldcup.host import TOURNAMENT_HOSTS
from worldcup.profiles import PROFILES, resolve_min_date
from worldcup.trophies import nt_trophy_bonus

BASE = PROFILES["v4a-flat-mild"]


def ratings(kickoff, tournament, year, with_host: bool):
    max_d = kickoff - timedelta(days=1)
    min_d = resolve_min_date(BASE, anchor=kickoff)
    final_df, _ = elo_run(min_date=min_d, max_date=max_d,
                          k_factors=BASE.k_factors, default_k=BASE.default_k, quiet=True)
    raw = {row.team: float(row.rating) for row in final_df.itertuples(index=False)}
    fb = compute_form_bumps(as_of_date=kickoff)
    hosts = TOURNAMENT_HOSTS.get((tournament, year), {}) if with_host else {}
    out = {}
    for t, r in raw.items():
        r += fb.get(t, 0.0)
        r += nt_trophy_bonus(t, as_of_year=year)
        if with_host: r += hosts.get(t, 0.0)
        out[t] = r
    return out


def eval_pairs(rt, tournament, year):
    with connect(read_only=True) as con:
        df = con.execute(
            "SELECT home_team, away_team, neutral, home_score, away_score "
            "FROM matches WHERE tournament=? AND EXTRACT(YEAR FROM date)=? "
            "AND home_score IS NOT NULL ORDER BY date", [tournament, year]).fetchdf()
    P, y = [], []
    for r in df.itertuples(index=False):
        rh = rt.get(r.home_team); ra = rt.get(r.away_team)
        if rh is None or ra is None: continue
        eff_h = rh + (0.0 if r.neutral else HOME_ADVANTAGE)
        lh, la = expected_goals(eff_h - ra, neutral=bool(r.neutral))
        ph, pd, pa = wld_probs(float(lh), float(la))
        P.append((ph, pd, pa))
        y.append(actual_outcome(int(r.home_score), int(r.away_score)))
    return np.array(P), np.array(y)


def metrics(P, y):
    eps = 1e-12
    ll = -np.log(np.clip(P[np.arange(len(y)), y], eps, 1.0)).mean()
    oh = np.eye(3)[y]
    brier = ((P - oh) ** 2).sum(axis=1).mean()
    return ll, brier, float((P.argmax(axis=1) == y).mean())


def main():
    t0 = time.time()
    for label, with_host in [("elo_form_nt", False), ("full", True)]:
        Ps, ys = [], []
        for tournament, year, kickoff in TOURNAMENTS:
            rt = ratings(kickoff, tournament, year, with_host)
            P, y = eval_pairs(rt, tournament, year)
            Ps.append(P); ys.append(y)
        P_all = np.concatenate(Ps); y_all = np.concatenate(ys)
        ll, br, tp = metrics(P_all, y_all)
        print(f"{label:>12}  N={len(y_all)}  LL={ll:.5f}  Brier={br:.5f}  top-pick={tp:.3f}")
    print(f"\n{time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
