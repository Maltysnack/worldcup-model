"""Sweep the K-factor for "Friendly" matches in the shipped profile
(v4a-flat-mild) on the 16-tournament phase-3 set.

Holds all other K-factors at their v4a-flat-mild values, varies K_friendly
across a grid, computes pooled log-loss / Brier / ECE / top-pick accuracy.
Picks the empirical optimum. ~1-2 minute wallclock total.

Usage: uv run python scripts/sweep_friendly_k.py
"""
from __future__ import annotations

import time
from datetime import timedelta

import numpy as np

from worldcup.calibration import DIXON_COLES_RHO
from worldcup.comprehensive_validation import (
    TOURNAMENTS,
    actual_outcome,
    apply_layers,
)
from worldcup.db import connect
from worldcup.elo import HOME_ADVANTAGE
from worldcup.elo import run as elo_run
from worldcup.goals import expected_goals, wld_probs
from worldcup.profiles import PROFILES, resolve_min_date

BASE_PROFILE = PROFILES["v4a-flat-mild"]
K_GRID = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]


def ratings_for(k_friendly: float, kickoff):
    k_factors = dict(BASE_PROFILE.k_factors)
    k_factors["Friendly"] = float(k_friendly)
    max_d = kickoff - timedelta(days=1)
    min_d = resolve_min_date(BASE_PROFILE, anchor=kickoff)
    final_df, _ = elo_run(
        min_date=min_d, max_date=max_d,
        k_factors=k_factors, default_k=BASE_PROFILE.default_k,
        quiet=True,
    )
    return {row.team: float(row.rating) for row in final_df.itertuples(index=False)}


def per_tournament_pairs(ratings, tournament, year, kickoff):
    boosted = apply_layers(
        ratings, tournament=tournament, year=year, kickoff=kickoff,
        form=True, nt=True, host=True,
    )
    with connect(read_only=True) as con:
        df = con.execute(
            """
            SELECT home_team, away_team, neutral, home_score, away_score
            FROM matches
            WHERE tournament = ? AND EXTRACT(YEAR FROM date) = ?
              AND home_score IS NOT NULL
            ORDER BY date
            """,
            [tournament, year],
        ).fetchdf()
    P, y = [], []
    for r in df.itertuples(index=False):
        rh = boosted.get(r.home_team); ra = boosted.get(r.away_team)
        if rh is None or ra is None:
            continue
        eff_h = rh + (0.0 if r.neutral else HOME_ADVANTAGE)
        lh, la = expected_goals(eff_h - ra, neutral=bool(r.neutral))
        p_h, p_d, p_a = wld_probs(float(lh), float(la))
        P.append((p_h, p_d, p_a))
        y.append(actual_outcome(int(r.home_score), int(r.away_score)))
    return np.array(P), np.array(y)


def metrics(P, y):
    eps = 1e-12
    ll = -np.log(np.clip(P[np.arange(len(y)), y], eps, 1.0)).mean()
    oh = np.eye(3)[y]
    brier = ((P - oh) ** 2).sum(axis=1).mean()
    flat_p = np.concatenate([P[:, c] for c in range(3)])
    flat_o = np.concatenate([(y == c).astype(int) for c in range(3)])
    bins = np.linspace(0, 1, 11); w = 0; n = 0
    for i in range(10):
        lo, hi = bins[i], bins[i + 1]
        m = (flat_p >= lo) & (flat_p <= hi if i == 9 else flat_p < hi)
        k = m.sum()
        if k == 0: continue
        w += k * abs(flat_p[m].mean() - flat_o[m].mean()); n += k
    ece = w / n
    top_pick_acc = float((P.argmax(axis=1) == y).mean())
    return ll, brier, ece, top_pick_acc


def main():
    t0 = time.time()
    rows = []
    quartiles = {int(len(K_GRID) * f) for f in (0.25, 0.5, 0.75)}
    for i, k in enumerate(K_GRID):
        Ps, ys = [], []
        for tournament, year, kickoff in TOURNAMENTS:
            ratings = ratings_for(k, kickoff)
            P, y = per_tournament_pairs(ratings, tournament, year, kickoff)
            Ps.append(P); ys.append(y)
        P_all = np.concatenate(Ps); y_all = np.concatenate(ys)
        ll, br, ece, tpa = metrics(P_all, y_all)
        rows.append((k, ll, br, ece, tpa))
        if i in quartiles or i == len(K_GRID) - 1:
            print(f"[{time.time()-t0:5.1f}s] {(i+1)/len(K_GRID)*100:.0f}%", flush=True)
        print(f"  K_friendly={k:>3}  log-loss={ll:.4f}  Brier={br:.4f}  ECE={ece:.4f}  top-pick={tpa:.3f}", flush=True)

    print()
    base = next(r for r in rows if r[0] == 30)
    best_ll = min(rows, key=lambda r: r[1])
    print(f"Baseline (K=30): log-loss {base[1]:.4f}  Brier {base[2]:.4f}  ECE {base[3]:.4f}  top-pick {base[4]:.3f}")
    print(f"Best     (K={best_ll[0]}): log-loss {best_ll[1]:.4f}  Δ {best_ll[1]-base[1]:+.4f}")
    print(f"                Brier {best_ll[2]:.4f}  Δ {best_ll[2]-base[2]:+.4f}")
    print(f"                ECE   {best_ll[3]:.4f}  Δ {best_ll[3]-base[3]:+.4f}")


if __name__ == "__main__":
    main()
