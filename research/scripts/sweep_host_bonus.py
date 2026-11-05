"""Sweep a scaling factor on the host-bonus constants in host.py
(HOST_BONUS_SOLO=60, _PRIMARY=50, _SECONDARY=30) on the 16-tournament
phase-3 set. Holds solo/primary/secondary ratios fixed; varies the
overall magnitude. Picks the empirical optimum.

Usage: uv run python scripts/sweep_host_bonus.py
"""
from __future__ import annotations

import time
from datetime import timedelta

import numpy as np

from worldcup.comprehensive_validation import TOURNAMENTS, actual_outcome
from worldcup.db import connect
from worldcup.elo import HOME_ADVANTAGE
from worldcup.elo import run as elo_run
from worldcup.form import compute_form_bumps
from worldcup.goals import expected_goals, wld_probs
from worldcup.host import TOURNAMENT_HOSTS
from worldcup.profiles import PROFILES, resolve_min_date
from worldcup.trophies import nt_trophy_bonus

BASE_PROFILE = PROFILES["v4a-flat-mild"]
SCALE_GRID = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]


def ratings_with_host_scale(kickoff, tournament, year, scale: float):
    max_d = kickoff - timedelta(days=1)
    min_d = resolve_min_date(BASE_PROFILE, anchor=kickoff)
    final_df, _ = elo_run(
        min_date=min_d, max_date=max_d,
        k_factors=BASE_PROFILE.k_factors, default_k=BASE_PROFILE.default_k,
        quiet=True,
    )
    raw = {row.team: float(row.rating) for row in final_df.itertuples(index=False)}
    form_bumps = compute_form_bumps(as_of_date=kickoff)
    hosts = TOURNAMENT_HOSTS.get((tournament, year), {})
    out = {}
    for t, r in raw.items():
        r += form_bumps.get(t, 0.0)
        r += nt_trophy_bonus(t, as_of_year=year)
        r += hosts.get(t, 0.0) * scale
        out[t] = r
    return out


def per_tournament_pairs(ratings, tournament, year):
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
        rh = ratings.get(r.home_team); ra = ratings.get(r.away_team)
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
    quartiles = {int(len(SCALE_GRID) * f) for f in (0.25, 0.5, 0.75)}
    for i, scale in enumerate(SCALE_GRID):
        Ps, ys = [], []
        for tournament, year, kickoff in TOURNAMENTS:
            ratings = ratings_with_host_scale(kickoff, tournament, year, scale)
            P, y = per_tournament_pairs(ratings, tournament, year)
            Ps.append(P); ys.append(y)
        P_all = np.concatenate(Ps); y_all = np.concatenate(ys)
        ll, br, ece, tpa = metrics(P_all, y_all)
        rows.append((scale, ll, br, ece, tpa))
        if i in quartiles or i == len(SCALE_GRID) - 1:
            print(f"[{time.time()-t0:5.1f}s] {(i+1)/len(SCALE_GRID)*100:.0f}%", flush=True)
        eff_solo = int(60 * scale)
        print(f"  scale={scale:.2f}  solo={eff_solo:>3} Elo   log-loss={ll:.4f}  Brier={br:.4f}  ECE={ece:.4f}  top-pick={tpa:.3f}", flush=True)

    print()
    base = next(r for r in rows if r[0] == 1.0)
    best_ll = min(rows, key=lambda r: r[1])
    print(f"Baseline (scale=1.0, solo=60): log-loss {base[1]:.4f}  Brier {base[2]:.4f}  ECE {base[3]:.4f}")
    print(f"Best     (scale={best_ll[0]}, solo={int(60*best_ll[0])}): log-loss {best_ll[1]:.4f}  Δ {best_ll[1]-base[1]:+.4f}")
    print(f"                Brier {best_ll[2]:.4f}  Δ {best_ll[2]-base[2]:+.4f}")
    print(f"                ECE   {best_ll[3]:.4f}  Δ {best_ll[3]-base[3]:+.4f}")


if __name__ == "__main__":
    main()
