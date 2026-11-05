"""Sweep the HOME_ADVANTAGE Elo constant against the 16-tournament
phase-3 set. Mirrors sweep_host_bonus.py. Picks the empirical
log-loss optimum on non-neutral matches as well as the full set.

Usage: uv run python scripts/sweep_home_advantage.py
"""
from __future__ import annotations

import time
from datetime import timedelta

import numpy as np

from worldcup.comprehensive_validation import TOURNAMENTS, actual_outcome
from worldcup.db import connect
from worldcup.elo import run as elo_run
from worldcup.form import compute_form_bumps
from worldcup.goals import expected_goals, wld_probs
from worldcup.host import TOURNAMENT_HOSTS
from worldcup.profiles import PROFILES, resolve_min_date
from worldcup.trophies import nt_trophy_bonus

BASE_PROFILE = PROFILES["v4a-flat-mild"]
HA_GRID = [0.0, 15.0, 25.0, 35.0, 50.0, 65.0, 80.0, 100.0]


def ratings_at(kickoff, tournament, year):
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
        r += hosts.get(t, 0.0)
        out[t] = r
    return out


def pairs(ratings, tournament, year, ha: float):
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
    P, y, neu = [], [], []
    for r in df.itertuples(index=False):
        rh = ratings.get(r.home_team); ra = ratings.get(r.away_team)
        if rh is None or ra is None:
            continue
        eff_h = rh + (0.0 if r.neutral else ha)
        lh, la = expected_goals(eff_h - ra, neutral=bool(r.neutral))
        p_h, p_d, p_a = wld_probs(float(lh), float(la))
        P.append((p_h, p_d, p_a))
        y.append(actual_outcome(int(r.home_score), int(r.away_score)))
        neu.append(bool(r.neutral))
    return np.array(P), np.array(y), np.array(neu)


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
    # home gap: mean P(home_win) - empirical home_win rate
    home_gap = float(P[:, 0].mean() - (y == 0).mean())
    return ll, brier, ece, home_gap, len(y)


def main():
    t0 = time.time()
    # cache ratings once per tournament (ha-independent)
    print("Computing per-tournament ratings (once)...", flush=True)
    cache = {}
    for tournament, year, kickoff in TOURNAMENTS:
        cache[(tournament, year)] = ratings_at(kickoff, tournament, year)
    print(f"  done in {time.time()-t0:.1f}s", flush=True)

    rows = []
    for ha in HA_GRID:
        Ps, ys, neus = [], [], []
        for tournament, year, kickoff in TOURNAMENTS:
            P, y, neu = pairs(cache[(tournament, year)], tournament, year, ha)
            Ps.append(P); ys.append(y); neus.append(neu)
        P_all = np.concatenate(Ps); y_all = np.concatenate(ys); neu_all = np.concatenate(neus)
        ll_all, br_all, ece_all, gap_all, n_all = metrics(P_all, y_all)
        nonneu = ~neu_all
        if nonneu.sum() > 0:
            ll_nn, br_nn, ece_nn, gap_nn, n_nn = metrics(P_all[nonneu], y_all[nonneu])
        else:
            ll_nn = br_nn = ece_nn = gap_nn = float("nan"); n_nn = 0
        rows.append((ha, ll_all, br_all, ece_all, gap_all, ll_nn, gap_nn, n_nn))
        print(
            f"  HA={ha:>5.1f}  all: LL={ll_all:.4f} Brier={br_all:.4f} ECE={ece_all:.4f} home_gap={gap_all:+.3f}"
            f"  | non-neu (n={n_nn}): LL={ll_nn:.4f} home_gap={gap_nn:+.3f}",
            flush=True,
        )

    print()
    base = next(r for r in rows if r[0] == 50.0)
    best_all = min(rows, key=lambda r: r[1])
    best_nn = min(rows, key=lambda r: r[5])
    zero_gap = min(rows, key=lambda r: abs(r[6]))
    print(f"Baseline HA=50: LL_all={base[1]:.4f} home_gap_all={base[4]:+.3f} LL_nn={base[5]:.4f} home_gap_nn={base[6]:+.3f}")
    print(f"Best LL_all     -> HA={best_all[0]:>5.1f}  LL_all={best_all[1]:.4f}  Δ={best_all[1]-base[1]:+.4f}")
    print(f"Best LL_nn      -> HA={best_nn[0]:>5.1f}  LL_nn={best_nn[5]:.4f}  Δ={best_nn[5]-base[5]:+.4f}")
    print(f"Min |home_gap_nn|-> HA={zero_gap[0]:>5.1f}  home_gap_nn={zero_gap[6]:+.3f}")
    print(f"\nTotal: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
