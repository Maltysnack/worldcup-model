"""Sweep an altitude bonus: extra Elo added to the home team when the
match is played in their own country at a venue ≥1500m elevation.

Hypothesis: acclimatised home teams have a real edge at altitude that
the base home-advantage doesn't capture. Examples include Bolivia in
La Paz, Ecuador in Quito, Mexico in Mexico City.

Evaluates pooled log-loss across the 868 altitude matches in the DB.

Usage: uv run python scripts/sweep_altitude_bonus.py
"""
from __future__ import annotations

import time
from datetime import timedelta

import numpy as np

from worldcup.db import connect
from worldcup.elo import HOME_ADVANTAGE
from worldcup.elo import run as elo_run
from worldcup.goals import expected_goals, wld_probs
from worldcup.profiles import PROFILES, resolve_min_date

BASE = PROFILES["v4a-flat-mild"]

# City → (country_team_name, elevation_m). Only ≥1500m.
ALTITUDE_VENUES = {
    "Mexico City": ("Mexico", 2240),
    "Guadalajara": ("Mexico", 1560),
    "La Paz": ("Bolivia", 3640),
    "Quito": ("Ecuador", 2850),
    "Bogotá": ("Colombia", 2640),
    "Bogota": ("Colombia", 2640),
    "Addis Ababa": ("Ethiopia", 2355),
    "Asmara": ("Eritrea", 2325),
    "Cusco": ("Peru", 3400),
}
BONUS_GRID = [0.0, 25.0, 50.0, 75.0, 100.0, 150.0, 200.0]


def fetch_alt_matches():
    with connect(read_only=True) as con:
        df = con.execute(
            f"""
            select date, home_team, away_team, neutral, city, country,
                   home_score, away_score
            from matches
            where home_score is not null
              and city in ({','.join(['?']*len(ALTITUDE_VENUES))})
            order by date
            """,
            list(ALTITUDE_VENUES.keys()),
        ).fetchdf()
    return df


def actual_outcome(h, a):
    return 0 if h > a else (2 if h < a else 1)


def main():
    df = fetch_alt_matches()
    print(f"altitude matches: {len(df)}")
    # We need an Elo rating for each team AS OF the match date. The cheap
    # approximation: walk Elo to a cutoff date per-decade, reuse. Even
    # cheaper: walk once to today, use those ratings (regression toward
    # current, biased but consistent across the sweep).
    print("  walking Elo to today (one pass)...")
    final_df, _ = elo_run(min_date=None, max_date=None,
                          k_factors=BASE.k_factors, default_k=BASE.default_k, quiet=True)
    rt = {row.team: float(row.rating) for row in final_df.itertuples(index=False)}

    # Build per-row metadata: home-is-altitude-native flag
    home_at_alt = []
    for r in df.itertuples(index=False):
        country_team, _elev = ALTITUDE_VENUES[r.city]
        home_at_alt.append(r.home_team == country_team)
    home_at_alt = np.array(home_at_alt)
    print(f"  rows where home team is altitude-native: {home_at_alt.sum()}/{len(df)}")

    print(f"\n{'bonus':>6}  {'log-loss':>9}  {'Brier':>7}  N")
    rows = []
    for bonus in BONUS_GRID:
        P, y = [], []
        for i, r in enumerate(df.itertuples(index=False)):
            rh = rt.get(r.home_team); ra = rt.get(r.away_team)
            if rh is None or ra is None:
                continue
            eff_h = rh + (0.0 if r.neutral else HOME_ADVANTAGE)
            if home_at_alt[i] and not r.neutral:
                eff_h += bonus
            lh, la = expected_goals(eff_h - ra, neutral=bool(r.neutral))
            ph, pd, pa = wld_probs(float(lh), float(la))
            P.append((ph, pd, pa))
            y.append(actual_outcome(int(r.home_score), int(r.away_score)))
        P = np.array(P); y = np.array(y)
        eps = 1e-12
        ll = -np.log(np.clip(P[np.arange(len(y)), y], eps, 1.0)).mean()
        oh = np.eye(3)[y]; br = ((P - oh) ** 2).sum(axis=1).mean()
        rows.append((bonus, ll, br, len(y)))
        print(f"  {bonus:>5.0f}  {ll:.5f}  {br:.5f}  {len(y)}")

    base = next(r for r in rows if r[0] == 0.0)
    best = min(rows, key=lambda r: r[1])
    print(f"\nBaseline (no altitude bonus): LL={base[1]:.5f}")
    print(f"Best: +{best[0]:.0f} Elo  LL={best[1]:.5f}  Δ={best[1]-base[1]:+.5f}")


if __name__ == "__main__":
    main()
