"""Backtest the Poisson goals model against historical World Cups.

For each match in the WC under test:
  1. Look up the rating each team had IMMEDIATELY BEFORE that match
     (we get this for free from the `predictions` table, which records
     rating_home_pre and rating_away_pre as the chronological Elo walk
     went through history)
  2. Apply the calibration constants → expected goals
  3. Sample 10k scorelines → empirical W/D/L probabilities
  4. Compare to the actual result via log-loss + Brier

We sweep a `slope_mult` multiplier on the Elo-to-goals slope. A
multiplier of 1.0 keeps the current calibration; lower values flatten
the relationship (less confident predictions); higher values steepen
it. The best multiplier on the backtest tells us whether we're
over- or under-confident.

Comparison baselines:
  - uniform: every match gets P(W)=P(D)=P(L)=1/3
  - elo_winner: assign 0.55 to the higher-Elo team, 0.30 draw, 0.15
    to the underdog (a flat heuristic, no calibration)

Run: `uv run worldcup-backtest 2018`
"""

from __future__ import annotations

import sys

import numpy as np

from worldcup.calibration import (
    LAMBDA_AWAY_AT_PARITY,
    LAMBDA_HOME_AT_PARITY,
    SLOPE_AWAY,
    SLOPE_HOME,
)
from worldcup.db import connect
from worldcup.elo import HOME_ADVANTAGE

EPS = 1e-12


def actual_outcome(home_g: int, away_g: int) -> int:
    """0 = home win, 1 = draw, 2 = away win."""
    if home_g > away_g:
        return 0
    if home_g < away_g:
        return 2
    return 1


def wld_probs_batch(
    rating_h: np.ndarray,
    rating_a: np.ndarray,
    neutral: np.ndarray,
    *,
    slope_mult: float = 1.0,
    rho: float | None = None,
) -> np.ndarray:
    """Return shape (n_matches, 3) of [p_home, p_draw, p_away] using
    the Dixon-Coles-corrected joint distribution. Analytic, no
    sampling needed."""
    from worldcup.calibration import DIXON_COLES_RHO
    from worldcup.goals import wld_probs

    if rho is None:
        rho = DIXON_COLES_RHO
    eff_h = rating_h + np.where(neutral, 0.0, HOME_ADVANTAGE)
    eff_a = rating_a
    gap = eff_h - eff_a
    lh = LAMBDA_HOME_AT_PARITY * np.exp(SLOPE_HOME * slope_mult * gap)
    la = LAMBDA_AWAY_AT_PARITY * np.exp(SLOPE_AWAY * slope_mult * gap)
    out = np.empty((len(lh), 3))
    for i in range(len(lh)):
        out[i] = wld_probs(float(lh[i]), float(la[i]), rho)
    return out


def log_loss(probs: np.ndarray, actuals: np.ndarray) -> float:
    """Mean negative-log-likelihood across matches.
    probs: (n, 3); actuals: (n,) of class indices in {0,1,2}."""
    chosen = probs[np.arange(len(actuals)), actuals]
    chosen = np.clip(chosen, EPS, 1.0)
    return float(-np.log(chosen).mean())


def brier(probs: np.ndarray, actuals: np.ndarray) -> float:
    """Mean Brier across matches: sum (p_i - I_actual)^2 averaged."""
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(len(actuals)), actuals] = 1.0
    return float(((probs - one_hot) ** 2).sum(axis=1).mean())


def fetch_world_cup(year: int):
    """Pull every match from the given World Cup as (rating_h_pre,
    rating_a_pre, neutral, actual_outcome) arrays. Uses the
    `predictions` table populated by the chronological Elo walk."""
    with connect(read_only=True) as con:
        df = con.execute(
            """
            SELECT
                match_date,
                home_team,
                away_team,
                neutral,
                rating_home_pre,
                rating_away_pre,
                actual_home_goals,
                actual_away_goals
            FROM predictions
            WHERE tournament = 'FIFA World Cup'
              AND EXTRACT(YEAR FROM match_date) = ?
            ORDER BY match_date
            """,
            [year],
        ).fetchdf()
    return df


def main() -> None:
    year = int(sys.argv[1]) if len(sys.argv) > 1 else 2018
    df = fetch_world_cup(year)
    if df.empty:
        print(f"No matches found for World Cup {year}")
        sys.exit(1)

    rating_h = df["rating_home_pre"].to_numpy()
    rating_a = df["rating_away_pre"].to_numpy()
    neutral = df["neutral"].to_numpy().astype(bool)
    actuals = np.array([
        actual_outcome(int(h), int(a))
        for h, a in zip(df["actual_home_goals"], df["actual_away_goals"])
    ])

    print(f"Backtesting {len(df)} matches from World Cup {year}")
    print(f"Outcome distribution: {dict(zip(*np.unique(actuals, return_counts=True)))}")
    print()

    # Baselines
    uniform = np.full((len(df), 3), 1 / 3)
    print(f"  Uniform (1/3 each)        log-loss={log_loss(uniform, actuals):.4f}  Brier={brier(uniform, actuals):.4f}")

    # Flat heuristic: 0.55 to higher-Elo team, 0.30 draw, 0.15 to underdog
    eff_h = rating_h + np.where(neutral, 0.0, HOME_ADVANTAGE)
    flat = np.zeros((len(df), 3))
    home_better = eff_h >= rating_a
    flat[home_better] = [0.55, 0.30, 0.15]
    flat[~home_better] = [0.15, 0.30, 0.55]
    print(f"  Flat heuristic (55/30/15) log-loss={log_loss(flat, actuals):.4f}  Brier={brier(flat, actuals):.4f}")

    # Slope sweep at the calibrated rho
    print()
    print(f"  Slope sweep (Dixon-Coles rho = {-0.10:.2f}):")
    print(f"  {'slope_mult':>10}  {'log-loss':>10}  {'Brier':>8}")
    print(f"  {'-' * 32}")
    best = (None, float("inf"), None)
    for mult in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]:
        probs = wld_probs_batch(rating_h, rating_a, neutral, slope_mult=mult)
        ll = log_loss(probs, actuals)
        br = brier(probs, actuals)
        marker = ""
        if ll < best[1]:
            best = (mult, ll, br)
            marker = "  <- best so far"
        print(f"  {mult:>10.2f}  {ll:>10.4f}  {br:>8.4f}{marker}")

    # Dixon-Coles rho sweep at slope_mult=1.0
    print()
    print(f"  Dixon-Coles rho sweep (slope_mult = 1.00):")
    print(f"  {'rho':>10}  {'log-loss':>10}  {'Brier':>8}")
    print(f"  {'-' * 32}")
    best_rho = (None, float("inf"), None)
    for rho in [-0.20, -0.15, -0.10, -0.05, 0.0, 0.05, 0.10]:
        probs = wld_probs_batch(rating_h, rating_a, neutral, rho=rho)
        ll = log_loss(probs, actuals)
        br = brier(probs, actuals)
        marker = ""
        if ll < best_rho[1]:
            best_rho = (rho, ll, br)
            marker = "  <- best so far"
        print(f"  {rho:>+10.2f}  {ll:>10.4f}  {br:>8.4f}{marker}")

    print()
    print(f"Best slope multiplier: {best[0]:.2f}  (log-loss {best[1]:.4f})")
    print(f"Best rho:              {best_rho[0]:+.2f}  (log-loss {best_rho[1]:.4f})")
    print()
    print(f"Reference: bookmaker log-loss for WC was around 0.95-1.00.")
    print(f"  - Beats uniform if log-loss < {-np.log(1/3):.4f}")
    print(f"  - Beats the flat heuristic if log-loss < {log_loss(flat, actuals):.4f}")


if __name__ == "__main__":
    main()
