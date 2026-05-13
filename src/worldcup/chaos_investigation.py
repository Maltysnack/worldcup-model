"""Investigate why our model's simulated tournaments are less chaotic
than real ones.

Recap: historical WC median chaos = 1.19 (per-team avg deviation between
expected and actual round). Our model's sim distribution maxes at 1.13.
Real WCs are systematically more chaotic than our sims.

This script tests three candidate hypotheses, each with a concrete
diagnostic:

  H1: Poisson goal model underestimates variance.
      Test: bin matches by predicted lambda, compute Var(actual_goals)
      per bin, compare to Poisson's predicted variance (which equals
      lambda exactly). If observed variance > lambda, we have
      overdispersion.

  H2: Match-level probabilities are calibrated but extreme outcomes
      (favourite collapses) are underrepresented in tails.
      Test: bin matches by favourite's predicted win probability,
      check actual win rate. The calibration audit found these well-
      calibrated overall; here we focus on whether the high-confidence
      bin (favourite > 80%) has the right rate of collapses.

  H3: Match outcomes are positively correlated within a tournament
      (a chaotic atmosphere triggers cascading upsets).
      Test: within each tournament, compute whether an upset in one
      match is followed by elevated upset rates in subsequent matches.
      A naive iid model would show no autocorrelation.

Run: `uv run worldcup-chaos-investigation`
"""

from __future__ import annotations

import sys
from collections import defaultdict
from datetime import date

import numpy as np

from worldcup.calibration import (
    LAMBDA_AWAY_AT_PARITY,
    LAMBDA_HOME_AT_PARITY,
    SLOPE_AWAY,
    SLOPE_HOME,
)
from worldcup.db import connect
from worldcup.elo import HOME_ADVANTAGE
from worldcup.goals import wld_probs


MAJOR_TOURNAMENTS = (
    "FIFA World Cup",
    "UEFA Euro",
    "Copa América",
    "African Cup of Nations",
    "AFC Asian Cup",
    "Gold Cup",
)


def load_predictions_with_actuals():
    """Pull every historical match with reconstructed lambdas and the
    actual goal counts. Restrict to major tournaments only."""
    with connect(read_only=True) as con:
        df = con.execute(
            f"""
            SELECT match_date, tournament, neutral,
                   rating_home_pre, rating_away_pre,
                   actual_home_goals, actual_away_goals
            FROM predictions
            WHERE tournament IN ({','.join(['?']*len(MAJOR_TOURNAMENTS))})
              AND actual_home_goals IS NOT NULL
            ORDER BY match_date
            """,
            list(MAJOR_TOURNAMENTS),
        ).fetchdf()
    rh = df["rating_home_pre"].to_numpy(dtype=float)
    ra = df["rating_away_pre"].to_numpy(dtype=float)
    neutral = df["neutral"].to_numpy().astype(bool)
    eff_h = rh + np.where(neutral, 0.0, HOME_ADVANTAGE)
    gap = eff_h - ra
    lh = LAMBDA_HOME_AT_PARITY * np.exp(SLOPE_HOME * gap)
    la = LAMBDA_AWAY_AT_PARITY * np.exp(SLOPE_AWAY * gap)
    return df, lh, la, gap


# ── H1: variance check ────────────────────────────────────────────────

def test_overdispersion(df, lh, la):
    """For each lambda bin, compute Var(actual) and compare to lambda
    (Poisson predicts Var == Mean == lambda)."""
    print("=" * 70)
    print("H1: Is the Poisson goal model underdispersed vs real data?")
    print("=" * 70)
    print("Bin matches by predicted lambda; check actual Var(goals).")
    print("Poisson predicts Var(goals) == lambda. If observed > lambda")
    print("then real football has more variance than Poisson allows.\n")

    home_goals = df["actual_home_goals"].to_numpy(dtype=float)
    away_goals = df["actual_away_goals"].to_numpy(dtype=float)

    # Pool home and away into one big sample
    all_lambdas = np.concatenate([lh, la])
    all_actuals = np.concatenate([home_goals, away_goals])

    bins = [0.5, 0.8, 1.1, 1.4, 1.7, 2.0, 2.5, 3.0, 5.0]
    print(f"  {'λ range':>12s}  {'n':>6s}  {'mean actual':>12s}  {'var actual':>11s}  "
          f"{'Poisson var (=λ)':>17s}  {'ratio':>7s}")
    print(f"  {'-'*12}  {'-'*6}  {'-'*12}  {'-'*11}  {'-'*17}  {'-'*7}")
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        mask = (all_lambdas >= lo) & (all_lambdas < hi)
        n = int(mask.sum())
        if n < 100:
            continue
        mean_lambda = float(all_lambdas[mask].mean())
        mean_actual = float(all_actuals[mask].mean())
        var_actual = float(all_actuals[mask].var(ddof=1))
        ratio = var_actual / mean_lambda
        print(f"  {lo:>5.2f}-{hi:<5.2f}  {n:>6}  {mean_actual:>12.3f}  "
              f"{var_actual:>11.3f}  {mean_lambda:>17.3f}  {ratio:>6.3f}x")
    print()
    print("  Reading: ratio > 1.0 means observed variance EXCEEDS Poisson's")
    print("  prediction → real football is over-dispersed → our Monte Carlo")
    print("  underestimates extreme scorelines.\n")


# ── H2: high-confidence collapse rate ─────────────────────────────────

def test_favourite_collapse_rate(df, lh, la):
    """For matches where we predict the favourite at p > X, how often
    does the favourite actually win?"""
    print("=" * 70)
    print("H2: Do extreme favourites collapse at the predicted rate?")
    print("=" * 70)
    print("If we predict an 85% favourite and 100 such matches happen,")
    print("we expect them to win 85 times. Less = real football has more")
    print("upsets at the extremes than we model.\n")

    home_goals = df["actual_home_goals"].to_numpy(dtype=int)
    away_goals = df["actual_away_goals"].to_numpy(dtype=int)
    probs = np.array([wld_probs(float(lh[i]), float(la[i]))
                       for i in range(len(lh))])

    # Actual outcome class: 0=home, 1=draw, 2=away
    actuals = np.where(home_goals > away_goals, 0,
                       np.where(home_goals < away_goals, 2, 1))
    # Probability the model assigned to the favoured class
    fav_class = probs.argmax(axis=1)
    fav_prob = probs.max(axis=1)
    fav_won = (fav_class == actuals).astype(float)

    bands = [(0.50, 0.60), (0.60, 0.70), (0.70, 0.80),
             (0.80, 0.90), (0.90, 1.01)]
    print(f"  {'predicted':>15s}  {'n':>5s}  {'mean pred':>10s}  "
          f"{'actual fav win':>16s}  {'gap':>7s}")
    print(f"  {'-'*15}  {'-'*5}  {'-'*10}  {'-'*16}  {'-'*7}")
    for lo, hi in bands:
        mask = (fav_prob >= lo) & (fav_prob < hi)
        n = int(mask.sum())
        if n < 30:
            continue
        mean_pred = float(fav_prob[mask].mean())
        obs = float(fav_won[mask].mean())
        gap = obs - mean_pred
        print(f"  {lo:.2f}-{hi:.2f}      {n:>5}  {mean_pred:>10.3f}  "
              f"{obs:>16.3f}  {gap:>+7.3f}")
    print()
    print("  Reading: large negative gap at high-prob bins = favourites")
    print("  collapse more than the model says → underestimates upsets.\n")


# ── H3: within-tournament autocorrelation ─────────────────────────────

def test_tournament_autocorrelation(df, lh, la):
    """Within each tournament, is an upset followed by more upsets?
    Compare: among matches that follow an upset within the same
    tournament, what's the upset rate? vs among matches that follow
    a favourite winning?"""
    print("=" * 70)
    print("H3: Are upsets temporally clustered within a tournament?")
    print("=" * 70)
    print("If chaos is a within-tournament effect (red cards, weather,")
    print("crowd psychology), upsets should cluster. If our iid Monte")
    print("Carlo misses this, here's where.\n")

    home_goals = df["actual_home_goals"].to_numpy(dtype=int)
    away_goals = df["actual_away_goals"].to_numpy(dtype=int)
    probs = np.array([wld_probs(float(lh[i]), float(la[i]))
                       for i in range(len(lh))])
    actuals = np.where(home_goals > away_goals, 0,
                       np.where(home_goals < away_goals, 2, 1))
    fav_class = probs.argmax(axis=1)
    fav_prob = probs.max(axis=1)
    # An "upset" is when the favourite (predicted > 60%) lost or drew
    is_upset = (fav_prob > 0.60) & (fav_class != actuals)
    is_chalk = (fav_prob > 0.60) & (fav_class == actuals)

    df = df.reset_index(drop=True)
    # Tournament edition = (tournament, year)
    df["tournament_year"] = list(zip(df["tournament"],
                                      df["match_date"].apply(lambda d: d.year)))

    # For each tournament, walk matches in order. After an upset, count
    # the next-N matches' upset rate; same for after a chalk result.
    after_upset = []
    after_chalk = []
    WINDOW = 5  # next 5 matches within the same tournament
    by_tourn = df.groupby("tournament_year")
    for tname, group in by_tourn:
        idx_list = group.index.tolist()
        for j, i in enumerate(idx_list):
            if not (is_upset[i] or is_chalk[i]):
                continue
            window = idx_list[j + 1: j + 1 + WINDOW]
            if not window:
                continue
            window_high_conf = [k for k in window if fav_prob[k] > 0.60]
            if not window_high_conf:
                continue
            upset_rate = (sum(1 for k in window_high_conf if is_upset[k])
                          / len(window_high_conf))
            (after_upset if is_upset[i] else after_chalk).append(upset_rate)

    if after_upset and after_chalk:
        print(f"  After an upset (within same tournament, next {WINDOW} matches):")
        print(f"    n: {len(after_upset)}  mean subsequent-upset rate: "
              f"{np.mean(after_upset):.3f}")
        print(f"  After a chalk result:")
        print(f"    n: {len(after_chalk)}  mean subsequent-upset rate: "
              f"{np.mean(after_chalk):.3f}")
        diff = np.mean(after_upset) - np.mean(after_chalk)
        print(f"\n  Difference: {diff:+.3f}")
        if diff > 0.02:
            print(f"  Reading: upsets ARE temporally clustered "
                  f"(+{diff*100:.1f}pp). Real momentum effect.")
        elif diff < -0.02:
            print(f"  Reading: actually less likely after upset "
                  f"({diff*100:.1f}pp). Regression to mean.")
        else:
            print(f"  Reading: no meaningful clustering. iid is fine.")
    print()


# ── Synthesis ─────────────────────────────────────────────────────────

def main() -> None:
    df, lh, la, gap = load_predictions_with_actuals()
    print(f"Loaded {len(df):,} major-tournament matches with reconstructed predictions\n")

    test_overdispersion(df, lh, la)
    test_favourite_collapse_rate(df, lh, la)
    test_tournament_autocorrelation(df, lh, la)

    print("=" * 70)
    print("SUMMARY: where is the chaos gap?")
    print("=" * 70)
    print("Run the three diagnostics above and look for: any test where")
    print("the model's prediction systematically misses the actual data.")
    print("If H1 fails (overdispersion), the goals model needs widening.")
    print("If H2 fails (high-prob collapses), the favouritism is too strong.")
    print("If H3 fails (autocorrelation), we need a within-tournament shock.")
    print("Multiple failures = multiple things to address.")


if __name__ == "__main__":
    sys.exit(main())
