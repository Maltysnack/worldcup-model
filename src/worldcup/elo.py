"""Elo rating system for international football.

A rolling Elo model. We walk through every match since MIN_DATE in
chronological order, and for each one:

  1. Look up both teams' current ratings (1500 if we've never seen them)
  2. Compute the Elo expected score for the home team, accounting for
     home advantage (skipped if neutral venue)
  3. Record that prediction
  4. Update both teams' ratings based on the actual outcome

K-factors vary by tournament tier so a friendly moves ratings less than
a World Cup match. Recency is handled implicitly: each new match nudges
the rating, gradually overwriting the influence of older matches.

Run: `uv run worldcup-elo`
"""

from __future__ import annotations

import sys
from datetime import date

import pandas as pd

from worldcup.db import connect

MODEL_VERSION = "elo-v1"
DEFAULT_RATING = 1500.0
HOME_ADVANTAGE = 50.0       # added to home team's effective rating, only if not neutral
MIN_DATE = date(1990, 1, 1) # ignore older data: too different from modern football

# K-factor: how much each match moves ratings. Bigger = matters more.
K_FACTORS: dict[str, float] = {
    "FIFA World Cup": 60,
    "UEFA Euro": 50,
    "Copa América": 50,
    "African Cup of Nations": 50,
    "AFC Asian Cup": 50,
    "Gold Cup": 50,
    "Confederations Cup": 50,
    "Oceania Nations Cup": 45,
    "FIFA World Cup qualification": 40,
    "UEFA Nations League": 35,
    "CONCACAF Nations League": 35,
    "UEFA Euro qualification": 30,
    "Copa América qualification": 30,
    "African Cup of Nations qualification": 30,
    "AFC Asian Cup qualification": 30,
    "Friendly": 20,
}
DEFAULT_K = 25  # fallback for tournaments not in the table


def expected_score(rating_home_eff: float, rating_away_eff: float) -> float:
    """Elo expected score for the home team (a number between 0 and 1).

    Counts a draw as 0.5. Not a literal win probability — that requires
    a separate draw model, which we'll add later.
    """
    return 1.0 / (1.0 + 10 ** ((rating_away_eff - rating_home_eff) / 400.0))


def k_for(tournament: str) -> float:
    return K_FACTORS.get(tournament, DEFAULT_K)


def actual_score(home_goals: int, away_goals: int) -> tuple[float, float]:
    """Convert final score to Elo result: win=1.0, draw=0.5, loss=0.0."""
    if home_goals > away_goals:
        return 1.0, 0.0
    if home_goals < away_goals:
        return 0.0, 1.0
    return 0.5, 0.5


def run() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Walk through history, return (final_ratings_df, predictions_df)."""
    with connect(read_only=True) as con:
        matches = con.execute(
            """
            SELECT date, home_team, away_team, home_score, away_score,
                   tournament, neutral
            FROM matches
            WHERE home_score IS NOT NULL
              AND date >= ?
            ORDER BY date, home_team
            """,
            [MIN_DATE],
        ).fetchdf()

    print(f"  walking {len(matches):,} matches from {MIN_DATE} onwards")

    ratings: dict[str, float] = {}
    match_counts: dict[str, int] = {}
    last_match: dict[str, date] = {}
    predictions: list[dict] = []

    for row in matches.itertuples(index=False):
        home, away = row.home_team, row.away_team
        r_home = ratings.get(home, DEFAULT_RATING)
        r_away = ratings.get(away, DEFAULT_RATING)

        # Effective ratings include home advantage
        eff_home = r_home + (0.0 if row.neutral else HOME_ADVANTAGE)
        eff_away = r_away

        exp_home = expected_score(eff_home, eff_away)

        predictions.append(
            {
                "match_date": row.date,
                "home_team": home,
                "away_team": away,
                "tournament": row.tournament,
                "neutral": bool(row.neutral),
                "model_version": MODEL_VERSION,
                "rating_home_pre": r_home,
                "rating_away_pre": r_away,
                "elo_expected_home": exp_home,
                "actual_home_goals": int(row.home_score),
                "actual_away_goals": int(row.away_score),
            }
        )

        # Update with actual outcome
        actual_home, actual_away = actual_score(row.home_score, row.away_score)
        k = k_for(row.tournament)
        ratings[home] = r_home + k * (actual_home - exp_home)
        ratings[away] = r_away + k * (actual_away - (1.0 - exp_home))
        match_counts[home] = match_counts.get(home, 0) + 1
        match_counts[away] = match_counts.get(away, 0) + 1
        last_match[home] = row.date
        last_match[away] = row.date

    final = pd.DataFrame(
        [
            {
                "team": team,
                "rating": rating,
                "n_matches": match_counts[team],
                "last_match": last_match[team],
                "model_version": MODEL_VERSION,
            }
            for team, rating in ratings.items()
        ]
    )
    preds = pd.DataFrame(predictions)
    return final, preds


def main() -> None:
    print(f"Building Elo ratings (model: {MODEL_VERSION})")
    final_df, preds_df = run()

    with connect() as con:
        # Replace tables for this model version. When we add new model
        # versions later, we'll keep history by appending to predictions
        # rather than dropping. For elo-v1 specifically this is fine.
        con.execute("DROP TABLE IF EXISTS elo_current")
        con.execute("CREATE TABLE elo_current AS SELECT * FROM final_df")

        con.execute("DROP TABLE IF EXISTS predictions")
        con.execute("CREATE TABLE predictions AS SELECT * FROM preds_df")

        print(f"\n  wrote elo_current: {len(final_df):,} teams")
        print(f"  wrote predictions: {len(preds_df):,} rows")

        print("\nTop 20 active teams (last match within past 18 months):")
        top = con.execute(
            """
            SELECT team, ROUND(rating) AS rating, n_matches, last_match
            FROM elo_current
            WHERE last_match >= (CURRENT_DATE - INTERVAL 18 MONTH)
            ORDER BY rating DESC
            LIMIT 20
            """
        ).fetchdf()
        print(top.to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())
