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
from worldcup.profiles import current_profile, table_suffix

# Resolve which profile we're running under (set via WC_PROFILE env var,
# defaults to v1). All K-factors come from the profile so a single
# `WC_PROFILE=v2-continental ...` run produces a parallel set of outputs
# without touching the v1 baseline.
_PROFILE = current_profile()
MODEL_VERSION = f"elo-{_PROFILE.name}"
DEFAULT_RATING = 1500.0
HOME_ADVANTAGE = 50.0       # added to home team's effective rating, only if not neutral
MIN_DATE = date(1990, 1, 1) # ignore older data: too different from modern football

K_FACTORS: dict[str, float] = _PROFILE.k_factors
DEFAULT_K: float = _PROFILE.default_k


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


def run(
    *,
    min_date=None,
    max_date=None,
    k_factors: dict[str, float] | None = None,
    default_k: float | None = None,
    quiet: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Walk through history, return (final_ratings_df, predictions_df).

    Defaults come from the active profile (via module-level constants),
    but each can be overridden — this is what the comparison tool uses
    to evaluate alternative profiles without touching the saved tables.
    """
    if min_date is None:
        min_date = MIN_DATE
    if k_factors is None:
        k_factors = K_FACTORS
    if default_k is None:
        default_k = DEFAULT_K

    sql = """
        SELECT date, home_team, away_team, home_score, away_score,
               tournament, neutral
        FROM matches
        WHERE home_score IS NOT NULL
          AND date >= ?
    """
    params: list = [min_date]
    if max_date is not None:
        sql += " AND date <= ?"
        params.append(max_date)
    sql += " ORDER BY date, home_team"

    with connect(read_only=True) as con:
        matches = con.execute(sql, params).fetchdf()

    if not quiet:
        print(f"  walking {len(matches):,} matches from {min_date} onwards")

    ratings: dict[str, float] = {}
    match_counts: dict[str, int] = {}
    last_match: dict[str, date] = {}
    predictions: list[dict] = []

    def k_lookup(tournament: str) -> float:
        return k_factors.get(tournament, default_k)

    for row in matches.itertuples(index=False):
        home, away = row.home_team, row.away_team
        r_home = ratings.get(home, DEFAULT_RATING)
        r_away = ratings.get(away, DEFAULT_RATING)

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

        actual_home, actual_away = actual_score(row.home_score, row.away_score)
        k = k_lookup(row.tournament)
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

    suffix = table_suffix(_PROFILE)
    elo_table = f"elo_current{suffix}"
    pred_table = f"predictions{suffix}"

    with connect() as con:
        con.execute(f"DROP TABLE IF EXISTS {elo_table}")
        con.execute(f"CREATE TABLE {elo_table} AS SELECT * FROM final_df")
        con.execute(f"DROP TABLE IF EXISTS {pred_table}")
        con.execute(f"CREATE TABLE {pred_table} AS SELECT * FROM preds_df")

        print(f"\n  wrote {elo_table}: {len(final_df):,} teams")
        print(f"  wrote {pred_table}: {len(preds_df):,} rows")

        print(f"\nTop 20 active teams (last match within past 18 months):")
        top = con.execute(
            f"""
            SELECT team, ROUND(rating) AS rating, n_matches, last_match
            FROM {elo_table}
            WHERE last_match >= (CURRENT_DATE - INTERVAL 18 MONTH)
            ORDER BY rating DESC
            LIMIT 20
            """
        ).fetchdf()
        print(top.to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())
