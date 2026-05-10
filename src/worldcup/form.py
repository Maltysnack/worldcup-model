"""Recent-form layer.

Captures the case where a team is over- or under-performing what their
Elo predicts (a hot streak Elo hasn't fully absorbed yet, or a slump).

For each team, we look at their matches over the last LOOKBACK_DAYS and
compute the average residual goal difference vs the goals model:

    residual_per_match = (actual_h - actual_a) - (lambda_h - lambda_a)
    (sign flipped for the away team)

Average across matches → form_score in goals/game.
Convert to an Elo bump via the calibration's "Elo per goal of expected
diff" gradient, with a SHRINKAGE factor (we don't fully trust 12 months
of small-sample international form) and a hard CAP (so a team that
played one freak match in a year doesn't move the planet).

Apply on top of squad-blended Elo. Result: ratings used by the simulator.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import timedelta

import numpy as np

from worldcup.calibration import (
    LAMBDA_AWAY_AT_PARITY,
    LAMBDA_HOME_AT_PARITY,
    SLOPE_AWAY,
    SLOPE_HOME,
)
from worldcup.db import connect
from worldcup.elo import HOME_ADVANTAGE
from worldcup.profiles import current_profile, table_suffix

_PRED_TABLE = f"predictions{table_suffix(current_profile())}"

LOOKBACK_DAYS = 365
MIN_MATCHES = 5

# Form metric. Three implementations:
#   "goals":            per-match residual = actual goal-diff - expected
#                       goal-diff, winsorised at +/-RESIDUAL_CLIP.
#   "goals-filtered":   same as "goals" but skip matches where the model
#                       expected a lopsided result (|expected diff| >
#                       LOPSIDED_THRESHOLD). Teams are not punished for
#                       routinely beating Andorra by "only" 2-0 when the
#                       model expected 7-0. Matches against peers are
#                       still scored normally.
#   "points":           per-match residual = actual W/D/L points -
#                       expected points. Robust to lopsided games but
#                       loses scoreline info; tends to under-penalise
#                       big-name teams that lose narrowly to peers.
FORM_METRIC = "goals"  # see backtest in 2026-05-11 session for the comparison

SHRINKAGE = 0.40
ELO_CAP = 40.0

# Goals metric only
ELO_PER_GOAL_DIFF = 180.0
RESIDUAL_CLIP = 2.0
# Skip matches where the model expected a goal-diff bigger than this in
# magnitude. Anything wider tends to be "minnow blowout" territory where
# real teams stop trying once the result is secured, so the residual
# carries no signal about the strong team's actual quality.
LOPSIDED_THRESHOLD = 4.0

# Points metric only
ELO_PER_POINT_DIFF = 80.0


def _load_recent(as_of_date=None):
    """Pull recent matches with rating_pre values from the predictions table."""
    with connect(read_only=True) as con:
        if as_of_date is None:
            as_of_date = con.execute(
                f"SELECT MAX(match_date) FROM {_PRED_TABLE} WHERE actual_home_goals IS NOT NULL"
            ).fetchone()[0]
        cutoff = as_of_date - timedelta(days=LOOKBACK_DAYS)
        df = con.execute(
            f"""
            SELECT match_date, home_team, away_team, neutral,
                   rating_home_pre, rating_away_pre,
                   actual_home_goals, actual_away_goals
            FROM {_PRED_TABLE}
            WHERE match_date >= ? AND match_date <= ?
              AND actual_home_goals IS NOT NULL
            """,
            [cutoff, as_of_date],
        ).fetchdf()
    return df


def _form_bumps_goals(df, *, filter_lopsided: bool = False) -> dict[str, float]:
    eff_h = df["rating_home_pre"].to_numpy() + np.where(
        df["neutral"].to_numpy(), 0.0, HOME_ADVANTAGE
    )
    eff_a = df["rating_away_pre"].to_numpy()
    gap = eff_h - eff_a
    lh = LAMBDA_HOME_AT_PARITY * np.exp(SLOPE_HOME * gap)
    la = LAMBDA_AWAY_AT_PARITY * np.exp(SLOPE_AWAY * gap)

    actual_diff = (
        df["actual_home_goals"].to_numpy() - df["actual_away_goals"].to_numpy()
    )
    expected_diff = lh - la
    residual_home = actual_diff - expected_diff
    residual_home = np.clip(residual_home, -RESIDUAL_CLIP, RESIDUAL_CLIP)
    # If filtering, drop matches where the model expected a lopsided result.
    keep = (
        np.abs(expected_diff) <= LOPSIDED_THRESHOLD
        if filter_lopsided
        else np.ones(len(df), dtype=bool)
    )

    per_team: dict[str, list[float]] = defaultdict(list)
    homes = df["home_team"].tolist()
    aways = df["away_team"].tolist()
    for i in range(len(df)):
        if not keep[i]:
            continue
        per_team[homes[i]].append(float(residual_home[i]))
        per_team[aways[i]].append(float(-residual_home[i]))

    bumps: dict[str, float] = {}
    for team, residuals in per_team.items():
        if len(residuals) < MIN_MATCHES:
            bumps[team] = 0.0
            continue
        avg = float(np.mean(residuals))
        bump = SHRINKAGE * ELO_PER_GOAL_DIFF * avg
        bumps[team] = float(np.clip(bump, -ELO_CAP, ELO_CAP))
    return bumps


def _form_bumps_points(df) -> dict[str, float]:
    """Form via expected-vs-actual W/D/L points instead of goals.
    Naturally robust to lopsided expected scorelines (Andorra, etc):
    winning a match the model said you'd win is a 0 residual no matter
    what scoreline."""
    from worldcup.goals import wld_probs  # avoid circular at module load

    eff_h = df["rating_home_pre"].to_numpy() + np.where(
        df["neutral"].to_numpy(), 0.0, HOME_ADVANTAGE
    )
    eff_a = df["rating_away_pre"].to_numpy()
    gap = eff_h - eff_a
    lh = LAMBDA_HOME_AT_PARITY * np.exp(SLOPE_HOME * gap)
    la = LAMBDA_AWAY_AT_PARITY * np.exp(SLOPE_AWAY * gap)

    h_goals = df["actual_home_goals"].to_numpy()
    a_goals = df["actual_away_goals"].to_numpy()

    per_team: dict[str, list[float]] = defaultdict(list)
    homes = df["home_team"].tolist()
    aways = df["away_team"].tolist()

    for i in range(len(df)):
        p_home, p_draw, p_away = wld_probs(float(lh[i]), float(la[i]))
        exp_home_pts = 3 * p_home + 1 * p_draw
        exp_away_pts = 3 * p_away + 1 * p_draw
        if h_goals[i] > a_goals[i]:
            actual_home_pts, actual_away_pts = 3, 0
        elif h_goals[i] < a_goals[i]:
            actual_home_pts, actual_away_pts = 0, 3
        else:
            actual_home_pts, actual_away_pts = 1, 1
        per_team[homes[i]].append(actual_home_pts - exp_home_pts)
        per_team[aways[i]].append(actual_away_pts - exp_away_pts)

    bumps: dict[str, float] = {}
    for team, residuals in per_team.items():
        if len(residuals) < MIN_MATCHES:
            bumps[team] = 0.0
            continue
        avg = float(np.mean(residuals))
        bump = SHRINKAGE * ELO_PER_POINT_DIFF * avg
        bumps[team] = float(np.clip(bump, -ELO_CAP, ELO_CAP))
    return bumps


def compute_form_bumps(as_of_date=None) -> dict[str, float]:
    """Return {team: elo_bump} based on the active FORM_METRIC."""
    df = _load_recent(as_of_date)
    if FORM_METRIC == "points":
        return _form_bumps_points(df)
    if FORM_METRIC == "goals-filtered":
        return _form_bumps_goals(df, filter_lopsided=True)
    return _form_bumps_goals(df, filter_lopsided=False)


def apply_form(ratings: dict[str, float], as_of_date=None) -> dict[str, float]:
    """Return a copy of `ratings` with each team's form bump added."""
    bumps = compute_form_bumps(as_of_date=as_of_date)
    return {t: r + bumps.get(t, 0.0) for t, r in ratings.items()}


def form_table(ratings: dict[str, float], teams: list[str] | None = None) -> list[dict]:
    """Diagnostic: per-team Elo + form bump."""
    bumps = compute_form_bumps()
    out = []
    keys = teams if teams is not None else list(ratings.keys())
    for t in keys:
        if t not in ratings:
            continue
        out.append({
            "team": t,
            "elo": round(ratings[t]),
            "form_bump": round(bumps.get(t, 0.0), 1),
            "after": round(ratings[t] + bumps.get(t, 0.0)),
        })
    return sorted(out, key=lambda r: r["form_bump"], reverse=True)
