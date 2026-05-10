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

# Approximate gradient: 100 Elo gap → ~0.55 goals of expected diff at the
# current calibration. So 1 goal of residual ≈ 180 Elo. We shrink that
# heavily because international form is noisy and 12 months is short.
ELO_PER_GOAL_DIFF = 180.0
SHRINKAGE = 0.35
ELO_CAP = 40.0

# Winsorize each match's residual before averaging. A single freak match
# (e.g. England-vs-Andorra where the goals model expects +7 and gets +1
# because the strong side stops trying once they're up 2-0) shouldn't
# dominate a team's form score. Capping per-match contribution at ±2
# means form is driven by the *consistent* trend, not outliers.
RESIDUAL_CLIP = 2.0


def compute_form_bumps(as_of_date=None) -> dict[str, float]:
    """Return {team: elo_bump} where bump is in raw Elo points."""
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
    residual_home = actual_diff - expected_diff  # from home team's POV
    # Winsorize per-match residuals so freak mismatches don't dominate
    residual_home = np.clip(residual_home, -RESIDUAL_CLIP, RESIDUAL_CLIP)

    per_team: dict[str, list[float]] = defaultdict(list)
    homes = df["home_team"].tolist()
    aways = df["away_team"].tolist()
    for i in range(len(df)):
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
