"""The Poisson goals model.

Maps an effective-Elo gap to per-side expected goals, then samples
scorelines. Used by both the bracket builder and the simulator so the
"deterministic" bracket and the Monte Carlo runs share one model.
"""

from __future__ import annotations

import numpy as np

from worldcup.calibration import (
    LAMBDA_AWAY_AT_PARITY,
    LAMBDA_HOME_AT_PARITY,
    SLOPE_AWAY,
    SLOPE_HOME,
)
from worldcup.elo import HOME_ADVANTAGE

# Penalty shootout: nearly a coin flip with a tiny Elo edge.
PEN_ELO_DIVISOR = 600.0


def expected_goals(eff_gap: float) -> tuple[float, float]:
    return (
        LAMBDA_HOME_AT_PARITY * np.exp(SLOPE_HOME * eff_gap),
        LAMBDA_AWAY_AT_PARITY * np.exp(SLOPE_AWAY * eff_gap),
    )


def sim_match(rating_a, rating_b, *, neutral: bool, ko: bool, rng: np.random.Generator):
    """Sample one match.

    Returns (score_a, score_b, winner_index).
    winner_index is 0 if team A wins, 1 if team B wins, -1 if drawn
    (only possible when ko=False; in knockouts draws go to penalties).
    """
    eff_a = rating_a + (0.0 if neutral else HOME_ADVANTAGE)
    eff_b = rating_b
    lh, la = expected_goals(eff_a - eff_b)
    sa = rng.poisson(lh)
    sb = rng.poisson(la)
    if sa > sb:
        return sa, sb, 0
    if sb > sa:
        return sa, sb, 1
    if not ko:
        return sa, sb, -1
    p_a = 1.0 / (1.0 + 10 ** ((eff_b - eff_a) / PEN_ELO_DIVISOR))
    return sa, sb, (0 if rng.random() < p_a else 1)
