"""Synthetic Sportsbet-style odds, used until real historical odds exist.

A synthetic book takes fair probabilities, perturbs them with log-normal
noise, applies a proportional margin, and quotes decimal odds. The closing
price is the opening fair price with a smaller zero-mean drift, so prices
never move toward the truth and CLV against the devigged close sits near
minus the margin by construction.

None of this says anything about real edge. It exercises the pipeline.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

SPORTSBET_MARGIN = 1.06   # typical 1X2 overround for intl football
OPEN_NOISE = 0.15         # sd of log-prob noise at the bet snapshot
CLOSE_DRIFT = 0.05        # sd of log-prob drift from bet snapshot to close
MIN_ODDS = 1.01


def perturb(p: Sequence[float], sd: float, rng: np.random.Generator) -> np.ndarray:
    """Multiply each probability by exp(N(0, sd)) and renormalise."""
    arr = np.asarray(p, dtype=float) * np.exp(rng.normal(0.0, sd, size=len(p)))
    return arr / arr.sum()


def quote(p_fair: Sequence[float], margin: float = SPORTSBET_MARGIN) -> list[float]:
    """Decimal odds with a proportional margin: odds_i = 1 / (p_i x margin),
    floored at MIN_ODDS like a real book."""
    return [max(MIN_ODDS, float(1.0 / (p * margin))) for p in p_fair]


def book_prices(
    p_fair: Sequence[float],
    rng: np.random.Generator,
    open_noise: float = OPEN_NOISE,
    close_drift: float = CLOSE_DRIFT,
    margin: float = SPORTSBET_MARGIN,
) -> tuple[list[float], list[float]]:
    """Return (bet-snapshot odds, closing odds) for one match."""
    p_open = perturb(p_fair, open_noise, rng)
    p_close = perturb(p_open, close_drift, rng)
    return quote(p_open, margin), quote(p_close, margin)
