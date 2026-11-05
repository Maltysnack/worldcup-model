"""The Poisson goals model with Dixon-Coles correction.

Maps an effective-Elo gap to per-side expected goals, then samples
scorelines from a Dixon-Coles-corrected joint distribution. Used by
both the bracket builder and the simulator so the "deterministic"
bracket and the Monte Carlo runs share one model.

The Dixon-Coles correction (Dixon & Coles 1997) re-weights the four
low-scoring cells of the joint PMF to fix Poisson's slight under-
prediction of 0-0 / 1-0 / 0-1 / 1-1. The other 117 cells (up to
MAX_GOALS) are unchanged and the joint is re-normalised.
"""

from __future__ import annotations

import os

import numpy as np

from worldcup.calibration import (
    DIXON_COLES_RHO,
    LAMBDA_AWAY_AT_PARITY,
    LAMBDA_HOME_AT_PARITY,
    LAMBDA_NEUTRAL_AT_PARITY,
    MAX_GOALS,
    SLOPE_AWAY,
    SLOPE_HOME,
    SLOPE_NEUTRAL,
    TEMPERATURE,
)
from worldcup.elo import HOME_ADVANTAGE

# Penalty shootout: nearly a coin flip with a tiny Elo edge.
PEN_ELO_DIVISOR = 600.0


def _resolve_temperature() -> float:
    raw = os.environ.get("WC_TEMPERATURE")
    if raw is None:
        return TEMPERATURE
    try:
        v = float(raw)
    except ValueError:
        return TEMPERATURE
    return v if v > 0 else 1.0


def apply_temperature(p_h: float, p_d: float, p_a: float, T: float | None = None) -> tuple[float, float, float]:
    """Flatten (T>1) or sharpen (T<1) a W/D/L probability triple."""
    if T is None:
        T = _resolve_temperature()
    if T == 1.0:
        return p_h, p_d, p_a
    eps = 1e-12
    arr = np.array([max(p_h, eps), max(p_d, eps), max(p_a, eps)])
    L = np.log(arr) / T
    L -= L.max()
    e = np.exp(L)
    e /= e.sum()
    return float(e[0]), float(e[1]), float(e[2])


def expected_goals(eff_gap, neutral=False):
    """Per-side expected goals from the effective Elo gap.

    When ``neutral`` is True, uses a symmetric lambda/slope so the first-listed
    team has no built-in home advantage. Otherwise uses the asymmetric
    home/away constants. Both ``eff_gap`` and ``neutral`` may be scalar or
    numpy array (broadcast together).
    """
    lh_home = LAMBDA_HOME_AT_PARITY * np.exp(SLOPE_HOME * eff_gap)
    la_home = LAMBDA_AWAY_AT_PARITY * np.exp(SLOPE_AWAY * eff_gap)
    if np.isscalar(neutral):
        if neutral:
            return (
                LAMBDA_NEUTRAL_AT_PARITY * np.exp(SLOPE_NEUTRAL * eff_gap),
                LAMBDA_NEUTRAL_AT_PARITY * np.exp(-SLOPE_NEUTRAL * eff_gap),
            )
        return lh_home, la_home
    lh_neut = LAMBDA_NEUTRAL_AT_PARITY * np.exp(SLOPE_NEUTRAL * eff_gap)
    la_neut = LAMBDA_NEUTRAL_AT_PARITY * np.exp(-SLOPE_NEUTRAL * eff_gap)
    return np.where(neutral, lh_neut, lh_home), np.where(neutral, la_neut, la_home)


def _poisson_pmf(lam: float, max_k: int) -> np.ndarray:
    """P(X=k) for k=0..max_k. Iterative, no scipy dependency."""
    pmf = np.empty(max_k + 1)
    pmf[0] = np.exp(-lam)
    for k in range(1, max_k + 1):
        pmf[k] = pmf[k - 1] * lam / k
    return pmf


def dc_joint_pmf(lh: float, la: float, rho: float = DIXON_COLES_RHO) -> np.ndarray:
    """Return shape (MAX_GOALS+1, MAX_GOALS+1) of P(home=i, away=j)."""
    h = _poisson_pmf(lh, MAX_GOALS)
    a = _poisson_pmf(la, MAX_GOALS)
    joint = np.outer(h, a)
    # Dixon-Coles tau correction on the four low-scoring cells
    joint[0, 0] *= 1.0 - lh * la * rho
    joint[0, 1] *= 1.0 + lh * rho
    joint[1, 0] *= 1.0 + la * rho
    joint[1, 1] *= 1.0 - rho
    s = joint.sum()
    if s > 0:
        joint /= s
    return joint


def wld_probs(lh: float, la: float, rho: float = DIXON_COLES_RHO) -> tuple[float, float, float]:
    """Analytic W/D/L probabilities under Dixon-Coles, with post-hoc
    temperature scaling (see calibration.TEMPERATURE)."""
    j = dc_joint_pmf(lh, la, rho)
    p_home = float(np.tril(j, -1).sum())
    p_away = float(np.triu(j, 1).sum())
    p_draw = float(np.trace(j))
    return apply_temperature(p_home, p_draw, p_away)


def _temperature_rescale_joint(joint: np.ndarray) -> np.ndarray:
    """Reweight a (G+1, G+1) joint PMF so its W/D/L marginals match the
    T-scaled wld_probs. Within each bucket, cell ratios are preserved.
    Keeps sim_match and wld_probs in agreement under temperature scaling."""
    T = _resolve_temperature()
    if T == 1.0:
        return joint
    p_h_raw = float(np.tril(joint, -1).sum())
    p_a_raw = float(np.triu(joint, 1).sum())
    p_d_raw = float(np.trace(joint))
    p_h, p_d, p_a = apply_temperature(p_h_raw, p_d_raw, p_a_raw, T=T)
    out = joint.copy()
    G = joint.shape[0]
    if p_h_raw > 0:
        out[np.tril_indices(G, -1)] *= p_h / p_h_raw
    if p_a_raw > 0:
        out[np.triu_indices(G, 1)] *= p_a / p_a_raw
    if p_d_raw > 0:
        out[np.diag_indices(G)] *= p_d / p_d_raw
    s = out.sum()
    if s > 0:
        out /= s
    return out


def sim_match(rating_a, rating_b, *, neutral: bool, ko: bool, rng: np.random.Generator):
    """Sample one match from the Dixon-Coles joint distribution.

    Returns (score_a, score_b, winner_index).
    winner_index is 0 if team A wins, 1 if team B wins, -1 if drawn
    (only possible when ko=False; in knockouts draws go to penalties).
    """
    eff_a = rating_a + (0.0 if neutral else HOME_ADVANTAGE)
    eff_b = rating_b
    lh, la = expected_goals(eff_a - eff_b, neutral=neutral)
    joint = _temperature_rescale_joint(dc_joint_pmf(lh, la))
    flat = joint.ravel()
    idx = rng.choice(flat.size, p=flat)
    sa, sb = divmod(int(idx), MAX_GOALS + 1)
    if sa > sb:
        return sa, sb, 0
    if sb > sa:
        return sa, sb, 1
    if not ko:
        return sa, sb, -1
    p_a = 1.0 / (1.0 + 10 ** ((eff_b - eff_a) / PEN_ELO_DIVISOR))
    return sa, sb, (0 if rng.random() < p_a else 1)
