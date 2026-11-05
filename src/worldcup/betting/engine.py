"""Core betting maths for 1X2 markets.

All triples are ordered (home, draw, away). Odds are decimal.

  - devig: proportional margin removal (implied 1/odds scaled to sum 1)
  - edge:  p_model x odds - 1 (only on prices up to MAX_ODDS)
  - stake: flat $10, or quarter-Kelly capped at 5% of bankroll
  - CLV:   bet odds against the devigged closing price
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

FLAT_UNIT = 10.0
KELLY_FRACTION = 0.25
KELLY_CAP = 0.05
DEFAULT_MIN_EDGE = 0.03
# No bets above this price. Elo is calibrated on results, but on longshots a
# small disagreement with the market reads as a huge edge, and bookmakers
# already overprice longshots (favourite-longshot bias). Dry run 2026-10-26
# would otherwise have backed Croatia at 14.0 and Serbia at 9.5.
MAX_ODDS = 6.0


def devig_proportional(odds: Sequence[float]) -> tuple[list[float], float]:
    """Return (fair probabilities, overround). Overround 1.06 means a 6% margin."""
    if any(o <= 1.0 for o in odds):
        raise ValueError(f"decimal odds must be > 1: {odds}")
    implied = [1.0 / o for o in odds]
    total = sum(implied)
    return [p / total for p in implied], total


def edge(p_model: float, odds: float) -> float:
    """Expected profit per $1 staked if the model is right."""
    return p_model * odds - 1.0


def pick_selection(
    p_model: Sequence[float], odds: Sequence[float], min_edge: float = DEFAULT_MIN_EDGE,
    max_odds: float = MAX_ODDS,
) -> int | None:
    """Index of the outcome with the largest edge among prices up to max_odds,
    or None if none clears min_edge. At most one bet per match."""
    eligible = [i for i, o in enumerate(odds) if o <= max_odds]
    if not eligible:
        return None
    best = max(eligible, key=lambda i: edge(p_model[i], odds[i]))
    return best if edge(p_model[best], odds[best]) >= min_edge else None


def stake_flat(bankroll: float, unit: float = FLAT_UNIT) -> float:
    return max(0.0, min(unit, bankroll))


def kelly_fraction(p_model: float, odds: float) -> float:
    """Full-Kelly fraction of bankroll; 0 when the edge is not positive."""
    return max(0.0, (p_model * odds - 1.0) / (odds - 1.0))


def stake_kelly(
    p_model: float,
    odds: float,
    bankroll: float,
    fraction: float = KELLY_FRACTION,
    cap: float = KELLY_CAP,
) -> float:
    """Fractional Kelly stake, capped at `cap` of the current bankroll."""
    if bankroll <= 0:
        return 0.0
    return bankroll * min(fraction * kelly_fraction(p_model, odds), cap)


def stake_for(rule: str, p_model: float, odds: float, bankroll: float) -> float:
    if rule == "flat":
        return stake_flat(bankroll)
    if rule == "kelly":
        return stake_kelly(p_model, odds, bankroll)
    raise ValueError(f"unknown staking rule: {rule!r}")


def profit(stake: float, odds: float, won: bool) -> float:
    return stake * (odds - 1.0) if won else -stake


def clv(bet_odds: float, closing_odds: Sequence[float], selection: int) -> float:
    """Closing line value against the devigged closing price.
    +0.05 means the bet was struck 5% above the fair closing odds."""
    fair, _ = devig_proportional(closing_odds)
    return bet_odds * fair[selection] - 1.0


def clv_raw(bet_odds: float, closing_odds: Sequence[float], selection: int) -> float:
    """CLV against the raw closing price, margin included."""
    return bet_odds / closing_odds[selection] - 1.0


@dataclass
class Bet:
    match_id: str
    kickoff: str
    home: str
    away: str
    selection: int
    odds: float
    stake: float
    model_p: float
    implied_p: float
    edge: float
    bankroll_before: float
    outcome: int | None = None
    pnl: float | None = None
    clv: float | None = None

    def settle(self, outcome: int, closing_odds: Sequence[float] | None = None) -> float:
        self.outcome = outcome
        self.pnl = profit(self.stake, self.odds, outcome == self.selection)
        if closing_odds is not None:
            self.clv = clv(self.odds, closing_odds, self.selection)
        return self.pnl
