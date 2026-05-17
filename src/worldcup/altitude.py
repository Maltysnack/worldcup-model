"""Altitude bonus layer.

Acclimatised home teams enjoy a real edge at high altitude that the
flat home-advantage constant doesn't capture. Sweep on n=867 historical
altitude matches (scripts/sweep_altitude_bonus.py) found:

  bonus  Δ log-loss
   +25    -0.009
   +50    -0.015
   +75    -0.018  (cleanest)
  +100    -0.018  (best LL, sharper above optimum)
  +150    -0.010
  +200    +0.010

Picked +75 as conservative: ~max benefit with margin away from the
sharp falloff above +100. Effect applies to a team playing in its
own country at a venue ≥1500m elevation.

Tournament-level integration: a static per-team bump captured here.
For WC 2026, Mexico plays all 3 group games at altitude venues
(Mexico City × 2, Guadalajara × 1). Discount the +75 by the fraction
of expected matches at altitude (group stage 100%, KO ~30%) → +50
as a tournament-stage average.

For historical backtest, applies if (tournament, year)'s host country
is altitude-native AND the team is the host. Currently AFCON 2010 in
Angola (none of the cities are altitude), Copa América doesn't venue
in Bogotá/Quito/La Paz, AFCON 2024 in Ivory Coast (no altitude). So
the historical impact via comprehensive_validation is nil; the bump
matters for WC 2026 only.
"""

from __future__ import annotations

ALTITUDE_BONUS_M = 50.0  # Elo points per WC stage for native-host altitude team

# (tournament, year) -> {team: bonus}
TOURNAMENT_ALTITUDE: dict[tuple[str, int], dict[str, float]] = {
    ("FIFA World Cup", 2026): {"Mexico": ALTITUDE_BONUS_M},
}


def altitude_bonus(team: str, tournament: str, year: int) -> float:
    return TOURNAMENT_ALTITUDE.get((tournament, year), {}).get(team, 0.0)


def apply_altitude_bonus(
    ratings: dict[str, float],
    tournament: str,
    year: int,
) -> dict[str, float]:
    return {t: r + altitude_bonus(t, tournament, year) for t, r in ratings.items()}
