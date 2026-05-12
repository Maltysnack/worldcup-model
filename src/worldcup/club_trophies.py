"""Authoritative 2025-26 club trophy results.

Source: Wikipedia search 2026-05-13. Updated as outcomes confirm.
Premier League and UEFA Champions League still being contested at
time of writing; placeholders flagged so they can be refreshed.

Each entry: club -> list of (trophy_id, weight) tuples.
Weights are calibrated against the scheme described in the trophies.py
module docstring:
  +5 per player at a UCL champion
  +3 per player at a UCL runner-up or top-5 league champion
  +2 per player at a UCL semi-finalist
  +1 per player at a UCL quarter-finalist
  +1 per player at a top-5 league runner-up
  +2 per player at a domestic cup winner
  +0.5 per player at any other meaningful trophy

Club names must match exactly what the Wikipedia squad scraper produces
(see wiki_rosters.py output). Add aliases as needed.
"""

from __future__ import annotations

# ── 2025-26 confirmed champions (as of 2026-05-13) ────────────────────
CLUB_TROPHIES_2025_26: dict[str, list[tuple[str, float]]] = {
    # Confirmed league champions
    "Barcelona":        [("la_liga_champion", 3.0)],          # confirmed May 10
    "Bayern Munich":    [("bundesliga_champion", 3.0)],       # confirmed Apr 19
    "Inter Milan":      [("serie_a_champion", 3.0)],          # confirmed May 3
    "Paris Saint-Germain": [("ligue_1_champion", 3.0)],       # confirmed mid-Apr

    # Premier League: not yet decided (Arsenal leads, Man City close).
    # Both clubs get partial credit until result is final; refresh after
    # final matchday (~24 May).
    "Arsenal":          [("premier_league_leader", 2.0)],
    "Manchester City":  [("premier_league_chasing", 1.5)],

    # UCL: PSG vs Arsenal final May 30. Both get UCL-final credit. Top
    # up after the result is known.
    # PSG already credited above via Ligue 1; append the UCL final line.
    # Arsenal already credited via PL leader; append the UCL final line.
    # (We do this by extending below.)

    # UCL semi-finalists (Bayern, Inter both eliminated by finalists)
    # — already credited at the league-champion level; add an extra
    # +1 for reaching the SF.
    # Done via the append loop below.
}

# 2025-26 UCL knockout-stage credit, sourced from
# https://en.wikipedia.org/wiki/2025%E2%80%9326_UEFA_Champions_League_knockout_phase
# verified 2026-05-13:
#   QF: PSG, Liverpool, Real Madrid, Bayern, Barcelona, Atletico, Sporting CP, Arsenal
#   SF: PSG, Bayern, Atletico, Arsenal
#   Final (30 May, not yet played): PSG vs Arsenal
_UCL_FINALISTS = ["Paris Saint-Germain", "Arsenal"]
_UCL_SEMIFINAL_LOSERS = ["Bayern Munich", "Atlético Madrid"]
_UCL_QUARTERFINAL_LOSERS = [
    "Liverpool", "Real Madrid", "Barcelona", "Sporting CP",
]

for c in _UCL_FINALISTS:
    CLUB_TROPHIES_2025_26.setdefault(c, []).append(("ucl_finalist", 3.0))
for c in _UCL_SEMIFINAL_LOSERS:
    CLUB_TROPHIES_2025_26.setdefault(c, []).append(("ucl_semifinal_loser", 1.5))
for c in _UCL_QUARTERFINAL_LOSERS:
    CLUB_TROPHIES_2025_26.setdefault(c, []).append(("ucl_quarterfinal_loser", 1.0))

# Club name aliases: Wikipedia squad pages often abbreviate or vary.
# Map common variants to the canonical name used in CLUB_TROPHIES_2025_26.
CLUB_ALIASES = {
    "PSG": "Paris Saint-Germain",
    "Bayern": "Bayern Munich",
    "FC Bayern Munich": "Bayern Munich",
    "Bayern Munchen": "Bayern Munich",
    "Bayern München": "Bayern Munich",
    "Inter": "Inter Milan",
    "Internazionale": "Inter Milan",
    "Inter Milano": "Inter Milan",
    "FC Barcelona": "Barcelona",
    "Manchester City": "Manchester City",
    "Man City": "Manchester City",
    "Arsenal F.C.": "Arsenal",
    "Real Madrid CF": "Real Madrid",
    "Atletico Madrid": "Atlético Madrid",
    "Atlético de Madrid": "Atlético Madrid",
}


def trophies_for_club(club: str) -> list[tuple[str, float]]:
    """Return list of (trophy_id, weight) for the given club, or empty
    list if the club won nothing notable in 2025-26."""
    canonical = CLUB_ALIASES.get(club, club)
    return CLUB_TROPHIES_2025_26.get(canonical, [])


def total_weight_for_club(club: str) -> float:
    return sum(w for _, w in trophies_for_club(club))


# Reference date for "as of" reporting
LAST_UPDATED = "2026-05-13"
PENDING = ["Premier League final standings (24 May)", "UCL final (30 May)"]
