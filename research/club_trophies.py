"""Authoritative 2025-26 club trophy results.

Source: Wikipedia search 2026-05-13. Updated as outcomes confirm.
UCL final confirmed 2026-05-31: PSG beat Arsenal.

Each entry: club -> list of (trophy_id, weight) tuples.

Weight scheme (in approximate "per-player" units of how much a squad
member at that club should contribute to their national team's trophy
aggregate):

  Tier 1 (top-tier elite):
    +5  UCL champion (per player on the WC squad at that club)
    +3  UCL runner-up or top-5 European league champion
    +2  UCL semi-finalist (also gets league credit if top-5)
    +1  UCL quarter-finalist
    +1  Top-5 European league runner-up
    +2  Major domestic cup winner (FA Cup, Copa del Rey, DFB Pokal etc)

  Tier 2 (strong but discounted):
    +1.5  Eredivisie / Primeira Liga champion (Dutch / Portuguese)
    +1.5  Brazilian Serie A (Brasileirão) champion (very strong league)

  Tier 3 (meaningful but heavily discounted):
    +0.8  MLS Cup winner (regular-season league = playoff knockout, hybrid)
    +0.8  Argentine Primera champion
    +0.5  Saudi Pro League champion
    +0.5  Liga MX (Apertura or Clausura) champion
    +0.4  MLS Supporters' Shield winner (regular season equivalent)

  Tier 4 (small leagues):
    +0.4  Belgian Pro League / Scottish Premiership / Turkish Süper Lig
    +0.3  Czech / Greek / other top-division champion

The discount on non-big-5 leagues reflects honest gap in quality of
opposition: an MLS championship is not the same signal as a Premier
League title. We still credit non-zero (a player who carried Inter
Miami to the title is a stronger WC contributor than a fringe player
at any club) but at a fraction of the weight.

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
    "Arsenal":          [("premier_league_champion", 3.0)],   # confirmed May 2026

    # UCL final confirmed 2026-05-31: PSG beat Arsenal.
    # PSG gets ucl_champion (+5); Arsenal gets ucl_finalist (+3).
    # Done via the append loops below.
}

# 2025-26 UCL knockout-stage credit, sourced from
# https://en.wikipedia.org/wiki/2025%E2%80%9326_UEFA_Champions_League_knockout_phase
# verified 2026-05-13; final confirmed 2026-05-31:
#   QF: PSG, Liverpool, Real Madrid, Bayern, Barcelona, Atletico, Sporting CP, Arsenal
#   SF: PSG, Bayern, Atletico, Arsenal
#   Final (30 May): PSG beat Arsenal
_UCL_CHAMPION = "Paris Saint-Germain"
_UCL_RUNNER_UP = "Arsenal"
_UCL_SEMIFINAL_LOSERS = ["Bayern Munich", "Atlético Madrid"]
_UCL_QUARTERFINAL_LOSERS = [
    "Liverpool", "Real Madrid", "Barcelona", "Sporting CP",
]

CLUB_TROPHIES_2025_26.setdefault(_UCL_CHAMPION, []).append(("ucl_champion", 5.0))
CLUB_TROPHIES_2025_26.setdefault(_UCL_RUNNER_UP, []).append(("ucl_finalist", 3.0))
for c in _UCL_SEMIFINAL_LOSERS:
    CLUB_TROPHIES_2025_26.setdefault(c, []).append(("ucl_semifinal_loser", 1.5))
for c in _UCL_QUARTERFINAL_LOSERS:
    CLUB_TROPHIES_2025_26.setdefault(c, []).append(("ucl_quarterfinal_loser", 1.0))


# ── Tier 2: Strong-but-discounted European leagues ────────────────────
# Verified Wikipedia search 2026-05-13.
#   Eredivisie 2025-26: PSV Eindhoven (27th title, 5 April 2026)
#   Primeira Liga 2025-26: FC Porto
CLUB_TROPHIES_2025_26.setdefault("PSV Eindhoven", []).append(("eredivisie_champion", 1.5))
CLUB_TROPHIES_2025_26.setdefault("Porto", []).append(("primeira_liga_champion", 1.5))

# Brazilian Serie A 2025: Flamengo (9th title, 3 December 2025)
CLUB_TROPHIES_2025_26.setdefault("Flamengo", []).append(("brasileirao_champion", 1.5))

# ── Tier 3: Meaningful-but-heavily-discounted leagues ─────────────────
# MLS Cup 2025: Inter Miami beat Vancouver Whitecaps 3-1
CLUB_TROPHIES_2025_26.setdefault("Inter Miami", []).append(("mls_cup_champion", 0.8))
CLUB_TROPHIES_2025_26.setdefault("Vancouver Whitecaps", []).append(("mls_cup_runner_up", 0.3))
# MLS Supporters' Shield 2025 (regular-season equivalent): Philadelphia Union
CLUB_TROPHIES_2025_26.setdefault("Philadelphia Union", []).append(("mls_shield_winner", 0.4))

# Liga MX 2025: Toluca won both Apertura and Clausura back-to-back
CLUB_TROPHIES_2025_26.setdefault("Toluca", []).append(("liga_mx_champion", 0.5))
CLUB_TROPHIES_2025_26.setdefault("Toluca", []).append(("liga_mx_apertura", 0.3))

# Saudi Pro League 2025-26: still being contested as of 2026-05-13.
# Will be refreshed once the season concludes.

# Argentine Primera 2025-26 / 2026 — TBD; currently no champion to credit.

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
LAST_UPDATED = "2026-05-31"
PENDING: list[str] = []
