"""Squad recent-trophies layer.

Captures the "Kane is on fire" / "Bellingham just won the Champions
League" effect that squad market value misses. Players' recent club
achievements signal current form and big-game readiness in a way that
salary-derived market value doesn't capture cleanly.

Per-WC-team score is a rough aggregate of player-club achievements in
the last ~24 months. Higher score = more squad members at clubs that
won major league titles, the UCL, or scoring/MVP awards.

Implicit weighting (very approximate):
  + ~3 per player at a top-5 European league champion (PL, La Liga,
    Bundesliga, Serie A, Ligue 1)
  + ~5 per player at a UCL semi-finalist or champion
  + ~3 per golden-boot or league-top-scorer in a major competition
  + ~2 per major domestic cup winner
  + ~1.5 for top-of-table positions in second-tier major leagues
  + reduced weights for Saudi Pro League, MLS, and other smaller
    competitions

These are honest estimates as of mid-2026. Without a player -> club ->
trophies scraping pipeline, this is the best a hobby project can do
quickly. Should be refined with real data when feasible.
"""

from __future__ import annotations

import numpy as np

# Aggregate "club-trophy density" per WC team. See module doc for the
# loose weighting scheme used to estimate these.
TEAM_TROPHY_SCORE: dict[str, float] = {
    # Top tier (large rosters at major-competition champions)
    "France": 38.0,        # Mbappe (Real), PSG core (UCL run), Tchouameni, Camavinga
    "Spain": 35.0,         # Yamal/Pedri (Barca champs), Rodri (City + Ballon d'Or), Carvajal
    "England": 32.0,       # Bellingham (Real), Foden/Walker (City), Kane (Bayern champ + golden boot), Saka
    "Germany": 30.0,       # Wirtz/Tah (Bayer), Musiala/Kimmich (Bayern), Havertz (Arsenal)
    "Brazil": 28.0,        # Vinicius/Rodrygo (Real), Bruno G. (Newcastle), Casemiro
    "Portugal": 25.0,      # Bruno Fernandes (United), Ruben Dias/Bernardo (City), Diogo Jota
    "Netherlands": 24.0,   # Van Dijk (Liverpool), De Jong (Barca), Reijnders (Milan)
    "Argentina": 22.0,     # Lautaro (Inter), Mac Allister (Liverpool), Julian Alvarez (Atleti)

    # Strong (multiple players at top European clubs)
    "Norway": 18.0,        # Haaland (City), Odegaard (Arsenal), Sorloth (Atleti)
    "Morocco": 16.0,       # Hakimi (PSG), En-Nesyri, Mazraoui (United)
    "Belgium": 16.0,       # KDB (Napoli), Lukaku (Roma), Doku (City)
    "South Korea": 16.0,   # Son (Spurs), Kim Min-jae (Bayern), Lee Kang-in (PSG)
    "Uruguay": 16.0,       # Nunez (Liverpool), Valverde (Real), Ugarte
    "Croatia": 14.0,       # Modric (Real, still active), Sucic (Salzburg), Brozovic (Saudi)
    "Turkey": 14.0,        # Hakan Calhanoglu (Inter), Guler (Real), Yildiz (Juve)
    "Senegal": 14.0,       # Sarr (Palace), Mendy (Saudi), Mane (Saudi), Koulibaly
    "Switzerland": 13.0,   # Sommer (Inter), Akanji (City), Xhaka (Bayer)
    "Sweden": 12.0,        # Isak (Newcastle), Gyokeres (Sporting/Arsenal)

    # Mid
    "Japan": 10.0,         # Mitoma, Endo, Tomiyasu, Kubo
    "Egypt": 10.0,         # Mostly Salah at Liverpool
    "Colombia": 10.0,      # Lucho Diaz, James, Sinisterra
    "Ivory Coast": 10.0,   # Various PL/Serie A
    "Austria": 9.0,        # Sabitzer, Arnautovic, Wober
    "United States": 9.0,  # Pulisic (Milan), McKennie (Juve), Adams
    "Canada": 8.0,         # Davies (Bayern champ) is the big one
    "Algeria": 8.0,        # Mahrez (Saudi), Bennacer (Milan)
    "Scotland": 8.0,       # McGinn (Villa), Tierney, Robertson (Liverpool)
    "Czech Republic": 7.0, # Hlozek (Bayer), Soucek (West Ham)
    "Ghana": 6.0,          # Various but no major champion contributors

    # Lower
    "Mexico": 5.0,         # Mostly Liga MX
    "Tunisia": 5.0,
    "Bosnia and Herzegovina": 5.0,
    "Ecuador": 5.0,
    "Iran": 4.0,
    "Australia": 4.0,
    "DR Congo": 4.0,
    "Saudi Arabia": 4.0,
    "Paraguay": 4.0,
    "Iraq": 3.0,
    "Jordan": 3.0,
    "Uzbekistan": 3.0,
    "Qatar": 3.0,
    "Cape Verde": 3.0,
    "South Africa": 3.0,
    "New Zealand": 3.0,
    "Panama": 3.0,
    "Haiti": 2.0,
    "Curaçao": 2.0,
}

TROPHY_DEFAULT = 3.0


def trophy_z_scores(team_universe: list[str]) -> dict[str, float]:
    """Return team -> z-score of log(trophy score) across the WC field."""
    vals = np.array([TEAM_TROPHY_SCORE.get(t, TROPHY_DEFAULT) for t in team_universe])
    log_vals = np.log(vals + 1.0)  # +1 so the zeros don't blow up
    mean = log_vals.mean()
    std = log_vals.std() or 1.0
    return {t: float((log_vals[i] - mean) / std) for i, t in enumerate(team_universe)}


def adjustment_table(elo: dict[str, float]) -> list[dict]:
    """Diagnostic: per-team trophy score, z-score, and notes."""
    teams = sorted(TEAM_TROPHY_SCORE)
    z = trophy_z_scores(teams)
    rows = [
        {"team": t, "trophy_score": TEAM_TROPHY_SCORE[t], "z": round(z[t], 2)}
        for t in teams
    ]
    return sorted(rows, key=lambda r: -r["trophy_score"])
