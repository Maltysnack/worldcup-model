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


# ─────────────────────────────────────────────────────────────────────
# National-team trophies layer
# ─────────────────────────────────────────────────────────────────────
# Recent finishes at major international tournaments. Verified from
# our matches table (no guesses). Each entry: (tournament, year,
# placement). Only champion and runner-up are recorded here, per the
# project decision to keep this layer focused.

NT_FINISHES: dict[str, list[tuple[str, int, str]]] = {
    # ─── Argentina (most decorated recent NT in our window) ───
    "Argentina": [
        ("WC", 2022, "champion"),
        ("Copa", 2024, "champion"),
        ("Copa", 2021, "champion"),
        ("Copa", 2016, "runner_up"),
        ("Copa", 2015, "runner_up"),
        ("WC", 2014, "runner_up"),
        ("Copa", 2007, "runner_up"),
        ("Copa", 2004, "runner_up"),
    ],
    # ─── European powers ───
    "Spain": [
        ("EURO", 2024, "champion"),
        ("WC", 2010, "champion"),
        ("EURO", 2012, "champion"),
        ("EURO", 2008, "champion"),
    ],
    "France": [
        ("WC", 2022, "runner_up"),
        ("WC", 2018, "champion"),
        ("EURO", 2016, "runner_up"),
        ("EURO", 2000, "champion"),
    ],
    "England": [
        ("EURO", 2024, "runner_up"),
        ("EURO", 2020, "runner_up"),
    ],
    "Italy": [  # not in 2026 WC, but still gives historical context
        ("EURO", 2020, "champion"),
        ("EURO", 2012, "runner_up"),
        ("WC", 2006, "champion"),
    ],
    "Germany": [
        ("WC", 2014, "champion"),
        ("EURO", 2008, "runner_up"),
        ("WC", 2002, "runner_up"),
    ],
    "Croatia": [
        ("WC", 2018, "runner_up"),
    ],
    "Netherlands": [
        ("WC", 2010, "runner_up"),
    ],
    "Portugal": [
        ("EURO", 2016, "champion"),
        ("EURO", 2004, "runner_up"),
    ],
    "Greece": [  # historical context only
        ("EURO", 2004, "champion"),
    ],
    # ─── South America ───
    "Brazil": [
        ("WC", 2002, "champion"),
        ("Copa", 2019, "champion"),
        ("Copa", 2007, "champion"),
        ("Copa", 2004, "champion"),
        ("Copa", 2021, "runner_up"),
    ],
    "Colombia": [
        ("Copa", 2024, "runner_up"),
        ("Copa", 2001, "champion"),
    ],
    "Uruguay": [
        ("Copa", 2011, "champion"),
    ],
    "Paraguay": [
        ("Copa", 2011, "runner_up"),
    ],
    "Chile": [  # not in 2026 WC, historical context
        ("Copa", 2016, "champion"),
        ("Copa", 2015, "champion"),
    ],
    "Peru": [
        ("Copa", 2019, "runner_up"),
    ],
    # ─── Africa ───
    "Morocco":       [("AFCON", 2026, "champion"),  ("AFCON", 2004, "runner_up")],
    "Senegal":       [("AFCON", 2026, "runner_up"), ("AFCON", 2022, "champion"),
                      ("AFCON", 2019, "runner_up"), ("AFCON", 2002, "runner_up")],
    "Ivory Coast":   [("AFCON", 2024, "champion"),  ("AFCON", 2015, "champion"),
                      ("AFCON", 2012, "runner_up"), ("AFCON", 2006, "runner_up")],
    "Egypt":         [("AFCON", 2022, "runner_up"), ("AFCON", 2017, "runner_up"),
                      ("AFCON", 2010, "champion"),  ("AFCON", 2008, "champion"),
                      ("AFCON", 2006, "champion")],
    "Cameroon":      [("AFCON", 2017, "champion"),  ("AFCON", 2008, "runner_up"),
                      ("AFCON", 2002, "champion"),  ("AFCON", 2000, "champion")],
    "Algeria":       [("AFCON", 2019, "champion")],
    "Ghana":         [("AFCON", 2015, "runner_up"), ("AFCON", 2010, "runner_up")],
    "Tunisia":       [("AFCON", 2004, "champion")],
    # Zambia, Burkina Faso, Nigeria 2013 etc not in 2026 WC: skip
    # ─── Asia / Oceania ───
    "Qatar":         [("Asian", 2024, "champion"),  ("Asian", 2019, "champion")],
    "Jordan":        [("Asian", 2024, "runner_up")],
    "Australia":     [("Asian", 2015, "champion"),  ("Asian", 2011, "runner_up")],
    "South Korea":   [("Asian", 2015, "runner_up")],
    "Japan":         [("Asian", 2019, "runner_up"), ("Asian", 2011, "champion"),
                      ("Asian", 2004, "champion"),  ("Asian", 2000, "champion")],
    "Saudi Arabia":  [("Asian", 2007, "runner_up"), ("Asian", 2000, "runner_up")],
    "Iraq":          [("Asian", 2007, "champion")],
    # ─── CONCACAF / Gold Cup ───
    "Mexico": [
        ("Gold Cup", 2025, "champion"), ("Gold Cup", 2023, "champion"),
        ("Gold Cup", 2019, "champion"), ("Gold Cup", 2015, "champion"),
        ("Gold Cup", 2011, "champion"), ("Gold Cup", 2009, "champion"),
        ("Gold Cup", 2003, "champion"), ("Gold Cup", 2021, "runner_up"),
        ("Gold Cup", 2017, "runner_up"), ("Gold Cup", 2007, "runner_up"),
    ],
    "United States": [
        ("Gold Cup", 2021, "champion"), ("Gold Cup", 2017, "champion"),
        ("Gold Cup", 2013, "champion"), ("Gold Cup", 2007, "champion"),
        ("Gold Cup", 2025, "runner_up"), ("Gold Cup", 2019, "runner_up"),
        ("Gold Cup", 2011, "runner_up"), ("Gold Cup", 2009, "runner_up"),
        ("Gold Cup", 2003, "runner_up"),
    ],
    "Canada":        [("Gold Cup", 2000, "champion")],
    "Panama":        [("Gold Cup", 2023, "runner_up"), ("Gold Cup", 2013, "runner_up")],
}

# Continental tournaments happen every 2-4 years and recent winners tend
# to be in current form. The World Cup is on a 4-year cycle and history
# shows defending champions frequently decline (France 2002, Italy 2010,
# Spain 2014, Germany 2018 all went out in groups). So we treat WC
# bonuses as roughly half-weight vs continentals, AND use a fast 2-year
# half-life so older results fade quickly.
NT_TOURNAMENT_TYPE = {
    "EURO": "CONTINENTAL",
    "Copa": "CONTINENTAL",
    "AFCON": "CONTINENTAL",
    "Asian": "CONTINENTAL",
    "Gold Cup": "CONTINENTAL",
    "WC": "WC",
}
NT_BONUS_BY_TYPE = {
    "CONTINENTAL": {"champion": 30.0, "runner_up": 12.0},
    "WC":          {"champion": 15.0, "runner_up":  6.0},
}
NT_HALFLIFE_YEARS = 2.0  # 2y old = 50%, 4y old = 25%, 6y old = 12.5%
NT_REFERENCE_YEAR = 2026


def nt_trophy_bonus(team: str, as_of_year: int | None = None) -> float:
    """Total Elo bonus for `team` from NT trophy finishes earlier than
    `as_of_year`. Continental and WC results use different base bonuses
    (continental winners weighted higher; WC winners get half because
    they're 4 years old by definition AND defending champs historically
    decline). Time-decayed at NT_HALFLIFE_YEARS."""
    ref_year = NT_REFERENCE_YEAR if as_of_year is None else as_of_year
    finishes = NT_FINISHES.get(team, [])
    total = 0.0
    for tournament, year, place in finishes:
        if year > ref_year:
            continue  # future tournament; hadn't happened yet
        type_ = NT_TOURNAMENT_TYPE.get(tournament, "CONTINENTAL")
        base = NT_BONUS_BY_TYPE[type_].get(place, 0.0)
        years_ago = max(0, ref_year - year)
        decay = 0.5 ** (years_ago / NT_HALFLIFE_YEARS)
        total += base * decay
    return total


def apply_nt_bonus(ratings: dict[str, float], as_of_year: int | None = None) -> dict[str, float]:
    """Add the NT trophy bonus to each rating. Teams not in NT_FINISHES
    are unchanged. Pass as_of_year for historical backtests."""
    return {t: r + nt_trophy_bonus(t, as_of_year=as_of_year) for t, r in ratings.items()}


def nt_bonus_table(teams: list[str] | None = None) -> list[dict]:
    """Diagnostic: per-team NT bonus breakdown."""
    universe = teams or list(NT_FINISHES.keys())
    rows = []
    for t in universe:
        finishes = NT_FINISHES.get(t, [])
        bonus = nt_trophy_bonus(t)
        labels = "; ".join(
            f"{tournament} {year} {place}" for tournament, year, place in finishes
        ) or "(none)"
        rows.append({"team": t, "bonus": round(bonus, 1), "finishes": labels})
    return sorted(rows, key=lambda r: -r["bonus"])


COMPUTED_COVERAGE_THRESHOLD = 0.5  # require ≥50% of WC teams before switching


def team_trophy_score(team: str) -> float:
    """Return the best-available trophy score for a team.

    Preference order:
      1. Computed score from real Wikipedia rosters + 2025-26 club
         trophy data, BUT only if coverage is broad enough that
         comparing computed-vs-handcoded across teams would be unfair.
      2. Hand-coded TEAM_TROPHY_SCORE dict (the legacy fallback).
      3. TROPHY_DEFAULT.

    Why the threshold: the computed scale and the hand-coded scale are
    not directly comparable. If we use computed for Argentina (squad
    published) while France/Brazil still get hand-coded numbers, we
    inadvertently penalise early-publishers. So we wait until enough
    teams have rosters that the field is broadly on the new scale.
    """
    computed = _load_computed_scores()
    if not _computed_coverage_sufficient(computed):
        # Not enough teams yet for the computed values to be fair.
        return TEAM_TROPHY_SCORE.get(team, TROPHY_DEFAULT)
    if team in computed:
        return computed[team]
    return TEAM_TROPHY_SCORE.get(team, TROPHY_DEFAULT)


def _computed_coverage_sufficient(computed: dict[str, float]) -> bool:
    """True once a critical mass of WC teams have computed scores.
    Uses the hand-coded TEAM_TROPHY_SCORE keyset as a proxy for the WC
    field (48 teams)."""
    if not computed:
        return False
    field = set(TEAM_TROPHY_SCORE)
    covered = sum(1 for t in computed if t in field)
    return covered / max(len(field), 1) >= COMPUTED_COVERAGE_THRESHOLD


_COMPUTED_CACHE: dict[str, float] | None = None


def _load_computed_scores() -> dict[str, float]:
    """Memoised lookup of the computed scores JSON. Returns empty dict
    if the file doesn't exist yet (e.g. fresh checkout before the
    scrapers have run)."""
    global _COMPUTED_CACHE
    if _COMPUTED_CACHE is not None:
        return _COMPUTED_CACHE
    import json
    from worldcup.db import PROJECT_ROOT
    path = PROJECT_ROOT / "data" / "raw" / "wc2026_trophy_scores.json"
    if not path.exists():
        _COMPUTED_CACHE = {}
        return _COMPUTED_CACHE
    data = json.loads(path.read_text())
    _COMPUTED_CACHE = {
        team: info["computed_score"]
        for team, info in data.get("teams", {}).items()
    }
    return _COMPUTED_CACHE


def trophy_z_scores(team_universe: list[str]) -> dict[str, float]:
    """Return team -> z-score of log(trophy score) across the WC field.

    Uses computed-when-available scores and falls back to hand-coded
    values for teams whose squads haven't been published yet.
    """
    vals = np.array([team_trophy_score(t) for t in team_universe])
    log_vals = np.log(vals + 1.0)  # +1 so the zeros don't blow up
    mean = log_vals.mean()
    std = log_vals.std() or 1.0
    return {t: float((log_vals[i] - mean) / std) for i, t in enumerate(team_universe)}


def adjustment_table(elo: dict[str, float]) -> list[dict]:
    """Diagnostic: per-team trophy score, z-score, and source."""
    teams = sorted(set(TEAM_TROPHY_SCORE) | set(_load_computed_scores()))
    z = trophy_z_scores(teams)
    computed = _load_computed_scores()
    rows = []
    for t in teams:
        score = team_trophy_score(t)
        rows.append({
            "team": t,
            "trophy_score": round(score, 2),
            "z": round(z[t], 2),
            "source": "computed" if t in computed else "hand-coded",
        })
    return sorted(rows, key=lambda r: -r["trophy_score"])
