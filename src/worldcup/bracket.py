"""Generate a deterministic pre-tournament bracket prediction using
the official FIFA 2026 bracket structure.

Approach:
  1. Pull all 72 group-stage fixtures for WC 2026
  2. Derive the 12 groups by graph-clustering the match list, then map
     each derived group to its official FIFA letter (A-L) by team
     membership against FIFA_GROUPS
  3. Simulate each group: in every match, higher-Elo team wins;
     standings sorted by points then Elo
  4. The 8 best third-place teams (ranked by Elo) qualify alongside
     the 24 group winners and runners-up
  5. Apply the FIFA-published Round-of-32 pairings, then the deterministic
     mapping forward to Final
  6. Higher Elo wins each knockout match; Final winner is the champion
  7. Write data/bracket.json

Run: `uv run worldcup-bracket`
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime

from worldcup.db import PROJECT_ROOT, connect
from worldcup.elo import HOME_ADVANTAGE, MODEL_VERSION, expected_score
from worldcup.squad import ELO_WEIGHT, blended_ratings

OUTPUT_PATH = PROJECT_ROOT / "data" / "bracket.json"

# ISO 3166-1 alpha-2 codes (with subdivision codes for England/Scotland)
# used to fetch flags from https://flagcdn.com/<size>/<code>.png
TEAM_ISO: dict[str, str] = {
    "Algeria": "dz", "Argentina": "ar", "Australia": "au", "Austria": "at",
    "Belgium": "be", "Bosnia and Herzegovina": "ba", "Brazil": "br",
    "Canada": "ca", "Cape Verde": "cv", "Colombia": "co", "Croatia": "hr",
    "Curaçao": "cw", "Czech Republic": "cz", "DR Congo": "cd",
    "Ecuador": "ec", "Egypt": "eg", "England": "gb-eng", "France": "fr",
    "Germany": "de", "Ghana": "gh", "Haiti": "ht", "Iran": "ir", "Iraq": "iq",
    "Ivory Coast": "ci", "Japan": "jp", "Jordan": "jo", "Mexico": "mx",
    "Morocco": "ma", "Netherlands": "nl", "New Zealand": "nz", "Norway": "no",
    "Panama": "pa", "Paraguay": "py", "Portugal": "pt", "Qatar": "qa",
    "Saudi Arabia": "sa", "Scotland": "gb-sct", "Senegal": "sn",
    "South Africa": "za", "South Korea": "kr", "Spain": "es", "Sweden": "se",
    "Switzerland": "ch", "Tunisia": "tn", "Turkey": "tr",
    "United States": "us", "Uruguay": "uy", "Uzbekistan": "uz",
}

# Official FIFA 2026 group draw
FIFA_GROUPS: dict[str, list[str]] = {
    "A": ["Mexico", "South Africa", "South Korea", "Czech Republic"],
    "B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["United States", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Curaçao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Iraq", "Norway"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "DR Congo", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}

# Round-of-32 pairings, official FIFA structure (matches 73-88).
# Each entry: (slot_a_spec, slot_b_spec)
#   ("W", "X")  = winner of group X
#   ("R", "X")  = runner-up of group X
#   ("3", "ABCDE") = best 3rd from groups in the string
R32_PAIRINGS: list[tuple[tuple[str, str], tuple[str, str]]] = [
    (("R", "A"), ("R", "B")),                  # 73
    (("W", "E"), ("3", "ABCDF")),              # 74
    (("W", "F"), ("R", "C")),                  # 75
    (("W", "C"), ("R", "F")),                  # 76
    (("W", "I"), ("3", "CDFGH")),              # 77
    (("R", "E"), ("R", "I")),                  # 78
    (("W", "A"), ("3", "CEFHI")),              # 79
    (("W", "L"), ("3", "EHIJK")),              # 80
    (("W", "D"), ("3", "BEFIJ")),              # 81
    (("W", "G"), ("3", "AEHIJ")),              # 82
    (("R", "K"), ("R", "L")),                  # 83
    (("W", "H"), ("R", "J")),                  # 84
    (("W", "B"), ("3", "EFGIJ")),              # 85
    (("W", "J"), ("R", "H")),                  # 86
    (("W", "K"), ("3", "DEIJL")),              # 87
    (("R", "D"), ("R", "G")),                  # 88
]

# Forward bracket: Match N -> (Match A, Match B). Both feeders must be
# resolved before this match.
R16_FEEDS = [
    (89, 74, 77),
    (90, 73, 75),
    (91, 76, 78),
    (92, 79, 80),
    (93, 83, 84),
    (94, 81, 82),
    (95, 86, 88),
    (96, 85, 87),
]
QF_FEEDS = [
    (97, 89, 90),
    (98, 93, 94),
    (99, 91, 92),
    (100, 95, 96),
]
SF_FEEDS = [
    (101, 97, 98),
    (102, 99, 100),
]
FINAL_FEED = (104, 101, 102)


def load_data():
    with connect(read_only=True) as con:
        ratings = dict(
            con.execute("SELECT team, rating FROM elo_current").fetchall()
        )
        fixtures = con.execute(
            """
            SELECT date, home_team, away_team, neutral
            FROM matches
            WHERE tournament = 'FIFA World Cup'
              AND date >= DATE '2026-06-11'
              AND home_score IS NULL
            ORDER BY date, home_team
            """
        ).fetchdf()
    return ratings, fixtures


def derive_groups_from_fixtures(fixtures) -> list[list[str]]:
    adj: dict[str, set[str]] = defaultdict(set)
    for row in fixtures.itertuples(index=False):
        adj[row.home_team].add(row.away_team)
        adj[row.away_team].add(row.home_team)
    seen, groups = set(), []
    for team in sorted(adj):
        if team in seen:
            continue
        group = sorted({team} | adj[team])
        seen.update(group)
        groups.append(group)
    return groups


def label_groups_by_fifa(derived: list[list[str]]) -> dict[str, list[str]]:
    """Match each derived group to a FIFA group letter by exact set match."""
    derived_sets = {frozenset(g): g for g in derived}
    labelled = {}
    for letter, teams in FIFA_GROUPS.items():
        key = frozenset(teams)
        if key not in derived_sets:
            raise ValueError(
                f"Group {letter} mismatch: FIFA says {teams} "
                f"but no matching derived group found."
            )
        labelled[letter] = derived_sets[key]
    return labelled


def simulate_group(teams, fixtures, ratings):
    """Deterministic group sim: higher Elo wins each match. 3 pts for a
    win, 0 for loss (no draws under deterministic rule)."""
    members = set(teams)
    points = {t: 0 for t in teams}
    group_fixtures = fixtures[
        fixtures["home_team"].isin(members) & fixtures["away_team"].isin(members)
    ]
    for row in group_fixtures.itertuples(index=False):
        rh = ratings.get(row.home_team, 1500.0)
        ra = ratings.get(row.away_team, 1500.0)
        eff_h = rh + (0.0 if row.neutral else HOME_ADVANTAGE)
        winner = row.home_team if expected_score(eff_h, ra) >= 0.5 else row.away_team
        points[winner] += 3
    standings = sorted(
        teams, key=lambda t: (points[t], ratings.get(t, 1500.0)), reverse=True
    )
    return [
        {
            "team": t,
            "rating": round(ratings.get(t, 1500.0)),
            "predicted_points": points[t],
            "position": i + 1,
        }
        for i, t in enumerate(standings)
    ]


def winner_of(team_a: str, team_b: str, ratings: dict[str, float]) -> str:
    """Higher Elo wins. Knockouts are at neutral venues."""
    return team_a if ratings.get(team_a, 1500) >= ratings.get(team_b, 1500) else team_b


def assign_best_thirds(
    third_place_by_group: dict[str, dict],
    advancing_groups: set[str],
    ratings: dict[str, float],
):
    """For each R32 slot needing a "best 3rd from {groups}", pick the
    highest-Elo eligible 3rd-place team that hasn't been assigned yet.
    Slots are processed in order so highest-priority slots get first pick.

    This is an approximation of FIFA's exact lookup table but produces
    sensible deterministic pairings with the right teams advancing.
    """
    # Identify the 8 best third-place teams overall by Elo
    eligible = sorted(
        advancing_groups,
        key=lambda g: ratings.get(third_place_by_group[g]["team"], 1500),
        reverse=True,
    )[:8]

    third_assignments: dict[int, str] = {}
    used_groups: set[str] = set()

    for i, (a, b) in enumerate(R32_PAIRINGS, start=73):
        for spec in (a, b):
            if spec[0] != "3":
                continue
            allowed = set(spec[1])
            # Pick highest-Elo eligible third-placer from allowed groups
            candidates = [
                g for g in eligible if g in allowed and g not in used_groups
            ]
            if not candidates:
                # Fallback: any unused 3rd placer
                candidates = [g for g in eligible if g not in used_groups]
            if not candidates:
                raise RuntimeError(f"No 3rd-place team available for match {i}")
            best = max(
                candidates,
                key=lambda g: ratings.get(third_place_by_group[g]["team"], 1500),
            )
            third_assignments[i] = third_place_by_group[best]["team"]
            used_groups.add(best)

    return third_assignments


def resolve_slot(spec, group_winners, group_runners, third_assignments, match_no):
    kind, key = spec
    if kind == "W":
        return group_winners[key]
    if kind == "R":
        return group_runners[key]
    if kind == "3":
        return third_assignments[match_no]
    raise ValueError(f"Unknown slot spec: {spec}")


def build_match(team_a, team_b, ratings):
    return {
        "team_a": team_a,
        "team_b": team_b,
        "rating_a": round(ratings.get(team_a, 1500)),
        "rating_b": round(ratings.get(team_b, 1500)),
        "winner": winner_of(team_a, team_b, ratings),
    }


def main() -> None:
    model_id = f"{MODEL_VERSION}+squad{int(round((1 - ELO_WEIGHT) * 100))}"
    print(f"Generating bracket using official FIFA 2026 structure (model: {model_id})")
    raw_ratings, fixtures = load_data()
    ratings = blended_ratings(raw_ratings)
    print(f"  loaded {len(ratings)} team ratings, {len(fixtures)} group fixtures")

    derived = derive_groups_from_fixtures(fixtures)
    if len(derived) != 12:
        raise RuntimeError(f"Expected 12 groups, found {len(derived)}")

    groups_by_letter = label_groups_by_fifa(derived)

    # Simulate each group
    group_results = []
    group_winners: dict[str, str] = {}
    group_runners: dict[str, str] = {}
    third_place_by_group: dict[str, dict] = {}

    for letter in "ABCDEFGHIJKL":
        standings = simulate_group(groups_by_letter[letter], fixtures, ratings)
        group_results.append({"name": letter, "standings": standings})
        group_winners[letter] = standings[0]["team"]
        group_runners[letter] = standings[1]["team"]
        third_place_by_group[letter] = standings[2]

    # 8 best third-place teams advance
    best_thirds = sorted(
        third_place_by_group.items(),
        key=lambda kv: ratings.get(kv[1]["team"], 1500),
        reverse=True,
    )[:8]
    advancing_third_groups = {letter for letter, _ in best_thirds}

    # Assign best 3rds to R32 slots
    third_assignments = assign_best_thirds(
        third_place_by_group, advancing_third_groups, ratings
    )

    # Mark qualification status on group standings
    for g in group_results:
        for s in g["standings"]:
            if s["position"] == 1:
                s["qualifies_as"] = "winner"
            elif s["position"] == 2:
                s["qualifies_as"] = "runner_up"
            elif s["position"] == 3 and g["name"] in advancing_third_groups:
                s["qualifies_as"] = "best_third"
            else:
                s["qualifies_as"] = "eliminated"

    # Build R32 matches
    matches: dict[int, dict] = {}
    for i, (a, b) in enumerate(R32_PAIRINGS, start=73):
        team_a = resolve_slot(a, group_winners, group_runners, third_assignments, i)
        team_b = resolve_slot(b, group_winners, group_runners, third_assignments, i)
        matches[i] = build_match(team_a, team_b, ratings)

    # Walk forward through the bracket
    for feeds in (R16_FEEDS, QF_FEEDS, SF_FEEDS):
        for match_no, fa, fb in feeds:
            matches[match_no] = build_match(
                matches[fa]["winner"], matches[fb]["winner"], ratings
            )

    fno, ffa, ffb = FINAL_FEED
    matches[fno] = build_match(
        matches[ffa]["winner"], matches[ffb]["winner"], ratings
    )

    knockout = {
        "round_of_32": [matches[i] for i in range(73, 89)],
        "round_of_16": [matches[i] for i, _, _ in R16_FEEDS],
        "quarter_finals": [matches[i] for i, _, _ in QF_FEEDS],
        "semi_finals": [matches[i] for i, _, _ in SF_FEEDS],
        "final": [matches[FINAL_FEED[0]]],
        "champion": matches[FINAL_FEED[0]]["winner"],
    }

    output = {
        "model_version": model_id,
        "elo_version": MODEL_VERSION,
        "elo_weight": ELO_WEIGHT,
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "bracket_format": "fifa-2026",
        "flags": TEAM_ISO,
        "groups": group_results,
        "knockout": knockout,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2))
    print(f"\n  wrote {OUTPUT_PATH}")
    print(f"\nPredicted champion: {knockout['champion']}")
    final_match = knockout["final"][0]
    print(f"Predicted final: {final_match['team_a']} vs {final_match['team_b']}")
    sf_teams = []
    for m in knockout["semi_finals"]:
        sf_teams += [m["team_a"], m["team_b"]]
    print(f"Predicted semi-finalists: {', '.join(sf_teams)}")


if __name__ == "__main__":
    sys.exit(main())
