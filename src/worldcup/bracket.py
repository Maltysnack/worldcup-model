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
import math
import sys
from collections import defaultdict
from datetime import datetime

import numpy as np

from worldcup.db import PROJECT_ROOT, connect
from worldcup.elo import MODEL_VERSION
from worldcup.profiles import current_profile, file_suffix, table_suffix
from worldcup.form import apply_form
from worldcup.goals import dc_joint_pmf, expected_goals, sim_match
from worldcup.squad import ELO_WEIGHT, blended_ratings
from worldcup.host import apply_host_bonus
from worldcup.trophies import apply_nt_bonus

# Number of group sims when computing predicted standings. The bracket
# is a single deterministic prediction, but the group standings driving
# it come from the average of GROUP_SIM_N Monte Carlo runs of the 6
# group fixtures, so the predicted bracket reflects the simulator's
# probability model rather than a naive "higher Elo wins" rule.
GROUP_SIM_N = 4000

_PROFILE = current_profile()
OUTPUT_PATH = PROJECT_ROOT / "data" / f"bracket{file_suffix(_PROFILE)}.json"
_ELO_TABLE = f"elo_current{table_suffix(_PROFILE)}"

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

# Visual ordering: top-to-bottom so adjacent matches in each round
# feed into the same match in the next round. Derived by walking the
# tree backwards from the Final. With this order, R16 matches sit
# midway between the two R32 matches that feed them, etc.
R32_VISUAL_ORDER = [74, 77, 73, 75, 83, 84, 81, 82, 76, 78, 79, 80, 86, 88, 85, 87]
R16_VISUAL_ORDER = [89, 90, 93, 94, 91, 92, 95, 96]
QF_VISUAL_ORDER = [97, 98, 99, 100]
SF_VISUAL_ORDER = [101, 102]


def load_data():
    with connect(read_only=True) as con:
        ratings = dict(
            con.execute(f"SELECT team, rating FROM {_ELO_TABLE}").fetchall()
        )
        fixtures = con.execute(
            """
            SELECT date, home_team, away_team, neutral, city, country
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


def simulate_group_mc(teams, fixtures, ratings, rng):
    """Monte Carlo group sim. Plays the 6 fixtures GROUP_SIM_N times,
    averages points/goal-diff/goals-for per team. Uses the same Poisson
    goals model as the simulator, so the predicted standings reflect
    real probabilities (draws happen, upsets happen) rather than a
    naive "higher-Elo-always-wins" rule."""
    members = set(teams)
    pts_total = {t: 0.0 for t in teams}
    gd_total = {t: 0.0 for t in teams}
    gf_total = {t: 0.0 for t in teams}

    group_fixtures = fixtures[
        fixtures["home_team"].isin(members) & fixtures["away_team"].isin(members)
    ]
    fixture_rows = list(group_fixtures.itertuples(index=False))

    for _ in range(GROUP_SIM_N):
        sim_pts = {t: 0 for t in teams}
        sim_gd = {t: 0 for t in teams}
        sim_gf = {t: 0 for t in teams}
        for row in fixture_rows:
            h, a = row.home_team, row.away_team
            sa, sb, win = sim_match(
                ratings[h], ratings[a],
                neutral=bool(row.neutral), ko=False, rng=rng,
            )
            if win == 0:
                sim_pts[h] += 3
            elif win == 1:
                sim_pts[a] += 3
            else:
                sim_pts[h] += 1
                sim_pts[a] += 1
            sim_gd[h] += sa - sb
            sim_gd[a] += sb - sa
            sim_gf[h] += sa
            sim_gf[a] += sb
        for t in teams:
            pts_total[t] += sim_pts[t]
            gd_total[t] += sim_gd[t]
            gf_total[t] += sim_gf[t]

    avg_pts = {t: pts_total[t] / GROUP_SIM_N for t in teams}
    avg_gd = {t: gd_total[t] / GROUP_SIM_N for t in teams}
    avg_gf = {t: gf_total[t] / GROUP_SIM_N for t in teams}

    # FIFA tiebreaker order: points → goal difference → goals for
    standings = sorted(
        teams,
        key=lambda t: (avg_pts[t], avg_gd[t], avg_gf[t]),
        reverse=True,
    )
    return [
        {
            "team": t,
            "rating": round(ratings.get(t, 1500.0)),
            "predicted_points": round(avg_pts[t], 1),
            "predicted_goal_diff": round(avg_gd[t], 2),
            "predicted_goals_for": round(avg_gf[t], 2),
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
):
    """For each R32 slot needing a "best 3rd from {groups}", pick the
    eligible 3rd-place team with the highest predicted points (then
    goal diff, then goals for) that hasn't been assigned yet. This is
    an approximation of FIFA's exact 8-of-12 lookup table but produces
    sensible pairings.
    """
    def rank_key(letter):
        s = third_place_by_group[letter]
        return (s["predicted_points"], s["predicted_goal_diff"], s["predicted_goals_for"])

    third_assignments: dict[int, str] = {}
    used_groups: set[str] = set()

    for i, (a, b) in enumerate(R32_PAIRINGS, start=73):
        for spec in (a, b):
            if spec[0] != "3":
                continue
            allowed = set(spec[1])
            candidates = [
                g for g in advancing_groups
                if g in allowed and g not in used_groups
            ]
            if not candidates:
                candidates = [g for g in advancing_groups if g not in used_groups]
            if not candidates:
                raise RuntimeError(f"No 3rd-place team available for match {i}")
            best = max(candidates, key=rank_key)
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
    """Deterministic knockout match prediction with a bolder scoreline.

    Higher-Elo team wins (winner_of). The score uses an asymmetric
    rounding of expected goals that scales boldness with the Elo gap:

      - Favourite: ceil(lambda_winner - 0.2). Lifts 1.4 -> 2, 2.4 -> 3.
        A team expected to score 1.5+ goals predicts at least 2.
      - Underdog:  floor(lambda_loser). Drops 1.07 -> 1, 0.7 -> 0.
        An opponent expected to score under 1 predicts a clean sheet.

    Result: close matches stay at 1-1 (pens), modest favourites go 2-0
    or 2-1, strong favourites go 3-0, blowouts go 4-0. Reads as the
    model's actual confidence rather than the timid "everything's a
    1-0" you get from picking the joint-PMF mode (which we tried and
    converged to 1-0 across 75% of ties).

    If the chosen scores match (genuine parity in the lambdas) we
    mark pens=True and the higher-Elo team progresses, matching the
    simulator's tiny-Elo-edge shootout.
    """
    ra = ratings.get(team_a, 1500)
    rb = ratings.get(team_b, 1500)
    winner = winner_of(team_a, team_b, ratings)
    loser = team_b if winner == team_a else team_a
    rw = ratings.get(winner, 1500)
    rl = ratings.get(loser, 1500)

    lh, la = expected_goals(rw - rl)
    # Asymmetric rounding: round winner up, loser down.
    score_w = max(1, math.ceil(float(lh) - 0.2))
    score_l = max(0, math.floor(float(la)))
    pens = False
    if score_w == score_l:
        pens = True  # parity tie: shootout, higher Elo wins
    elif score_w < score_l:
        # Should not happen with positive Elo gap; guard regardless.
        score_w = score_l + 1

    if winner == team_a:
        score_a, score_b = score_w, score_l
    else:
        score_a, score_b = score_l, score_w

    return {
        "team_a": team_a,
        "team_b": team_b,
        "rating_a": round(ra),
        "rating_b": round(rb),
        "winner": winner,
        "score_a": int(score_a),
        "score_b": int(score_b),
        "pens": pens,
    }


def main() -> None:
    model_id = f"{MODEL_VERSION}+squad{int(round((1 - ELO_WEIGHT) * 100))}+form"
    print(f"Generating bracket using official FIFA 2026 structure (model: {model_id})")
    raw_ratings, fixtures = load_data()
    ratings = apply_host_bonus(
        apply_nt_bonus(apply_form(blended_ratings(raw_ratings))),
        "FIFA World Cup", 2026,
    )
    print(f"  loaded {len(ratings)} team ratings, {len(fixtures)} group fixtures")

    derived = derive_groups_from_fixtures(fixtures)
    if len(derived) != 12:
        raise RuntimeError(f"Expected 12 groups, found {len(derived)}")

    groups_by_letter = label_groups_by_fifa(derived)

    # Simulate each group via Monte Carlo
    rng = np.random.default_rng(123)
    group_results = []
    group_winners: dict[str, str] = {}
    group_runners: dict[str, str] = {}
    third_place_by_group: dict[str, dict] = {}

    for letter in "ABCDEFGHIJKL":
        standings = simulate_group_mc(
            groups_by_letter[letter], fixtures, ratings, rng
        )
        group_results.append({"name": letter, "standings": standings})
        group_winners[letter] = standings[0]["team"]
        group_runners[letter] = standings[1]["team"]
        third_place_by_group[letter] = standings[2]

    # 8 best third-place teams advance, ranked by FIFA criteria:
    # average points → goal diff → goals for (all from MC).
    best_thirds = sorted(
        third_place_by_group.items(),
        key=lambda kv: (
            kv[1]["predicted_points"],
            kv[1]["predicted_goal_diff"],
            kv[1]["predicted_goals_for"],
        ),
        reverse=True,
    )[:8]
    advancing_third_groups = {letter for letter, _ in best_thirds}

    # Assign best 3rds to R32 slots
    third_assignments = assign_best_thirds(
        third_place_by_group, advancing_third_groups
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
        "round_of_32": [matches[i] for i in R32_VISUAL_ORDER],
        "round_of_16": [matches[i] for i in R16_VISUAL_ORDER],
        "quarter_finals": [matches[i] for i in QF_VISUAL_ORDER],
        "semi_finals": [matches[i] for i in SF_VISUAL_ORDER],
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
