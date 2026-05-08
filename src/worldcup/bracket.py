"""Generate a deterministic pre-tournament bracket prediction.

Approach (deliberately simple for v1, will be replaced by Monte Carlo
later):
  1. Pull all 72 group-stage fixtures for WC 2026
  2. Derive the 12 groups by graph-clustering the match list
  3. Simulate each group: in every match, higher-Elo team wins; rank by
     points, then Elo as tiebreaker
  4. Top 2 per group qualify (24 teams) plus 8 best third-place teams
     (ranked by Elo) = 32 advancing teams
  5. Seed the 32 by Elo (1 vs 32, 2 vs 31, ...) — note this is NOT the
     official FIFA bracket structure, just a clean placeholder
  6. Walk the bracket round by round, higher Elo always wins
  7. Write the result to data/bracket.json

Run: `uv run worldcup-bracket`
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from worldcup.db import PROJECT_ROOT, connect
from worldcup.elo import HOME_ADVANTAGE, MODEL_VERSION, expected_score

OUTPUT_PATH = PROJECT_ROOT / "data" / "bracket.json"


def load_data():
    with connect(read_only=True) as con:
        ratings = dict(
            con.execute(
                "SELECT team, rating FROM elo_current"
            ).fetchall()
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


def derive_groups(fixtures) -> list[list[str]]:
    """Find the 12 groups by treating fixtures as a graph: teams are
    nodes, every fixture is an edge. Each connected component is a
    group of 4."""
    adj: dict[str, set[str]] = defaultdict(set)
    for row in fixtures.itertuples(index=False):
        adj[row.home_team].add(row.away_team)
        adj[row.away_team].add(row.home_team)

    seen: set[str] = set()
    groups: list[list[str]] = []
    for team in sorted(adj):
        if team in seen:
            continue
        # Each group is fully connected (each team plays 3 others), so
        # the group is just the team plus its neighbors.
        group = sorted({team} | adj[team])
        seen.update(group)
        groups.append(group)
    return groups


def name_groups(groups: list[list[str]]) -> dict[str, list[str]]:
    """Label groups A..L by date order of their first match.
    Also returns a host-anchored ordering when possible (e.g. Mexico's
    group as A) — but we'll just use alphabetical letters for now."""
    # Sort groups by first appearance (alphabetical key on members)
    return {chr(ord("A") + i): g for i, g in enumerate(groups)}


def simulate_group(
    teams: list[str],
    fixtures,
    ratings: dict[str, float],
) -> list[dict]:
    """For one group: simulate the 6 matches deterministically.
    Returns standings sorted by (points DESC, rating DESC)."""
    members = set(teams)
    points = {t: 0 for t in teams}
    goals_for = {t: 0 for t in teams}  # tracked but unused for now

    # Find the 6 fixtures for this group
    group_fixtures = fixtures[
        fixtures["home_team"].isin(members)
        & fixtures["away_team"].isin(members)
    ]

    for row in group_fixtures.itertuples(index=False):
        h, a = row.home_team, row.away_team
        rh = ratings.get(h, 1500.0)
        ra = ratings.get(a, 1500.0)
        eff_h = rh + (0.0 if row.neutral else HOME_ADVANTAGE)
        if expected_score(eff_h, ra) >= 0.5:
            points[h] += 3
        else:
            points[a] += 3

    standings = sorted(
        teams,
        key=lambda t: (points[t], ratings.get(t, 1500.0)),
        reverse=True,
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
    """Higher Elo wins. Knockouts are at neutral venues so no home bonus."""
    return team_a if ratings.get(team_a, 1500) >= ratings.get(team_b, 1500) else team_b


def build_knockout(advancing: list[str], ratings: dict[str, float]) -> dict:
    """Seeded 32-team bracket: 1 v 32, 2 v 31, ... and walk it."""
    # Seed by Elo descending
    seeded = sorted(advancing, key=lambda t: ratings.get(t, 1500), reverse=True)

    def pair(round_teams: list[str]) -> list[dict]:
        n = len(round_teams)
        out = []
        for i in range(n // 2):
            a, b = round_teams[i], round_teams[n - 1 - i]
            w = winner_of(a, b, ratings)
            out.append(
                {
                    "team_a": a,
                    "team_b": b,
                    "rating_a": round(ratings.get(a, 1500)),
                    "rating_b": round(ratings.get(b, 1500)),
                    "winner": w,
                }
            )
        return out

    rounds = {}
    current = seeded
    for label in ["round_of_32", "round_of_16", "quarter_finals", "semi_finals", "final"]:
        matches = pair(current)
        rounds[label] = matches
        current = [m["winner"] for m in matches]

    rounds["champion"] = current[0]
    return rounds


def main() -> None:
    print("Generating placeholder bracket from current Elo")
    ratings, fixtures = load_data()
    print(f"  loaded {len(ratings)} team ratings, {len(fixtures)} group-stage fixtures")

    raw_groups = derive_groups(fixtures)
    if len(raw_groups) != 12:
        print(f"  WARNING: expected 12 groups, found {len(raw_groups)}")

    named_groups = name_groups(raw_groups)

    group_results = []
    third_place_pool = []
    advancing = []

    for label, members in named_groups.items():
        standings = simulate_group(members, fixtures, ratings)
        group_results.append({"name": label, "standings": standings})
        # Top 2 advance; 3rd goes to best-thirds pool
        advancing.extend([s["team"] for s in standings[:2]])
        third_place_pool.append(standings[2])

    # Pick 8 best third-place teams by Elo
    best_thirds = sorted(
        third_place_pool, key=lambda s: s["rating"], reverse=True
    )[:8]
    advancing.extend([s["team"] for s in best_thirds])

    print(f"  {len(advancing)} teams advancing to knockouts")

    knockout = build_knockout(advancing, ratings)

    # Mark third-place teams' qualification status in group_results
    advancing_set = set(advancing)
    for g in group_results:
        for s in g["standings"]:
            if s["position"] <= 2:
                s["qualifies_as"] = ["winner", "runner_up"][s["position"] - 1]
            elif s["team"] in advancing_set:
                s["qualifies_as"] = "best_third"
            else:
                s["qualifies_as"] = "eliminated"

    output = {
        "model_version": MODEL_VERSION,
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "groups": group_results,
        "knockout": knockout,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2))
    print(f"\n  wrote {OUTPUT_PATH}")
    print(f"\nPredicted champion: {knockout['champion']}")
    print(f"Predicted final: {knockout['final'][0]['team_a']} vs {knockout['final'][0]['team_b']}")
    print(f"Predicted semi-finalists: {[m['winner'] for m in knockout['semi_finals']]}")


if __name__ == "__main__":
    sys.exit(main())
