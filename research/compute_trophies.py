"""Compute per-team trophy scores from real roster + club-trophy data.

Replaces the hand-coded TEAM_TROPHY_SCORE in trophies.py with values
derived from:
  - Wikipedia squad rosters (player -> club mapping)
  - 2025-26 confirmed club trophy results (club -> trophy weight)

For each WC team that has a published roster, sums the trophy weights
across the squad: each player contributes the sum of all trophy weights
their club won. No minutes weighting yet (treats every squad member as
contributing equally), that's a follow-up via FBref player stats.

For teams without a published roster, the function returns None and
the caller should fall back to the hand-coded TEAM_TROPHY_SCORE dict.

Run: `uv run worldcup-compute-trophies`

Output goes to data/raw/wc2026_trophy_scores.json with both the
computed value and a per-player breakdown for inspection.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timezone

from club_trophies import (
    CLUB_TROPHIES_2025_26,
    LAST_UPDATED,
    PENDING,
    trophies_for_club,
)
from wiki_rosters import fetch_and_parse

from worldcup.db import PROJECT_ROOT

OUTPUT_PATH = PROJECT_ROOT / "data" / "raw" / "wc2026_trophy_scores.json"


# Baseline trophy contribution given to every WC qualifier, on top of
# their actual trophy-weight aggregation. Rationale: WC qualification
# itself is a competence signal, a squad of professional internationals
# who beat their continent's qualifying field. Pure "0 trophy weight"
# would mean we treat a team like Bosnia (no big-club players, but
# clearly elite within their region) as having zero squad strength via
# this layer, which is too punitive. Setting BASELINE to the value our
# hand-coded scheme gave to the smallest WC teams (Haiti, Cape Verde,
# Curaçao, Saudi Arabia were all 2-3 hand-coded) is the principled
# floor.
TROPHY_BASELINE = 3.0


def normalise_score(raw: float, players_in_squad: int) -> float:
    """Scale the raw sum + add the WC-qualification baseline.

    Without the baseline a team whose published squad happens to lack
    any 2025-26 trophy-winning-club players (e.g. Bosnia, May 2026)
    scores 0, which produces a 50+ Elo drop relative to hand-coded -
    far too sensitive to one signal layer.

    The 0.85 multiplier on raw is calibrated so France/Spain/Brazil
    land near their old hand-coded values (~30-38) once their squads
    are scraped. Will be re-tuned post-FBref minutes data.
    """
    if players_in_squad == 0:
        return TROPHY_BASELINE
    return TROPHY_BASELINE + raw * 0.85


def compute() -> dict:
    rosters = fetch_and_parse(refresh=False)
    breakdown = {}
    for r in rosters:
        per_player = []
        raw_total = 0.0
        for p in r.players:
            trophies = trophies_for_club(p.club)
            player_score = sum(w for _, w in trophies)
            raw_total += player_score
            if player_score > 0:
                per_player.append({
                    "name": p.name,
                    "club": p.club,
                    "trophies": [t for t, _ in trophies],
                    "score": round(player_score, 2),
                })
        normalised = normalise_score(raw_total, len(r.players))
        # Sort player breakdown by contribution
        per_player.sort(key=lambda x: -x["score"])
        breakdown[r.team] = {
            "raw_total": round(raw_total, 2),
            "computed_score": round(normalised, 2),
            "n_players": len(r.players),
            "is_provisional": r.is_provisional,
            "n_contributing": len(per_player),
            "contributing_players": per_player[:15],  # top 15 for inspection
        }

    out = {
        "computed_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "club_trophy_data_last_updated": LAST_UPDATED,
        "pending_trophies": PENDING,
        "teams": breakdown,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    return out


def get_computed_score(team: str) -> float | None:
    """Look up the computed trophy score for a team. Returns None if
    we don't have a roster for them yet."""
    if not OUTPUT_PATH.exists():
        return None
    data = json.loads(OUTPUT_PATH.read_text())
    info = data.get("teams", {}).get(team)
    if info is None:
        return None
    return info["computed_score"]


def main() -> None:
    print("Computing per-team trophy scores from Wikipedia rosters + 2025-26 club trophies\n")
    out = compute()

    teams = out["teams"]
    print(f"{len(teams)} teams with computed scores:\n")
    print(f"  {'team':30s} {'computed':>9} {'raw':>7} {'players':>9} {'contrib':>9}")
    print(f"  {'-'*30} {'-'*9} {'-'*7} {'-'*9} {'-'*9}")
    for team in sorted(teams, key=lambda t: -teams[t]["computed_score"]):
        d = teams[team]
        print(f"  {team:30s} {d['computed_score']:>9.2f} {d['raw_total']:>7.2f} "
              f"{d['n_players']:>9} {d['n_contributing']:>9}")

    print("\nTop contributing players overall (per team):")
    for team in sorted(teams, key=lambda t: -teams[t]["computed_score"])[:5]:
        d = teams[team]
        print(f"\n  {team}:")
        for p in d["contributing_players"][:5]:
            print(f"    {p['name']:25s} {p['club']:20s}  +{p['score']:.1f}  {p['trophies']}")

    print(f"\nData last updated: {out['club_trophy_data_last_updated']}")
    print(f"Still pending: {', '.join(out['pending_trophies'])}")
    print(f"\nWrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
