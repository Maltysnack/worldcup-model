"""Historical upset-score corpus for tournament chaos calibration.

For each historical tournament we compute one number: how chaotic it
was relative to what would have been expected from the pre-tournament
Elo rankings. Lower = favourites mostly held; higher = lots of upsets.

Definition: for each team in the tournament, compare actual round
reached to the round their pre-tournament Elo rank would have predicted.
Sum the absolute differences across teams. Normalise by team count so
brackets of different sizes are comparable.

  upset_score = sum_t |actual_round(t) - expected_round(t)| / n_teams

Round ordinals (small = early exit, large = won it):
  0  group stage exit
  1  R32 loss (only in 48-team format like 2026 WC)
  2  R16 loss
  3  QF loss
  4  SF loss
  5  Final loss (runner-up)
  6  Champion

Expected round comes from Elo rank within the field at kickoff:
  - rank 1                         -> Champion (6)
  - rank 2                         -> Runner-up (5)
  - ranks 3..4                     -> SF (4)
  - ranks 5..8                     -> QF (3)
  - ranks 9..16                    -> R16 (2)
  - ranks 17..32 (48-team only)    -> R32 (1)
  - rest                           -> group stage (0)

This means each "expected slot" is the same size as the actual slot
counts, so a no-upset tournament would score 0.

Run: `uv run worldcup-build-historical-corpus`
Output: data/raw/historical_upset_corpus.json
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime, timezone

import numpy as np

from worldcup.db import PROJECT_ROOT, connect


# Tournament editions to include in the corpus. We restrict to high-
# quality international tournaments since 1998 (post-format-change era
# for WC). AFCON, Asian Cup, Gold Cup are deliberately excluded — their
# upset patterns differ from WC/Euro/Copa and would skew the median.
ELIGIBLE_TOURNAMENTS = {
    "FIFA World Cup",
    "UEFA Euro",
    "Copa América",
}
EARLIEST_YEAR = 1998
HOLDOUT_YEARS = {("FIFA World Cup", 2022)}  # locked WC test set

ROUND_GROUPS = 0
ROUND_R32 = 1
ROUND_R16 = 2
ROUND_QF = 3
ROUND_SF = 4
ROUND_FINAL = 5  # runner-up
ROUND_CHAMPION = 6


@dataclass
class TournamentResult:
    tournament: str
    year: int
    n_teams: int
    n_rounds: int  # number of distinct round levels above groups (KO depth)
    teams: dict[str, int]   # team -> actual round ordinal
    elo_rank: dict[str, int]  # team -> pre-tournament Elo rank (1 = best)
    upset_score: float       # sum |actual - expected| / n_teams
    raw_upset_sum: int       # sum |actual - expected| (un-normalised)


def fetch_pretourn_elo(tournament: str, year: int) -> dict[str, float]:
    """Each team's Elo immediately before their first match in this
    tournament edition."""
    with connect(read_only=True) as con:
        df = con.execute(
            """
            WITH first_match AS (
                SELECT team, MIN(match_date) AS d FROM (
                    SELECT match_date, home_team AS team FROM predictions
                    WHERE tournament = ? AND EXTRACT(YEAR FROM match_date) = ?
                    UNION ALL
                    SELECT match_date, away_team AS team FROM predictions
                    WHERE tournament = ? AND EXTRACT(YEAR FROM match_date) = ?
                ) GROUP BY team
            )
            SELECT t.team,
                   CASE WHEN p.home_team = t.team
                        THEN p.rating_home_pre ELSE p.rating_away_pre END AS rating
            FROM first_match t
            JOIN predictions p ON p.match_date = t.d
                AND p.tournament = ?
                AND (p.home_team = t.team OR p.away_team = t.team)
            """,
            [tournament, year, tournament, year, tournament],
        ).fetchdf()
    return {row.team: float(row.rating) for row in df.itertuples()}


def classify_actual_rounds(tournament: str, year: int) -> tuple[dict[str, int], int, int]:
    """Walk through this tournament's matches and assign each team the
    deepest round they reached. Returns (round_map, n_teams, ko_depth).

    Strategy: find the Final (last match by date), work backward. Each
    earlier KO round has 2x as many matches as the next. Group-stage
    teams have their last match before the first knockout match.
    """
    with connect(read_only=True) as con:
        df = con.execute(
            """
            SELECT date, home_team, away_team, home_score, away_score
            FROM matches
            WHERE tournament = ? AND EXTRACT(YEAR FROM date) = ?
              AND home_score IS NOT NULL
            ORDER BY date, home_team
            """,
            [tournament, year],
        ).fetchdf()
    if df.empty:
        return {}, 0, 0

    teams = sorted(set(df["home_team"]) | set(df["away_team"]))
    n_teams = len(teams)

    # Determine knockout depth from team count
    # Standard mappings:
    #   16 teams: groups -> QF -> SF -> F          (depth 3, KO rounds = QF/SF/F)
    #   24 teams: groups -> R16 -> QF -> SF -> F   (depth 4)
    #   32 teams: groups -> R16 -> QF -> SF -> F   (depth 4)
    #   48 teams: groups -> R32 -> R16 -> QF -> SF -> F  (depth 5)
    if n_teams <= 16:
        ko_depth = 3
    elif n_teams <= 32:
        ko_depth = 4
    else:
        ko_depth = 5

    # Final = chronologically last match (by row order, since we sorted by date)
    final_row = df.iloc[-1]
    champion, runner_up = (
        (final_row["home_team"], final_row["away_team"])
        if final_row["home_score"] > final_row["away_score"]
        else (final_row["away_team"], final_row["home_team"])
    )

    actual_round: dict[str, int] = {}
    actual_round[champion] = ROUND_CHAMPION
    actual_round[runner_up] = ROUND_FINAL

    # SFs: 2 matches, located 2 or 3 spots from end (depending on whether
    # there's a 3rd place playoff). Identify by tracing the SF participants
    # who lost.
    # Heuristic: any non-final, non-3rd-place match where one of the two
    # teams is a finalist = SF. Look at the last few matches.
    sf_losers: set[str] = set()
    # Skip the final, then look for matches with finalist participation
    finalists = {champion, runner_up}
    for idx in range(len(df) - 2, max(-1, len(df) - 6), -1):
        m = df.iloc[idx]
        # SF match: one team is a finalist (they advanced from this match)
        teams_in = {m["home_team"], m["away_team"]}
        finalist_in_match = teams_in & finalists
        if finalist_in_match:
            loser = (m["away_team"] if m["home_score"] > m["away_score"]
                     else m["home_team"])
            if loser in finalists:
                continue  # this is the final itself, already handled
            sf_losers.add(loser)
        if len(sf_losers) >= 2:
            break

    for t in sf_losers:
        actual_round[t] = ROUND_SF

    # QF/R16/R32 losers: find each team's deepest match.
    # Then assign rounds based on:
    #   - team played in groups only -> ROUND_GROUPS
    #   - team's last match was in KO and we know the round structure
    # Simpler: map each match to a round based on its sequential position
    # for KO matches.

    # Find first KO match (first match after group stage)
    # Group stage: each team plays exactly (group_size - 1) matches
    # For 4-team groups: 3 group matches each; for n_teams=24 with 6 groups of 4 = 36 group matches
    n_groups = (n_teams // 4) if n_teams in (16, 32, 48) else (n_teams // 4) if n_teams >= 16 else 4
    # Special-case the typical structures:
    if n_teams == 16:
        n_group_matches = 24  # 4 groups * 6 matches
    elif n_teams == 24:
        n_group_matches = 36  # 6 groups * 6 matches
    elif n_teams == 32:
        n_group_matches = 48  # 8 groups * 6 matches
    elif n_teams == 48:
        n_group_matches = 72  # 12 groups * 6 matches
    else:
        # Approximation for other sizes
        n_groups = max(1, n_teams // 4)
        n_group_matches = n_groups * 6

    # For each team, find their last match index
    last_match_idx: dict[str, int] = {}
    for idx in range(len(df)):
        m = df.iloc[idx]
        last_match_idx[m["home_team"]] = idx
        last_match_idx[m["away_team"]] = idx

    # If a team's last match was in the group-stage section, they exited groups.
    # If KO and they aren't already classified, assign the round based on their
    # match position relative to the KO sequence.
    # The KO matches start at index n_group_matches and there are 2^ko_depth - 1
    # of them (for clean brackets). For a 32-team WC: 8 R16 + 4 QF + 2 SF + 1 3rd + 1 F = 16
    for t in teams:
        if t in actual_round:
            continue
        idx = last_match_idx.get(t)
        if idx is None:
            actual_round[t] = ROUND_GROUPS
            continue
        if idx < n_group_matches:
            actual_round[t] = ROUND_GROUPS
        else:
            # This team's last match was in the KO phase; figure out which round
            ko_pos = idx - n_group_matches  # 0-indexed within KO
            # KO match counts: R32 (16), R16 (8), QF (4), SF (2), [3rd], F (1)
            # Walk through round levels
            if n_teams >= 48:  # has R32
                if ko_pos < 16:
                    actual_round[t] = ROUND_R32
                    continue
                ko_pos -= 16
            if n_teams >= 24:  # has R16
                if ko_pos < 8:
                    actual_round[t] = ROUND_R16
                    continue
                ko_pos -= 8
            if ko_pos < 4:
                actual_round[t] = ROUND_QF
                continue
            ko_pos -= 4
            # SF + Final + maybe 3rd place
            actual_round[t] = ROUND_SF  # already covered above ideally
    return actual_round, n_teams, ko_depth


def expected_rounds_from_elo(elo: dict[str, float], n_teams: int,
                             ko_depth: int) -> dict[str, int]:
    """Map each team's Elo rank to an expected round, with slot counts
    matching the actual round-occupancy in a no-upset tournament."""
    teams_sorted = sorted(elo.items(), key=lambda kv: -kv[1])

    # Expected slot count per round (matches actual occupancy)
    # Champion: 1, RU: 1, SF: 2, QF: 4, R16: 8, R32: 16, Groups: rest
    if n_teams >= 48:
        slot_assignment = (
            [ROUND_CHAMPION] * 1 + [ROUND_FINAL] * 1 + [ROUND_SF] * 2
            + [ROUND_QF] * 4 + [ROUND_R16] * 8 + [ROUND_R32] * 16
            + [ROUND_GROUPS] * (n_teams - 32)
        )
    elif n_teams >= 24:
        slot_assignment = (
            [ROUND_CHAMPION] * 1 + [ROUND_FINAL] * 1 + [ROUND_SF] * 2
            + [ROUND_QF] * 4 + [ROUND_R16] * 8
            + [ROUND_GROUPS] * (n_teams - 16)
        )
    else:  # 16 teams
        slot_assignment = (
            [ROUND_CHAMPION] * 1 + [ROUND_FINAL] * 1 + [ROUND_SF] * 2
            + [ROUND_QF] * 4
            + [ROUND_GROUPS] * (n_teams - 8)
        )

    if len(slot_assignment) != n_teams:
        # Pad/trim safely
        if len(slot_assignment) < n_teams:
            slot_assignment += [ROUND_GROUPS] * (n_teams - len(slot_assignment))
        else:
            slot_assignment = slot_assignment[:n_teams]

    return {team: slot_assignment[i] for i, (team, _) in enumerate(teams_sorted)}


def compute_tournament(tournament: str, year: int) -> TournamentResult | None:
    elo = fetch_pretourn_elo(tournament, year)
    if not elo:
        return None
    actual, n_teams, ko_depth = classify_actual_rounds(tournament, year)
    if not actual:
        return None
    # Restrict to teams we have BOTH ratings AND results for
    teams = sorted(set(elo) & set(actual))
    if len(teams) < 8:
        return None
    elo_subset = {t: elo[t] for t in teams}
    actual_subset = {t: actual[t] for t in teams}
    expected = expected_rounds_from_elo(elo_subset, len(teams), ko_depth)
    raw_sum = sum(abs(actual_subset[t] - expected[t]) for t in teams)
    score = raw_sum / len(teams)
    # Elo rank for transparency
    teams_sorted = sorted(teams, key=lambda t: -elo[t])
    elo_rank = {t: i + 1 for i, t in enumerate(teams_sorted)}
    return TournamentResult(
        tournament=tournament, year=year,
        n_teams=len(teams), n_rounds=ko_depth,
        teams=actual_subset, elo_rank=elo_rank,
        upset_score=round(score, 4),
        raw_upset_sum=int(raw_sum),
    )


def main() -> None:
    print(f"Building historical upset-score corpus")
    print(f"Tournaments: {sorted(ELIGIBLE_TOURNAMENTS)}")
    print(f"Earliest year: {EARLIEST_YEAR}, hold-outs: {HOLDOUT_YEARS}\n")

    with connect(read_only=True) as con:
        editions = con.execute(
            """
            SELECT tournament, EXTRACT(YEAR FROM date)::int AS yr
            FROM matches
            WHERE tournament = ANY(?)
              AND home_score IS NOT NULL
              AND date >= MAKE_DATE(?, 1, 1)
            GROUP BY tournament, yr
            ORDER BY tournament, yr
            """,
            [list(ELIGIBLE_TOURNAMENTS), EARLIEST_YEAR],
        ).fetchall()

    results: list[TournamentResult] = []
    for tname, yr in editions:
        if (tname, yr) in HOLDOUT_YEARS:
            print(f"  skip {tname} {yr} (hold-out)")
            continue
        r = compute_tournament(tname, yr)
        if r is None:
            print(f"  skip {tname} {yr} (insufficient data)")
            continue
        # Show the standout upsets in this tournament
        upsets = []
        teams_sorted_by_dev = sorted(
            r.teams.keys(),
            key=lambda t: -abs(r.teams[t] - expected_rounds_from_elo(
                {x: 0 for x in r.teams},  # dummy; we'll get expected via re-rank
                r.n_teams, r.n_rounds,
            )[t]) if False else 0,  # placeholder; we'll compute below
        )
        expected = {}
        teams_by_rank = sorted(r.elo_rank.keys(), key=lambda t: r.elo_rank[t])
        if r.n_teams >= 48:
            slots = ([ROUND_CHAMPION]*1 + [ROUND_FINAL]*1 + [ROUND_SF]*2 +
                     [ROUND_QF]*4 + [ROUND_R16]*8 + [ROUND_R32]*16 +
                     [ROUND_GROUPS]*(r.n_teams - 32))
        elif r.n_teams >= 24:
            slots = ([ROUND_CHAMPION]*1 + [ROUND_FINAL]*1 + [ROUND_SF]*2 +
                     [ROUND_QF]*4 + [ROUND_R16]*8 +
                     [ROUND_GROUPS]*(r.n_teams - 16))
        else:
            slots = ([ROUND_CHAMPION]*1 + [ROUND_FINAL]*1 + [ROUND_SF]*2 +
                     [ROUND_QF]*4 + [ROUND_GROUPS]*(r.n_teams - 8))
        slots = slots[:r.n_teams]
        for i, t in enumerate(teams_by_rank):
            expected[t] = slots[i]
        notable = sorted(
            r.teams.keys(),
            key=lambda t: -abs(r.teams[t] - expected[t]),
        )[:3]
        notable_str = ", ".join(
            f"{t} (rank {r.elo_rank[t]}, expected R{expected[t]}, got R{r.teams[t]})"
            for t in notable
        )
        print(f"  {tname:20s} {yr}  n={r.n_teams:>2}  upset_score={r.upset_score:.3f}  raw={r.raw_upset_sum}")
        print(f"      notable: {notable_str}")
        results.append(r)

    print(f"\n{len(results)} tournaments in corpus")
    scores = np.array([r.upset_score for r in results])
    print(f"\nUpset-score statistics (per-team avg deviation):")
    print(f"  min:    {scores.min():.3f}")
    print(f"  q25:    {np.quantile(scores, 0.25):.3f}")
    print(f"  median: {np.quantile(scores, 0.5):.3f}")
    print(f"  mean:   {scores.mean():.3f}")
    print(f"  q75:    {np.quantile(scores, 0.75):.3f}")
    print(f"  max:    {scores.max():.3f}")

    # Per-tournament breakdown
    print(f"\nBy tournament family:")
    by_t: dict[str, list[float]] = defaultdict(list)
    for r in results:
        by_t[r.tournament].append(r.upset_score)
    for t in sorted(by_t):
        arr = np.array(by_t[t])
        print(f"  {t:20s} n={len(arr):>2}  median={np.median(arr):.3f}  range=[{arr.min():.3f}, {arr.max():.3f}]")

    out_path = PROJECT_ROOT / "data" / "raw" / "historical_upset_corpus.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "computed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tournaments_included": sorted(ELIGIBLE_TOURNAMENTS),
        "earliest_year": EARLIEST_YEAR,
        "holdout_years": [list(t) for t in HOLDOUT_YEARS],
        "n_tournaments": len(results),
        "score_stats": {
            "min": float(scores.min()),
            "q25": float(np.quantile(scores, 0.25)),
            "median": float(np.quantile(scores, 0.5)),
            "mean": float(scores.mean()),
            "q75": float(np.quantile(scores, 0.75)),
            "max": float(scores.max()),
        },
        "tournaments": [asdict(r) for r in results],
    }, indent=2, ensure_ascii=False))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    sys.exit(main())
