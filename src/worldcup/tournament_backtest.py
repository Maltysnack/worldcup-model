"""Tournament-level backtest.

Different from worldcup.backtest, which scores per-match W/D/L
predictions. This one runs the FULL simulator (groups + knockouts +
penalties) against a past World Cup using only data available before
kickoff, and asks: did the model think the actual winner was a real
contender? Did it have the actual semifinalists in its top 10? How
confident were we in the actual group winners?

Uses the legacy 32-team WC bracket structure (1998-2022).

Run: `uv run worldcup-backtest-tournament 2018`
Skip 2022 (locked hold-out test set).
"""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from datetime import date

import numpy as np

from worldcup.db import connect
from worldcup.goals import sim_match
from worldcup.profiles import current_profile, table_suffix

_PRED_TABLE = f"predictions{table_suffix(current_profile())}"

N_SIMS = 10_000
LOCKED_YEARS = {2022}  # never touch these

# 32-team WC bracket (1998-2022 format): top 2 from each of 8 groups
# advance, R16 pairings cross between groups, then standard bracket.
GROUPS_LEGACY = "ABCDEFGH"

# (match number, slot_a, slot_b) where slot is (kind, group_letter)
# kind: 'W' = group winner, 'R' = runner-up
R16_LEGACY = [
    (49, ('W', 'C'), ('R', 'D')),
    (50, ('W', 'A'), ('R', 'B')),
    (51, ('W', 'B'), ('R', 'A')),
    (52, ('W', 'D'), ('R', 'C')),
    (53, ('W', 'E'), ('R', 'F')),
    (54, ('W', 'G'), ('R', 'H')),
    (55, ('W', 'F'), ('R', 'E')),
    (56, ('W', 'H'), ('R', 'G')),
]
QF_LEGACY = [
    (57, 49, 50),
    (58, 53, 54),
    (59, 51, 52),
    (60, 55, 56),
]
SF_LEGACY = [
    (61, 57, 58),
    (62, 59, 60),
]
FINAL_LEGACY = (64, 61, 62)


def fetch_wc_teams_and_groups(year: int):
    """Derive the group assignments for a past WC by graph-clustering
    its group-stage matches. Returns {letter: [team1, ..., team4]}.

    We use the canonical FIFA convention: group A = the host's group
    (Mexico for 1986/2026, etc); for older WCs we just label them in
    discovery order. The labels themselves don't matter for the
    backtest results, only that each set of 4 mutually-playing teams
    forms one group.
    """
    with connect(read_only=True) as con:
        # Group stage matches only: each team plays exactly 3 others
        # (the other group members) and 0 from outside the group.
        df = con.execute(
            """
            SELECT date, home_team, away_team
            FROM matches
            WHERE tournament = 'FIFA World Cup'
              AND EXTRACT(YEAR FROM date) = ?
              AND home_score IS NOT NULL
            ORDER BY date
            """,
            [year],
        ).fetchdf()

    if df.empty:
        return None

    # 32-team WC has exactly 48 group-stage matches (8 groups x 6).
    # Sort by date and take the first 48; everything after is knockout.
    group_df = df.sort_values("date").head(48)

    adj: dict[str, set[str]] = defaultdict(set)
    for row in group_df.itertuples(index=False):
        adj[row.home_team].add(row.away_team)
        adj[row.away_team].add(row.home_team)

    seen: set[str] = set()
    groups: list[list[str]] = []
    for team in sorted(adj):
        if team in seen:
            continue
        # Each group is a 4-clique: team + its 3 opponents
        members = sorted({team} | adj[team])
        seen.update(members)
        if len(members) == 4:
            groups.append(members)

    if len(groups) != 8:
        return None
    return {chr(ord("A") + i): g for i, g in enumerate(groups)}


def fetch_pre_wc_ratings(year: int, teams: list[str]) -> dict[str, float]:
    """For each team, return their Elo immediately before their first
    WC match that year. Pulled from the predictions table populated by
    the chronological Elo walk."""
    with connect(read_only=True) as con:
        df = con.execute(
            f"""
            WITH first_match AS (
              SELECT team, MIN(match_date) AS d FROM (
                SELECT match_date, home_team AS team FROM {_PRED_TABLE}
                WHERE tournament = 'FIFA World Cup'
                  AND EXTRACT(YEAR FROM match_date) = ?
                UNION ALL
                SELECT match_date, away_team AS team FROM {_PRED_TABLE}
                WHERE tournament = 'FIFA World Cup'
                  AND EXTRACT(YEAR FROM match_date) = ?
              ) GROUP BY team
            )
            SELECT t.team,
              CASE WHEN p.home_team = t.team
                   THEN p.rating_home_pre ELSE p.rating_away_pre END AS rating
            FROM first_match t
            JOIN {_PRED_TABLE} p ON p.match_date = t.d
              AND (p.home_team = t.team OR p.away_team = t.team)
            """,
            [year, year],
        ).fetchdf()
    return {row.team: float(row.rating) for row in df.itertuples()}


def fetch_actual_results(year: int):
    """Return the actual outcomes: champion, finalists, semifinalists,
    group winners, plus 4th place."""
    with connect(read_only=True) as con:
        # Find the final (match with the latest date in the WC)
        df = con.execute(
            """
            SELECT date, home_team, away_team, home_score, away_score
            FROM matches
            WHERE tournament = 'FIFA World Cup'
              AND EXTRACT(YEAR FROM date) = ?
              AND home_score IS NOT NULL
            ORDER BY date
            """,
            [year],
        ).fetchdf()
    if df.empty:
        return None
    final = df.iloc[-1]
    third_place = df.iloc[-2]
    # SF losers are the 3rd/4th place participants
    finalist_a, finalist_b = final.home_team, final.away_team
    if final.home_score > final.away_score:
        champion, runner_up = finalist_a, finalist_b
    else:
        champion, runner_up = finalist_b, finalist_a
    # SFs: the two matches before the 3rd-place match
    sfs = df.iloc[-4:-2]
    semifinalists = set()
    for _, m in sfs.iterrows():
        semifinalists.add(m.home_team)
        semifinalists.add(m.away_team)
    return {
        "champion": champion,
        "runner_up": runner_up,
        "semifinalists": semifinalists,
    }


def simulate_tournament(
    ratings: dict[str, float],
    groups: dict[str, list[str]],
    n_sims: int = N_SIMS,
    seed: int = 42,
):
    """Return per-team round counts across n_sims tournaments."""
    rng = np.random.default_rng(seed)
    counts: dict[str, Counter] = defaultdict(Counter)
    group_winner_counts: dict[str, Counter] = defaultdict(Counter)

    for _ in range(n_sims):
        # Group stage
        winners, runners = {}, {}
        for letter, teams in groups.items():
            pts = {t: 0 for t in teams}
            gd = {t: 0 for t in teams}
            gf = {t: 0 for t in teams}
            for i in range(len(teams)):
                for j in range(i + 1, len(teams)):
                    sa, sb, win = sim_match(
                        ratings[teams[i]], ratings[teams[j]],
                        neutral=True, ko=False, rng=rng,
                    )
                    if win == 0:
                        pts[teams[i]] += 3
                    elif win == 1:
                        pts[teams[j]] += 3
                    else:
                        pts[teams[i]] += 1
                        pts[teams[j]] += 1
                    gd[teams[i]] += sa - sb
                    gd[teams[j]] += sb - sa
                    gf[teams[i]] += sa
                    gf[teams[j]] += sb
            sorted_teams = sorted(
                teams,
                key=lambda t: (pts[t], gd[t], gf[t], ratings[t]),
                reverse=True,
            )
            winners[letter] = sorted_teams[0]
            runners[letter] = sorted_teams[1]
            group_winner_counts[letter][sorted_teams[0]] += 1
            for t in sorted_teams[2:]:
                counts[t]["group_stage"] += 1

        # Knockouts
        match_winners = {}

        def resolve(spec):
            return winners[spec[1]] if spec[0] == 'W' else runners[spec[1]]

        for matchno, a, b in R16_LEGACY:
            ta, tb = resolve(a), resolve(b)
            sa, sb, win = sim_match(ratings[ta], ratings[tb], neutral=True, ko=True, rng=rng)
            w = ta if win == 0 else tb
            l = tb if win == 0 else ta
            match_winners[matchno] = w
            counts[l]["round_of_16"] += 1
        for matchno, fa, fb in QF_LEGACY:
            ta, tb = match_winners[fa], match_winners[fb]
            sa, sb, win = sim_match(ratings[ta], ratings[tb], neutral=True, ko=True, rng=rng)
            w = ta if win == 0 else tb
            l = tb if win == 0 else ta
            match_winners[matchno] = w
            counts[l]["quarter_finals"] += 1
        for matchno, fa, fb in SF_LEGACY:
            ta, tb = match_winners[fa], match_winners[fb]
            sa, sb, win = sim_match(ratings[ta], ratings[tb], neutral=True, ko=True, rng=rng)
            w = ta if win == 0 else tb
            l = tb if win == 0 else ta
            match_winners[matchno] = w
            counts[l]["semi_finals"] += 1
        matchno, fa, fb = FINAL_LEGACY
        ta, tb = match_winners[fa], match_winners[fb]
        sa, sb, win = sim_match(ratings[ta], ratings[tb], neutral=True, ko=True, rng=rng)
        w = ta if win == 0 else tb
        l = tb if win == 0 else ta
        counts[l]["runner_up"] += 1
        counts[w]["champion"] += 1

    return counts, group_winner_counts


def _round_probabilities(counts, n_sims):
    """For each team, compute cumulative P(reach round X)."""
    out = {}
    for team, c in counts.items():
        n = sum(c.values()) or 1
        p_champion = c["champion"] / n * 100
        p_final = (c["champion"] + c["runner_up"]) / n * 100
        p_semis = (c["champion"] + c["runner_up"] + c["semi_finals"]) / n * 100
        p_qf = (c["champion"] + c["runner_up"] + c["semi_finals"] + c["quarter_finals"]) / n * 100
        p_r16 = 100 - (c["group_stage"] / n * 100)
        out[team] = {
            "champion": p_champion,
            "final": p_final,
            "semis": p_semis,
            "qf": p_qf,
            "r16": p_r16,
        }
    return out


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: worldcup-backtest-tournament <year>")
        sys.exit(1)
    year = int(sys.argv[1])
    if year in LOCKED_YEARS:
        print(f"WC {year} is locked (held-out test set). Refusing to backtest.")
        sys.exit(1)

    groups = fetch_wc_teams_and_groups(year)
    if groups is None:
        print(f"Could not derive 8 groups for WC {year}")
        sys.exit(1)

    teams = [t for g in groups.values() for t in g]
    ratings = fetch_pre_wc_ratings(year, teams)
    missing = [t for t in teams if t not in ratings]
    if missing:
        print(f"WARNING: missing pre-WC rating for: {missing}")

    actual = fetch_actual_results(year)

    print(f"Tournament-level backtest: WC {year}")
    print(f"  Teams: {len(teams)}, Groups: {len(groups)}")
    print(f"  Actual: {actual['champion']} won, {actual['runner_up']} runner-up")
    print(f"  Actual semifinalists: {sorted(actual['semifinalists'])}")
    print(f"\n  Running {N_SIMS:,} simulations...")

    counts, group_winners = simulate_tournament(ratings, groups, n_sims=N_SIMS)
    probs = _round_probabilities(counts, N_SIMS)

    # Top 12 by championship probability
    print(f"\n  Top 12 by championship probability:")
    print(f"  {'team':25s} {'rating':>6} {'champ':>7} {'final':>7} {'semi':>7} {'QF':>7}")
    ranked = sorted(probs.items(), key=lambda kv: -kv[1]["champion"])
    for team, p in ranked[:12]:
        marker = ""
        if team == actual["champion"]:
            marker = "  <-- ACTUAL CHAMPION"
        elif team == actual["runner_up"]:
            marker = "  <-- runner-up"
        elif team in actual["semifinalists"]:
            marker = "  <-- SF"
        print(f"  {team:25s} {round(ratings.get(team, 1500)):>6} "
              f"{p['champion']:>6.1f}% {p['final']:>6.1f}% "
              f"{p['semis']:>6.1f}% {p['qf']:>6.1f}%{marker}")

    print(f"\n  Where the actual top 4 finished in our prediction:")
    for team in [actual["champion"], actual["runner_up"], *sorted(actual["semifinalists"] - {actual["champion"], actual["runner_up"]})]:
        if team not in probs:
            continue
        rank = next(i + 1 for i, (t, _) in enumerate(ranked) if t == team)
        p = probs[team]
        print(f"  {team:25s} ranked #{rank:2d} by champ%, "
              f"P(SF)={p['semis']:5.1f}%, P(F)={p['final']:5.1f}%, P(W)={p['champion']:5.1f}%")

    # Group winner predictions vs actuals
    print(f"\n  Group-winner predictions:")
    for letter in sorted(group_winners.keys()):
        sorted_winners = sorted(group_winners[letter].items(), key=lambda kv: -kv[1])
        top = sorted_winners[0]
        print(f"  Group {letter}: predicted {top[0]} ({100*top[1]/N_SIMS:.0f}%)  members={groups[letter]}")


if __name__ == "__main__":
    main()
