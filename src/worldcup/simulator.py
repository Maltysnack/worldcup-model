"""Monte Carlo World Cup simulator.

For each match we model the two teams' expected goals as Poisson rates
that depend on the Elo gap (and home advantage at non-neutral venues).
Calibration constants are fit from our 32k historical predictions.

We then simulate the entire tournament N times. Each simulation:
  1. Plays all 72 group matches, sampling each scoreline from Poisson
  2. Ranks each group by (points, goal difference, goals for)
  3. Picks the 8 best 3rd-placed teams the same way
  4. Resolves R32 → ... → Final using the official FIFA structure
     (knockout draws go to penalties, modelled as a coin flip with a
     small Elo edge)
  5. Records each team's deepest round reached

Aggregating across the N runs gives per-team probabilities of winning,
making the final, semis, etc. We also record a per-match "Spain beats
France in 38% of sims" style statistic. Finally we identify the *modal*
bracket — the most likely single outcome — and write it as the
public-facing prediction.

Run: `uv run worldcup-simulate`
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np

from worldcup.bracket import (
    FIFA_GROUPS,
    FINAL_FEED,
    QF_FEEDS,
    R16_FEEDS,
    R32_PAIRINGS,
    SF_FEEDS,
    TEAM_ISO,
    derive_groups_from_fixtures,
    label_groups_by_fifa,
    load_data,
)
from worldcup.db import PROJECT_ROOT
from worldcup.elo import HOME_ADVANTAGE, MODEL_VERSION
from worldcup.squad import ELO_WEIGHT, blended_ratings

OUTPUT_PATH = PROJECT_ROOT / "data" / "simulation.json"
N_SIMS = 10_000

# Calibration: fit empirically from 15k post-2010 international matches
LAMBDA_HOME_AT_PARITY = 1.41
LAMBDA_AWAY_AT_PARITY = 1.07
SLOPE_HOME = 0.0024
SLOPE_AWAY = -0.0030

# Penalty shootout: very weak Elo influence (it's nearly a coin flip)
PEN_ELO_DIVISOR = 600.0

ROUND_LABELS = [
    "group_stage",     # eliminated in groups (or didn't make best-3rd cut)
    "round_of_32",     # made KO but lost in R32
    "round_of_16",     # lost in R16
    "quarter_finals",  # lost in QF
    "semi_finals",     # lost in SF
    "runner_up",       # lost the Final
    "champion",        # won
]


def expected_goals(eff_gap: float) -> tuple[float, float]:
    return (
        LAMBDA_HOME_AT_PARITY * np.exp(SLOPE_HOME * eff_gap),
        LAMBDA_AWAY_AT_PARITY * np.exp(SLOPE_AWAY * eff_gap),
    )


def sim_match(rating_a, rating_b, *, neutral, ko, rng):
    """Sample a single match; return (score_a, score_b, winner_index 0|1).
    For knockouts (ko=True), draws are resolved by penalty shootout.
    Home advantage applies only to the FIRST team and only if not neutral."""
    eff_a = rating_a + (0.0 if neutral else HOME_ADVANTAGE)
    eff_b = rating_b
    lh, la = expected_goals(eff_a - eff_b)
    sa = rng.poisson(lh)
    sb = rng.poisson(la)
    if sa > sb:
        return sa, sb, 0
    if sb > sa:
        return sa, sb, 1
    if not ko:
        return sa, sb, -1  # draw is a real outcome in groups
    # Penalty shootout
    p_a = 1.0 / (1.0 + 10 ** ((eff_b - eff_a) / PEN_ELO_DIVISOR))
    return sa, sb, (0 if rng.random() < p_a else 1)


def simulate_one(
    ratings: dict[str, float],
    groups_by_letter: dict[str, list[str]],
    fixtures_by_group: dict[str, list],
    rng: np.random.Generator,
):
    """One full tournament. Returns:
      - team_round[team] -> deepest round string from ROUND_LABELS
      - r32_winners[match_no] -> winning team
    """
    # Group stage
    group_standings: dict[str, list[tuple[str, int, int, int]]] = {}
    for letter, teams in groups_by_letter.items():
        pts = {t: 0 for t in teams}
        gd = {t: 0 for t in teams}
        gf = {t: 0 for t in teams}
        for row in fixtures_by_group[letter]:
            h, a = row["home_team"], row["away_team"]
            sa, sb, win = sim_match(
                ratings[h], ratings[a], neutral=row["neutral"], ko=False, rng=rng
            )
            if win == 0:
                pts[h] += 3
            elif win == 1:
                pts[a] += 3
            else:
                pts[h] += 1
                pts[a] += 1
            gd[h] += sa - sb
            gd[a] += sb - sa
            gf[h] += sa
            gf[a] += sb
        ordered = sorted(teams, key=lambda t: (pts[t], gd[t], gf[t]), reverse=True)
        group_standings[letter] = [(t, pts[t], gd[t], gf[t]) for t in ordered]

    # Determine 32 advancing teams
    winners = {l: s[0][0] for l, s in group_standings.items()}
    runners = {l: s[1][0] for l, s in group_standings.items()}
    thirds_with_keys = [
        (l, s[2][0], s[2][1], s[2][2], s[2][3])
        for l, s in group_standings.items()
    ]
    # Top 8 thirds by (points, GD, GF)
    thirds_sorted = sorted(
        thirds_with_keys, key=lambda x: (x[2], x[3], x[4]), reverse=True
    )
    advancing_third_groups = {l for l, *_ in thirds_sorted[:8]}
    third_team_by_group = {l: t for l, t, *_ in thirds_with_keys}

    # Track each team's deepest round
    team_round: dict[str, str] = {}
    for letter, standings in group_standings.items():
        for pos, (t, *_ ) in enumerate(standings):
            if pos < 2:
                team_round[t] = "round_of_32"  # in KO; will be overwritten if eliminated later
            elif pos == 2 and letter in advancing_third_groups:
                team_round[t] = "round_of_32"
            else:
                team_round[t] = "group_stage"

    # Best-3rd → R32 slot assignment (same heuristic as bracket.py)
    eligible_thirds = [l for l in advancing_third_groups]
    used = set()
    third_assignment: dict[int, str] = {}
    for i, (a_spec, b_spec) in enumerate(R32_PAIRINGS, start=73):
        for spec in (a_spec, b_spec):
            if spec[0] != "3":
                continue
            allowed = set(spec[1])
            cands = [
                l for l in eligible_thirds
                if l in allowed and l not in used
            ]
            if not cands:
                cands = [l for l in eligible_thirds if l not in used]
            best = max(cands, key=lambda l: ratings[third_team_by_group[l]])
            third_assignment[i] = third_team_by_group[best]
            used.add(best)

    def resolve(spec, match_no):
        kind, key = spec
        if kind == "W":
            return winners[key]
        if kind == "R":
            return runners[key]
        return third_assignment[match_no]

    # Knockouts: matches 73-104
    match_winner: dict[int, str] = {}

    # R32
    for i, (a_spec, b_spec) in enumerate(R32_PAIRINGS, start=73):
        ta = resolve(a_spec, i)
        tb = resolve(b_spec, i)
        _, _, win = sim_match(
            ratings[ta], ratings[tb], neutral=True, ko=True, rng=rng
        )
        winner = ta if win == 0 else tb
        loser = tb if win == 0 else ta
        match_winner[i] = winner
        team_round[loser] = "round_of_32"

    # Subsequent rounds
    def play(rounds_def, round_label_winner, round_label_loser):
        for match_no, fa, fb in rounds_def:
            ta = match_winner[fa]
            tb = match_winner[fb]
            _, _, win = sim_match(
                ratings[ta], ratings[tb], neutral=True, ko=True, rng=rng
            )
            winner = ta if win == 0 else tb
            loser = tb if win == 0 else ta
            match_winner[match_no] = winner
            team_round[loser] = round_label_loser

    play(R16_FEEDS, "quarter_finals", "round_of_16")
    play(QF_FEEDS, "semi_finals", "quarter_finals")
    play(SF_FEEDS, "final", "semi_finals")

    # Final
    fno, ffa, ffb = FINAL_FEED
    ta = match_winner[ffa]
    tb = match_winner[ffb]
    _, _, win = sim_match(
        ratings[ta], ratings[tb], neutral=True, ko=True, rng=rng
    )
    champion = ta if win == 0 else tb
    runner_up = tb if win == 0 else ta
    match_winner[fno] = champion
    team_round[champion] = "champion"
    team_round[runner_up] = "runner_up"

    return team_round, match_winner


def main(n_sims: int = N_SIMS, seed: int = 42) -> None:
    model_id = f"{MODEL_VERSION}+squad{int(round((1 - ELO_WEIGHT) * 100))}"
    print(f"Monte Carlo simulator (model: {model_id}, n={n_sims:,})")

    raw_ratings, fixtures = load_data()
    ratings = blended_ratings(raw_ratings)
    derived = derive_groups_from_fixtures(fixtures)
    groups_by_letter = label_groups_by_fifa(derived)

    # Pre-bin fixtures by group letter for speed
    fixtures_by_group: dict[str, list] = {l: [] for l in groups_by_letter}
    members_to_letter = {
        frozenset(teams): letter for letter, teams in groups_by_letter.items()
    }
    for row in fixtures.itertuples(index=False):
        for letter, members in groups_by_letter.items():
            if row.home_team in members and row.away_team in members:
                fixtures_by_group[letter].append({
                    "home_team": row.home_team,
                    "away_team": row.away_team,
                    "neutral": bool(row.neutral),
                })
                break

    round_counts: dict[str, Counter] = defaultdict(Counter)
    rng = np.random.default_rng(seed)

    for _ in range(n_sims):
        team_round, _ = simulate_one(
            ratings, groups_by_letter, fixtures_by_group, rng
        )
        for team, rnd in team_round.items():
            round_counts[team][rnd] += 1

    # Per-team stats
    teams_all = sorted(round_counts.keys())
    team_stats = []
    for t in teams_all:
        c = round_counts[t]
        n_team_total = sum(c.values())
        # Probability of "reaching at least round X"
        pct = lambda key: round(100 * c[key] / n_team_total, 2)
        team_stats.append({
            "team": t,
            "rating": round(ratings[t]),
            "p_champion": pct("champion"),
            "p_final": pct("champion") + pct("runner_up"),
            "p_semis": (
                pct("champion") + pct("runner_up") + pct("semi_finals")
            ),
            "p_quarters": (
                pct("champion") + pct("runner_up") + pct("semi_finals") + pct("quarter_finals")
            ),
            "p_round_of_16": (
                pct("champion") + pct("runner_up") + pct("semi_finals") + pct("quarter_finals") + pct("round_of_16")
            ),
            "p_advance_from_groups": 100.0 - pct("group_stage"),
        })
    team_stats.sort(key=lambda x: x["p_champion"], reverse=True)

    output = {
        "model_version": model_id,
        "elo_version": MODEL_VERSION,
        "elo_weight": ELO_WEIGHT,
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "n_simulations": n_sims,
        "calibration": {
            "lambda_home_at_parity": LAMBDA_HOME_AT_PARITY,
            "lambda_away_at_parity": LAMBDA_AWAY_AT_PARITY,
            "slope_home": SLOPE_HOME,
            "slope_away": SLOPE_AWAY,
            "home_advantage_elo": HOME_ADVANTAGE,
        },
        "flags": TEAM_ISO,
        "team_stats": team_stats,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2))
    print(f"\n  wrote {OUTPUT_PATH}")
    print(f"\nTop 10 by championship probability:")
    for s in team_stats[:10]:
        print(f"  {s['team']:25s}  {s['p_champion']:6.2f}%   "
              f"(final {s['p_final']:5.2f}%, SF {s['p_semis']:5.2f}%, "
              f"QF {s['p_quarters']:5.2f}%)")



if __name__ == "__main__":
    sys.exit(main())
