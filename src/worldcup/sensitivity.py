"""Sensitivity tests for squad-weight blend and Friendly K-factor.

Two sweeps:

  1. Squad weight (Elo / Squad / Trophy split)
     Holds Trophy at 0.15, slides Elo down / Squad up. Asks: how
     squishy is the Top 12 if we trust market values more or less?

  2. Friendly K-factor (within v4a-flat-mild)
     Drops the Friendly K from 30 down through 25, 20, 15, 10. Asks:
     does discounting friendlies harder change which teams are at the
     top, and does it improve historical log-loss?

For each variant we run a forward 2026 Monte Carlo (3000 sims) and
report the Top 12 by championship probability, plus a side-by-side
delta vs the current production model. K-variants also get scored on
historical per-match log-loss (2010+ major tournaments).

Squad-weight sweep reuses one Elo walk (the blend changes only post-
Elo). K-sweep walks Elo fresh for each variant.

Run: `uv run python -m worldcup.sensitivity`
"""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass
from datetime import date

import numpy as np

from worldcup import squad as squad_mod
from worldcup.bracket import (
    FIFA_GROUPS,
    label_groups_by_fifa,
    derive_groups_from_fixtures,
)
from worldcup.db import connect
from worldcup.elo import run as elo_run
from worldcup.form import apply_form
from worldcup.host import apply_host_bonus
from worldcup.iterate_profiles import score_profile
from worldcup.profiles import PROFILES, Profile, V4A_FLAT_MILD
from worldcup.simulator import simulate_one
from worldcup.squad import blended_ratings
from worldcup.trophies import apply_nt_bonus

N_SIMS = 3000  # lighter than the production 10k since we run many variants
TODAY = date(2026, 5, 12)
WC_YEAR = 2026


# --------------------------------------------------------------------- #
# Variant definitions
# --------------------------------------------------------------------- #

@dataclass
class SquadVariant:
    label: str
    elo_w: float
    squad_w: float
    trophy_w: float

    @property
    def is_baseline(self) -> bool:
        return (
            abs(self.elo_w - squad_mod.ELO_WEIGHT) < 1e-9
            and abs(self.squad_w - squad_mod.SQUAD_WEIGHT) < 1e-9
            and abs(self.trophy_w - squad_mod.TROPHY_WEIGHT) < 1e-9
        )


SQUAD_SWEEP: list[SquadVariant] = [
    SquadVariant("elo80/sqd05/tr15", 0.80, 0.05, 0.15),
    SquadVariant("elo75/sqd10/tr15", 0.75, 0.10, 0.15),
    SquadVariant("elo70/sqd15/tr15", 0.70, 0.15, 0.15),
    SquadVariant("elo65/sqd20/tr15", 0.65, 0.20, 0.15),  # current
    SquadVariant("elo60/sqd25/tr15", 0.60, 0.25, 0.15),
    SquadVariant("elo55/sqd30/tr15", 0.55, 0.30, 0.15),
    SquadVariant("elo50/sqd35/tr15", 0.50, 0.35, 0.15),
    SquadVariant("elo40/sqd45/tr15", 0.40, 0.45, 0.15),
]


def make_friendly_variant(friendly_k: int) -> Profile:
    kf = dict(V4A_FLAT_MILD.k_factors)
    kf["Friendly"] = friendly_k
    return Profile(
        name=f"v4a-friendly-{friendly_k:02d}",
        k_factors=kf,
        default_k=V4A_FLAT_MILD.default_k,
    )


FRIENDLY_K_SWEEP: list[Profile] = [
    make_friendly_variant(k) for k in [30, 25, 20, 15, 10]
]


# --------------------------------------------------------------------- #
# Forward simulation helpers (mirrors simulator.main but parameterised)
# --------------------------------------------------------------------- #

def load_wc_fixtures():
    """Pull WC 2026 group-stage fixtures (the only ones simulate_one
    needs to know about — knockouts come from R32_PAIRINGS etc)."""
    with connect(read_only=True) as con:
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
    return fixtures


def ratings_dict_from_walk(final_ratings_df) -> dict[str, float]:
    return {row.team: float(row.rating) for row in final_ratings_df.itertuples(index=False)}


def build_groups_and_fixtures():
    fixtures = load_wc_fixtures()
    derived = derive_groups_from_fixtures(fixtures)
    if len(derived) != 12:
        raise RuntimeError(f"Expected 12 groups, found {len(derived)}")
    groups_by_letter = label_groups_by_fifa(derived)

    fixtures_by_group: dict[str, list] = {l: [] for l in groups_by_letter}
    for row in fixtures.itertuples(index=False):
        for letter, members in groups_by_letter.items():
            if row.home_team in members and row.away_team in members:
                fixtures_by_group[letter].append({
                    "home_team": row.home_team,
                    "away_team": row.away_team,
                    "neutral": bool(row.neutral),
                })
                break
    return groups_by_letter, fixtures_by_group


def run_forward_sim(
    raw_ratings: dict[str, float],
    groups_by_letter,
    fixtures_by_group,
    *,
    elo_w: float,
    squad_w: float,
    trophy_w: float,
    n_sims: int = N_SIMS,
    seed: int = 42,
):
    """Apply blend (with given weights) + form + NT + host, then sim."""
    # Monkey-patch the module-level blend weights, restore on return.
    saved = (squad_mod.ELO_WEIGHT, squad_mod.SQUAD_WEIGHT, squad_mod.TROPHY_WEIGHT)
    squad_mod.ELO_WEIGHT = elo_w
    squad_mod.SQUAD_WEIGHT = squad_w
    squad_mod.TROPHY_WEIGHT = trophy_w
    try:
        ratings = apply_host_bonus(
            apply_nt_bonus(apply_form(blended_ratings(raw_ratings))),
            "FIFA World Cup", WC_YEAR,
        )
    finally:
        squad_mod.ELO_WEIGHT, squad_mod.SQUAD_WEIGHT, squad_mod.TROPHY_WEIGHT = saved

    round_counts: dict[str, Counter] = defaultdict(Counter)
    rng = np.random.default_rng(seed)
    for _ in range(n_sims):
        team_round, _ = simulate_one(ratings, groups_by_letter, fixtures_by_group, rng)
        for team, rnd in team_round.items():
            round_counts[team][rnd] += 1

    stats = []
    for t, c in round_counts.items():
        n_t = sum(c.values()) or 1
        p_champ = 100 * c["champion"] / n_t
        p_final = 100 * (c["champion"] + c["runner_up"]) / n_t
        p_semi = 100 * (c["champion"] + c["runner_up"] + c["semi_finals"]) / n_t
        stats.append({
            "team": t,
            "rating": round(ratings[t]),
            "p_champion": round(p_champ, 2),
            "p_final": round(p_final, 2),
            "p_semis": round(p_semi, 2),
        })
    stats.sort(key=lambda s: -s["p_champion"])
    return stats


# --------------------------------------------------------------------- #
# Output helpers
# --------------------------------------------------------------------- #

def print_top_n(stats, n=12, marker_team=None):
    print(f"  {'rank':>4}  {'team':25s} {'rating':>6} {'champ%':>7} {'final%':>7} {'semi%':>7}")
    for i, s in enumerate(stats[:n], start=1):
        mark = "  <--" if (marker_team and s["team"] == marker_team) else ""
        print(f"  {i:>4d}  {s['team']:25s} {s['rating']:>6} "
              f"{s['p_champion']:>6.2f}% {s['p_final']:>6.2f}% {s['p_semis']:>6.2f}%{mark}")


def delta_table(baseline_stats, variant_stats, top_n=12):
    """Show p_champion delta for the union of baseline-top-N and variant-top-N."""
    base_lookup = {s["team"]: s for s in baseline_stats}
    var_lookup = {s["team"]: s for s in variant_stats}
    base_top = {s["team"] for s in baseline_stats[:top_n]}
    var_top = {s["team"] for s in variant_stats[:top_n]}
    union = base_top | var_top

    base_rank = {s["team"]: i + 1 for i, s in enumerate(baseline_stats)}
    var_rank = {s["team"]: i + 1 for i, s in enumerate(variant_stats)}

    rows = []
    for team in union:
        b = base_lookup.get(team, {"p_champion": 0.0})
        v = var_lookup.get(team, {"p_champion": 0.0})
        rows.append({
            "team": team,
            "base_pct": b["p_champion"],
            "var_pct": v["p_champion"],
            "delta": v["p_champion"] - b["p_champion"],
            "base_rank": base_rank.get(team, 99),
            "var_rank": var_rank.get(team, 99),
        })
    rows.sort(key=lambda r: -r["var_pct"])

    print(f"  {'team':25s} {'base':>8} {'variant':>8} {'delta':>8}  rank-shift")
    for r in rows:
        rank_shift = ""
        if r["base_rank"] != r["var_rank"]:
            rank_shift = f"  #{r['base_rank']:>2} → #{r['var_rank']:>2}"
        sign = "+" if r["delta"] >= 0 else ""
        print(f"  {r['team']:25s} {r['base_pct']:>6.2f}%  {r['var_pct']:>6.2f}%  "
              f"{sign}{r['delta']:>5.2f}pp{rank_shift}")


# --------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------- #

def main() -> None:
    print("=" * 90)
    print("SENSITIVITY TEST  ·  squad-weight + Friendly-K sweeps  ·  Top 12 forward sim")
    print("=" * 90)

    print("\n[setup] Loading WC 2026 fixtures + groups...")
    groups_by_letter, fixtures_by_group = build_groups_and_fixtures()

    # ============================================================== #
    # Sweep 1: Squad weights (single Elo walk, vary blend)
    # ============================================================== #
    print("\n[setup] Walking Elo once with current production profile (v4a-flat-mild)...")
    final_df, _ = elo_run(
        min_date=date(1990, 1, 1),
        max_date=TODAY,
        k_factors=V4A_FLAT_MILD.k_factors,
        default_k=V4A_FLAT_MILD.default_k,
        quiet=True,
    )
    raw_ratings = ratings_dict_from_walk(final_df)
    # Restrict to teams we have squad data for (the WC field)
    wc_teams = {t for teams in FIFA_GROUPS.values() for t in teams}
    raw_ratings = {t: r for t, r in raw_ratings.items() if t in wc_teams}
    print(f"           {len(raw_ratings)} WC team ratings ready")

    print("\n" + "=" * 90)
    print("SWEEP 1: SQUAD-WEIGHT SENSITIVITY")
    print("=" * 90)
    print("Blend = w_elo · z(Elo) + w_squad · z(log market value) + w_trophy · z(log club trophies)")
    print(f"Current production: Elo=0.65 / Squad=0.20 / Trophy=0.15\n")

    squad_results: dict[str, list[dict]] = {}
    for v in SQUAD_SWEEP:
        tag = "  (current)" if v.is_baseline else ""
        print(f"\n--- {v.label}{tag} ---")
        stats = run_forward_sim(
            raw_ratings, groups_by_letter, fixtures_by_group,
            elo_w=v.elo_w, squad_w=v.squad_w, trophy_w=v.trophy_w,
            n_sims=N_SIMS,
        )
        squad_results[v.label] = stats
        print_top_n(stats, n=12)

    # Side-by-side deltas vs baseline
    baseline = next(v for v in SQUAD_SWEEP if v.is_baseline)
    base_stats = squad_results[baseline.label]

    print("\n" + "-" * 90)
    print(f"SQUAD SWEEP DELTAS vs baseline ({baseline.label})")
    print("-" * 90)
    for v in SQUAD_SWEEP:
        if v.is_baseline:
            continue
        print(f"\n{v.label}  (Elo {v.elo_w:.2f} / Squad {v.squad_w:.2f} / Trophy {v.trophy_w:.2f})")
        delta_table(base_stats, squad_results[v.label], top_n=12)

    # ============================================================== #
    # Sweep 2: Friendly-K (full Elo re-walk per variant)
    # ============================================================== #
    print("\n\n" + "=" * 90)
    print("SWEEP 2: FRIENDLY-K SENSITIVITY")
    print("=" * 90)
    print("v4a-flat-mild baseline (WC=50, Cont=45, WCQ=40, NL=40, CQ=38).")
    print("Only the Friendly K is varied. All other layers (form/NT/host/blend) held constant.\n")

    fr_results: dict[str, list[dict]] = {}
    fr_logloss: dict[str, tuple[float, float, int]] = {}

    for prof in FRIENDLY_K_SWEEP:
        fr_k = prof.k_factors["Friendly"]
        tag = "  (current v4a)" if fr_k == 30 else ""
        print(f"\n--- Friendly K = {fr_k}  ({prof.name}){tag} ---")
        # Per-match log-loss on 2010+ majors
        ll, brier, n = score_profile(prof)
        fr_logloss[prof.name] = (ll, brier, n)
        print(f"  per-match log-loss (2010+ majors, n={n}): ll={ll:.4f}  brier={brier:.4f}")

        # Forward sim
        final_df_v, _ = elo_run(
            min_date=date(1990, 1, 1),
            max_date=TODAY,
            k_factors=prof.k_factors,
            default_k=prof.default_k,
            quiet=True,
        )
        raw_v = ratings_dict_from_walk(final_df_v)
        raw_v = {t: r for t, r in raw_v.items() if t in wc_teams}
        stats = run_forward_sim(
            raw_v, groups_by_letter, fixtures_by_group,
            elo_w=squad_mod.ELO_WEIGHT,
            squad_w=squad_mod.SQUAD_WEIGHT,
            trophy_w=squad_mod.TROPHY_WEIGHT,
            n_sims=N_SIMS,
        )
        fr_results[prof.name] = stats
        print_top_n(stats, n=12)

    # Friendly K deltas vs Friendly=30
    base_fr = FRIENDLY_K_SWEEP[0]
    base_fr_stats = fr_results[base_fr.name]

    print("\n" + "-" * 90)
    print(f"FRIENDLY-K SWEEP DELTAS vs Friendly=30 (current v4a)")
    print("-" * 90)
    for prof in FRIENDLY_K_SWEEP[1:]:
        fr_k = prof.k_factors["Friendly"]
        print(f"\nFriendly K = {fr_k}")
        delta_table(base_fr_stats, fr_results[prof.name], top_n=12)

    # ============================================================== #
    # Headline summary
    # ============================================================== #
    print("\n\n" + "=" * 90)
    print("HEADLINE SUMMARY")
    print("=" * 90)

    print("\nSquad sweep — top-3 champion% across variants:")
    print(f"  {'variant':22s}  {'1st':30s}  {'2nd':30s}  {'3rd':30s}")
    for v in SQUAD_SWEEP:
        s = squad_results[v.label]
        line = f"  {v.label:22s}  "
        for k in range(3):
            line += f"{s[k]['team']:14s} {s[k]['p_champion']:5.2f}%   "
        print(line)

    print("\nFriendly-K sweep — log-loss (lower=better) + top-3 champion%:")
    print(f"  {'Fr K':>4}  {'log-loss':>9}  {'1st':30s}  {'2nd':30s}  {'3rd':30s}")
    for prof in FRIENDLY_K_SWEEP:
        fr_k = prof.k_factors["Friendly"]
        ll, _, _ = fr_logloss[prof.name]
        s = fr_results[prof.name]
        line = f"  {fr_k:>4d}  {ll:>9.4f}  "
        for k in range(3):
            line += f"{s[k]['team']:14s} {s[k]['p_champion']:5.2f}%   "
        print(line)

    print("\nNotes on interpretation:")
    print("  - Squad sweep changes only the post-Elo blend, so it stresses 'how much do we")
    print("    trust market values vs match results' without re-fitting any Elo history.")
    print("  - Friendly-K sweep re-walks 35 years of matches per variant; lower K means")
    print("    friendlies barely move the rating (treats them as noise).")
    print("  - Deltas under ±0.5pp on champion% are within Monte Carlo noise at n=3000.")
    print("  - Look for: rank-order changes in the top 6, England's % movement, and whether")
    print("    log-loss meaningfully improves on the K sweep.")


if __name__ == "__main__":
    main()
