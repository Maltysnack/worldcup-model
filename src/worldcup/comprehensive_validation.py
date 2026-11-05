"""Multi-tournament × multi-K-factor × multi-layer × multi-N comprehensive
validation. Designed as a credibility check on the entire architecture.

Grid:
  - 6 K-factor profiles: v1, v3a-mild, v4a (default), v4b, v4c, v5a-recent12
  - 4 layer configs: elo-only / +form / +form+NT / full (+host)
  - 6 tournaments: WC 2010, 2014, 2018  +  Euro 2012, 2016, 2020
  - 2 sim counts: 2000 (screening) and 10000 (high-precision)

Per evaluation:
  - Per-match log-loss, Brier, ECE
  - Champion rank, champion probability of actual winner
  - Top-4 in our top-8 hit rate
  - Group/seed-stage winner accuracy

Output: data/raw/validation_results.csv + data/raw/validation_summary.md
Run: `uv run worldcup-comprehensive-validation`
"""

from __future__ import annotations

# ── Config ────────────────────────────────────────────────────────────
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import UTC, date, datetime

import numpy as np

from worldcup.db import PROJECT_ROOT, connect
from worldcup.elo import HOME_ADVANTAGE
from worldcup.elo import run as elo_run
from worldcup.form import compute_form_bumps
from worldcup.goals import expected_goals, sim_match, wld_probs
from worldcup.host import host_bonus
from worldcup.profiles import PROFILES, resolve_min_date
from worldcup.trophies import nt_trophy_bonus

# Phase-aware scope. The default is the maximal grid for overnight runs;
# pass WC_VAL_QUICK=1 to use the smaller phase-1 grid (kept as a sanity
# check that ran first while we built confidence in the harness).
_FULL = os.environ.get("WC_VAL_QUICK") != "1"

PROFILES_TO_TEST = (
    [
        "v1", "v2-continental",
        "v3a-tournament-mild", "v3b-tournament-mod", "v3c-tournament-extreme",
        "v4a-flat-mild", "v4b-flat-mod", "v4c-flat-uniform",
        "v5a-recent-12", "v5b-recent-18", "v5c-recent-24",
    ]
    if _FULL else [
        "v1",
        "v3a-tournament-mild",
        "v4a-flat-mild",
        "v4b-flat-mod",
        "v4c-flat-uniform",
        "v5a-recent-12",
    ]
)

LAYER_CONFIGS = {
    "elo_only":     dict(form=False, nt=False, host=False),
    "elo_form":     dict(form=True,  nt=False, host=False),
    "elo_form_nt":  dict(form=True,  nt=True,  host=False),
    "full":         dict(form=True,  nt=True,  host=True),
}

# (tournament_name, year_played, kickoff_date). year_played matches the
# YEAR column in the matches table. Per-match metrics are robust across
# any team count; tournament-level MC only fires when the team count is
# one of the supported formats (16 / 24 / 32).
TOURNAMENTS = [
    ("FIFA World Cup",         2010, date(2010, 6, 10)),
    ("FIFA World Cup",         2014, date(2014, 6, 11)),
    ("FIFA World Cup",         2018, date(2018, 6, 13)),
    ("UEFA Euro",              2012, date(2012, 6, 7)),
    ("UEFA Euro",              2016, date(2016, 6, 9)),
    ("UEFA Euro",              2021, date(2021, 6, 10)),  # "Euro 2020" held in 2021
    ("UEFA Euro",              2024, date(2024, 6, 13)),
    # Continental tournaments, expand the per-match base and stress-test
    # generalisation outside Europe / WC.
    ("African Cup of Nations", 2017, date(2017, 1, 14)),
    ("African Cup of Nations", 2019, date(2019, 6, 21)),
    ("African Cup of Nations", 2022, date(2022, 1, 9)),   # AFCON 2021 played 2022
    ("African Cup of Nations", 2024, date(2024, 1, 13)),  # AFCON 2023 played 2024
    ("AFC Asian Cup",          2015, date(2015, 1, 9)),
    ("AFC Asian Cup",          2019, date(2019, 1, 5)),
    ("AFC Asian Cup",          2024, date(2024, 1, 12)),  # held 2024
    ("Copa América",           2016, date(2016, 6, 3)),
    ("Copa América",           2024, date(2024, 6, 20)),
]

SIM_COUNTS = [2000, 10000]

CSV_OUT = PROJECT_ROOT / "data" / "raw" / "validation_results.csv"
MD_OUT  = PROJECT_ROOT / "data" / "raw" / "validation_summary.md"


# ── Ratings construction ─────────────────────────────────────────────

def get_pre_tournament_ratings(profile_name: str, kickoff: date) -> tuple[dict, dict]:
    """Walk Elo up to the day before kickoff, return (raw_ratings, preds_df).
    Cached per (profile, kickoff)."""
    profile = PROFILES[profile_name]
    # max_date = day before kickoff so the tournament's first match is not
    # yet in the rating walk (avoids leakage).
    from datetime import timedelta
    max_d = kickoff - timedelta(days=1)
    # min_date = profile-aware. v5x recent-only profiles anchor on the
    # kickoff date so the recency window is "the N days before the
    # tournament started," not "the N days before today."
    min_d = resolve_min_date(profile, anchor=kickoff) or date(1990, 1, 1)
    final_df, preds = elo_run(
        min_date=min_d,
        max_date=max_d,
        k_factors=profile.k_factors,
        default_k=profile.default_k,
        quiet=True,
    )
    ratings = {row.team: float(row.rating) for row in final_df.itertuples(index=False)}
    return ratings, preds


def apply_layers(
    raw_ratings: dict[str, float],
    *,
    tournament: str,
    year: int,
    kickoff: date,
    form: bool,
    nt: bool,
    host: bool,
) -> dict[str, float]:
    """Apply the active layers on top of raw Elo. Each layer can be
    toggled independently. Squad-value blend and trophy-density layers
    are NOT applied historically (2026-specific data, would leak)."""
    boosted = dict(raw_ratings)
    if form:
        form_bumps = compute_form_bumps(as_of_date=kickoff)
        for t in boosted:
            boosted[t] += form_bumps.get(t, 0.0)
    if nt:
        for t in boosted:
            boosted[t] += nt_trophy_bonus(t, as_of_year=year)
    if host:
        for t in boosted:
            boosted[t] += host_bonus(t, tournament, year)
    return boosted


# ── Per-match scoring ─────────────────────────────────────────────────

def actual_outcome(h: int, a: int) -> int:
    return 0 if h > a else (2 if h < a else 1)


def per_match_metrics(tournament: str, year: int, ratings: dict[str, float]) -> dict:
    """For every match in this tournament, reconstruct W/D/L prediction
    using the post-layer ratings, score against actuals."""
    with connect(read_only=True) as con:
        df = con.execute(
            """
            SELECT date, home_team, away_team, neutral,
                   home_score, away_score
            FROM matches
            WHERE tournament = ?
              AND EXTRACT(YEAR FROM date) = ?
              AND home_score IS NOT NULL
            ORDER BY date
            """,
            [tournament, year],
        ).fetchdf()
    if df.empty:
        return None
    probs_list, actuals_list = [], []
    for r in df.itertuples(index=False):
        rh = ratings.get(r.home_team)
        ra = ratings.get(r.away_team)
        if rh is None or ra is None:
            continue
        eff_h = rh + (0.0 if r.neutral else HOME_ADVANTAGE)
        gap = eff_h - ra
        lh, la = expected_goals(gap, neutral=bool(r.neutral))
        p_h, p_d, p_a = wld_probs(float(lh), float(la))
        probs_list.append((p_h, p_d, p_a))
        actuals_list.append(actual_outcome(int(r.home_score), int(r.away_score)))
    if not probs_list:
        return None
    probs = np.array(probs_list)
    actuals = np.array(actuals_list)
    eps = 1e-12

    # Log-loss
    chosen = np.clip(probs[np.arange(len(actuals)), actuals], eps, 1.0)
    log_loss = float(-np.log(chosen).mean())
    # Brier
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(len(actuals)), actuals] = 1.0
    brier = float(((probs - one_hot) ** 2).sum(axis=1).mean())
    # ECE (10 bins, pooled)
    flat_p = np.concatenate([probs[:, 0], probs[:, 1], probs[:, 2]])
    flat_o = np.concatenate([(actuals == c).astype(int) for c in range(3)])
    bins = np.linspace(0, 1, 11)
    total = 0
    w_err = 0.0
    for i in range(10):
        lo, hi = bins[i], bins[i + 1]
        mask = (flat_p >= lo) & (flat_p <= hi if i == 9 else flat_p < hi)
        n = int(mask.sum())
        if n == 0:
            continue
        w_err += n * abs(float(flat_p[mask].mean()) - float(flat_o[mask].mean()))
        total += n
    ece = w_err / max(1, total)
    # Top-pick accuracy
    top_pick = probs.argmax(axis=1)
    top_pick_acc = float((top_pick == actuals).mean())
    return {
        "n_matches": len(probs_list),
        "log_loss": log_loss,
        "brier": brier,
        "ece": ece,
        "top_pick_acc": top_pick_acc,
    }


# ── Tournament-level Monte Carlo ──────────────────────────────────────

def fetch_groups_and_actual(tournament: str, year: int):
    """Return (groups_dict, actual_dict). Handles both 32-team WC and
    16/24-team Euro formats via the team count."""
    with connect(read_only=True) as con:
        df = con.execute(
            """
            SELECT date, home_team, away_team, home_score, away_score
            FROM matches
            WHERE tournament = ?
              AND EXTRACT(YEAR FROM date) = ?
              AND home_score IS NOT NULL
            ORDER BY date
            """,
            [tournament, year],
        ).fetchdf()
    if df.empty:
        return None, None

    teams_all = sorted(set(df["home_team"]) | set(df["away_team"]))
    n_teams = len(teams_all)

    # Group matches: first n_groups * 6 matches (each group plays 6 round-robin)
    if n_teams == 32:
        n_groups, gms = 8, 48
    elif n_teams == 24:
        n_groups, gms = 6, 36
    elif n_teams == 16:
        n_groups, gms = 4, 24
    else:
        # Unknown format; bail
        return None, None
    group_df = df.head(gms)

    adj: dict[str, set[str]] = defaultdict(set)
    for r in group_df.itertuples(index=False):
        adj[r.home_team].add(r.away_team)
        adj[r.away_team].add(r.home_team)
    seen: set[str] = set()
    groups: list[list[str]] = []
    for team in sorted(adj):
        if team in seen:
            continue
        members = sorted({team} | adj[team])
        if len(members) == 4 and all(m not in seen for m in members):
            groups.append(members)
            seen.update(members)
    if len(groups) != n_groups:
        return None, None
    groups_dict = {chr(ord("A") + i): g for i, g in enumerate(groups)}

    # Actual winner / runner-up
    final = df.iloc[-1]
    if final.home_score > final.away_score:
        champion, runner_up = final.home_team, final.away_team
    else:
        champion, runner_up = final.away_team, final.home_team
    actual = {"champion": champion, "runner_up": runner_up}
    return groups_dict, actual


def simulate_mc(
    ratings: dict[str, float],
    groups: dict[str, list[str]],
    n_sims: int,
    seed: int = 42,
) -> dict:
    """Run N tournament sims. Supports 16-team (groups→QF→SF→F),
    24-team (groups→R16→QF→SF→F), and 32-team (groups→R16→QF→SF→F).
    Returns {team: {rounds reached counts}}."""
    n_groups = len(groups)
    n_teams = sum(len(v) for v in groups.values())

    rng = np.random.default_rng(seed)
    counts: dict[str, Counter] = defaultdict(Counter)

    for _ in range(n_sims):
        # Group stage
        winners, runners, thirds = {}, {}, {}
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
                    if win == 0: pts[teams[i]] += 3
                    elif win == 1: pts[teams[j]] += 3
                    else:
                        pts[teams[i]] += 1
                        pts[teams[j]] += 1
                    gd[teams[i]] += sa - sb
                    gd[teams[j]] += sb - sa
                    gf[teams[i]] += sa
                    gf[teams[j]] += sb
            sorted_t = sorted(teams, key=lambda t: (pts[t], gd[t], gf[t]), reverse=True)
            winners[letter] = sorted_t[0]
            runners[letter] = sorted_t[1]
            thirds[letter] = sorted_t[2]
            for t in sorted_t[(2 if n_groups == 8 else 2):]:
                # 32-team and 24-team: top 2 advance (+ best 3rds for 24); 16: top 2 advance
                pass

        # Determine R16/QF entrants by format
        if n_teams == 32:
            # Top 2 from each group: 16 → R16
            ro16 = []
            pair_order = [
                (winners["A"], runners["B"]), (winners["C"], runners["D"]),
                (winners["E"], runners["F"]), (winners["G"], runners["H"]),
                (winners["B"], runners["A"]), (winners["D"], runners["C"]),
                (winners["F"], runners["E"]), (winners["H"], runners["G"]),
            ]
            for a, b in pair_order:
                ro16.append((a, b))
            # Track group exits
            for letter, teams in groups.items():
                for t in teams[2:]:  # 3rd, 4th place from the sorted order
                    pass  # we don't have explicit 3rd/4th tracking; skip
        elif n_teams == 24:
            # Euro 2016/2020 format: top 2 + 4 best 3rds → R16
            sorted_thirds = sorted(thirds.items(),
                                   key=lambda kv: 0, reverse=True)[:4]
            best_third_letters = {l for l, _ in sorted_thirds}
            advancing_thirds = [thirds[l] for l in best_third_letters]
            ro16 = []
            # Simplified pairings: top 2 from each group + advancing thirds
            top16 = [winners[l] for l in groups] + [runners[l] for l in groups] + advancing_thirds
            top16 = top16[:16]
            rng.shuffle(top16)
            for i in range(0, 16, 2):
                ro16.append((top16[i], top16[i + 1]))
        else:  # 16 teams
            ro16 = None  # skip R16; go straight to QF

        # Knockouts (R16 → QF → SF → F)
        def run_round(pairs):
            next_round = []
            losers = []
            for a, b in pairs:
                sa, sb, win = sim_match(ratings[a], ratings[b],
                                         neutral=True, ko=True, rng=rng)
                w = a if win == 0 else b
                l = b if win == 0 else a
                next_round.append(w)
                losers.append(l)
            return next_round, losers

        if ro16 is not None:
            winners_r16, losers_r16 = run_round(ro16)
            for l in losers_r16: counts[l]["r16"] += 1
            qf_pairs = [(winners_r16[i], winners_r16[i + 1]) for i in range(0, len(winners_r16), 2)]
        else:
            # 16-team: top 2 per group go to QF
            top8 = [winners[l] for l in groups] + [runners[l] for l in groups]
            rng.shuffle(top8)
            qf_pairs = [(top8[i], top8[i + 1]) for i in range(0, 8, 2)]

        winners_qf, losers_qf = run_round(qf_pairs)
        for l in losers_qf: counts[l]["qf"] += 1
        sf_pairs = [(winners_qf[i], winners_qf[i + 1]) for i in range(0, 4, 2)]
        winners_sf, losers_sf = run_round(sf_pairs)
        for l in losers_sf: counts[l]["sf"] += 1
        final_pair = [(winners_sf[0], winners_sf[1])]
        winners_f, losers_f = run_round(final_pair)
        counts[losers_f[0]]["ru"] += 1
        counts[winners_f[0]]["champ"] += 1

    return dict(counts)


# ── Per-tournament eval ───────────────────────────────────────────────

def eval_tournament(
    profile_name: str,
    layer_name: str,
    tournament: str,
    year: int,
    kickoff: date,
    sim_count: int,
) -> dict | None:
    """Run one (profile, layer, tournament, N) evaluation. Return metrics."""
    raw_ratings, _ = get_pre_tournament_ratings(profile_name, kickoff)
    layer_cfg = LAYER_CONFIGS[layer_name]
    ratings = apply_layers(
        raw_ratings, tournament=tournament, year=year, kickoff=kickoff,
        **layer_cfg,
    )

    # Per-match
    pm = per_match_metrics(tournament, year, ratings)
    if pm is None:
        return None

    # Tournament-level MC
    groups, actual = fetch_groups_and_actual(tournament, year)
    if groups is None:
        return {**pm, "champion_rank": None, "champion_p": None}

    counts = simulate_mc(ratings, groups, sim_count)
    champ_count = {t: c.get("champ", 0) for t, c in counts.items()}
    ranked = sorted(champ_count.items(), key=lambda kv: -kv[1])

    actual_champ = actual["champion"]
    rank = next((i + 1 for i, (t, _) in enumerate(ranked) if t == actual_champ), None)
    p_champ = champ_count.get(actual_champ, 0) / sim_count * 100 if sim_count > 0 else 0.0

    return {
        **pm,
        "champion_rank": rank,
        "champion_p_actual": round(p_champ, 2),
        "actual_champion": actual_champ,
        "our_top_pick": ranked[0][0] if ranked else None,
        "top_pick_won": (ranked[0][0] == actual_champ) if ranked else False,
    }


# ── Orchestration ─────────────────────────────────────────────────────

def main() -> None:
    t0 = time.time()
    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    total = len(PROFILES_TO_TEST) * len(LAYER_CONFIGS) * len(TOURNAMENTS) * len(SIM_COUNTS)
    print(f"Comprehensive validation: {total} evaluations")
    print(f"  {len(PROFILES_TO_TEST)} profiles × {len(LAYER_CONFIGS)} layer configs × "
          f"{len(TOURNAMENTS)} tournaments × {len(SIM_COUNTS)} sim counts\n")

    idx = 0
    for profile_name in PROFILES_TO_TEST:
        # Pre-compute ratings cache for this profile across all tournament cutoffs
        for tournament, year, kickoff in TOURNAMENTS:
            for layer_name in LAYER_CONFIGS:
                for sim_count in SIM_COUNTS:
                    idx += 1
                    t_start = time.time()
                    print(f"[{idx:>3}/{total}] {profile_name:24s}  {layer_name:14s}  "
                          f"{tournament:15s} {year}  N={sim_count}", end=" ", flush=True)
                    try:
                        metrics = eval_tournament(
                            profile_name, layer_name, tournament, year, kickoff, sim_count,
                        )
                        if metrics is None:
                            print("  no data")
                            continue
                        row = {
                            "profile": profile_name,
                            "layer_config": layer_name,
                            "tournament": tournament,
                            "year": year,
                            "sim_count": sim_count,
                            **metrics,
                        }
                        rows.append(row)
                        elapsed = time.time() - t_start
                        print(f"  ll={metrics['log_loss']:.4f} rank=#{metrics.get('champion_rank') or '?':>2} "
                              f"p_winner={metrics.get('champion_p_actual', 0):.1f}%  ({elapsed:.1f}s)")
                    except Exception as e:
                        print(f"  ERROR: {e}")
                        continue

    # ── Aggregate + rank ─────────────────────────────────────────────
    print(f"\n{'='*70}\nWriting results...")
    import csv
    if rows:
        keys = list(rows[0].keys())
        with CSV_OUT.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f"  CSV: {CSV_OUT}  ({len(rows)} rows)")

    # Per-(profile, layer) aggregate (averaging across tournaments at N=10000)
    high_n = [r for r in rows if r["sim_count"] == max(SIM_COUNTS)]
    agg: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in high_n:
        agg[(r["profile"], r["layer_config"])].append(r)

    summary_rows = []
    for (profile, layer), records in agg.items():
        ll = np.mean([r["log_loss"] for r in records])
        ece = np.mean([r["ece"] for r in records])
        brier = np.mean([r["brier"] for r in records])
        ranks = [r["champion_rank"] for r in records if r.get("champion_rank")]
        mean_rank = np.mean(ranks) if ranks else None
        champ_pct = np.mean([r.get("champion_p_actual", 0) for r in records])
        top_pick_hits = sum(1 for r in records if r.get("top_pick_won"))
        summary_rows.append({
            "profile": profile,
            "layer": layer,
            "n_tournaments": len(records),
            "avg_log_loss": ll,
            "avg_ece": ece,
            "avg_brier": brier,
            "avg_champ_rank": mean_rank,
            "avg_champ_pct_winner": champ_pct,
            "top_pick_hits": top_pick_hits,
        })

    # Sort by log-loss (lower better)
    summary_rows.sort(key=lambda r: r["avg_log_loss"])

    md_lines = []
    md_lines.append("# Comprehensive validation summary\n")
    md_lines.append(f"\n_Generated {datetime.now(UTC).isoformat(timespec='seconds')}_\n")
    md_lines.append(f"\nGrid: {len(PROFILES_TO_TEST)} profiles × {len(LAYER_CONFIGS)} layer configs × "
                    f"{len(TOURNAMENTS)} tournaments × {len(SIM_COUNTS)} sim counts = "
                    f"{total} evaluations ({len(rows)} successful)\n")
    md_lines.append(f"\nElapsed: {(time.time() - t0)/60:.1f} minutes\n\n")
    md_lines.append("## Ranked by mean per-match log-loss (lower = better) at N=10000\n\n")
    md_lines.append("| Rank | Profile | Layer | Avg log-loss | Avg ECE | Avg Brier | Avg champ rank | Champ% of winner | Top-pick hits |\n")
    md_lines.append("|---:|---|---|---:|---:|---:|---:|---:|---:|\n")
    for i, r in enumerate(summary_rows, start=1):
        rank_str = f"{r['avg_champ_rank']:.1f}" if r['avg_champ_rank'] is not None else "n/a"
        md_lines.append(
            f"| {i} | {r['profile']} | {r['layer']} | "
            f"{r['avg_log_loss']:.4f} | {r['avg_ece']:.4f} | {r['avg_brier']:.4f} | "
            f"{rank_str} | {r['avg_champ_pct_winner']:.1f}% | "
            f"{r['top_pick_hits']}/{r['n_tournaments']} |\n"
        )

    # Per-profile aggregate (across layer configs at full)
    md_lines.append("\n## Profile comparison (layer=full, N=10000)\n\n")
    profile_full = [r for r in summary_rows if r["layer"] == "full"]
    profile_full.sort(key=lambda r: r["avg_log_loss"])
    md_lines.append("| Rank | Profile | Avg log-loss | Avg ECE | Avg champ rank |\n")
    md_lines.append("|---:|---|---:|---:|---:|\n")
    for i, r in enumerate(profile_full, start=1):
        rank_str = f"{r['avg_champ_rank']:.1f}" if r['avg_champ_rank'] else "n/a"
        md_lines.append(f"| {i} | {r['profile']} | {r['avg_log_loss']:.4f} | "
                        f"{r['avg_ece']:.4f} | {rank_str} |\n")

    # Layer ablation per profile
    md_lines.append("\n## Layer ablation by profile (N=10000)\n\n")
    md_lines.append("Does each layer help? Compare elo_only → elo_form → elo_form_nt → full.\n\n")
    for profile in PROFILES_TO_TEST:
        md_lines.append(f"\n### {profile}\n\n")
        md_lines.append("| Layer | Log-loss | ECE | Champ rank |\n|---|---:|---:|---:|\n")
        for layer in LAYER_CONFIGS:
            match = [s for s in summary_rows
                     if s["profile"] == profile and s["layer"] == layer]
            if not match:
                continue
            r = match[0]
            rank_str = f"{r['avg_champ_rank']:.1f}" if r['avg_champ_rank'] else "n/a"
            md_lines.append(f"| {layer} | {r['avg_log_loss']:.4f} | {r['avg_ece']:.4f} | {rank_str} |\n")

    # Stability check (N comparison for top 3 configs)
    md_lines.append("\n## Simulation-count stability\n\n")
    md_lines.append("Does the top config's ranking hold when N goes from 2000 to 10000?\n\n")
    top3 = summary_rows[:3]
    for top in top3:
        md_lines.append(f"\n### {top['profile']} + {top['layer']}\n\n")
        md_lines.append("| Tournament | N | Log-loss | Champ rank | Top pick |\n|---|---:|---:|---:|---|\n")
        for tournament, year, _ in TOURNAMENTS:
            for n in SIM_COUNTS:
                match = [r for r in rows
                         if r["profile"] == top["profile"]
                         and r["layer_config"] == top["layer"]
                         and r["tournament"] == tournament
                         and r["year"] == year
                         and r["sim_count"] == n]
                if not match:
                    continue
                r = match[0]
                rank_str = f"#{r['champion_rank']}" if r['champion_rank'] else "?"
                top_pick = r.get("our_top_pick", "?")
                md_lines.append(f"| {tournament} {year} | {n:>5} | "
                                f"{r['log_loss']:.4f} | {rank_str} | {top_pick} |\n")

    MD_OUT.write_text("".join(md_lines))
    print(f"  MD : {MD_OUT}")
    print(f"\nTotal time: {(time.time() - t0)/60:.1f} minutes")
    print("\nTop 5 (profile, layer) combos by log-loss:")
    for i, r in enumerate(summary_rows[:5], start=1):
        rank_str = f"{r['avg_champ_rank']:.1f}" if r['avg_champ_rank'] is not None else "n/a"
        print(f"  {i}. {r['profile']:24s}  {r['layer']:14s}  ll={r['avg_log_loss']:.4f}  "
              f"ECE={r['avg_ece']:.4f}  champ-rank={rank_str}  "
              f"top-pick-hits={r['top_pick_hits']}/{r['n_tournaments']}")


if __name__ == "__main__":
    sys.exit(main())
