"""Deep holdout evaluation: WC 2022.

The 2022 WC was locked out from the start as our test set. This module
evaluates the model's actual performance against it across multiple
dimensions:

  1. Per-match calibration. For each of the 64 matches, what did the
     model predict W/D/L and how often was the predicted-favourite
     correct? Log-loss, Brier, ECE.
  2. Match-by-match outcomes. Win, draw, loss accuracy if we naively
     picked the model's top class.
  3. Tournament-level simulation. Run 10k Monte Carlo tournaments at
     the 2022 pre-kickoff state. Where did the actual champion
     (Argentina) rank in our predicted championship probabilities?
     Where did each top-4 team end up in our distribution?
  4. Bracket comparison. Modal predicted bracket vs actual bracket,
     side-by-side, by round.

Layers active in this evaluation:
  - Elo walk (min_date 1990-01-01, max_date 2022-11-19)
  - Form (recent 12-month residual GD, as of 2022-11-19)
  - National-team trophy bonus (NT_FINISHES, as_of_year=2022)
  - Host bonus (Qatar for WC 2022)

Layers explicitly skipped, they're 2026-specific data that would
leak future info into the 2022 evaluation:
  - Squad market-value blend (2026 values only, since removed)
  - Computed trophy aggregation (2025-26 club trophies, current rosters)
  - Hand-coded TEAM_TROPHY_SCORE (estimated for 2026 squads)

So this is "Elo + form + NT trophies + host", the core date-aware
model. About 80% of the current architecture, evaluated honestly
against unseen data.

Run: `uv run worldcup-wc2022-evaluation`
"""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from datetime import date

import numpy as np

from worldcup.calibration import (
    LAMBDA_AWAY_AT_PARITY,
    LAMBDA_HOME_AT_PARITY,
    SLOPE_AWAY,
    SLOPE_HOME,
)
from worldcup.db import connect
from worldcup.elo import HOME_ADVANTAGE
from worldcup.elo import run as elo_run
from worldcup.form import compute_form_bumps
from worldcup.goals import expected_goals, wld_probs
from worldcup.host import host_bonus
from worldcup.profiles import V4A_FLAT_MILD
from worldcup.tournament_backtest import (
    fetch_actual_results,
    fetch_wc_teams_and_groups,
    simulate_tournament,
)
from worldcup.trophies import nt_trophy_bonus

WC_YEAR = 2022
KICKOFF = date(2022, 11, 20)
PRE_KICKOFF = date(2022, 11, 19)  # last match date included in walk
N_SIMS = 10_000


def get_pre_kickoff_ratings():
    """Walk Elo from 1990 to the day BEFORE WC 2022 kickoff (so no WC
    2022 match leaks into the model), then layer form + NT bonus + host.
    """
    print(f"  walking Elo to {PRE_KICKOFF.isoformat()} (pre-kickoff)...")
    final_df, _preds = elo_run(
        min_date=date(1990, 1, 1),
        max_date=PRE_KICKOFF,
        k_factors=V4A_FLAT_MILD.k_factors,
        default_k=V4A_FLAT_MILD.default_k,
        quiet=True,
    )
    ratings = {row.team: float(row.rating)
               for row in final_df.itertuples(index=False)}

    form_bumps = compute_form_bumps(as_of_date=PRE_KICKOFF)
    boosted = {}
    for t, r in ratings.items():
        f = form_bumps.get(t, 0.0)
        nt = nt_trophy_bonus(t, as_of_year=WC_YEAR)
        host = host_bonus(t, "FIFA World Cup", WC_YEAR)
        boosted[t] = r + f + nt + host
    return ratings, boosted


def actual_outcome(home_g: int, away_g: int) -> int:
    if home_g > away_g:
        return 0
    if home_g < away_g:
        return 2
    return 1


def fetch_wc_2022_matches():
    with connect(read_only=True) as con:
        return con.execute(
            """
            SELECT date, home_team, away_team, neutral,
                   home_score, away_score
            FROM matches
            WHERE tournament = 'FIFA World Cup'
              AND EXTRACT(YEAR FROM date) = ?
              AND home_score IS NOT NULL
            ORDER BY date
            """,
            [WC_YEAR],
        ).fetchdf()


def evaluate_per_match(matches, boosted_ratings):
    """For each match, compute model probabilities at the time and
    compare to actual outcome."""
    rows = []
    for m in matches.itertuples(index=False):
        rh = boosted_ratings.get(m.home_team)
        ra = boosted_ratings.get(m.away_team)
        if rh is None or ra is None:
            continue
        eff_h = rh + (0.0 if m.neutral else HOME_ADVANTAGE)
        gap = eff_h - ra
        lh, la = expected_goals(gap, neutral=bool(m.neutral))
        p_h, p_d, p_a = wld_probs(float(lh), float(la))
        actual = actual_outcome(int(m.home_score), int(m.away_score))
        rows.append({
            "date": m.date,
            "home_team": m.home_team,
            "away_team": m.away_team,
            "p_home": p_h,
            "p_draw": p_d,
            "p_away": p_a,
            "actual": actual,
            "home_score": int(m.home_score),
            "away_score": int(m.away_score),
            "top_pick_class": int(np.argmax([p_h, p_d, p_a])),
            "top_pick_correct": int(np.argmax([p_h, p_d, p_a])) == actual,
            "p_assigned_to_actual": [p_h, p_d, p_a][actual],
        })
    return rows


def log_loss_rows(rows):
    eps = 1e-12
    return float(-np.mean(np.log(np.clip([r["p_assigned_to_actual"] for r in rows],
                                           eps, 1.0))))


def brier_rows(rows):
    sq_err = []
    for r in rows:
        p = np.array([r["p_home"], r["p_draw"], r["p_away"]])
        oh = np.zeros(3)
        oh[r["actual"]] = 1
        sq_err.append(((p - oh) ** 2).sum())
    return float(np.mean(sq_err))


def ece_rows(rows, n_bins=10):
    """ECE pooled across W/D/L classes (each match contributes 3 pairs)."""
    flat_p = []
    flat_o = []
    for r in rows:
        flat_p.extend([r["p_home"], r["p_draw"], r["p_away"]])
        flat_o.extend([1 if r["actual"] == c else 0 for c in (0, 1, 2)])
    flat_p = np.array(flat_p)
    flat_o = np.array(flat_o)
    bins = np.linspace(0, 1, n_bins + 1)
    total = 0
    weighted = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (flat_p >= lo) & (flat_p <= hi if i == n_bins - 1 else flat_p < hi)
        n = int(mask.sum())
        if n == 0:
            continue
        weighted += n * abs(float(flat_p[mask].mean()) - float(flat_o[mask].mean()))
        total += n
    return weighted / max(1, total)


def main() -> None:
    print("=" * 72)
    print("WC 2022 DEEP HOLDOUT EVALUATION")
    print("=" * 72)
    print(f"Cutoff: {KICKOFF.isoformat()}  (day before kickoff)")
    print("Layers active: Elo (v4a-flat-mild) + form + NT bonus + host")
    print("Layers SKIPPED (2026-specific, would leak future info):")
    print("  - Squad market-value blend")
    print("  - Trophy density (computed + hand-coded)")
    print()

    raw_ratings, boosted_ratings = get_pre_kickoff_ratings()

    matches = fetch_wc_2022_matches()
    print(f"  {len(matches)} WC 2022 matches with results in DB")

    actual = fetch_actual_results(WC_YEAR)
    print("\n  Actual outcome:")
    print(f"    Champion:        {actual['champion']}")
    print(f"    Runner-up:       {actual['runner_up']}")
    print(f"    Semifinalists:   {sorted(actual['semifinalists'])}")

    # ── Per-match evaluation ────────────────────────────────────────
    print("\n" + "=" * 72)
    print("PER-MATCH PREDICTION ACCURACY")
    print("=" * 72)
    rows = evaluate_per_match(matches, boosted_ratings)
    n_correct = sum(r["top_pick_correct"] for r in rows)
    print(f"\n  Top-pick W/D/L accuracy:  {n_correct}/{len(rows)} "
          f"= {100*n_correct/len(rows):.1f}%")
    print(f"  Per-match log-loss:       {log_loss_rows(rows):.4f}")
    print(f"  Per-match Brier:          {brier_rows(rows):.4f}")
    print(f"  Per-match ECE:            {ece_rows(rows):.4f}")
    print("  (Reference: overall historical log-loss is 0.91, ECE 0.008)")

    # ── Notable per-match calls ────────────────────────────────────
    print("\n  Most CONFIDENT correct calls:")
    correct = [r for r in rows if r["top_pick_correct"]]
    correct_sorted = sorted(correct, key=lambda r: -r["p_assigned_to_actual"])
    for r in correct_sorted[:5]:
        prob = r["p_assigned_to_actual"] * 100
        print(f"    {r['date']}  {r['home_team']:18s} {r['home_score']}-{r['away_score']} "
              f"{r['away_team']:18s}  (p={prob:.0f}%)")

    print("\n  Most CONFIDENT misses (model said X with high conviction, "
          "got it wrong):")
    wrong = [r for r in rows if not r["top_pick_correct"]]
    wrong_sorted = sorted(
        wrong,
        key=lambda r: -max(r["p_home"], r["p_draw"], r["p_away"]),
    )
    for r in wrong_sorted[:8]:
        top_class = ["H", "D", "A"][r["top_pick_class"]]
        actual_class = ["H", "D", "A"][r["actual"]]
        max_p = max(r["p_home"], r["p_draw"], r["p_away"]) * 100
        print(f"    {r['date']}  {r['home_team']:18s} {r['home_score']}-{r['away_score']} "
              f"{r['away_team']:18s}  (said {top_class} at {max_p:.0f}%, got {actual_class})")

    # ── Tournament-level Monte Carlo ───────────────────────────────
    print("\n" + "=" * 72)
    print(f"TOURNAMENT-LEVEL MONTE CARLO ({N_SIMS:,} sims)")
    print("=" * 72)
    groups = fetch_wc_teams_and_groups(WC_YEAR)
    if groups is None:
        print("Could not derive groups for WC 2022; skipping MC sims")
        return

    # Only sim teams we have ratings for
    teams_in_groups = [t for g in groups.values() for t in g]
    missing = [t for t in teams_in_groups if t not in boosted_ratings]
    if missing:
        print(f"  WARNING: missing ratings for: {missing}")

    counts, group_winners = simulate_tournament(
        boosted_ratings, groups, n_sims=N_SIMS,
    )
    # Round probabilities, simulate_tournament uses LONG key names.
    probs = {}
    for team, c in counts.items():
        n = sum(c.values()) or 1
        ch = c.get("champion", 0)
        ru = c.get("runner_up", 0)
        sf = c.get("semi_finals", 0)
        qf = c.get("quarter_finals", 0)
        probs[team] = {
            "champion": ch / n * 100,
            "final": (ch + ru) / n * 100,
            "semis": (ch + ru + sf) / n * 100,
            "qf": (ch + ru + sf + qf) / n * 100,
        }

    ranked = sorted(probs.items(), key=lambda kv: -kv[1]["champion"])
    print("\n  Top 12 by championship probability:")
    print(f"  {'team':25s}  {'rating':>6}  {'champ%':>7}  {'final%':>7}  "
          f"{'semi%':>7}  {'QF%':>7}")
    for team, p in ranked[:12]:
        marker = ""
        if team == actual["champion"]:
            marker = "  ← ACTUAL CHAMPION"
        elif team == actual["runner_up"]:
            marker = "  ← runner-up"
        elif team in actual["semifinalists"]:
            marker = "  ← SF"
        print(f"  {team:25s}  {round(boosted_ratings.get(team, 1500)):>6}  "
              f"{p['champion']:>6.1f}%  {p['final']:>6.1f}%  "
              f"{p['semis']:>6.1f}%  {p['qf']:>6.1f}%{marker}")

    print("\n  Where each actual top-4 team ranked in our forecast:")
    top4_actual = [actual["champion"], actual["runner_up"], *sorted(
        actual["semifinalists"] - {actual["champion"], actual["runner_up"]}
    )]
    for team in top4_actual:
        if team not in probs:
            continue
        rank = next(i + 1 for i, (t, _) in enumerate(ranked) if t == team)
        p = probs[team]
        print(f"    {team:25s}  rank #{rank:2d}  champ={p['champion']:.1f}%  "
              f"final={p['final']:.1f}%  semi={p['semis']:.1f}%  QF={p['qf']:.1f}%")

    # ── Group-stage prediction accuracy ─────────────────────────────
    print("\n  Group winners, predicted vs actual:")
    # Determine actual group winners from match results
    actual_group_winners = {}
    actual_group_runners = {}
    with connect(read_only=True) as con:
        gdf = con.execute(
            """
            SELECT date, home_team, away_team, home_score, away_score
            FROM matches
            WHERE tournament = 'FIFA World Cup'
              AND EXTRACT(YEAR FROM date) = ?
              AND home_score IS NOT NULL
            ORDER BY date
            LIMIT 48
            """,
            [WC_YEAR],
        ).fetchdf()
    # Compute group standings from these 48 group-stage matches
    for letter, group_teams in groups.items():
        gpts = {t: 0 for t in group_teams}
        ggd = {t: 0 for t in group_teams}
        ggf = {t: 0 for t in group_teams}
        for m in gdf.itertuples(index=False):
            if m.home_team in group_teams and m.away_team in group_teams:
                h, a, hs, as_ = m.home_team, m.away_team, m.home_score, m.away_score
                if hs > as_:
                    gpts[h] += 3
                elif hs < as_:
                    gpts[a] += 3
                else:
                    gpts[h] += 1
                    gpts[a] += 1
                ggd[h] += hs - as_
                ggd[a] += as_ - hs
                ggf[h] += hs
                ggf[a] += as_
        ranked_g = sorted(group_teams, key=lambda t: (gpts[t], ggd[t], ggf[t]),
                          reverse=True)
        actual_group_winners[letter] = ranked_g[0]
        actual_group_runners[letter] = ranked_g[1]

    n_winner_correct = 0
    n_top2_correct = 0
    for letter in sorted(group_winners.keys()):
        sorted_winners = sorted(group_winners[letter].items(),
                                 key=lambda kv: -kv[1])
        predicted_winner = sorted_winners[0][0]
        winner_pct = 100 * sorted_winners[0][1] / N_SIMS
        actual_w = actual_group_winners[letter]
        actual_r = actual_group_runners[letter]
        match = predicted_winner == actual_w
        if match:
            n_winner_correct += 1
        top2_predicted = {sorted_winners[0][0], sorted_winners[1][0]}
        actual_top2 = {actual_w, actual_r}
        n_top2_correct += len(top2_predicted & actual_top2)
        flag = "✓" if match else "✗"
        print(f"    Group {letter}: predicted {predicted_winner:18s} "
              f"({winner_pct:>4.0f}%)  →  actual: {actual_w} {flag}")
    print(f"\n  Group winner accuracy:  {n_winner_correct}/8")
    print(f"  Top-2 advancing teams:  {n_top2_correct}/16 correctly forecast")

    # ── Summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    actual_champ_rank = next(
        i + 1 for i, (t, _) in enumerate(ranked) if t == actual["champion"]
    )
    actual_champ_prob = probs[actual["champion"]]["champion"]
    print(f"\n  Actual champion: {actual['champion']}")
    print(f"    Ranked #{actual_champ_rank} in our pre-tournament model")
    print(f"    Champion probability: {actual_champ_prob:.1f}%")
    print(f"    P(reach SF) by our model: {probs[actual['champion']]['semis']:.1f}%")
    print()
    print(f"  Per-match top-pick accuracy: "
          f"{n_correct}/{len(rows)} = {100*n_correct/len(rows):.1f}%")
    print(f"  Per-match log-loss:          {log_loss_rows(rows):.4f}")
    print(f"  Per-match ECE:               {ece_rows(rows):.4f}")
    print(f"  Group winners correct:       {n_winner_correct}/8")
    print(f"  Top-2 advancers correct:     {n_top2_correct}/16")


if __name__ == "__main__":
    sys.exit(main())
