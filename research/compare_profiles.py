"""Run all profiles through historical World Cups and compare.

For each (profile, year) combination:
  1. Walk Elo with that profile's K-factors and recency window
  2. Score WC matches by per-match log-loss / Brier
  3. Run a tournament-level Monte Carlo sim and record where the
     actual top 4 ranked

Outputs a single comparison table.

Run: `uv run worldcup-compare-profiles`
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import date, timedelta

import numpy as np

from worldcup.elo import run as elo_run
from worldcup.goals import sim_match, wld_probs
from worldcup.profiles import PROFILES, Profile
from worldcup.tournament_backtest import (
    GROUPS_LEGACY,
    R16_LEGACY,
    QF_LEGACY,
    SF_LEGACY,
    FINAL_LEGACY,
    fetch_actual_results,
    fetch_wc_teams_and_groups,
)

# Order matters for output; v1 first as the baseline anchor.
PROFILES_TO_COMPARE = [
    "v1",
    "v3a-tournament-mild",
    "v3b-tournament-mod",
    "v3c-tournament-extreme",
    "v4a-flat-mild",
    "v4b-flat-mod",
    "v4c-flat-uniform",
    "v5a-recent-12",
    "v5b-recent-18",
    "v5c-recent-24",
]
YEARS = [2010, 2014, 2018]

# Use the actual WC kickoff dates (matches our DB)
WC_KICKOFF = {
    2010: date(2010, 6, 11),
    2014: date(2014, 6, 12),
    2018: date(2018, 6, 14),
}


def resolve_min_date(profile: Profile, prediction_date: date) -> date:
    """For v5 profiles, return prediction_date - offset; otherwise default."""
    if profile.min_date_offset_days is None:
        return date(1990, 1, 1)
    return prediction_date - timedelta(days=profile.min_date_offset_days)


def actual_outcome(home_g: int, away_g: int) -> int:
    if home_g > away_g:
        return 0
    if home_g < away_g:
        return 2
    return 1


def per_match_score(preds, year: int):
    """Compute log-loss + Brier for WC matches in `year`."""
    df = preds[
        (preds["tournament"] == "FIFA World Cup")
        & (preds["match_date"].apply(lambda d: d.year == year))
    ]
    if df.empty:
        return None, None, 0
    rh = df["rating_home_pre"].to_numpy()
    ra = df["rating_away_pre"].to_numpy()
    neutral = df["neutral"].to_numpy().astype(bool)
    eff_h = rh + np.where(neutral, 0.0, 50.0)
    eff_a = ra
    gap = eff_h - eff_a
    # Use baked-in calibration: pull from worldcup.calibration
    from worldcup.calibration import (
        LAMBDA_AWAY_AT_PARITY,
        LAMBDA_HOME_AT_PARITY,
        SLOPE_AWAY,
        SLOPE_HOME,
    )

    lh = LAMBDA_HOME_AT_PARITY * np.exp(SLOPE_HOME * gap)
    la = LAMBDA_AWAY_AT_PARITY * np.exp(SLOPE_AWAY * gap)
    probs = np.array([wld_probs(float(lh[i]), float(la[i])) for i in range(len(lh))])
    actuals = np.array([
        actual_outcome(int(h), int(a))
        for h, a in zip(df["actual_home_goals"], df["actual_away_goals"])
    ])
    eps = 1e-12
    chosen = probs[np.arange(len(actuals)), actuals]
    chosen = np.clip(chosen, eps, 1.0)
    log_loss = float(-np.log(chosen).mean())
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(len(actuals)), actuals] = 1.0
    brier = float(((probs - one_hot) ** 2).sum(axis=1).mean())
    return log_loss, brier, len(df)


def get_pre_wc_ratings_from(preds, year):
    """For each team that played in WC `year`, find their rating immediately
    before their first WC match that year."""
    wc = preds[
        (preds["tournament"] == "FIFA World Cup")
        & (preds["match_date"].apply(lambda d: d.year == year))
    ].sort_values("match_date")
    ratings = {}
    for row in wc.itertuples(index=False):
        if row.home_team not in ratings:
            ratings[row.home_team] = float(row.rating_home_pre)
        if row.away_team not in ratings:
            ratings[row.away_team] = float(row.rating_away_pre)
    return ratings


def simulate_tournament(ratings, groups, n_sims=5000, seed=42):
    rng = np.random.default_rng(seed)
    counts: dict[str, Counter] = defaultdict(Counter)
    for _ in range(n_sims):
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
                    if win == 0: pts[teams[i]] += 3
                    elif win == 1: pts[teams[j]] += 3
                    else: pts[teams[i]] += 1; pts[teams[j]] += 1
                    gd[teams[i]] += sa - sb
                    gd[teams[j]] += sb - sa
                    gf[teams[i]] += sa
                    gf[teams[j]] += sb
            sorted_t = sorted(teams, key=lambda t: (pts[t], gd[t], gf[t], ratings[t]), reverse=True)
            winners[letter] = sorted_t[0]
            runners[letter] = sorted_t[1]
            for t in sorted_t[2:]:
                counts[t]["group"] += 1

        match_winners = {}
        def resolve(spec):
            return winners[spec[1]] if spec[0] == 'W' else runners[spec[1]]

        for matchno, a, b in R16_LEGACY:
            ta, tb = resolve(a), resolve(b)
            sa, sb, win = sim_match(ratings[ta], ratings[tb], neutral=True, ko=True, rng=rng)
            w = ta if win == 0 else tb
            l = tb if win == 0 else ta
            match_winners[matchno] = w
            counts[l]["r16"] += 1
        for matchno, fa, fb in QF_LEGACY:
            ta, tb = match_winners[fa], match_winners[fb]
            sa, sb, win = sim_match(ratings[ta], ratings[tb], neutral=True, ko=True, rng=rng)
            w = ta if win == 0 else tb
            l = tb if win == 0 else ta
            match_winners[matchno] = w
            counts[l]["qf"] += 1
        for matchno, fa, fb in SF_LEGACY:
            ta, tb = match_winners[fa], match_winners[fb]
            sa, sb, win = sim_match(ratings[ta], ratings[tb], neutral=True, ko=True, rng=rng)
            w = ta if win == 0 else tb
            l = tb if win == 0 else ta
            match_winners[matchno] = w
            counts[l]["sf"] += 1
        matchno, fa, fb = FINAL_LEGACY
        ta, tb = match_winners[fa], match_winners[fb]
        sa, sb, win = sim_match(ratings[ta], ratings[tb], neutral=True, ko=True, rng=rng)
        w = ta if win == 0 else tb
        l = tb if win == 0 else ta
        counts[l]["ru"] += 1
        counts[w]["champ"] += 1
    return counts


def champ_rank(counts, target_team, n):
    """Probability target_team won + their rank by champ %."""
    pct = {t: c.get('champ', 0) / n * 100 for t, c in counts.items()}
    sorted_teams = sorted(pct.items(), key=lambda kv: -kv[1])
    rank = next((i + 1 for i, (t, _) in enumerate(sorted_teams) if t == target_team), None)
    p = pct.get(target_team, 0.0)
    return rank, p


def main() -> None:
    rows = []
    print("Comparison run: 10 profiles x 3 World Cups\n")
    for pname in PROFILES_TO_COMPARE:
        profile = PROFILES[pname]
        for year in YEARS:
            kickoff = WC_KICKOFF[year]
            min_d = resolve_min_date(profile, kickoff)
            print(f"  Building Elo for {pname} -> WC {year} (from {min_d})...", end=" ", flush=True)
            _, preds = elo_run(
                min_date=min_d,
                max_date=kickoff + timedelta(days=40),  # include the WC itself
                k_factors=profile.k_factors,
                default_k=profile.default_k,
                quiet=True,
            )
            ll, brier, n = per_match_score(preds, year)

            # Tournament-level sim
            groups = fetch_wc_teams_and_groups(year)
            actual = fetch_actual_results(year)
            tourn_rank = tourn_pct = None
            if groups is not None:
                ratings = get_pre_wc_ratings_from(preds, year)
                missing = [t for g in groups.values() for t in g if t not in ratings]
                if not missing:
                    counts = simulate_tournament(ratings, groups, n_sims=4000, seed=42)
                    tourn_rank, tourn_pct = champ_rank(counts, actual["champion"], 4000)
            rows.append({
                "profile": pname,
                "year": year,
                "log_loss": ll,
                "brier": brier,
                "n_matches": n,
                "winner": actual["champion"] if actual else "?",
                "winner_rank": tourn_rank,
                "winner_pct": tourn_pct,
            })
            print("done")

    # Print summary table
    print("\n" + "=" * 100)
    print("RESULTS")
    print("=" * 100)
    header = f"\n{'profile':28s} {'year':>4} {'log-loss':>9} {'Brier':>7} {'winner':14s} {'rank':>5} {'p_win':>7}"
    print(header)
    print("-" * 80)
    for r in rows:
        ll = f"{r['log_loss']:.4f}" if r['log_loss'] is not None else "n/a"
        br = f"{r['brier']:.4f}" if r['brier'] is not None else "n/a"
        rk = f"#{r['winner_rank']}" if r['winner_rank'] else "n/a"
        pc = f"{r['winner_pct']:.1f}%" if r['winner_pct'] is not None else "n/a"
        print(f"{r['profile']:28s} {r['year']:>4d} {ll:>9} {br:>7} {r['winner']:14s} {rk:>5} {pc:>7}")

    # Aggregate per profile
    print("\n" + "=" * 100)
    print("AVERAGED ACROSS YEARS (lower log-loss = better; higher winner_pct = better)")
    print("=" * 100)
    agg: dict[str, list] = defaultdict(list)
    agg_rank: dict[str, list] = defaultdict(list)
    agg_pct: dict[str, list] = defaultdict(list)
    for r in rows:
        if r['log_loss'] is not None:
            agg[r['profile']].append(r['log_loss'])
        if r['winner_rank'] is not None:
            agg_rank[r['profile']].append(r['winner_rank'])
        if r['winner_pct'] is not None:
            agg_pct[r['profile']].append(r['winner_pct'])

    print(f"\n{'profile':28s} {'avg log-loss':>13} {'avg winner rank':>16} {'avg p(actual W)':>16}")
    print("-" * 78)
    summary = []
    for pname in PROFILES_TO_COMPARE:
        ll_avg = np.mean(agg[pname]) if agg[pname] else None
        rk_avg = np.mean(agg_rank[pname]) if agg_rank[pname] else None
        pc_avg = np.mean(agg_pct[pname]) if agg_pct[pname] else None
        summary.append((pname, ll_avg, rk_avg, pc_avg))
    # Sort by log-loss for readable ranking
    summary.sort(key=lambda x: (x[1] if x[1] is not None else 99))
    for pname, ll, rk, pc in summary:
        ll_s = f"{ll:.4f}" if ll is not None else "n/a"
        rk_s = f"{rk:.1f}" if rk is not None else "n/a"
        pc_s = f"{pc:.1f}%" if pc is not None else "n/a"
        print(f"{pname:28s} {ll_s:>13} {rk_s:>16} {pc_s:>16}")


if __name__ == "__main__":
    main()
