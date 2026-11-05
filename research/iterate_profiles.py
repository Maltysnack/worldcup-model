"""Iterative K-factor tuning across major tournaments since 2010.

Each iteration:
  1. Score every active profile on per-match log-loss across all major
     tournament matches since 2010 (excluding WC 2022, the locked
     hold-out). Cached: profiles already scored aren't re-walked.
  2. Sort by log-loss, keep top KEEP_PER_ITERATION, drop the rest.
  3. Generate new variants:
       - PERT_BEST perturbations of #1
       - PERT_SECOND perturbations of #2
       - N_RANDOM fully random profiles (K-factors uniform [10, 80])
  4. Repeat.

The fully random profiles act as a negative control: if they never win
we have confidence the leader is real; if they often win we're chasing
noise.

Run: `uv run worldcup-iterate-profiles`
"""

from __future__ import annotations

from datetime import date

import numpy as np

from worldcup.calibration import (
    LAMBDA_AWAY_AT_PARITY,
    LAMBDA_HOME_AT_PARITY,
    SLOPE_AWAY,
    SLOPE_HOME,
)
from worldcup.elo import HOME_ADVANTAGE
from worldcup.elo import run as elo_run
from worldcup.goals import wld_probs
from worldcup.profiles import PROFILES, V1_BASELINE, Profile, resolve_min_date

TARGET_TOURNAMENTS = [
    "FIFA World Cup",
    "UEFA Euro",
    "Copa América",
    "African Cup of Nations",
    "AFC Asian Cup",
    "Gold Cup",
    "Confederations Cup",
]
EARLIEST_YEAR = 2010
HOLDOUT_YEAR = 2022  # excluded for FIFA World Cup only

POOL_SIZE = 20            # profiles tested each iteration
KEEP_PER_ITERATION = 10   # top N kept; bottom (POOL_SIZE - KEEP) dropped
PERT_BEST = 4             # perturbations of #1
PERT_SECOND = 1           # perturbations of #2
N_RANDOM = 5              # fully random variants
N_ITERATIONS = 20
TODAY = date(2026, 5, 11)


def actual_outcome(home_g: int, away_g: int) -> int:
    if home_g > away_g:
        return 0
    if home_g < away_g:
        return 2
    return 1


def score_profile(profile: Profile) -> tuple[float, float, int]:
    # Honour profile.min_date_offset_days so v5-style recent-only profiles
    # actually use a sliding window instead of silently behaving like v1.
    min_d = resolve_min_date(profile, anchor=TODAY) or date(1990, 1, 1)
    _, preds = elo_run(
        min_date=min_d,
        max_date=TODAY,
        k_factors=profile.k_factors,
        default_k=profile.default_k,
        quiet=True,
    )
    mask = preds["tournament"].isin(TARGET_TOURNAMENTS)
    mask &= preds["match_date"].apply(lambda d: d.year >= EARLIEST_YEAR)
    mask &= ~(
        (preds["tournament"] == "FIFA World Cup")
        & (preds["match_date"].apply(lambda d: d.year == HOLDOUT_YEAR))
    )
    df = preds[mask].copy()
    if df.empty:
        return float("inf"), float("inf"), 0

    rh = df["rating_home_pre"].to_numpy()
    ra = df["rating_away_pre"].to_numpy()
    neutral = df["neutral"].to_numpy().astype(bool)
    eff_h = rh + np.where(neutral, 0.0, HOME_ADVANTAGE)
    gap = eff_h - ra
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


def perturb_variant(base: Profile, name: str, seed: int) -> Profile:
    """Generate one perturbed K-factor profile near `base`."""
    rng = np.random.default_rng(seed)
    scale = float(rng.uniform(0.85, 1.15))
    new_kf = {}
    for k, v in base.k_factors.items():
        noise = float(rng.uniform(-4, 4))
        new_kf[k] = max(5, round(v * scale + noise))
    new_default = max(5, round(base.default_k * scale))
    return Profile(name=name, k_factors=new_kf, default_k=new_default)


def random_variant(name: str, seed: int) -> Profile:
    """Generate one fully random profile: each K independently uniform [10, 80]."""
    rng = np.random.default_rng(seed)
    new_kf = {k: int(rng.integers(10, 81)) for k in V1_BASELINE.k_factors}
    return Profile(name=name, k_factors=new_kf, default_k=int(rng.integers(10, 81)))


def k_summary(p: Profile) -> str:
    kf = p.k_factors
    return (
        f"WC={kf.get('FIFA World Cup', '?'):>3} "
        f"Cont={kf.get('UEFA Euro', '?'):>3} "
        f"WCQ={kf.get('FIFA World Cup qualification', '?'):>3} "
        f"NL={kf.get('UEFA Nations League', '?'):>3} "
        f"CQ={kf.get('UEFA Euro qualification', '?'):>3} "
        f"Fr={kf.get('Friendly', '?'):>3}"
    )


def category(name: str) -> str:
    """Short label for printing: pert / rand / orig."""
    if name.startswith("rand-") or "-rand" in name:
        return "RAND"
    if name.startswith("pert-") or "-pert" in name or "-seed-" in name:
        return "pert"
    return "orig"


def main() -> None:
    # Initial pool: 4 named profiles spanning the K-factor spectrum + 16 random
    rng_init = np.random.default_rng(7)
    initial_named = [
        PROFILES["v1"],
        PROFILES["v4a-flat-mild"],
        PROFILES["v4b-flat-mod"],
        PROFILES["v4c-flat-uniform"],
    ]
    initial_random = [
        random_variant(f"rand-init-{i+1}", seed=int(rng_init.integers(0, 10**8)))
        for i in range(POOL_SIZE - len(initial_named))
    ]
    active = initial_named + initial_random

    cache: dict[str, tuple[float, float, int]] = {}
    history = []

    for it in range(1, N_ITERATIONS + 1):
        # Score uncached profiles
        n_new_walks = 0
        for p in active:
            if p.name not in cache:
                cache[p.name] = score_profile(p)
                n_new_walks += 1

        # Sort
        ranked = sorted(active, key=lambda p: cache[p.name][0])
        winner = ranked[0]
        winner_ll = cache[winner.name][0]
        runner = ranked[1]

        # Random vs perturbed/orig in this iteration
        n_rand = sum(1 for p in active if category(p.name) == "RAND")
        rand_in_top10 = sum(
            1 for p in ranked[:KEEP_PER_ITERATION] if category(p.name) == "RAND"
        )
        rand_won = category(winner.name) == "RAND"

        print(f"\nIter {it:>2}  (walked {n_new_walks:>2} new, total profiles seen: {len(cache):>3})")
        print(f"  Best:    {winner.name:24s} ll={winner_ll:.4f}  | {k_summary(winner)}")
        print(f"  #2:      {runner.name:24s} ll={cache[runner.name][0]:.4f}")
        print(f"  Random:  {n_rand:>2} in pool, {rand_in_top10:>2} survived to top {KEEP_PER_ITERATION}, "
              f"{'YES' if rand_won else 'no':>3} won")

        history.append({
            "iter": it,
            "winner_name": winner.name,
            "winner_ll": winner_ll,
            "winner_kf": dict(winner.k_factors),
            "winner_kind": category(winner.name),
            "ranked": [(p.name, cache[p.name][0], category(p.name)) for p in ranked],
        })

        if it < N_ITERATIONS:
            keep = ranked[:KEEP_PER_ITERATION]
            new_variants = []
            for i in range(PERT_BEST):
                new_variants.append(
                    perturb_variant(winner, f"iter{it}-pert1-{i+1}", seed=it * 10_000 + i)
                )
            for i in range(PERT_SECOND):
                new_variants.append(
                    perturb_variant(runner, f"iter{it}-pert2-{i+1}", seed=it * 10_000 + 100 + i)
                )
            for i in range(N_RANDOM):
                new_variants.append(
                    random_variant(f"iter{it}-rand-{i+1}", seed=it * 10_000 + 200 + i)
                )
            active = keep + new_variants

    # ---- Final summary ----
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    # Best ever
    best_ll = min(cache.values(), key=lambda v: v[0])[0]
    best_names = [n for n, v in cache.items() if v[0] == best_ll]
    print(f"\nBest log-loss seen: {best_ll:.4f}  ({len(best_names)} profiles tied)")
    for name in best_names[:5]:
        p = next((p for h in history for p in [PROFILES.get(name)] if p), None)
        # Find the actual profile object (not all are in PROFILES; use history)
        prof = None
        for h in history:
            for n, ll, _ in h["ranked"]:
                if n == name:
                    # Reconstruct from winner_kf if it was the winner
                    if h["winner_name"] == name:
                        prof_kf = h["winner_kf"]
                        prof = Profile(name=name, k_factors=prof_kf, default_k=0)
                    break
        if prof is None:
            print(f"  {name:24s}  (k-factors not retained)")
        else:
            print(f"  {name:24s}  | {k_summary(prof)}")

    # Iteration-by-iteration winner trajectory
    print("\nWinner per iteration:")
    print(f"{'iter':>4}  {'name':24s} {'kind':5s} {'log-loss':>9}  notes")
    print("-" * 70)
    prev = None
    for h in history:
        delta = ""
        if prev is not None and h["winner_ll"] < prev:
            delta = f"  improved by {prev - h['winner_ll']:+.4f}"
        elif prev is not None and h["winner_ll"] > prev:
            delta = f"  regressed by {prev - h['winner_ll']:+.4f}"
        print(f"{h['iter']:>4d}  {h['winner_name']:24s} {h['winner_kind']:5s} "
              f"{h['winner_ll']:>9.4f}{delta}")
        prev = h["winner_ll"]

    # How often did random variants win or finish top-3?
    rand_wins = sum(1 for h in history if h["winner_kind"] == "RAND")
    rand_top3 = sum(
        1 for h in history
        if any(kind == "RAND" for _, _, kind in h["ranked"][:3])
    )
    print("\nRandom variants in 20 iterations:")
    print(f"  Won iteration:        {rand_wins:>2} times")
    print(f"  Finished top 3:       {rand_top3:>2} times")
    print("  (Negative-control check: if RAND often beats perturbations, we are in pure noise.)")

    # Pool-level: across all unique profiles tested, how do random vs
    # perturbations vs originals compare?
    rand_lls = [v[0] for n, v in cache.items() if category(n) == "RAND"]
    pert_lls = [v[0] for n, v in cache.items() if category(n) == "pert"]
    orig_lls = [v[0] for n, v in cache.items() if category(n) == "orig"]
    print(f"\nLog-loss distribution across all unique profiles tested ({len(cache)} total):")
    for label, lls in [("Originals (named)", orig_lls), ("Perturbations", pert_lls), ("Random", rand_lls)]:
        if not lls:
            continue
        a = np.array(lls)
        print(f"  {label:22s} n={len(lls):>3}  min={a.min():.4f}  median={np.median(a):.4f}  "
              f"max={a.max():.4f}  mean={a.mean():.4f}")


if __name__ == "__main__":
    main()
