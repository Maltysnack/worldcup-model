"""Where is the friendly-tier miscalibration concentrated?

Calibration audit found: friendly favourites at predicted 70-85% win
only ~58-77% of the time (mild overconfidence, ECE 0.039 vs majors'
0.013). This script slices that miscalibration by two natural axes:

  1. Date-distance to the next major tournament. Pre-tournament
     friendlies (within ~90 days of a WC/Euro/Copa) tend to be taken
     more seriously — first-choice XIs, full-effort. Off-season
     friendlies (180+ days from any major) are often rotated.
  2. Pre-match rating gap. Mismatched friendlies (gap > 300 Elo) often
     feature B-teams from the favourite or wholly experimental
     lineups, which distort outcomes more than the model's K-factor
     can absorb.

If miscalibration is concentrated in one slice, that subset becomes
the target for a per-match K modification.

Run: `uv run worldcup-friendly-diagnostic`
"""

from __future__ import annotations

import sys
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
from worldcup.goals import wld_probs


MAJOR_TOURNAMENTS = (
    "FIFA World Cup",
    "UEFA Euro",
    "Copa América",
    "African Cup of Nations",
    "AFC Asian Cup",
    "Gold Cup",
    "Confederations Cup",
)


def actual_outcome(home_g: int, away_g: int) -> int:
    if home_g > away_g:
        return 0
    if home_g < away_g:
        return 2
    return 1


def load_friendlies_with_distance():
    """Pull every historical friendly with: rating gap, days-to-next-major,
    actual W/D/L outcome, and reconstructed model probabilities."""
    with connect(read_only=True) as con:
        # Major-tournament start dates (per edition)
        majors = con.execute(
            f"""
            SELECT MIN(date) AS start_date
            FROM matches
            WHERE tournament IN ({','.join(['?'] * len(MAJOR_TOURNAMENTS))})
              AND home_score IS NOT NULL
            GROUP BY tournament, EXTRACT(YEAR FROM date)
            ORDER BY start_date
            """,
            list(MAJOR_TOURNAMENTS),
        ).fetchall()
        major_starts = sorted(d[0] for d in majors)

        # All friendlies with predictions
        df = con.execute(
            """
            SELECT match_date, home_team, away_team, neutral,
                   rating_home_pre, rating_away_pre,
                   actual_home_goals, actual_away_goals
            FROM predictions
            WHERE tournament = 'Friendly'
              AND actual_home_goals IS NOT NULL
            ORDER BY match_date
            """
        ).fetchdf()

    # Compute days_to_next_major per friendly
    major_dates = np.array([np.datetime64(d) for d in major_starts])

    def days_to_next(d):
        d64 = np.datetime64(d)
        future = major_dates[major_dates >= d64]
        if len(future) == 0:
            return None
        return int((future[0] - d64).astype("timedelta64[D]").astype(int))

    days = []
    for d in df["match_date"]:
        days.append(days_to_next(d))
    df["days_to_next_major"] = days
    return df


def reconstruct_probs(df):
    rh = df["rating_home_pre"].to_numpy(dtype=float)
    ra = df["rating_away_pre"].to_numpy(dtype=float)
    neutral = df["neutral"].to_numpy().astype(bool)
    eff_h = rh + np.where(neutral, 0.0, HOME_ADVANTAGE)
    gap = eff_h - ra
    lh = LAMBDA_HOME_AT_PARITY * np.exp(SLOPE_HOME * gap)
    la = LAMBDA_AWAY_AT_PARITY * np.exp(SLOPE_AWAY * gap)
    probs = np.array(
        [wld_probs(float(lh[i]), float(la[i])) for i in range(len(lh))]
    )
    actuals = np.array(
        [actual_outcome(int(h), int(a))
         for h, a in zip(df["actual_home_goals"], df["actual_away_goals"])]
    )
    return probs, actuals, np.abs(eff_h - ra)


def reliability_for_subset(probs, actuals, n_bins=10):
    flat_p = np.concatenate([probs[:, 0], probs[:, 1], probs[:, 2]])
    flat_o = np.concatenate([
        (actuals == 0).astype(int),
        (actuals == 1).astype(int),
        (actuals == 2).astype(int),
    ])
    bins = np.linspace(0, 1, n_bins + 1)
    rows = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (flat_p >= lo) & (flat_p < hi if i < n_bins - 1 else flat_p <= hi)
        n = int(mask.sum())
        if n == 0:
            rows.append((lo, hi, n, np.nan, np.nan, np.nan))
            continue
        mean_p = float(flat_p[mask].mean())
        obs = float(flat_o[mask].mean())
        rows.append((lo, hi, n, mean_p, obs, obs - mean_p))
    return rows


def ece(rows):
    total = sum(r[2] for r in rows)
    if total == 0:
        return float("nan")
    return sum(r[2] * abs(r[5]) for r in rows if r[2] > 0) / total


def eps_clip(p):
    return np.clip(p, 1e-12, 1.0)


def log_loss(probs, actuals):
    chosen = eps_clip(probs[np.arange(len(actuals)), actuals])
    return float(-np.log(chosen).mean())


def main() -> None:
    print("Loading friendlies with date-to-next-major distance...")
    df = load_friendlies_with_distance()
    df = df[df["days_to_next_major"].notna()]
    probs, actuals, abs_gap = reconstruct_probs(df)
    print(f"  loaded {len(df):,} friendlies")
    print(f"  overall log-loss: {log_loss(probs, actuals):.4f}")
    print(f"  overall ECE:      {ece(reliability_for_subset(probs, actuals)):.4f}")

    # Slice by days-to-next-major
    print("\n" + "=" * 70)
    print("BY DAYS-TO-NEXT-MAJOR (when does next WC/Euro/Copa/etc start)")
    print("=" * 70)
    days_arr = np.array(df["days_to_next_major"].tolist())
    bands = [
        ("0-30 days   (right before)",  (0, 30)),
        ("30-90 days  (pre-tourn)",     (30, 90)),
        ("90-180 days (early prep)",    (90, 180)),
        ("180-365 days (mid-cycle)",    (180, 365)),
        ("365+ days  (off-season)",     (365, 100000)),
    ]
    for label, (lo, hi) in bands:
        mask = (days_arr >= lo) & (days_arr < hi)
        n = int(mask.sum())
        if n < 50:
            print(f"  {label:30s}  n={n:>4} (skipped)")
            continue
        sub_probs = probs[mask]
        sub_actuals = actuals[mask]
        rows = reliability_for_subset(sub_probs, sub_actuals)
        print(f"  {label:30s}  n={n:>5}  log-loss={log_loss(sub_probs, sub_actuals):.4f}  ECE={ece(rows):.4f}")
        # Show high-confidence bin gap
        for r in rows:
            if r[2] > 0 and r[3] > 0.6:
                print(f"      pred={r[3]:.3f} obs={r[4]:.3f} gap={r[5]:+.3f} (n={r[2]})")
                break

    # Slice by rating gap
    print("\n" + "=" * 70)
    print("BY RATING GAP (absolute Elo difference between teams)")
    print("=" * 70)
    bands = [
        ("0-100 Elo (close)",     (0, 100)),
        ("100-200 Elo (modest)",  (100, 200)),
        ("200-300 Elo (clear)",   (200, 300)),
        ("300-450 Elo (large)",   (300, 450)),
        ("450+ Elo (mismatch)",   (450, 100000)),
    ]
    for label, (lo, hi) in bands:
        mask = (abs_gap >= lo) & (abs_gap < hi)
        n = int(mask.sum())
        if n < 50:
            print(f"  {label:30s}  n={n:>4} (skipped)")
            continue
        sub_probs = probs[mask]
        sub_actuals = actuals[mask]
        rows = reliability_for_subset(sub_probs, sub_actuals)
        print(f"  {label:30s}  n={n:>5}  log-loss={log_loss(sub_probs, sub_actuals):.4f}  ECE={ece(rows):.4f}")

    # 2D slice: rating gap x days-to-next-major
    print("\n" + "=" * 70)
    print("HIGH-CONFIDENCE FAVOURITES ONLY (predicted favourite > 60%)")
    print("Cross-tab: rating gap x date-distance, observed-vs-predicted gap")
    print("=" * 70)
    # For each match, take the favourite's predicted prob (max across H/D/A)
    fav_probs = probs.max(axis=1)
    fav_class = probs.argmax(axis=1)
    fav_actual = (fav_class == actuals).astype(int)
    high_conf = fav_probs > 0.60

    rating_bands = [(0, 200), (200, 400), (400, 100000)]
    rating_labels = ["close (<200)", "wide (200-400)", "huge (400+)"]
    days_bands = [(0, 90), (90, 180), (180, 100000)]
    days_labels = ["≤90 days", "90-180 days", "180+ days"]

    print(f"\n  {'rating gap':18s}  {'≤90 days':>14s}  {'90-180 days':>14s}  {'180+ days':>14s}")
    for r_label, (rlo, rhi) in zip(rating_labels, rating_bands):
        cells = []
        r_mask = high_conf & (abs_gap >= rlo) & (abs_gap < rhi)
        for dlo, dhi in days_bands:
            d_mask = (days_arr >= dlo) & (days_arr < dhi)
            mask = r_mask & d_mask
            n = int(mask.sum())
            if n < 30:
                cells.append(f"{'n='+str(n):>14s}")
            else:
                pred = float(fav_probs[mask].mean())
                obs = float(fav_actual[mask].mean())
                gap = obs - pred
                cells.append(f"p={pred:.2f} o={obs:.2f}".ljust(14))
        print(f"  {r_label:18s}  " + "  ".join(cells))


if __name__ == "__main__":
    sys.exit(main())
