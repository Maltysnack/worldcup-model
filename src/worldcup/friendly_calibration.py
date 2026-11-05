"""Per-tier Platt-scaling calibration for friendly predictions.

Diagnostic (friendly_diagnostic.py) showed the model is uniformly
overconfident on friendly favourites by ~5-8 pp regardless of rating
gap, date-to-next-major, or opponent quality. The fingerprint is a
calibration slope < 1: probabilities are too extreme.

Fix: fit a logistic Platt-scaling map on past friendlies and apply it
to displayed friendly probabilities. Doesn't touch the Elo walk or
the K-factor for friendlies (which is K=30 from v4a-flat-mild), this
is purely a post-hoc correction to the prob output for the friendly
tier.

Fitting:
  - 80/20 random split (stable seed)
  - Fit logistic regression: occurred ~ a + b * logit(predicted_prob)
    on the train set
  - Evaluate ECE + log-loss on both splits before/after
  - Save (a, b) to data/friendly_calibration.json

Application (at use sites in simulator.py): for friendly predictions
only, transform each of (p_home, p_draw, p_away) via
  p_cal = sigmoid(a + b * logit(p))
then renormalise so the three probabilities sum to 1.

Run: `uv run worldcup-fit-friendly-calibration`
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime

import numpy as np

from worldcup.calibration import (
    LAMBDA_AWAY_AT_PARITY,
    LAMBDA_HOME_AT_PARITY,
    SLOPE_AWAY,
    SLOPE_HOME,
)
from worldcup.db import PROJECT_ROOT, connect
from worldcup.elo import HOME_ADVANTAGE
from worldcup.goals import wld_probs

OUTPUT_PATH = PROJECT_ROOT / "data" / "friendly_calibration.json"
TRAIN_FRACTION = 0.8
SPLIT_SEED = 42
EPS = 1e-6


def actual_outcome(h: int, a: int) -> int:
    return 0 if h > a else (2 if h < a else 1)


def load_friendlies():
    with connect(read_only=True) as con:
        df = con.execute(
            """
            SELECT match_date, neutral,
                   rating_home_pre, rating_away_pre,
                   actual_home_goals, actual_away_goals
            FROM predictions
            WHERE tournament = 'Friendly'
              AND actual_home_goals IS NOT NULL
            ORDER BY match_date
            """
        ).fetchdf()
    return df


def compute_probs_and_actuals(df):
    rh = df["rating_home_pre"].to_numpy(dtype=float)
    ra = df["rating_away_pre"].to_numpy(dtype=float)
    neutral = df["neutral"].to_numpy().astype(bool)
    eff_h = rh + np.where(neutral, 0.0, HOME_ADVANTAGE)
    gap = eff_h - ra
    lh = LAMBDA_HOME_AT_PARITY * np.exp(SLOPE_HOME * gap)
    la = LAMBDA_AWAY_AT_PARITY * np.exp(SLOPE_AWAY * gap)
    probs = np.array([
        wld_probs(float(lh[i]), float(la[i])) for i in range(len(lh))
    ])
    actuals = np.array([
        actual_outcome(int(h), int(a))
        for h, a in zip(df["actual_home_goals"], df["actual_away_goals"])
    ])
    return probs, actuals


def fit_platt(probs: np.ndarray, actuals: np.ndarray) -> tuple[float, float]:
    """Newton-Raphson logistic fit. Flat (probability, occurred) data
    across all three outcome classes (each match contributes 3 pairs)."""
    flat_p = np.concatenate([probs[:, 0], probs[:, 1], probs[:, 2]])
    flat_o = np.concatenate([
        (actuals == c).astype(int) for c in range(3)
    ])
    flat_p = np.clip(flat_p, EPS, 1 - EPS)
    logit = np.log(flat_p / (1 - flat_p))

    a, b = 0.0, 1.0
    for _ in range(100):
        z = a + b * logit
        p_pred = 1.0 / (1.0 + np.exp(-z))
        ga = (p_pred - flat_o).sum()
        gb = ((p_pred - flat_o) * logit).sum()
        w = p_pred * (1 - p_pred)
        haa = w.sum()
        hab = (w * logit).sum()
        hbb = (w * logit * logit).sum()
        det = haa * hbb - hab * hab
        if abs(det) < 1e-12:
            break
        da = (hbb * ga - hab * gb) / det
        db = (-hab * ga + haa * gb) / det
        a -= da
        b -= db
        if abs(da) + abs(db) < 1e-9:
            break
    return float(a), float(b)


def apply_calibration(probs: np.ndarray, a: float, b: float) -> np.ndarray:
    """Apply Platt scaling per-class then renormalise so each row sums to 1.

    Works on either a single match (probs of shape (3,)) or a batch
    (shape (N, 3)). Returns same shape.
    """
    p = np.asarray(probs, dtype=float)
    one_match = (p.ndim == 1)
    if one_match:
        p = p.reshape(1, 3)
    clipped = np.clip(p, EPS, 1 - EPS)
    logit = np.log(clipped / (1 - clipped))
    z = a + b * logit
    raw = 1.0 / (1.0 + np.exp(-z))
    out = raw / raw.sum(axis=1, keepdims=True)
    return out[0] if one_match else out


def ece_multi(probs: np.ndarray, actuals: np.ndarray, n_bins: int = 10) -> float:
    flat_p = np.concatenate([probs[:, 0], probs[:, 1], probs[:, 2]])
    flat_o = np.concatenate([(actuals == c).astype(int) for c in range(3)])
    bins = np.linspace(0, 1, n_bins + 1)
    total = 0
    weighted_err = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (flat_p >= lo) & (flat_p <= hi if i == n_bins - 1 else flat_p < hi)
        n = int(mask.sum())
        if n == 0:
            continue
        weighted_err += n * abs(float(flat_p[mask].mean()) - float(flat_o[mask].mean()))
        total += n
    return weighted_err / max(1, total)


def log_loss(probs: np.ndarray, actuals: np.ndarray) -> float:
    chosen = np.clip(probs[np.arange(len(actuals)), actuals], EPS, 1.0)
    return float(-np.log(chosen).mean())


# ── Runtime loader, used by simulator.py ─────────────────────────────

_CACHED_PARAMS: tuple[float, float] | None = None


def get_params() -> tuple[float, float]:
    """Return (intercept, slope) for the fitted calibration, or
    (0.0, 1.0) as a no-op if no calibration file exists yet."""
    global _CACHED_PARAMS
    if _CACHED_PARAMS is not None:
        return _CACHED_PARAMS
    if not OUTPUT_PATH.exists():
        _CACHED_PARAMS = (0.0, 1.0)
        return _CACHED_PARAMS
    data = json.loads(OUTPUT_PATH.read_text())
    _CACHED_PARAMS = (float(data["intercept"]), float(data["slope"]))
    return _CACHED_PARAMS


def calibrate_friendly_probs(p_home: float, p_draw: float, p_away: float
                             ) -> tuple[float, float, float]:
    """Public helper called by simulator.py when displaying friendly
    predictions. Returns the post-Platt probabilities (renormalised)."""
    a, b = get_params()
    if a == 0.0 and b == 1.0:
        return p_home, p_draw, p_away
    cal = apply_calibration(np.array([p_home, p_draw, p_away]), a, b)
    return float(cal[0]), float(cal[1]), float(cal[2])


# ── Fit pipeline ─────────────────────────────────────────────────────

def main() -> None:
    print("Fitting friendly-tier calibration\n")
    df = load_friendlies()
    probs, actuals = compute_probs_and_actuals(df)
    print(f"Loaded {len(df):,} friendlies from predictions table")

    rng = np.random.default_rng(SPLIT_SEED)
    perm = rng.permutation(len(df))
    n_train = int(TRAIN_FRACTION * len(df))
    train_idx, test_idx = perm[:n_train], perm[n_train:]
    p_train, a_train = probs[train_idx], actuals[train_idx]
    p_test, a_test = probs[test_idx], actuals[test_idx]
    print(f"  train: {len(train_idx):,}  test: {len(test_idx):,}")

    print("\nBefore calibration:")
    print(f"  Train  log-loss = {log_loss(p_train, a_train):.4f}   ECE = {ece_multi(p_train, a_train):.4f}")
    print(f"  Test   log-loss = {log_loss(p_test, a_test):.4f}   ECE = {ece_multi(p_test, a_test):.4f}")

    a_fit, b_fit = fit_platt(p_train, a_train)
    print("\nFitted parameters:")
    print(f"  intercept = {a_fit:+.4f}")
    print(f"  slope     = {b_fit:+.4f}  (1.0 = no adjustment; < 1.0 = pulls toward 0.5)")

    p_train_cal = apply_calibration(p_train, a_fit, b_fit)
    p_test_cal = apply_calibration(p_test, a_fit, b_fit)

    print("\nAfter calibration:")
    print(f"  Train  log-loss = {log_loss(p_train_cal, a_train):.4f}   ECE = {ece_multi(p_train_cal, a_train):.4f}")
    print(f"  Test   log-loss = {log_loss(p_test_cal, a_test):.4f}   ECE = {ece_multi(p_test_cal, a_test):.4f}")

    # Sample illustration: show how a 70%/15%/15% prediction transforms
    sample = np.array([0.70, 0.15, 0.15])
    cal_sample = apply_calibration(sample, a_fit, b_fit)
    print(f"\nExample: raw [70%, 15%, 15%] -> calibrated "
          f"[{cal_sample[0]*100:.0f}%, {cal_sample[1]*100:.0f}%, {cal_sample[2]*100:.0f}%]")

    sample2 = np.array([0.85, 0.10, 0.05])
    cal_sample2 = apply_calibration(sample2, a_fit, b_fit)
    print(f"Example: raw [85%, 10%, 5%]  -> calibrated "
          f"[{cal_sample2[0]*100:.0f}%, {cal_sample2[1]*100:.0f}%, {cal_sample2[2]*100:.0f}%]")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps({
        "fitted_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "tournament_tier": "Friendly",
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "split_seed": SPLIT_SEED,
        "intercept": float(a_fit),
        "slope": float(b_fit),
        "metrics_before": {
            "train_log_loss": log_loss(p_train, a_train),
            "test_log_loss": log_loss(p_test, a_test),
            "train_ece": ece_multi(p_train, a_train),
            "test_ece": ece_multi(p_test, a_test),
        },
        "metrics_after": {
            "train_log_loss": log_loss(p_train_cal, a_train),
            "test_log_loss": log_loss(p_test_cal, a_test),
            "train_ece": ece_multi(p_train_cal, a_train),
            "test_ece": ece_multi(p_test_cal, a_test),
        },
    }, indent=2))
    print(f"\nWrote {OUTPUT_PATH}")


if __name__ == "__main__":
    sys.exit(main())
