"""Calibration audit: are our predicted probabilities believable?

For every match in the predictions table we already have the pre-match
ratings the model would have used. Reconstruct the W/D/L probabilities
exactly the way the website computes them today, then ask: when we
say 70%, does it actually happen 70% of the time?

Method (standard reliability analysis for multi-class probabilistic
forecasts):

  1. For every match, compute (p_home, p_draw, p_away).
  2. Each match contributes 3 (probability, occurred 0/1) data points
     — one per outcome class.
  3. Bin these by predicted probability into N_BINS equal-width bins.
  4. Per bin: mean predicted prob, observed frequency, count.
  5. ECE (Expected Calibration Error) = sum_b (n_b/N) * |mean_p_b - freq_b|.
     Lower is better. Below ~0.02 = well calibrated; 0.02-0.05 = mild
     miscalibration; >0.05 = needs a calibration layer.
  6. Compare to a "no-skill" baseline (always predict the marginal
     class frequencies) so the absolute log-loss number has a yardstick.
  7. Slice by outcome class and tournament tier to find where any
     miscalibration is concentrated.

Run: `uv run python -m worldcup.calibration_audit`
"""

from __future__ import annotations

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

N_BINS = 10
TIER_MAJOR = {
    "FIFA World Cup",
    "UEFA Euro",
    "Copa América",
    "African Cup of Nations",
    "AFC Asian Cup",
    "Gold Cup",
    "Confederations Cup",
}
TIER_QUALIFIER_PREFIXES = (
    "FIFA World Cup qualification",
    "UEFA Euro qualification",
    "Copa América qualification",
    "African Cup of Nations qualification",
    "AFC Asian Cup qualification",
)


# --------------------------------------------------------------------- #
# Data loading + probability reconstruction
# --------------------------------------------------------------------- #

def load_predictions():
    with connect(read_only=True) as con:
        df = con.execute(
            """
            SELECT match_date, tournament, neutral,
                   rating_home_pre, rating_away_pre,
                   actual_home_goals, actual_away_goals
            FROM predictions
            WHERE actual_home_goals IS NOT NULL
            ORDER BY match_date
            """
        ).fetchdf()
    return df


def compute_probs(df):
    """Return (probs Nx3, actuals N, era N, tier N).
    probs columns are [p_home, p_draw, p_away]; actuals are 0/1/2."""
    rh = df["rating_home_pre"].to_numpy(dtype=float)
    ra = df["rating_away_pre"].to_numpy(dtype=float)
    neutral = df["neutral"].to_numpy().astype(bool)
    eff_h = rh + np.where(neutral, 0.0, HOME_ADVANTAGE)
    gap = eff_h - ra
    lh = LAMBDA_HOME_AT_PARITY * np.exp(SLOPE_HOME * gap)
    la = LAMBDA_AWAY_AT_PARITY * np.exp(SLOPE_AWAY * gap)
    probs = np.array([wld_probs(float(lh[i]), float(la[i])) for i in range(len(lh))])

    hg = df["actual_home_goals"].to_numpy(dtype=int)
    ag = df["actual_away_goals"].to_numpy(dtype=int)
    actuals = np.where(hg > ag, 0, np.where(hg < ag, 2, 1))

    years = df["match_date"].apply(lambda d: d.year).to_numpy()
    return probs, actuals, years, df["tournament"].to_numpy()


def tier_of(t: str) -> str:
    if t in TIER_MAJOR:
        return "major"
    if t.startswith(TIER_QUALIFIER_PREFIXES):
        return "qualifier"
    if t in ("UEFA Nations League", "CONCACAF Nations League"):
        return "nations_lg"
    if t == "Friendly":
        return "friendly"
    return "other"


# --------------------------------------------------------------------- #
# Reliability + metrics
# --------------------------------------------------------------------- #

def reliability_table(probs_flat, occurred_flat, n_bins=N_BINS):
    """Return list of (bin_lo, bin_hi, n, mean_pred, observed_freq) per bin."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    out = []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (probs_flat >= lo) & (probs_flat <= hi)
        else:
            mask = (probs_flat >= lo) & (probs_flat < hi)
        n = int(mask.sum())
        if n == 0:
            out.append((lo, hi, 0, np.nan, np.nan))
        else:
            mean_p = float(probs_flat[mask].mean())
            obs = float(occurred_flat[mask].mean())
            out.append((lo, hi, n, mean_p, obs))
    return out


def ece(table) -> float:
    """Expected Calibration Error: weighted mean |observed - predicted|."""
    total = sum(row[2] for row in table)
    if total == 0:
        return float("nan")
    return sum(row[2] * abs(row[4] - row[3]) for row in table if row[2] > 0) / total


def log_loss(probs, actuals) -> float:
    eps = 1e-12
    chosen = np.clip(probs[np.arange(len(actuals)), actuals], eps, 1.0)
    return float(-np.log(chosen).mean())


def brier(probs, actuals) -> float:
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(len(actuals)), actuals] = 1.0
    return float(((probs - one_hot) ** 2).sum(axis=1).mean())


def baseline_log_loss(actuals) -> float:
    """Always-predict-the-marginal-rates baseline."""
    base = np.bincount(actuals, minlength=3) / len(actuals)
    chosen = base[actuals]
    return float(-np.log(np.clip(chosen, 1e-12, 1.0)).mean())


# --------------------------------------------------------------------- #
# Output formatters
# --------------------------------------------------------------------- #

def print_reliability(label, table, ece_val):
    print(f"\n{label}")
    print(f"  {'bin':>11} {'n':>8} {'mean pred':>10} {'observed':>10} {'gap':>8}  bar (target = .)")
    print(f"  {'-'*11} {'-'*8} {'-'*10} {'-'*10} {'-'*8}  {'-'*42}")
    for lo, hi, n, mean_p, obs in table:
        if n == 0:
            print(f"  {lo:>4.1f}-{hi:<4.1f} {n:>8d}    {'-':>7}    {'-':>7}    {'-':>5}  (empty bin)")
            continue
        gap = obs - mean_p
        # ASCII bar: target marker '.' at mean_p, '#' at obs
        scale = 40
        target = int(round(mean_p * scale))
        actual = int(round(obs * scale))
        bar_chars = [" "] * (scale + 1)
        bar_chars[min(target, scale)] = "."
        bar_chars[min(actual, scale)] = "#"
        bar = "".join(bar_chars)
        print(f"  {lo:>4.1f}-{hi:<4.1f} {n:>8d}    {mean_p:>6.3f}    {obs:>6.3f}   {gap:>+5.3f}  |{bar}|")
    print(f"  ECE = {ece_val:.4f}")


def calibration_verdict(ece_val):
    if ece_val < 0.02:
        return "WELL CALIBRATED — no fix needed"
    if ece_val < 0.05:
        return "MILD MISCALIBRATION — Platt scaling would tidy it up"
    return "POOR CALIBRATION — calibration layer recommended"


# --------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------- #

def main() -> None:
    print("=" * 86)
    print("CALIBRATION AUDIT  ·  reliability of W/D/L probabilities across all history")
    print("=" * 86)

    print("\n[load] Pulling predictions...")
    df = load_predictions()
    probs, actuals, years, tournaments = compute_probs(df)
    print(f"       {len(df):,} matches,  {df['match_date'].min().date()} → {df['match_date'].max().date()}")

    # Headline metrics
    ll = log_loss(probs, actuals)
    bs = brier(probs, actuals)
    base_ll = baseline_log_loss(actuals)
    skill = 1 - ll / base_ll
    marginals = np.bincount(actuals, minlength=3) / len(actuals)

    print(f"\n[headline]")
    print(f"  Log-loss               : {ll:.4f}")
    print(f"  Brier (multi-class)    : {bs:.4f}")
    print(f"  Baseline log-loss      : {base_ll:.4f}  (always predict marginals: H {marginals[0]:.2f} / D {marginals[1]:.2f} / A {marginals[2]:.2f})")
    print(f"  Skill score (1 - ll/base): {skill:+.3f}  (positive = better than dumb baseline)")

    # ── Overall reliability across all (prob, occurred) pairs ──────────
    n = len(probs)
    flat_probs = np.concatenate([probs[:, 0], probs[:, 1], probs[:, 2]])
    flat_occurred = np.concatenate([
        (actuals == 0).astype(int),
        (actuals == 1).astype(int),
        (actuals == 2).astype(int),
    ])
    table_all = reliability_table(flat_probs, flat_occurred)
    ece_all = ece(table_all)

    print("\n" + "-" * 86)
    print("OVERALL RELIABILITY  (each match contributes 3 points: one per outcome class)")
    print("-" * 86)
    print_reliability("All outcomes pooled, all years", table_all, ece_all)
    print(f"\n  Verdict: {calibration_verdict(ece_all)}")

    # ── Per outcome class ─────────────────────────────────────────────
    print("\n" + "-" * 86)
    print("BY OUTCOME CLASS  (does the model overconfidently predict draws? home wins?)")
    print("-" * 86)
    for cls_idx, cls_name in enumerate(["HOME WIN", "DRAW", "AWAY WIN"]):
        p = probs[:, cls_idx]
        o = (actuals == cls_idx).astype(int)
        t = reliability_table(p, o)
        e = ece(t)
        print_reliability(f"{cls_name}  (mean p = {p.mean():.3f},  observed = {o.mean():.3f})", t, e)

    # ── Per tournament tier ───────────────────────────────────────────
    print("\n" + "-" * 86)
    print("BY TOURNAMENT TIER  (does noise in friendlies hurt calibration of friendlies?)")
    print("-" * 86)
    tiers = np.array([tier_of(t) for t in tournaments])
    for tier_name in ["major", "qualifier", "nations_lg", "friendly"]:
        mask = tiers == tier_name
        if mask.sum() < 100:
            continue
        sub_probs = probs[mask]
        sub_actuals = actuals[mask]
        flat_p = np.concatenate([sub_probs[:, 0], sub_probs[:, 1], sub_probs[:, 2]])
        flat_o = np.concatenate([
            (sub_actuals == 0).astype(int),
            (sub_actuals == 1).astype(int),
            (sub_actuals == 2).astype(int),
        ])
        t = reliability_table(flat_p, flat_o)
        e = ece(t)
        ll_tier = log_loss(sub_probs, sub_actuals)
        print(f"\n[{tier_name}]  n_matches={mask.sum():>6}  log-loss={ll_tier:.4f}  ECE={e:.4f}  ({calibration_verdict(e)})")
        # Compact bin summary (one line per bin)
        for lo, hi, n_b, mean_p, obs in t:
            if n_b == 0:
                continue
            print(f"    {lo:>4.1f}-{hi:<4.1f}  n={n_b:>6}  pred={mean_p:.3f}  obs={obs:.3f}  gap={obs - mean_p:+.3f}")

    # ── Era split: recent 5 years vs older ────────────────────────────
    print("\n" + "-" * 86)
    print("BY ERA  (has the model drifted? recent vs historical calibration)")
    print("-" * 86)
    cutoff = years.max() - 5
    for era_name, mask in [
        (f"recent (≥ {cutoff})", years >= cutoff),
        (f"older  (< {cutoff})", years < cutoff),
    ]:
        if mask.sum() < 100:
            continue
        sub_probs = probs[mask]
        sub_actuals = actuals[mask]
        flat_p = np.concatenate([sub_probs[:, 0], sub_probs[:, 1], sub_probs[:, 2]])
        flat_o = np.concatenate([
            (sub_actuals == 0).astype(int),
            (sub_actuals == 1).astype(int),
            (sub_actuals == 2).astype(int),
        ])
        t = reliability_table(flat_p, flat_o)
        e = ece(t)
        ll_era = log_loss(sub_probs, sub_actuals)
        print(f"\n[{era_name}]  n_matches={mask.sum():>6}  log-loss={ll_era:.4f}  ECE={e:.4f}  ({calibration_verdict(e)})")

    # ── Calibration slope/intercept (Platt-scaling diagnostic) ────────
    # Fit logistic regression: occurred ~ logit(predicted). Slope = 1.0
    # and intercept = 0 means well calibrated. Slope > 1 = underconfident
    # (probabilities too close to 0.5); slope < 1 = overconfident.
    print("\n" + "-" * 86)
    print("CALIBRATION SLOPE / INTERCEPT  (logistic of occurred ~ logit(predicted))")
    print("  slope = 1.0, intercept = 0.0 → perfectly calibrated")
    print("  slope < 1.0 → model OVERCONFIDENT (push toward 0.5)")
    print("  slope > 1.0 → model UNDERCONFIDENT (push away from 0.5)")
    print("-" * 86)
    eps = 1e-6
    pl = np.clip(flat_probs := np.concatenate([probs[:, 0], probs[:, 1], probs[:, 2]]), eps, 1 - eps)
    fo = np.concatenate([
        (actuals == 0).astype(int),
        (actuals == 1).astype(int),
        (actuals == 2).astype(int),
    ])
    logit_p = np.log(pl / (1 - pl))
    # Closed-form-ish: simple Newton-Raphson logistic regression in 2 vars.
    a, b = 0.0, 1.0  # intercept, slope
    for _ in range(50):
        z = a + b * logit_p
        p_pred = 1.0 / (1.0 + np.exp(-z))
        # gradient
        ga = (p_pred - fo).sum()
        gb = ((p_pred - fo) * logit_p).sum()
        # Hessian
        w = p_pred * (1 - p_pred)
        haa = w.sum()
        hab = (w * logit_p).sum()
        hbb = (w * logit_p * logit_p).sum()
        det = haa * hbb - hab * hab
        if abs(det) < 1e-12:
            break
        da = (hbb * ga - hab * gb) / det
        db = (-hab * ga + haa * gb) / det
        a -= da
        b -= db
        if abs(da) + abs(db) < 1e-8:
            break
    print(f"  fitted intercept = {a:+.4f}")
    print(f"  fitted slope     = {b:+.4f}")
    if b < 0.9:
        print(f"  → model is OVERCONFIDENT by ~{(1 - b) * 100:.0f}%; Platt scaling with these params would help")
    elif b > 1.1:
        print(f"  → model is UNDERCONFIDENT by ~{(b - 1) * 100:.0f}%; Platt scaling would help")
    else:
        print(f"  → slope is within 10% of perfect; calibration is fine, no Platt layer needed")


if __name__ == "__main__":
    main()
