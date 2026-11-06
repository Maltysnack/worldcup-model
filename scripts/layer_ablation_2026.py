"""Which tournament layers helped on the 2026 World Cup?

Rebuilds each layer from raw Elo at the 2026-06-10 cutoff (form is computed
as of the cutoff, so no result leaks in), scores all 104 matches 3-way with
the same goals model as scorecard_2026, and reports:
  - cumulative: elo -> +squad -> +form -> +nt -> +host -> +altitude
  - leave-one-out: all layers minus one
  - paired bootstrap 95% CI of the log-loss change vs plain Elo

Writes data/capsule/layer_ablation_2026.md.
Run: uv run python scripts/layer_ablation_2026.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from score_capsule_2026 import CUTOFF, ROOT, outcome_idx, probs_from_ratings
from worldcup.altitude import apply_altitude_bonus
from worldcup.db import connect
from worldcup.elo import run as elo_run
from worldcup.form import apply_form
from worldcup.host import apply_host_bonus
from worldcup.squad import blended_ratings
from worldcup.trophies import apply_nt_bonus

WC = "FIFA World Cup"
LAYERS = {
    "squad": blended_ratings,
    "form": lambda r: apply_form(r, as_of_date=CUTOFF),
    "nt": apply_nt_bonus,
    "host": lambda r: apply_host_bonus(r, WC, 2026),
    "altitude": lambda r: apply_altitude_bonus(r, WC, 2026),
}
ORDER = list(LAYERS)


def build(raw: dict, on: list[str]) -> dict:
    r = raw
    for name in ORDER:  # fixed application order, as in the simulator
        if name in on:
            r = LAYERS[name](r)
    return r


def losses(ratings: dict, matches) -> np.ndarray:
    out = []
    for m in matches.itertuples(index=False):
        p = probs_from_ratings(ratings[m.home_team], ratings[m.away_team], bool(m.neutral))
        out.append(-math.log(max(p[outcome_idx(int(m.home_score), int(m.away_score))], 1e-12)))
    return np.array(out)


def main() -> None:
    raw_df, _ = elo_run(max_date=CUTOFF, quiet=True)
    raw = dict(zip(raw_df["team"], raw_df["rating"]))
    with connect(read_only=True) as con:
        matches = con.execute(
            "SELECT home_team, away_team, home_score, away_score, neutral FROM matches "
            "WHERE tournament = 'FIFA World Cup' AND date >= '2026-06-01' ORDER BY date, home_team"
        ).fetchdf()
    assert len(matches) == 104

    # Sanity: compare the rebuild with the frozen adjusted ratings.
    frozen = {t["team"]: float(t["rating"]) for t in
              json.loads((ROOT / "data" / "simulation-prediction.json").read_text())["team_stats"]}
    full = build(raw, ORDER)
    drift = max(abs(full[t] - v) for t, v in frozen.items())
    mdrift = float(np.mean([abs(full[t] - v) for t, v in frozen.items()]))

    base = losses(raw, matches)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(base), size=(5000, len(base)))

    def row(label: str, ls: np.ndarray) -> str:
        d = ls - base
        boot = d[idx].mean(axis=1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        return f"| {label} | {ls.mean():.4f} | {d.mean():+.4f} | [{lo:+.4f}, {hi:+.4f}] |"

    head = "| Config | Log-loss | vs plain Elo | 95% CI (paired bootstrap) |\n|---|---:|---:|---|"
    cum, on = [row("Elo only", base)], []
    for name in ORDER:
        on.append(name)
        cum.append(row("+ " + name, losses(build(raw, on), matches)))
    loo = [row("All layers", losses(full, matches))]
    for name in ORDER:
        loo.append(row("All minus " + name, losses(build(raw, [n for n in ORDER if n != name]), matches)))

    md = f"""# 2026 layer ablation

Ratings at {CUTOFF} (raw Elo, profile v4a-flat-mild), 104 matches, 3-way log-loss, lower is better.
Negative "vs plain Elo" means the layers helped. Intervals resample matches; with n = 104 they are wide.
Approximate rebuild: the frozen file was generated 2026-06-06 from slightly earlier inputs, so rebuilt ratings differ from it by up to {drift:.0f} Elo points (mean {mdrift:.0f}). Read the layer effects as indicative, not an exact replay.

## Cumulative (layers added in simulator order)

{head}
{chr(10).join(cum)}

## Leave-one-out

{head}
{chr(10).join(loo)}
"""
    out = Path(ROOT / "data" / "capsule" / "layer_ablation_2026.md")
    out.write_text(md)
    print(md)


if __name__ == "__main__":
    main()
