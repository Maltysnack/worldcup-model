"""Build per-match (predicted, actual) pairs for the shipped profile across
the 16 phase-3 tournaments, so we can iterate on reliability diagrams and
Platt/isotonic fits without rerunning the full 117-min validation.

Writes data/raw/calibration_audit_pairs.csv with one row per match:
  tournament, year, date, home, away, p_h, p_d, p_a, actual (0=H, 1=D, 2=A)
"""
from __future__ import annotations
import csv
import time

import numpy as np

from worldcup.comprehensive_validation import (
    TOURNAMENTS, apply_layers, get_pre_tournament_ratings, actual_outcome,
)
from worldcup.db import PROJECT_ROOT, connect
from worldcup.elo import HOME_ADVANTAGE
from worldcup.goals import expected_goals, wld_probs

PROFILE = "v4a-flat-mild"
LAYERS = dict(form=True, nt=True, host=True)  # "full"
OUT = PROJECT_ROOT / "data" / "raw" / "calibration_audit_pairs.csv"


def collect():
    rows = []
    t0 = time.time()
    for tournament, year, kickoff in TOURNAMENTS:
        print(f"[{time.time()-t0:5.1f}s] {tournament} {year} ...", flush=True)
        raw, _ = get_pre_tournament_ratings(PROFILE, kickoff)
        ratings = apply_layers(
            raw, tournament=tournament, year=year, kickoff=kickoff, **LAYERS,
        )
        with connect(read_only=True) as con:
            df = con.execute(
                """
                SELECT date, home_team, away_team, neutral, home_score, away_score
                FROM matches
                WHERE tournament = ? AND EXTRACT(YEAR FROM date) = ?
                  AND home_score IS NOT NULL
                ORDER BY date
                """,
                [tournament, year],
            ).fetchdf()
        for r in df.itertuples(index=False):
            rh = ratings.get(r.home_team)
            ra = ratings.get(r.away_team)
            if rh is None or ra is None:
                continue
            eff_h = rh + (0.0 if r.neutral else HOME_ADVANTAGE)
            gap = eff_h - ra
            lh, la = expected_goals(gap, neutral=bool(r.neutral))
            p_h, p_d, p_a = wld_probs(float(lh), float(la))
            rows.append({
                "tournament": tournament,
                "year": year,
                "date": str(r.date),
                "home": r.home_team,
                "away": r.away_team,
                "neutral": int(r.neutral),
                "p_h": p_h, "p_d": p_d, "p_a": p_a,
                "actual": actual_outcome(int(r.home_score), int(r.away_score)),
            })
    print(f"[{time.time()-t0:5.1f}s] done. {len(rows)} matches.", flush=True)
    return rows


def main():
    rows = collect()
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
