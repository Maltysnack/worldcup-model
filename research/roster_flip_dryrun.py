"""Dry-run the trophy-score coverage flip.

The trophy aggregation falls back to hand-coded TEAM_TROPHY_SCORE when
fewer than 50% of WC teams have computed scores from real rosters. When
coverage crosses that threshold, the system auto-switches to computed
scores, which can produce a discrete rating shift for several teams
simultaneously.

This dry-run forces the computed scores active (overrides the gate),
recomputes ratings, and prints a side-by-side diff of:
  - per-team Elo rating (hand-coded -> computed)
  - per-team trophy score (the source of the rating shift)

Doesn't touch any pipeline outputs. Read-only. Lets us see in advance
what the flip will look like when it lands automatically.

Run: `uv run worldcup-roster-flip-dryrun`
"""

from __future__ import annotations

import sys

import worldcup.trophies as trophies_mod
from worldcup.bracket import FIFA_GROUPS
from worldcup.form import apply_form
from worldcup.host import apply_host_bonus
from worldcup.squad import blended_ratings
from worldcup.trophies import (
    TEAM_TROPHY_SCORE,
    TROPHY_DEFAULT,
    _load_computed_scores,
    apply_nt_bonus,
    team_trophy_score,
)


def get_full_ratings():
    """Build the full ratings dict (Elo + blend + form + NT + host),
    using whatever trophy-score source `team_trophy_score()` currently
    selects (gated or forced)."""
    # Pull current Elo from DB
    from worldcup.db import connect
    from worldcup.profiles import current_profile, table_suffix
    _PROFILE = current_profile()
    _ELO_TABLE = f"elo_current{table_suffix(_PROFILE)}"
    with connect(read_only=True) as con:
        raw_ratings = dict(
            con.execute(f"SELECT team, rating FROM {_ELO_TABLE}").fetchall()
        )
    return apply_host_bonus(
        apply_nt_bonus(apply_form(blended_ratings(raw_ratings))),
        "FIFA World Cup", 2026,
    )


def main() -> None:
    computed = _load_computed_scores()
    coverage = len(computed) / len(TEAM_TROPHY_SCORE)
    print(f"Current coverage: {len(computed)}/{len(TEAM_TROPHY_SCORE)} "
          f"= {coverage * 100:.1f}% (threshold {trophies_mod.COMPUTED_COVERAGE_THRESHOLD * 100:.0f}%)\n")

    if not computed:
        print("No computed scores available, run worldcup-fetch-rosters + "
              "worldcup-compute-trophies first.")
        return

    # 1) Ratings with current gate (likely hand-coded fallback)
    print("Building ratings under CURRENT gate (likely hand-coded fallback)...")
    ratings_before = dict(get_full_ratings())

    # 2) Force computed scores by temporarily disabling the threshold
    original = trophies_mod.COMPUTED_COVERAGE_THRESHOLD
    trophies_mod.COMPUTED_COVERAGE_THRESHOLD = 0.0
    # Cache is already populated; team_trophy_score will now see >=0 coverage
    print("Forcing computed scores (threshold lowered to 0)...")
    ratings_after = dict(get_full_ratings())
    trophies_mod.COMPUTED_COVERAGE_THRESHOLD = original  # restore

    # 3) Diff per team for the WC field
    wc_teams = sorted({t for teams in FIFA_GROUPS.values() for t in teams})

    print("\nPer-team rating impact (only teams with a non-zero shift shown):\n")
    print(f"  {'team':30s}  {'trophy: hand → comp':<24s}  {'rating: before → after':<26s}  Δ")
    print(f"  {'-'*30}  {'-'*24}  {'-'*26}  {'-'*6}")
    rows = []
    for team in wc_teams:
        hand = TEAM_TROPHY_SCORE.get(team, TROPHY_DEFAULT)
        comp = computed.get(team)
        rb = ratings_before.get(team)
        ra = ratings_after.get(team)
        if rb is None or ra is None:
            continue
        delta = ra - rb
        rows.append((team, hand, comp, rb, ra, delta))
    rows.sort(key=lambda r: -abs(r[5]))
    nonzero = [r for r in rows if abs(r[5]) > 0.1]
    if not nonzero:
        print("  (no team has a rating shift > 0.1 Elo points)")
    else:
        for team, hand, comp, rb, ra, delta in nonzero:
            comp_str = f"{comp:5.2f}" if comp is not None else " n/a"
            trophy_str = f"{hand:5.1f} → {comp_str}"
            rating_str = f"{rb:5.0f} → {ra:5.0f}"
            sign = "+" if delta >= 0 else ""
            print(f"  {team:30s}  {trophy_str:<24s}  {rating_str:<26s}  {sign}{delta:6.1f}")

    # 4) Champion-percentage shift would require re-running the simulator.
    # The diff above is the upstream signal; the user can eyeball whether
    # the magnitudes are concerning before letting the live flip happen.
    print(f"\n{len(nonzero)} teams shift > 0.1 Elo. Largest magnitude: "
          f"{max((abs(r[5]) for r in rows), default=0):.1f} Elo points.")
    print(f"Coverage will auto-flip when scores reach "
          f"{trophies_mod.COMPUTED_COVERAGE_THRESHOLD * 100:.0f}%; today we are at "
          f"{coverage * 100:.1f}%.")


if __name__ == "__main__":
    sys.exit(main())
