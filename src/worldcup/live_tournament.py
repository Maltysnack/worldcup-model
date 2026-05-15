"""Generate live tournament state for the website's Live tab.

For each WC 2026 fixture (and each played match), produce:
  - Group standings derived from results (currently empty until kickoff)
  - Per-match: our pre-match prediction + actual result if played
  - Knockout bracket as it fills in (R32 brackets populate post-groups)

Output: data/live_tournament.json

Tournament-status enum used by the UI:
  "pre"        – no WC 2026 match has been played
  "groups"     – at least one group match played, all 72 not done
  "knockouts"  – all 72 group matches done, KO phase underway
  "complete"   – final played

Each match's prediction is the SAME calibration as the rest of the
model: pre-match rating (from the predictions table, populated by
the Elo walk), through the goals model. So as the tournament unfolds
and ratings update after each result, the model's prediction for the
NEXT match reflects the updated rating.

Run: `uv run worldcup-build-live-tournament`
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd

from worldcup.bracket import FIFA_GROUPS
from worldcup.calibration import (
    LAMBDA_AWAY_AT_PARITY,
    LAMBDA_HOME_AT_PARITY,
    SLOPE_AWAY,
    SLOPE_HOME,
)
from worldcup.db import PROJECT_ROOT, connect
from worldcup.elo import HOME_ADVANTAGE
from worldcup.goals import expected_goals, wld_probs

KICKOFF = date(2026, 6, 11)
OUTPUT_PATH = PROJECT_ROOT / "data" / "live_tournament.json"
N_GROUP_MATCHES = 72   # 12 groups × 6 matches each


def _date_str(d) -> str:
    if hasattr(d, "strftime"):
        return d.strftime("%Y-%m-%d")
    return str(d)


def load_wc_matches():
    """Return (matches_df, predictions_for_played, simulation_predictions).

    matches_df: every WC 2026 fixture (played + unplayed)
    predictions_for_played: rows from the predictions table for matches
        that have been played (each has rating_home_pre / rating_away_pre
        captured at the chronological walk's moment)
    simulation_predictions: list of dicts from simulation.json's
        match_predictions for WC fixtures; covers unplayed matches with
        current-rating-based probabilities
    """
    with connect(read_only=True) as con:
        matches = con.execute(
            """
            SELECT date, home_team, away_team, neutral, city, country,
                   home_score, away_score
            FROM matches
            WHERE tournament = 'FIFA World Cup'
              AND date >= DATE '2026-06-11'
            ORDER BY date, home_team
            """
        ).fetchdf()
        preds = con.execute(
            """
            SELECT match_date, home_team, away_team,
                   rating_home_pre, rating_away_pre
            FROM predictions
            WHERE tournament = 'FIFA World Cup'
              AND match_date >= DATE '2026-06-11'
            """
        ).fetchdf()

    # Pull current-rating-based predictions from simulation.json (these
    # cover every unplayed WC fixture). simulation.json may not exist
    # on a fresh checkout; treat that case gracefully.
    sim_path = PROJECT_ROOT / "data" / "simulation.json"
    sim_preds: list[dict] = []
    if sim_path.exists():
        try:
            sim_data = json.loads(sim_path.read_text())
            sim_preds = [
                m for m in sim_data.get("match_predictions", [])
                if m.get("tournament") == "FIFA World Cup"
            ]
        except Exception:
            pass
    return matches, preds, sim_preds


def compute_prediction(rating_h: float, rating_a: float, neutral: bool) -> dict:
    eff_h = float(rating_h) + (0.0 if neutral else HOME_ADVANTAGE)
    gap = eff_h - float(rating_a)
    lh, la = expected_goals(gap, neutral=neutral)
    p_h, p_d, p_a = wld_probs(float(lh), float(la))
    return {
        "p_home_win": round(float(p_h), 4),
        "p_draw": round(float(p_d), 4),
        "p_away_win": round(float(p_a), 4),
        "expected_home_goals": round(float(lh), 2),
        "expected_away_goals": round(float(la), 2),
        "modal_score": [max(0, round(float(lh))), max(0, round(float(la)))],
    }


def compute_standings(matches: list[dict], group_teams: list[str]) -> list[dict]:
    """Return standings sorted by FIFA criteria (points → GD → GF)."""
    stats = {t: {"played": 0, "won": 0, "drawn": 0, "lost": 0,
                 "gf": 0, "ga": 0, "gd": 0, "points": 0}
             for t in group_teams}
    for m in matches:
        if m["home_score"] is None or m["away_score"] is None:
            continue
        h, a = m["home_team"], m["away_team"]
        hs, as_ = m["home_score"], m["away_score"]
        for team in (h, a):
            if team not in stats:
                continue
        if h in stats:
            stats[h]["played"] += 1
            stats[h]["gf"] += hs
            stats[h]["ga"] += as_
        if a in stats:
            stats[a]["played"] += 1
            stats[a]["gf"] += as_
            stats[a]["ga"] += hs
        if hs > as_:
            stats[h]["won"] += 1
            stats[h]["points"] += 3
            stats[a]["lost"] += 1
        elif hs < as_:
            stats[a]["won"] += 1
            stats[a]["points"] += 3
            stats[h]["lost"] += 1
        else:
            stats[h]["drawn"] += 1
            stats[h]["points"] += 1
            stats[a]["drawn"] += 1
            stats[a]["points"] += 1
    for t in stats:
        stats[t]["gd"] = stats[t]["gf"] - stats[t]["ga"]

    sorted_teams = sorted(
        stats.keys(),
        key=lambda t: (stats[t]["points"], stats[t]["gd"], stats[t]["gf"]),
        reverse=True,
    )
    rows = []
    for i, t in enumerate(sorted_teams):
        rows.append({
            "team": t,
            "position": i + 1 if any(stats[x]["played"] for x in stats) else None,
            **stats[t],
        })
    return rows


def main() -> None:
    matches_df, preds_df, sim_preds = load_wc_matches()

    # Two prediction sources:
    # 1) predictions table — has rating_pre captured at walk time
    #    (only for played matches in our DB)
    # 2) simulation.json's match_predictions — current-rating-based
    #    forecasts for every WC fixture
    # Use (1) for played matches (lets us show "what we'd have said
    # going in") and (2) for unplayed matches.
    rating_pre_lookup = {}
    for row in preds_df.itertuples(index=False):
        key = (_date_str(row.match_date), row.home_team, row.away_team)
        rating_pre_lookup[key] = (
            float(row.rating_home_pre), float(row.rating_away_pre),
        )

    sim_pred_lookup = {}
    for sp in sim_preds:
        key = (sp.get("match_date"), sp.get("home_team"), sp.get("away_team"))
        sim_pred_lookup[key] = sp

    # Partition matches by group (group matches) vs knockout
    matches_by_group: dict[str, list[dict]] = defaultdict(list)
    knockout_matches: list[dict] = []
    for row in matches_df.itertuples(index=False):
        d = _date_str(row.date)
        h, a = row.home_team, row.away_team
        played = not (row.home_score is None or pd.isna(row.home_score))
        m_obj = {
            "date": d,
            "home_team": h,
            "away_team": a,
            "neutral": bool(row.neutral),
            "city": row.city,
            "country": row.country,
            "home_score": int(row.home_score) if played else None,
            "away_score": int(row.away_score) if played else None,
            "played": played,
        }
        # Attach pre-match prediction.
        # Played matches: reconstruct from the predictions table's
        # rating_pre, so the prediction reflects the rating going INTO
        # the match (not the post-match updated rating).
        # Unplayed matches: pull from simulation.json which has the
        # current-rating-based prediction.
        rating_pair = rating_pre_lookup.get((d, h, a))
        if rating_pair is not None:
            m_obj["prediction"] = compute_prediction(
                rating_pair[0], rating_pair[1], m_obj["neutral"],
            )
            m_obj["rating_home_pre"] = round(rating_pair[0])
            m_obj["rating_away_pre"] = round(rating_pair[1])
        else:
            sp = sim_pred_lookup.get((d, h, a))
            if sp is not None:
                m_obj["prediction"] = {
                    "p_home_win": sp.get("p_home_win"),
                    "p_draw": sp.get("p_draw"),
                    "p_away_win": sp.get("p_away_win"),
                    "expected_home_goals": sp.get("expected_home_goals"),
                    "expected_away_goals": sp.get("expected_away_goals"),
                    "modal_score": sp.get("modal_score"),
                }

        # Group classification
        group_letter = None
        for letter, teams in FIFA_GROUPS.items():
            if h in teams and a in teams:
                group_letter = letter
                break
        if group_letter:
            matches_by_group[group_letter].append(m_obj)
        else:
            knockout_matches.append(m_obj)

    # Build group output. Top 2 + best-third indicator computed only
    # once group is mathematically settled; for now we surface position.
    groups_out = []
    for letter in "ABCDEFGHIJKL":
        team_list = FIFA_GROUPS[letter]
        matches = sorted(matches_by_group.get(letter, []), key=lambda m: m["date"])
        standings = compute_standings(matches, team_list)
        groups_out.append({
            "name": letter,
            "teams": team_list,
            "standings": standings,
            "matches": matches,
        })

    # Bucket the knockout matches into rounds using chronological order.
    # WC 2026 schedule (48-team): 16 R32 → 8 R16 → 4 QF → 2 SF → 1 (3rd
    # place, ignored) → 1 Final. We sort by date and slice. As real KO
    # matches get scheduled and played they fall into the right bucket
    # without us needing to hard-code team-pair lookups.
    knockout_matches_sorted = sorted(knockout_matches, key=lambda m: m["date"])

    def _as_bracket_match(m: dict) -> dict:
        """Reshape a live match dict to match the bracket.json match
        shape so the site's existing renderBracketInto can render it."""
        winner = None
        pens = False
        if m.get("played"):
            sa = m.get("home_score") or 0
            sb = m.get("away_score") or 0
            if sa > sb:
                winner = m["home_team"]
            elif sb > sa:
                winner = m["away_team"]
            else:
                # Drawn after 90/ET → penalty shootout. We don't yet have
                # explicit pen-winner data in our matches table, so we
                # mark pens=True and leave winner null until the next
                # match's winner reveals who advanced.
                pens = True
        return {
            "team_a": m["home_team"],
            "team_b": m["away_team"],
            "rating_a": None,
            "rating_b": None,
            "winner": winner,
            "score_a": m.get("home_score"),
            "score_b": m.get("away_score"),
            "pens": pens,
            "played": bool(m.get("played")),
            "date": m["date"],
        }

    ko_shaped = [_as_bracket_match(m) for m in knockout_matches_sorted]
    knockout_bucketed = {
        "round_of_32": ko_shaped[0:16],
        "round_of_16": ko_shaped[16:24],
        "quarter_finals": ko_shaped[24:28],
        "semi_finals": ko_shaped[28:30],
        # ko_shaped[30] = 3rd-place playoff if scheduled; we ignore it
        # for the bracket. The final is the LAST chronological KO match.
        "final": ko_shaped[-1:] if len(ko_shaped) >= 31 else [],
        "champion": None,
    }
    # Champion: winner of the final match if it has been played
    if knockout_bucketed["final"]:
        f = knockout_bucketed["final"][0]
        if f.get("winner"):
            knockout_bucketed["champion"] = f["winner"]

    # Tournament status
    def _played(r) -> bool:
        return not (r.home_score is None or pd.isna(r.home_score))

    n_played = sum(1 for r in matches_df.itertuples(index=False) if _played(r))
    n_total = len(matches_df)
    n_groups_played = sum(
        1 for r in matches_df.itertuples(index=False)
        if _played(r) and any(
            r.home_team in teams and r.away_team in teams
            for teams in FIFA_GROUPS.values()
        )
    )
    if n_played == 0:
        status = "pre"
    elif n_groups_played < N_GROUP_MATCHES:
        status = "groups"
    elif n_played < n_total:
        status = "knockouts"
    else:
        status = "complete"

    # Next match (chronologically first unplayed)
    next_match = None
    for row in matches_df.itertuples(index=False):
        if not _played(row):
            next_match = {
                "date": _date_str(row.date),
                "home_team": row.home_team,
                "away_team": row.away_team,
            }
            break

    today_str = date.today().isoformat()
    days_to_kickoff = (KICKOFF - date.today()).days

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "kickoff": KICKOFF.isoformat(),
        "today": today_str,
        "days_to_kickoff": days_to_kickoff,
        "tournament_status": status,
        "n_played": int(n_played),
        "n_total": int(n_total),
        "n_group_matches_played": int(n_groups_played),
        "n_group_matches_total": N_GROUP_MATCHES,
        "next_match": next_match,
        "groups": groups_out,
        # Pre-bucketed knockout phase for the bracket renderer. Same
        # shape as bracket.json's `knockout` so the existing JS bracket
        # renderer can consume it directly.
        "knockout": knockout_bucketed,
        # Backwards-compat raw list (deprecated; remove when site no longer reads).
        "knockout_matches": knockout_matches,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2, ensure_ascii=False))

    print(f"Tournament status: {status}")
    print(f"Played: {n_played}/{n_total}  ({n_groups_played}/{N_GROUP_MATCHES} group, "
          f"{n_played - n_groups_played}/{n_total - N_GROUP_MATCHES} KO)")
    if days_to_kickoff > 0:
        print(f"Days to kickoff: {days_to_kickoff}")
    if next_match:
        print(f"Next match: {next_match['date']} {next_match['home_team']} v {next_match['away_team']}")
    print(f"\nWrote {OUTPUT_PATH}")


if __name__ == "__main__":
    sys.exit(main())
