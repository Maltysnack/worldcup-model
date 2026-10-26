"""Export the data files for the v2 site (Bankroll + 2026 tabs).

Usage: uv run python scripts/export_site_v2.py OUTDIR

Writes OUTDIR/bankroll.json and OUTDIR/capsule.json. Until the bot is live,
bankroll.json is an untouched $1,000 with `source: not-started`. The synthetic
backtest stays internal (`uv run worldcup-betting-backtest`), never on the page.
"""

import json
import sys
from datetime import date
from pathlib import Path

from worldcup.betting import backtest

ROOT = Path(__file__).resolve().parents[1]
SCORECARD = ROOT / "data" / "capsule" / "scorecard_2026.json"
FROZEN_SIM = ROOT / "data" / "simulation-prediction.json"

RULE, MIN_EDGE = "flat", 0.03
STAGE_SHORT = {"champion": "Won", "final": "Final", "semi_finals": "SF", "quarter_finals": "QF",
               "round_of_16": "R16", "round_of_32": "R32", "group": "Group"}


def bankroll() -> dict:
    # No live ledger yet and no real historical odds (paid plan declined), so
    # the page shows an untouched $1,000 rather than synthetic backtest numbers.
    start = backtest.START_BANKROLL
    return {
        "source": "not-started",
        "note": "No bets yet. The bot needs a Sportsbet odds feed (free Odds API key).",
        "generated_at": date.today().isoformat(),
        "rule": {"staking": RULE, "stake": 10, "min_edge": MIN_EDGE},
        "start": start,
        "first_kickoff": None,
        "last_kickoff": None,
        "n_bets": 0,
        "n_won": 0,
        "staked": 0.0,
        "pnl": 0.0,
        "roi": None,
        "mean_clv": None,
        "curve": [start],
        "next_bet": None,
        "recent": [],
    }


def capsule() -> dict:
    sc = json.loads(SCORECARD.read_text())
    frozen_at = json.loads(FROZEN_SIM.read_text())["generated_at"][:10]
    allm = sc["summary"]["all"]
    teams = sorted(sc["teams"], key=lambda t: -t["p_champion"])[:6]
    final = next(m for m in sc["matches"] if m["stage"] == "final")
    return {
        "frozen_at": frozen_at,
        "model_version": sc["frozen_model"],
        "champion": sc["champion"],
        "final": {"home": final["home"], "away": final["away"], "score": final["score"]},
        "n_matches": allm["model"]["n"],
        "log_loss": {k: round(allm[k]["log_loss"], 3) for k in ("model", "plain_elo", "uniform")},
        "accuracy": round(allm["model"]["accuracy"], 4),
        "top": [{"team": t["team"], "p_champion": t["p_champion"],
                 "stage": STAGE_SHORT.get(t["stage_reached"], t["stage_reached"])} for t in teams],
    }


def main() -> None:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    for name, data in (("bankroll", bankroll()), ("capsule", capsule())):
        (out / f"{name}.json").write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")
        print(f"wrote {out / name}.json")


if __name__ == "__main__":
    main()
