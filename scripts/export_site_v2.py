"""Export the data files for the v2 site (Bankroll + 2026 tabs).

Usage: uv run python scripts/export_site_v2.py OUTDIR

Writes OUTDIR/bankroll.json and OUTDIR/capsule.json. Until the bot is live,
bankroll.json is the synthetic-odds backtest (no_edge scenario, flat $10,
min edge 0.03, seed 1) and says so in its `source` field.
"""

import json
import sys
from datetime import date
from pathlib import Path

from worldcup.betting import backtest

ROOT = Path(__file__).resolve().parents[1]
SCORECARD = ROOT / "data" / "capsule" / "scorecard_2026.json"
FROZEN_SIM = ROOT / "data" / "simulation-prediction.json"

SCENARIO, RULE, MIN_EDGE, SEED = "no_edge", "flat", 0.03, 1
STAGE_SHORT = {"champion": "Won", "final": "Final", "semi_finals": "SF", "quarter_finals": "QF",
               "round_of_16": "R16", "round_of_32": "R32", "group": "Group"}
PICK = ["home", "draw", "away"]


def bankroll() -> dict:
    priced = backtest.attach_odds(backtest.load_matches(), SCENARIO, SEED)
    bets = backtest.run_bets(priced, RULE, MIN_EDGE)
    s = backtest.summarise_run(bets)
    return {
        "source": "backtest-synthetic",
        "note": "Synthetic Sportsbet prices (6% margin) on real results. Plumbing check, not evidence of edge.",
        "generated_at": date.today().isoformat(),
        "rule": {"staking": RULE, "stake": 10, "min_edge": MIN_EDGE, "scenario": SCENARIO, "seed": SEED},
        "start": backtest.START_BANKROLL,
        "first_kickoff": bets[0].kickoff[:10] if bets else None,
        "last_kickoff": bets[-1].kickoff[:10] if bets else None,
        "n_bets": s["n_bets"],
        "n_won": sum(b.outcome == b.selection for b in bets),
        "staked": round(s["staked"], 2),
        "pnl": round(s["pnl"], 2),
        "roi": round(s["roi"], 4),
        "mean_clv": round(s["mean_clv"], 4),
        "curve": [round(v, 2) for v in s["curve"]],
        "next_bet": None,
        "recent": [
            {"kickoff": b.kickoff[:10], "home": b.home, "away": b.away, "pick": PICK[b.selection],
             "odds": round(b.odds, 2), "stake": b.stake, "pnl": round(b.pnl, 2)}
            for b in bets[-5:]
        ],
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
