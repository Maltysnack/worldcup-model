"""Export the data files for the v2 site (Bankroll + 2026 tabs).

Usage: uv run python scripts/export_site_v2.py OUTDIR

Writes OUTDIR/bankroll.json (from ledger/) and OUTDIR/capsule.json. With no
bets yet, bankroll.json is an untouched $1,000 with `source: not-started`. The
synthetic backtest stays internal (`uv run worldcup-betting-backtest`).
"""

import json
import sys
from datetime import date
from pathlib import Path

from worldcup.betting import engine
from worldcup.betting.ledger import Ledger

ROOT = Path(__file__).resolve().parents[1]
SCORECARD = ROOT / "data" / "capsule" / "scorecard_2026.json"
FROZEN_SIM = ROOT / "data" / "simulation-prediction.json"

RULE, MIN_EDGE = "flat", engine.DEFAULT_MIN_EDGE
STAGE_SHORT = {"champion": "Won", "final": "Final", "semi_finals": "SF", "quarter_finals": "QF",
               "round_of_16": "R16", "round_of_32": "R32", "group": "Group"}


def bankroll(ledger_dir: Path = ROOT / "ledger") -> dict:
    """Live numbers from the hash-chained ledger. Until the first bet the page
    shows an untouched $1,000; synthetic backtests never reach the page."""
    led = Ledger(ledger_dir)
    start, bets, sets = led.start_bankroll, led.bets(), led.settlements()
    base = {
        "generated_at": date.today().isoformat(),
        "rule": {"staking": RULE, "stake": engine.FLAT_UNIT, "min_edge": MIN_EDGE, "max_odds": engine.MAX_ODDS},
        "start": start,
    }
    if not bets:
        return {**base, "source": "not-started", "note": "No bets yet.", "first_kickoff": None,
                "n_bets": 0, "n_settled": 0, "n_won": 0, "staked": 0.0, "pnl": 0.0,
                "roi": None, "mean_clv": None, "curve": [start], "next_bet": None, "recent": []}
    by_hash = {b["hash"]: b for b in bets}
    curve, cash = [start], start
    for st in sets:
        cash += st["pnl"]
        curve.append(round(cash, 2))
    staked = sum(by_hash[st["bet_hash"]]["stake"] for st in sets if st["result"] != "void")
    pnl = sum(st["pnl"] for st in sets)
    clvs = [st["clv"] for st in sets if st.get("clv") is not None]
    upcoming = sorted(led.open_bets(), key=lambda b: b["kickoff"])
    pick = lambda b: "Draw" if b["selection"] == "draw" else b[b["selection"]]
    nxt = upcoming[0] if upcoming else None
    return {
        **base,
        "source": "ledger",
        "first_kickoff": min(b["kickoff"] for b in bets)[:10],
        "n_bets": len(bets),
        "n_settled": len(sets),
        "n_won": sum(st["result"] == "won" for st in sets),
        "staked": round(staked, 2),
        "pnl": round(pnl, 2),
        "roi": round(pnl / staked, 4) if staked else None,
        "mean_clv": round(sum(clvs) / len(clvs), 4) if clvs else None,
        "curve": curve,
        "next_bet": {"home": nxt["home"], "away": nxt["away"], "kickoff": nxt["kickoff"][:10],
                     "pick_name": pick(nxt), "odds": nxt["odds"], "stake": nxt["stake"]} if nxt else None,
        "recent": [
            {"kickoff": by_hash[st["bet_hash"]]["kickoff"][:10], "home": by_hash[st["bet_hash"]]["home"],
             "away": by_hash[st["bet_hash"]]["away"], "pick": by_hash[st["bet_hash"]]["selection"],
             "odds": by_hash[st["bet_hash"]]["odds"], "result": st["result"], "pnl": st["pnl"]}
            for st in sets[-5:]
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
