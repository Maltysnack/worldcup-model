"""Weekly status for the owner, as Markdown on stdout. weekly.yml posts it as a
GitHub issue assigned to Maltysnack, which GitHub emails.

Usage: uv run python scripts/weekly_report.py [runs.json]
  runs.json: `gh run list --workflow bot.yml --json conclusion,createdAt,event,displayTitle,url`
  ODDS_API_KEY in env adds credits left (free /sports call).
"""

import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_site_v2 import ROOT, TITLE_ODDS, bankroll  # noqa: E402

from worldcup.betting.ledger import Ledger  # noqa: E402


def report(now: datetime, runs: list[dict], credits: int | None) -> str:
    b = bankroll(now=now)
    led = Ledger(ROOT / "ledger")
    since = (now - timedelta(days=7)).isoformat()
    week_bets = [x for x in led.bets() if x["placed_at"] >= since]
    week_sets = [x for x in led.settlements() if x.get("settled_at", "") >= since]
    week_pass = [x for x in led.passes() if x["decided_at"] >= since]
    cur = b["curve"][-1]
    L = [f"**Bankroll ${cur:,.2f}** ({cur - b['start']:+,.2f} from ${b['start']:,.0f}), {b['n_bets']} bets, "
         f"{b['n_settled']} settled, {b['n_won']} won."
         + (f" ROI {b['roi'] * 100:+.1f}%." if b["roi"] is not None else "")
         + (f" Mean CLV {b['mean_clv'] * 100:+.1f}%." if b["mean_clv"] is not None else ""), "",
         "## This week",
         f"- Bets placed: {len(week_bets)}; passes: {len(week_pass)}; settled: {len(week_sets)} "
         f"(P&L {sum(s['pnl'] for s in week_sets):+.2f})."]
    L += [f"  - {x['home']} v {x['away']} {x['kickoff'][:10]}: {x['selection']} @ {x['odds']}" for x in week_bets]
    if b["upcoming"]:
        L.append("- Upcoming: " + "; ".join(f"{g['home']} v {g['away']} ({g['decision']})" for g in b["upcoming"]))
    fails = [r for r in runs if r["conclusion"] == "failure" and r["createdAt"] >= since]
    L += ["", "## Health",
          f"- Bot runs in the last 7 days: {sum(r['createdAt'] >= since for r in runs)}, failed: {len(fails)}."]
    L += [f"  - [{r['createdAt'][:16]}Z failed]({r['url']})" for r in fails[:5]]
    L.append(f"- Odds API credits left (shared with FPL): {credits if credits is not None else 'unknown'}.")
    if TITLE_ODDS.exists():
        o = json.loads(TITLE_ODDS.read_text())
        L += ["", f"## Title odds (as of {o['as_of']})"]
        L += [f"- {t['name']}: " + ", ".join(f"{x['team']} {x['p_champion'] * 100:.0f}%" for x in t["top"][:3])
              for t in o["tournaments"]]
    handoff = (ROOT / "HANDOFF.md").read_text()
    if "## USER ACTION NEEDED" in handoff:
        block = handoff.split("## USER ACTION NEEDED", 1)[1].split("\n## ", 1)[0].strip()
        L += ["", "## Needs you", block]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    runs = json.loads(Path(sys.argv[1]).read_text()) if len(sys.argv) > 1 else []
    credits = None
    if os.environ.get("ODDS_API_KEY"):
        from worldcup.betting.bot import OddsClient

        c = OddsClient(os.environ["ODDS_API_KEY"])
        c.sports()
        credits = c.remaining
    print(report(datetime.now(timezone.utc), runs, credits), end="")
