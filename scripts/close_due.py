"""Print 1 if any open bet without a closing price kicks off within 3 h, else 0.

Stdlib only, so bot.yml can run it before installing anything and skip idle
hourly close runs. Mirrors the `due` filter in betting/bot.py run_close.
"""

import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path


def rows(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []


def due(ledger: Path, now: datetime) -> bool:
    done = {s["bet_hash"] for s in rows(ledger / "settlements.jsonl")}
    closed = {c["event_id"] for c in rows(ledger / "closing.jsonl")}
    for b in rows(ledger / "bets.jsonl"):
        ko = datetime.fromisoformat(b["kickoff"].replace("Z", "+00:00"))
        if b["hash"] not in done and b["event_id"] not in closed and now < ko <= now + timedelta(hours=3):
            return True
    return False


if __name__ == "__main__":
    ledger = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("ledger")
    print(int(due(ledger, datetime.now(UTC))))
