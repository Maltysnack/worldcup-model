"""Freeze the current pre-tournament prediction as an archive.

Once kickoff hits 11 June 2026 the nightly cron will start incorporating
real match results into the rolling Elo, which will gradually overwrite
the "what we thought before any games were played" forecast. Running
this script writes a snapshot of `simulation.json` + `bracket.json` to
`simulation-prediction.json` + `bracket-prediction.json` with a
`frozen_at` timestamp.

The website's Prediction tab can then point at the frozen archive
instead of the rolling output so our pre-kickoff forecast stays intact
as a permanent record.

Run: `uv run python -m worldcup.freeze_prediction`

Intended to be called manually shortly before kickoff (e.g. evening of
10 June 2026, after the last day's friendlies have been ingested), so
the archive includes every signal available pre-tournament.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from worldcup.db import PROJECT_ROOT

SOURCE_PATHS = {
    "simulation":  PROJECT_ROOT / "data" / "simulation.json",
    "bracket":     PROJECT_ROOT / "data" / "bracket.json",
    "top_scorers": PROJECT_ROOT / "data" / "top_scorers.json",
}
TARGET_PATHS = {
    "simulation":  PROJECT_ROOT / "data" / "simulation-prediction.json",
    "bracket":     PROJECT_ROOT / "data" / "bracket-prediction.json",
    "top_scorers": PROJECT_ROOT / "data" / "top_scorers-prediction.json",
}
FROZEN_SENTINEL = PROJECT_ROOT / "data" / ".frozen"


def main() -> None:
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")

    for kind, src in SOURCE_PATHS.items():
        if not src.exists():
            cmd = {
                "simulation": "worldcup-simulate",
                "bracket": "worldcup-bracket",
                "top_scorers": "worldcup-predict-top-scorer",
            }[kind]
            print(f"  ! {src} missing, run `uv run {cmd}` first")
            sys.exit(1)
        data = json.loads(src.read_text())
        data["frozen_at"] = timestamp
        data["frozen_label"] = "Pre-kickoff archive"
        target = TARGET_PATHS[kind]
        target.write_text(json.dumps(data, indent=2))
        print(f"  froze {src.name}  ->  {target.name}")

    FROZEN_SENTINEL.write_text(f"frozen_at: {timestamp}\n")
    print(f"\nWrote sentinel: {FROZEN_SENTINEL}")
    print("CI will now skip simulate/bracket/top-scorer steps and publish the")
    print("frozen archives instead of the live JSONs. To unfreeze (e.g. for a")
    print("late critical bug), `rm data/.frozen` and commit.")
    print("\nCommit + push to lock in:")
    print("  git add data/.frozen data/*-prediction.json")
    print('  git commit -m "Freeze pre-kickoff prediction"')
    print("  git push")


if __name__ == "__main__":
    main()
