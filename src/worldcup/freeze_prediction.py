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
    "simulation": PROJECT_ROOT / "data" / "simulation.json",
    "bracket":    PROJECT_ROOT / "data" / "bracket.json",
}
TARGET_PATHS = {
    "simulation": PROJECT_ROOT / "data" / "simulation-prediction.json",
    "bracket":    PROJECT_ROOT / "data" / "bracket-prediction.json",
}


def main() -> None:
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")

    for kind, src in SOURCE_PATHS.items():
        if not src.exists():
            print(f"  ! {src} missing, run worldcup-{kind.replace('simulation', 'simulate')} first")
            sys.exit(1)
        data = json.loads(src.read_text())
        # Stamp the snapshot so the website can detect "this is archived".
        data["frozen_at"] = timestamp
        data["frozen_label"] = "Pre-kickoff archive"
        target = TARGET_PATHS[kind]
        target.write_text(json.dumps(data, indent=2))
        print(f"  froze {src.name}  ->  {target.name}  (frozen_at = {timestamp})")

    print("\nDone. Copy these files to the site repo:")
    for kind, target in TARGET_PATHS.items():
        print(f"  cp {target} <site-repo>/projects/worldcup-{target.stem.replace('-prediction', '')}-prediction.json")
    print("\nThen update the Prediction tab fetch in worldcup.html to use the -prediction.json paths,")
    print("and the nightly cron's Live updates will not overwrite the archive.")


if __name__ == "__main__":
    main()
