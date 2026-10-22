"""Scrape Wikipedia squad pages for past tournaments to unlock the
squad-strength blend weight (backlog #7). Re-uses the parser from
worldcup.wiki_rosters; loops a list of (tournament, year, url).

Stores per-tournament JSON under data/raw/historical_rosters/.

Usage: uv run python scripts/scrape_historical_rosters.py
       uv run python scripts/scrape_historical_rosters.py --refresh
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import requests

from worldcup.db import PROJECT_ROOT
from wiki_rosters import USER_AGENT, parse_page

OUT_DIR = PROJECT_ROOT / "data" / "raw" / "historical_rosters"

# (tournament_name, year, wikipedia_url)
TARGETS = [
    ("FIFA World Cup", 2010, "https://en.wikipedia.org/wiki/2010_FIFA_World_Cup_squads"),
    ("FIFA World Cup", 2014, "https://en.wikipedia.org/wiki/2014_FIFA_World_Cup_squads"),
    ("FIFA World Cup", 2018, "https://en.wikipedia.org/wiki/2018_FIFA_World_Cup_squads"),
    ("FIFA World Cup", 2022, "https://en.wikipedia.org/wiki/2022_FIFA_World_Cup_squads"),
    ("UEFA Euro",      2012, "https://en.wikipedia.org/wiki/UEFA_Euro_2012_squads"),
    ("UEFA Euro",      2016, "https://en.wikipedia.org/wiki/UEFA_Euro_2016_squads"),
    ("UEFA Euro",      2021, "https://en.wikipedia.org/wiki/UEFA_Euro_2020_squads"),
    ("UEFA Euro",      2024, "https://en.wikipedia.org/wiki/UEFA_Euro_2024_squads"),
    ("Copa América",   2016, "https://en.wikipedia.org/wiki/Copa_Am%C3%A9rica_Centenario_squads"),
    ("Copa América",   2019, "https://en.wikipedia.org/wiki/2019_Copa_Am%C3%A9rica_squads"),
    ("Copa América",   2021, "https://en.wikipedia.org/wiki/2021_Copa_Am%C3%A9rica_squads"),
    ("Copa América",   2024, "https://en.wikipedia.org/wiki/2024_Copa_Am%C3%A9rica_squads"),
]


def slugify(tournament: str, year: int) -> str:
    s = tournament.lower().replace(" ", "_").replace("á", "a")
    return f"{s}_{year}"


def fetch_one(tournament: str, year: int, url: str, refresh: bool) -> tuple[int, int]:
    path = OUT_DIR / f"{slugify(tournament, year)}.json"
    if not refresh and path.exists():
        data = json.loads(path.read_text())
        return len(data["rosters"]), sum(len(r["players"]) for r in data["rosters"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
    resp.raise_for_status()
    rosters = parse_page(resp.text)
    payload = {
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tournament": tournament,
        "year": year,
        "source": url,
        "rosters": [
            {**asdict(r), "players": [asdict(p) for p in r.players]}
            for r in rosters
        ],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return len(rosters), sum(len(r.players) for r in rosters)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    print(f"Targets: {len(TARGETS)}")
    total_teams = total_players = 0
    for tournament, year, url in TARGETS:
        try:
            n_teams, n_players = fetch_one(tournament, year, url, args.refresh)
            print(f"  ✓ {tournament:18s} {year}  teams={n_teams:>3}  players={n_players:>4}")
            total_teams += n_teams; total_players += n_players
        except Exception as e:
            print(f"  ✗ {tournament:18s} {year}  ERROR  {e}")
        time.sleep(1.0)  # be nice to Wikipedia
    print(f"\nTotal: {total_teams} teams, {total_players} players")
    return 0


if __name__ == "__main__":
    sys.exit(main())
