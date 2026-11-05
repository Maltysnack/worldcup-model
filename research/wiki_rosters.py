"""Scrape WC 2026 squad rosters from Wikipedia.

Source: https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_squads

Page structure (verified 2026-05-12):
  - Each of the 48 WC teams has an <h3> heading with the team name
  - Below the heading is either a placeholder paragraph (not yet
    announced) or a <table class="wikitable"> with columns:
      No., Pos., Player, Date of birth (age), Caps, Goals, Club
  - Final squad deadline: 7 June 2026; preliminary squads of up to 55
    players land any time before that

This module pulls the page, walks each team heading, and extracts the
squad table when present. Output is cached to data/raw/wc2026_rosters.json
so re-running cheaply re-uses cached data unless --refresh is passed.

Run: `uv run worldcup-fetch-rosters`
Or: `uv run worldcup-fetch-rosters --refresh` to force a re-fetch.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timezone

import requests
from bs4 import BeautifulSoup, NavigableString, Tag

from worldcup.db import PROJECT_ROOT

WIKI_URL = "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_squads"
USER_AGENT = "maltysnack-worldcup-hobby/0.1 (https://github.com/Maltysnack/worldcup)"
OUTPUT_PATH = PROJECT_ROOT / "data" / "raw" / "wc2026_rosters.json"


# Names that need normalisation to match our internal team names
# (which come from the martj42 international results dataset).
WIKI_TO_INTERNAL = {
    "United States": "United States",
    "Korea Republic": "South Korea",
    "South Korea": "South Korea",
    "Ivory Coast": "Ivory Coast",
    "Côte d'Ivoire": "Ivory Coast",
    "Cape Verde": "Cape Verde",
    "Curaçao": "Curaçao",
    "DR Congo": "DR Congo",
    "Democratic Republic of the Congo": "DR Congo",
    "Czech Republic": "Czech Republic",
    "Czechia": "Czech Republic",
    "Bosnia and Herzegovina": "Bosnia and Herzegovina",
    "New Zealand": "New Zealand",
    "South Africa": "South Africa",
    "Saudi Arabia": "Saudi Arabia",
}


@dataclass
class Player:
    number: int | None
    position: str          # "GK", "DF", "MF", "FW"
    name: str
    club: str
    age: int | None = None
    caps: int | None = None
    goals: int | None = None


@dataclass
class TeamRoster:
    team: str               # normalised internal name
    wiki_name: str          # name as it appears on Wikipedia
    players: list[Player]
    is_provisional: bool    # >26 players means it's a long-list, not final 26


def _clean(text: str) -> str:
    return " ".join(text.split()).strip()


def _parse_player_row(cells: list[Tag]) -> Player | None:
    """Convert a tr's cells into a Player. Wikipedia squad tables use
    a consistent column order: No., Pos., Player, DOB(age), Caps, Goals,
    Club. Older pages (pre-2018) omit the Goals column (6 cells); some
    teams have extra columns (e.g. continent of birth) which we ignore.
    Position cells often have format "1GK" or "GK"."""
    if len(cells) < 6:
        return None
    has_goals = len(cells) >= 7
    club_idx = 6 if has_goals else 5

    def text(c: Tag) -> str:
        return _clean(c.get_text(" "))

    # Position cell: strip leading digits ("1GK" -> "GK") and stop at first
    # 2-3 letter abbreviation.
    pos_raw = text(cells[1])
    pos = "".join(ch for ch in pos_raw if ch.isalpha())[:2].upper()
    if pos not in {"GK", "DF", "MF", "FW"}:
        # Some pages use 3-letter codes or weird capitalisation. Fall back
        # to whatever the cleaned text was so we at least record something.
        pos = pos_raw[:3].upper()

    number_raw = text(cells[0])
    try:
        number = int(number_raw) if number_raw else None
    except ValueError:
        number = None

    name = text(cells[2])
    if not name:
        return None

    # Date-of-birth cell contains "(YYYY-MM-DD) Month D, YYYY (age 27)"
    # Pull the age out of the parenthesised "(age N)" suffix.
    age = None
    dob_text = text(cells[3])
    if "(age" in dob_text:
        try:
            age = int(dob_text.split("(age")[-1].strip(" )").split()[0])
        except (ValueError, IndexError):
            pass

    def try_int(s: str) -> int | None:
        try:
            return int(s)
        except (ValueError, TypeError):
            return None

    caps = try_int(text(cells[4]))
    goals = try_int(text(cells[5])) if has_goals else None
    club = text(cells[club_idx])
    # Club text may include a country flag word like "Real Madrid" + " Spain"
    # if the flag icon's alt-text got pulled in. Strip trailing country.
    # Heuristic: country names are short, and the actual club is everything
    # except the last token IF the last token is a recognised country word.
    # In practice the BeautifulSoup get_text already drops most of this; we
    # just strip excessive whitespace.
    if not club:
        return None

    return Player(
        number=number, position=pos, name=name, club=club,
        age=age, caps=caps, goals=goals,
    )


def _find_next_squad_table(heading: Tag) -> Tag | None:
    """Walk forward from a team heading until we hit either a squad
    wikitable or the next heading. Returns the table or None.

    Wikipedia wraps each h3 in <div class="mw-heading mw-heading3">,
    so we start from the wrapper div (the h3's parent) and walk its
    siblings, not the h3's siblings directly.
    """
    # Start at the heading's wrapper if it has one, else the heading itself.
    start: Tag = heading.parent if (
        heading.parent
        and heading.parent.name == "div"
        and "mw-heading" in (heading.parent.get("class") or [])
    ) else heading
    sib = start
    while True:
        sib = sib.find_next_sibling()
        if sib is None:
            return None
        if isinstance(sib, NavigableString):
            continue
        # Stop if we hit another section heading (h2/h3 in a wrapper div)
        if sib.name == "div" and any(
            c.startswith("mw-heading") for c in (sib.get("class") or [])
        ):
            return None
        if sib.name in {"h2", "h3"}:
            return None
        if sib.name == "table" and "wikitable" in (sib.get("class") or []):
            header_cells = [c.get_text(strip=True) for c in sib.find_all("th")[:8]]
            if any("Player" in c for c in header_cells):
                return sib
            # Some squad tables have "name" or other variants; keep looking
        # Look for nested tables inside containers
        nested = sib.find("table", class_="wikitable") if hasattr(sib, "find") else None
        if nested is not None:
            header_cells = [c.get_text(strip=True) for c in nested.find_all("th")[:8]]
            if any("Player" in c for c in header_cells):
                return nested


def parse_page(html: str) -> list[TeamRoster]:
    soup = BeautifulSoup(html, "html.parser")
    # The mw-heading wrappers contain the actual h3. Walk through h3s in
    # document order; for each, find the next wikitable.
    rosters: list[TeamRoster] = []
    for heading in soup.find_all(["h3"]):
        # Skip h3 that are not team names (e.g. "Group A" sections often
        # use h3 too). Team names are typically just one or two words and
        # not group labels.
        wiki_name = _clean(heading.get_text())
        # Strip the [edit] link text Wikipedia appends.
        wiki_name = wiki_name.replace("[edit]", "").strip()
        if not wiki_name or wiki_name.lower().startswith("group "):
            continue
        if wiki_name.lower() in {"references", "see also", "notes", "external links"}:
            continue

        # Map to internal team name
        internal = WIKI_TO_INTERNAL.get(wiki_name, wiki_name)

        table = _find_next_squad_table(heading)
        if table is None:
            continue

        players: list[Player] = []
        for row in table.find_all("tr"):
            cells = row.find_all(["td", "th"])
            # Skip header rows: header rows are all <th>
            if not row.find("td"):
                continue
            p = _parse_player_row(cells)
            if p is not None:
                players.append(p)

        if not players:
            continue

        rosters.append(TeamRoster(
            team=internal,
            wiki_name=wiki_name,
            players=players,
            is_provisional=len(players) > 27,  # final squads are 26
        ))
    return rosters


def fetch_and_parse(refresh: bool = False) -> list[TeamRoster]:
    if not refresh and OUTPUT_PATH.exists():
        data = json.loads(OUTPUT_PATH.read_text())
        rosters = [
            TeamRoster(
                team=r["team"], wiki_name=r["wiki_name"],
                is_provisional=r["is_provisional"],
                players=[Player(**p) for p in r["players"]],
            )
            for r in data["rosters"]
        ]
        return [r for r in rosters if not r.is_provisional]
    print(f"  fetching {WIKI_URL}")
    resp = requests.get(WIKI_URL, headers={"User-Agent": USER_AGENT}, timeout=30)
    resp.raise_for_status()
    # Provisional squads (>27 players) are dropped: oversized lists distort
    # downstream consumers (top scorer, squad strength). They reappear here
    # automatically once Wikipedia shows the final 26.
    rosters = [r for r in parse_page(resp.text) if not r.is_provisional]
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps({
        "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "source": WIKI_URL,
        "rosters": [
            {**asdict(r), "players": [asdict(p) for p in r.players]}
            for r in rosters
        ],
    }, indent=2, ensure_ascii=False))
    print(f"  wrote {OUTPUT_PATH}  ({len(rosters)} teams)")
    return rosters


def main() -> None:
    p = argparse.ArgumentParser(description="Scrape WC 2026 squad rosters from Wikipedia.")
    p.add_argument("--refresh", action="store_true", help="Force a fresh fetch")
    args = p.parse_args()
    rosters = fetch_and_parse(refresh=args.refresh)

    print(f"\n{len(rosters)} teams with published rosters:")
    for r in sorted(rosters, key=lambda r: r.team):
        tag = "(provisional)" if r.is_provisional else "(final)"
        print(f"  {r.team:30s} {len(r.players):>3} players  {tag}")

    # Sanity-check club coverage: list distinct clubs found
    all_clubs = sorted({p.club for r in rosters for p in r.players})
    print(f"\n{len(all_clubs)} distinct clubs across all rosters")
    if len(all_clubs) <= 30:
        for c in all_clubs:
            print(f"  {c}")


if __name__ == "__main__":
    sys.exit(main())
