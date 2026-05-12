"""Scrape per-player 2025-26 club season stats from Wikipedia.

For each club appearing in the WC rosters we resolve the club's
"2025-26 [Club] season" Wikipedia page and extract two tables:
  - Appearances by competition (Pld / League / Cup / Europe / Total)
  - Goals by competition  (Rank / Player / League / Cup / Europe / Total)

Output is cached per-club to data/raw/club_season_stats/{slug}.json so
re-running is cheap. Adds a polite ~0.8s delay between fetches.

Wikipedia doesn't expose minutes-played directly, so appearances is our
proxy for "how regularly a player featured" — a regular starter plays
30+ league matches, a fringe player plays <10. Good enough for the
WC trophy + top-scorer use cases until FBref (or similar minutes-level
data) becomes accessible.

URL resolution: Wikipedia is inconsistent about whether club season
pages use "F.C." / "FC" / no suffix / etc. We try a small set of
canonical patterns + a Wikipedia search API fallback.

Run: `uv run worldcup-fetch-club-stats`
       `uv run worldcup-fetch-club-stats --clubs "Inter Milan,Barcelona"`
       `uv run worldcup-fetch-club-stats --refresh`
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path

import requests
from bs4 import BeautifulSoup, Tag

from worldcup.db import PROJECT_ROOT
from worldcup.wiki_rosters import fetch_and_parse as fetch_rosters

USER_AGENT = "maltysnack-worldcup-hobby/0.1 (https://github.com/Maltysnack/worldcup)"
CACHE_DIR = PROJECT_ROOT / "data" / "raw" / "club_season_stats"
FETCH_DELAY_S = 0.8  # polite-citizen delay between requests


@dataclass
class PlayerStat:
    name: str
    position: str | None = None
    appearances: dict[str, int] = field(default_factory=dict)  # competition -> games
    goals: dict[str, int] = field(default_factory=dict)        # competition -> goals
    total_appearances: int = 0
    total_goals: int = 0

    def goals_per_appearance(self) -> float:
        if self.total_appearances == 0:
            return 0.0
        return self.total_goals / self.total_appearances


@dataclass
class ClubSeason:
    club: str               # normalised name as used by callers
    wiki_url: str           # the resolved Wikipedia URL
    season: str = "2025-26"
    players: list[PlayerStat] = field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────
# URL resolution
# ──────────────────────────────────────────────────────────────────────

# Manual overrides for clubs whose Wikipedia URL doesn't follow the
# generic pattern. Keys are normalised club names (as they appear in
# WC squad rosters from Wikipedia).
URL_OVERRIDES: dict[str, str] = {
    "Manchester City": "https://en.wikipedia.org/wiki/2025%E2%80%9326_Manchester_City_F.C._season",
    "Manchester United": "https://en.wikipedia.org/wiki/2025%E2%80%9326_Manchester_United_F.C._season",
    "Arsenal": "https://en.wikipedia.org/wiki/2025%E2%80%9326_Arsenal_F.C._season",
    "Liverpool": "https://en.wikipedia.org/wiki/2025%E2%80%9326_Liverpool_F.C._season",
    "Chelsea": "https://en.wikipedia.org/wiki/2025%E2%80%9326_Chelsea_F.C._season",
    "Tottenham Hotspur": "https://en.wikipedia.org/wiki/2025%E2%80%9326_Tottenham_Hotspur_F.C._season",
    "Aston Villa": "https://en.wikipedia.org/wiki/2025%E2%80%9326_Aston_Villa_F.C._season",
    "Newcastle United": "https://en.wikipedia.org/wiki/2025%E2%80%9326_Newcastle_United_F.C._season",
    "Bayern Munich": "https://en.wikipedia.org/wiki/2025%E2%80%9326_FC_Bayern_Munich_season",
    "Inter Milan": "https://en.wikipedia.org/wiki/2025%E2%80%9326_Inter_Milan_season",
    "Internazionale": "https://en.wikipedia.org/wiki/2025%E2%80%9326_Inter_Milan_season",
    "Real Madrid": "https://en.wikipedia.org/wiki/2025%E2%80%9326_Real_Madrid_CF_season",
    "Atlético Madrid": "https://en.wikipedia.org/wiki/2025%E2%80%9326_Atl%C3%A9tico_Madrid_season",
    "Barcelona": "https://en.wikipedia.org/wiki/2025%E2%80%9326_FC_Barcelona_season",
    "Paris Saint-Germain": "https://en.wikipedia.org/wiki/2025%E2%80%9326_Paris_Saint-Germain_FC_season",
    "Borussia Dortmund": "https://en.wikipedia.org/wiki/2025%E2%80%9326_Borussia_Dortmund_season",
    "Bayer Leverkusen": "https://en.wikipedia.org/wiki/2025%E2%80%9326_Bayer_04_Leverkusen_season",
    "Juventus": "https://en.wikipedia.org/wiki/2025%E2%80%9326_Juventus_FC_season",
    "AC Milan": "https://en.wikipedia.org/wiki/2025%E2%80%9326_AC_Milan_season",
    "Napoli": "https://en.wikipedia.org/wiki/2025%E2%80%9326_SSC_Napoli_season",
    "AS Roma": "https://en.wikipedia.org/wiki/2025%E2%80%9326_AS_Roma_season",
    "Roma": "https://en.wikipedia.org/wiki/2025%E2%80%9326_AS_Roma_season",
    "Sporting CP": "https://en.wikipedia.org/wiki/2025%E2%80%9326_Sporting_CP_season",
    "Benfica": "https://en.wikipedia.org/wiki/2025%E2%80%9326_S.L._Benfica_season",
    "Porto": "https://en.wikipedia.org/wiki/2025%E2%80%9326_FC_Porto_season",
}


def slugify(club: str) -> str:
    """Filesystem-safe slug for a club name."""
    s = unicodedata.normalize("NFKD", club).encode("ascii", "ignore").decode()
    return re.sub(r"[^\w\-]+", "_", s).strip("_").lower()


def _candidate_urls(club: str) -> list[str]:
    """Generate plausible Wikipedia URL patterns for a club season."""
    # Title-case + space-to-underscore + en-dash year range
    encoded = club.replace(" ", "_")
    season_prefix = "2025%E2%80%9326_"  # 2025–26 (en-dash)
    return [
        f"https://en.wikipedia.org/wiki/{season_prefix}{encoded}_season",
        f"https://en.wikipedia.org/wiki/{season_prefix}{encoded}_F.C._season",
        f"https://en.wikipedia.org/wiki/{season_prefix}{encoded}_FC_season",
        f"https://en.wikipedia.org/wiki/{season_prefix}FC_{encoded}_season",
    ]


def resolve_url(club: str) -> str | None:
    """Find the Wikipedia URL for a club's 2025-26 season page, or None
    if no plausible page exists (e.g. small clubs without season pages)."""
    if club in URL_OVERRIDES:
        return URL_OVERRIDES[club]
    for url in _candidate_urls(club):
        try:
            head = requests.head(url, headers={"User-Agent": USER_AGENT},
                                 timeout=15, allow_redirects=True)
        except requests.RequestException:
            continue
        if head.status_code == 200:
            return head.url  # follow redirects
    # Final fallback: Wikipedia search API
    try:
        api = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "opensearch",
                "search": f"2025-26 {club} season",
                "limit": "3",
                "format": "json",
            },
            headers={"User-Agent": USER_AGENT},
            timeout=15,
        ).json()
        if api and len(api) >= 4 and api[3]:
            return api[3][0]
    except Exception:
        pass
    return None


# ──────────────────────────────────────────────────────────────────────
# Table parsing
# ──────────────────────────────────────────────────────────────────────

# Possible headers indicating which table is which kind of stat.
def _clean(s: str) -> str:
    return " ".join(s.split()).strip()


def _is_player_stat_table(headers: list[str]) -> bool:
    """Tables we care about: include Player + Total + at least 2 competition columns."""
    h_low = [h.lower() for h in headers]
    if not any("player" in h for h in h_low):
        return False
    if not any(h.lower() == "total" for h in headers):
        return False
    competition_likes = sum(
        1 for h in headers
        if any(kw in h.lower() for kw in [
            "league", "cup", "champions", "europa", "conference", "bundesliga",
            "serie", "ligue", "liga", "primeira", "eredivisie", "süper",
        ])
    )
    return competition_likes >= 2


def _try_int(s: str) -> int:
    """Safe integer parse; returns 0 for blanks, dashes, etc."""
    s = _clean(s)
    if not s or s in {"—", "-", "–"}:
        return 0
    try:
        return int(s)
    except ValueError:
        return 0


def _looks_like_player_name(s: str) -> bool:
    """Filter out obvious non-player strings that get captured by the
    name-column heuristic: shirt numbers, totals rows, position headers,
    blank cells. A real name has at least one letter and isn't all
    digits / short separators."""
    s = _clean(s)
    if not s:
        return False
    if s.lower() in {"totals", "total", "own goals", "goalkeepers",
                     "defenders", "midfielders", "forwards", "own goal"}:
        return False
    if s.isdigit():
        return False
    # Must contain at least one alphabetic letter
    if not any(ch.isalpha() for ch in s):
        return False
    # Strip trailing "*" (Wikipedia uses asterisks for loaned/transferred)
    return True


def _find_section_for(table: Tag) -> str:
    """Walk back from a table to find the nearest enclosing h3/h4 section
    heading. Returns the heading text (lowercased) or empty string."""
    cur = table
    for _ in range(50):  # bounded walk to be safe
        cur = cur.find_previous(["h2", "h3", "h4"])
        if cur is None:
            return ""
        txt = _clean(cur.get_text()).lower().replace("[edit]", "").strip()
        if txt:
            return txt
    return ""


def _classify_table(table: Tag, headers: list[str], second_row_cells: list[str]) -> str:
    """Return 'combined', 'goals', 'appearances', 'assists', 'discipline',
    or 'unknown' based on table structure and surrounding section.

    A 'combined' table has paired Apps/Goals sub-headers in row 2.
    Otherwise we look at the nearest section heading to disambiguate."""
    # Combined tables: row 2 contains alternating Apps/Goals
    second_lower = [s.lower() for s in second_row_cells]
    if "apps" in second_lower and "goals" in second_lower:
        return "combined"

    # Use section context for single-stat tables
    section = _find_section_for(table)
    if "scorer" in section or "goalscorer" in section or "top scorer" in section:
        return "goals"
    if "appearance" in section or "matches played" in section:
        return "appearances"
    if "assist" in section:
        return "assists"
    if "discipline" in section or "card" in section:
        return "discipline"
    if "clean sheet" in section:
        return "clean_sheets"
    # Fallback: look at the table caption or first headings
    return "unknown"


def _parse_combined_table(table: Tag, headers: list[str]) -> list[PlayerStat]:
    """Parse a combined apps+goals table where each competition column
    splits into Apps / Goals sub-columns. Header pattern (Barcelona-style):

        No. | Pos. | Nat. | Player | La Liga | Cup | UCL | Total | Discipline
                                    [Apps|Goals| Apps|Goals|...]
    """
    rows = table.find_all("tr")
    if len(rows) < 2:
        return []

    # Column meanings: walk the top header, expand each competition
    # column into (comp, "Apps") and (comp, "Goals") pairs in order.
    # Pre-data columns (No, Pos, Nat, Player) stay 1-wide.
    pre_cols = []  # column indices that are metadata
    comp_cols = []  # list of competition names in order
    name_col = None
    pos_col = None
    h_idx = 0
    h_low = [h.lower() for h in headers]
    for i, h in enumerate(headers):
        hl = h.lower()
        if hl in {"no.", "no", "rank"}:
            pre_cols.append(i)
        elif hl in {"pos.", "pos"}:
            pre_cols.append(i)
            pos_col = i
        elif hl in {"nat.", "nat", "nationality"}:
            pre_cols.append(i)
        elif "player" in hl:
            pre_cols.append(i)
            name_col = i
        elif hl == "discipline" or hl == "cards":
            break  # discipline is after Total
        else:
            comp_cols.append(h)

    if name_col is None:
        return []

    # Each competition column expands to 2 actual data columns. So the
    # cell index for "La Liga Apps" = name_col + 1 + (comp_idx * 2)
    # "La Liga Goals" = name_col + 1 + (comp_idx * 2) + 1

    players: dict[str, PlayerStat] = {}
    for row in rows[2:]:  # skip header + sub-header row
        cells = row.find_all(["td", "th"])
        if len(cells) < name_col + 1 + len(comp_cols) * 2:
            continue
        # Position-section divider rows ("Goalkeepers", "Defenders" etc)
        # have only 1 cell with text spanning all columns.
        if len(cells) <= 2:
            continue

        name = _clean(cells[name_col].get_text(" ")).rstrip("*").strip()
        if not _looks_like_player_name(name):
            continue
        pos = None
        if pos_col is not None and pos_col < len(cells):
            ptxt = _clean(cells[pos_col].get_text())
            if ptxt in {"GK", "DF", "MF", "FW"}:
                pos = ptxt

        p = players.setdefault(name, PlayerStat(name=name, position=pos))

        apps_total = 0
        goals_total = 0
        for ci, comp in enumerate(comp_cols):
            ai = name_col + 1 + ci * 2
            gi = ai + 1
            if gi >= len(cells):
                break
            a = _try_int(cells[ai].get_text())
            g = _try_int(cells[gi].get_text())
            if comp.lower() != "total":  # don't double-count the Total column
                p.appearances[comp] = a
                p.goals[comp] = g
                apps_total += a
                goals_total += g
        # Prefer the explicit Total column if it exists at the end
        # (Total is the last comp_col by convention)
        if comp_cols and comp_cols[-1].lower() == "total":
            ti = name_col + 1 + (len(comp_cols) - 1) * 2
            if ti + 1 < len(cells):
                p.total_appearances = _try_int(cells[ti].get_text())
                p.total_goals = _try_int(cells[ti + 1].get_text())
            else:
                p.total_appearances = apps_total
                p.total_goals = goals_total
        else:
            p.total_appearances = apps_total
            p.total_goals = goals_total

    return list(players.values())


def _parse_single_stat_table(table: Tag, headers: list[str], stat_kind: str) -> list[tuple[str, str | None, dict[str, int], int]]:
    """Parse a Format-B table (single stat per competition column).
    Returns (name, position, comp_stats, total). stat_kind is 'goals'
    or 'appearances'."""
    rows = table.find_all("tr")[1:]  # skip header

    # Identify column meanings
    name_col = None
    pos_col = None
    comp_cols = []  # (idx, comp_name)
    total_col = None
    for i, h in enumerate(headers):
        hl = h.lower()
        if "player" in hl:
            name_col = i
        elif hl in {"pos.", "pos"}:
            pos_col = i
        elif hl in {"rank", "no.", "no", "nat.", "nat", "nationality"}:
            continue
        elif hl == "total":
            total_col = i
        else:
            comp_cols.append((i, h))

    if name_col is None:
        return []

    out = []
    for row in rows:
        cells = row.find_all(["td", "th"])
        if len(cells) < name_col + 1:
            continue
        name = _clean(cells[name_col].get_text(" ")).rstrip("*").strip()
        if not _looks_like_player_name(name):
            continue

        pos = None
        if pos_col is not None and pos_col < len(cells):
            ptxt = _clean(cells[pos_col].get_text())
            if ptxt in {"GK", "DF", "MF", "FW"}:
                pos = ptxt

        comp_stats = {}
        for ci, comp in comp_cols:
            if ci < len(cells):
                comp_stats[comp] = _try_int(cells[ci].get_text())
        total = sum(comp_stats.values())
        if total_col is not None and total_col < len(cells):
            t = _try_int(cells[total_col].get_text())
            if t > 0:
                total = t
        out.append((name, pos, comp_stats, total))
    return out


def parse_season_page(html: str) -> list[PlayerStat]:
    soup = BeautifulSoup(html, "html.parser")
    players: dict[str, PlayerStat] = {}

    def upsert(name: str, pos: str | None) -> PlayerStat:
        p = players.get(name)
        if p is None:
            p = PlayerStat(name=name, position=pos)
            players[name] = p
        if pos and not p.position:
            p.position = pos
        return p

    for t in soup.find_all("table", class_="wikitable"):
        rows = t.find_all("tr")
        if len(rows) < 2:
            continue
        headers = [_clean(c.get_text()) for c in rows[0].find_all(["th", "td"])]
        if not _is_player_stat_table(headers):
            continue
        second_row = [_clean(c.get_text()) for c in rows[1].find_all(["th", "td"])]

        kind = _classify_table(t, headers, second_row)

        if kind == "combined":
            parsed = _parse_combined_table(t, headers)
            for p in parsed:
                existing = upsert(p.name, p.position)
                # Combined tables provide everything
                existing.appearances = p.appearances
                existing.goals = p.goals
                existing.total_appearances = p.total_appearances
                existing.total_goals = p.total_goals
        elif kind in {"goals", "appearances"}:
            rows_parsed = _parse_single_stat_table(t, headers, kind)
            for name, pos, comp_stats, total in rows_parsed:
                p = upsert(name, pos)
                if kind == "goals":
                    p.goals = comp_stats
                    p.total_goals = total
                else:
                    p.appearances = comp_stats
                    p.total_appearances = total
        # We deliberately skip 'assists', 'discipline', 'clean_sheets',
        # 'unknown' for now.

    return list(players.values())


# ──────────────────────────────────────────────────────────────────────
# Orchestration
# ──────────────────────────────────────────────────────────────────────

def fetch_club_stats(club: str, refresh: bool = False) -> ClubSeason | None:
    cache_path = CACHE_DIR / f"{slugify(club)}.json"
    if not refresh and cache_path.exists():
        data = json.loads(cache_path.read_text())
        return ClubSeason(
            club=data["club"],
            wiki_url=data["wiki_url"],
            season=data.get("season", "2025-26"),
            players=[PlayerStat(**p) for p in data["players"]],
        )

    url = resolve_url(club)
    if url is None:
        return None
    try:
        time.sleep(FETCH_DELAY_S)
        resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
        if resp.status_code != 200:
            return None
        players = parse_season_page(resp.text)
    except Exception as e:
        print(f"    {club}: fetch failed ({e})")
        return None

    cs = ClubSeason(club=club, wiki_url=url, players=players)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps({
        **asdict(cs),
        "players": [asdict(p) for p in cs.players],
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, indent=2, ensure_ascii=False))
    return cs


def collect_clubs_from_rosters() -> list[str]:
    rosters = fetch_rosters(refresh=False)
    clubs = sorted({p.club for r in rosters for p in r.players})
    return clubs


def main() -> None:
    p = argparse.ArgumentParser(description="Scrape 2025-26 club season stats from Wikipedia.")
    p.add_argument("--clubs", help="Comma-separated club names; default = all clubs in current WC rosters")
    p.add_argument("--refresh", action="store_true", help="Force re-fetch even if cached")
    args = p.parse_args()

    if args.clubs:
        clubs = [c.strip() for c in args.clubs.split(",")]
    else:
        clubs = collect_clubs_from_rosters()
        print(f"Found {len(clubs)} unique clubs across published WC rosters")

    n_ok = n_missing = n_empty = 0
    for c in clubs:
        cs = fetch_club_stats(c, refresh=args.refresh)
        if cs is None:
            n_missing += 1
            print(f"  ✗ {c}: no Wikipedia season page found")
        elif not cs.players:
            n_empty += 1
            print(f"  ? {c}: page found but no player tables parsed")
        else:
            n_ok += 1
            top = sorted(cs.players, key=lambda x: -x.total_goals)[:3]
            top_str = ", ".join(f"{p.name} ({p.total_goals}g/{p.total_appearances}a)" for p in top)
            print(f"  ✓ {c}: {len(cs.players):>3} players  | top scorers: {top_str}")

    print(f"\n{n_ok} ok, {n_empty} empty, {n_missing} missing  ({len(clubs)} total)")
    print(f"Cache directory: {CACHE_DIR}")


if __name__ == "__main__":
    sys.exit(main())
