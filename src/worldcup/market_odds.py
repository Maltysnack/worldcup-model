"""Fetch WC 2026 outright champion market probabilities from Polymarket.

Polymarket is a prediction market: traders buy/sell shares that pay $1
if their team wins. The current share price equals the implied
probability the team wins. Compared to traditional bookmakers, this
has three advantages for our overlay:

  1. Free public API — no auth, no key
  2. Real-money traders -> prices are calibrated by arbitrage
  3. Probabilities are direct (no decimal-odds-to-probability math)

Source: https://gamma-api.polymarket.com/events?slug=2026-fifa-world-cup-winner-595

Mild de-vig step: with 48 active markets the YES prices sum to ~1.04,
so we divide each by the sum to renormalise. The market has Spain
around 16%, France around 18%, England around 12% as of 2026-05-13.

Output: data/raw/market_odds.json with per-team de-vigged probabilities
and timestamp. The simulator splices this into team_stats so the site
can render a "Market %" column alongside "Model %".

Run: `uv run worldcup-fetch-market-odds`
     `uv run worldcup-fetch-market-odds --refresh`  (ignore cache)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone

import requests

from worldcup.db import PROJECT_ROOT

POLYMARKET_URL = (
    "https://gamma-api.polymarket.com/events?slug=2026-fifa-world-cup-winner-595"
)
USER_AGENT = "maltysnack-worldcup-hobby/0.1 (https://github.com/Maltysnack/worldcup)"
OUTPUT_PATH = PROJECT_ROOT / "data" / "raw" / "market_odds.json"

# Map Polymarket's team naming to our internal DB names. Add as needed.
TEAM_NAME_MAP = {
    "USA": "United States",
    "Turkiye": "Turkey",
    "Czechia": "Czech Republic",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Congo DR": "DR Congo",
    "South Korea": "South Korea",
    "Cape Verde": "Cape Verde",
    "Saudi Arabia": "Saudi Arabia",
    "South Africa": "South Africa",
    "Ivory Coast": "Ivory Coast",
    "New Zealand": "New Zealand",
    "Curaçao": "Curaçao",
}


def fetch_raw_markets() -> list[dict]:
    resp = requests.get(POLYMARKET_URL, headers={"User-Agent": USER_AGENT}, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    if not data:
        raise RuntimeError("Polymarket returned no events for slug")
    return data[0].get("markets", [])


def parse_markets(markets: list[dict]) -> list[tuple[str, float]]:
    """Return [(team_name_internal, raw_yes_price), ...] for ACTIVE
    markets (closed/resolved markets are filtered out)."""
    out = []
    for m in markets:
        if m.get("closed") or not m.get("active"):
            continue
        op = m.get("outcomePrices")
        if not op:
            continue
        if isinstance(op, str):
            try:
                op = json.loads(op)
            except json.JSONDecodeError:
                continue
        try:
            yes_price = float(op[0])
        except (IndexError, TypeError, ValueError):
            continue
        q = m.get("question", "")
        match = re.match(r"^Will (.+) win the 2026 FIFA World Cup", q)
        if not match:
            continue
        team_raw = match.group(1).strip()
        if team_raw.startswith("Team ") or team_raw == "Any Other Team":
            continue
        team = TEAM_NAME_MAP.get(team_raw, team_raw)
        out.append((team, yes_price))
    return out


def devig(raw: list[tuple[str, float]]) -> dict[str, float]:
    """Renormalise so the per-team probabilities sum to 1.0 (removes the
    market's overround). Returns {team: probability}."""
    total = sum(p for _, p in raw) or 1.0
    return {team: p / total for team, p in raw}


def fetch_and_save() -> dict:
    markets = fetch_raw_markets()
    raw = parse_markets(markets)
    if not raw:
        raise RuntimeError("No active markets parsed from Polymarket response")
    raw_total = sum(p for _, p in raw)
    devigged = devig(raw)

    out = {
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": "Polymarket gamma-api",
        "source_url": POLYMARKET_URL,
        "n_teams": len(raw),
        "raw_price_sum": round(raw_total, 4),
        "vig_pct": round((raw_total - 1.0) * 100, 2),
        "market_probabilities": [
            {
                "team": team,
                "raw_price": round(raw_p, 4),
                "implied_probability": round(devigged[team], 4),
            }
            for team, raw_p in sorted(raw, key=lambda x: -x[1])
        ],
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    return out


def load_cached() -> dict[str, float]:
    """Return {team: implied_probability} from the latest fetch, or
    empty dict if no cache exists."""
    if not OUTPUT_PATH.exists():
        return {}
    data = json.loads(OUTPUT_PATH.read_text())
    return {row["team"]: row["implied_probability"]
            for row in data.get("market_probabilities", [])}


def main() -> None:
    p = argparse.ArgumentParser(description="Fetch WC 2026 market champion odds.")
    p.add_argument("--refresh", action="store_true",
                   help="Force a fresh fetch even if cache exists")
    args = p.parse_args()

    if OUTPUT_PATH.exists() and not args.refresh:
        print(f"Cache exists at {OUTPUT_PATH}, use --refresh to re-fetch")
        cached = load_cached()
        print(f"\nCached market (top 12):")
        for team, p in sorted(cached.items(), key=lambda kv: -kv[1])[:12]:
            print(f"  {team:30s} {p*100:>5.2f}%")
        return

    print(f"Fetching from {POLYMARKET_URL}\n")
    out = fetch_and_save()
    print(f"Parsed {out['n_teams']} active markets, raw vig {out['vig_pct']}%")
    print(f"\nMarket champion probabilities (de-vigged):")
    for row in out["market_probabilities"][:15]:
        print(f"  {row['team']:30s} raw={row['raw_price']*100:>5.2f}%  "
              f"de-vigged={row['implied_probability']*100:>5.2f}%")
    print(f"\nWrote {OUTPUT_PATH}")


if __name__ == "__main__":
    sys.exit(main())
