"""Paper-money betting bot: Sportsbet 1X2 via The Odds API.

Three tasks, each run by `.github/workflows/bot.yml`:

  bet     pull Sportsbet h2h odds for every active intl key; for each event
          kicking off within 48h that has no decision yet, bet (or record a
          pass). One decision per event, made at the first pull inside 48h.
  close   list events via the free events endpoint; for open bets kicking off
          within 90 min with no closing capture, pull odds for just those
          event ids and record the closing price. Exits early (0 credits)
          when nothing is due.
  settle  match open bets to results in the DB (run worldcup-ingest and
          worldcup-elo first) and settle them with the closing price.

Model: plain Elo (`elo_current`, v4a-flat-mild) through the Poisson goals
model, as in the backtest. Finals keys are treated as neutral venues,
qualifiers and Nations League as home/away. Staking defaults to the
provisional backtest rec (flat $10, min edge 0.03) until the USER locks it.

Results caveat: martj42 knockout scores include extra time, so a 90-minute
draw decided in ET would settle as a win. Matches in the shootouts table
settle as draws. Most live markets here are qualifiers (no ET).

Credit guard: stops pulling when `x-requests-remaining` drops below
`--min-credits` (default 50).

Run: uv run worldcup-bot {bet,close,settle} [--ledger ledger] [--dry]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

from worldcup.betting import engine
from worldcup.betting.backtest import outcome_idx, probs_from_ratings
from worldcup.betting.ledger import Ledger, _parse_ts, now_utc

API = "https://api.the-odds-api.com/v4"
BOOKMAKER = "sportsbet"
REGION = "au"
MARKET = "h2h"
BET_WINDOW = timedelta(hours=48)
CLOSE_WINDOW = timedelta(minutes=90)
SETTLE_AFTER = timedelta(hours=3)
VOID_AFTER = timedelta(days=14)
MIN_CREDITS = 50

FINALS_KEYS = {
    "soccer_fifa_world_cup",
    "soccer_uefa_european_championship",
    "soccer_conmebol_copa_america",
    "soccer_concacaf_gold_cup",
    "soccer_africa_cup_of_nations",
}
INTL_KEYS = FINALS_KEYS | {
    "soccer_fifa_world_cup_qualifiers_europe",
    "soccer_fifa_world_cup_qualifiers_south_america",
    "soccer_uefa_nations_league",
    "soccer_uefa_euro_qualification",
}

# Odds API name -> martj42 canonical name. Unconfirmed guesses until the
# item 3 spike shows real feed names; unmapped teams missing from
# elo_current are passed with reason "unknown team".
ALIASES = {
    "USA": "United States",
    "Czechia": "Czech Republic",
    "Türkiye": "Turkey",
    "Turkiye": "Turkey",
    "Korea Republic": "South Korea",
    "Côte d'Ivoire": "Ivory Coast",
    "Cote d'Ivoire": "Ivory Coast",
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
    "Congo DR": "DR Congo",
    "Cape Verde Islands": "Cape Verde",
    "IR Iran": "Iran",
    "China PR": "China",
    "Ireland": "Republic of Ireland",
    "Macedonia": "North Macedonia",
}

Fetch = Callable[[str, dict], tuple[bytes, dict]]


def canonical(name: str) -> str:
    return ALIASES.get(name, name)


class OddsClient:
    def __init__(self, api_key: str, fetch: Fetch | None = None, min_credits: int = MIN_CREDITS):
        self.api_key = api_key
        self.fetch = fetch or _http_get
        self.min_credits = min_credits
        self.remaining: int | None = None
        self.used_this_run = 0

    def get(self, path: str, **params) -> tuple[bytes, list | dict]:
        if self.remaining is not None and self.remaining < self.min_credits:
            raise RuntimeError(f"credit guard: {self.remaining} remaining < {self.min_credits}")
        raw, headers = self.fetch(f"{API}{path}", {"apiKey": self.api_key, **params})
        h = {k.lower(): v for k, v in headers.items()}
        if "x-requests-remaining" in h:
            self.remaining = int(float(h["x-requests-remaining"]))
        self.used_this_run += int(float(h.get("x-requests-last", 0)))
        return raw, json.loads(raw)

    def sports(self) -> list[dict]:
        return self.get("/sports")[1]

    def events(self, sport: str) -> list[dict]:
        return self.get(f"/sports/{sport}/events")[1]

    def odds(self, sport: str, event_ids: list[str] | None = None) -> tuple[bytes, list[dict]]:
        params = {"regions": REGION, "markets": MARKET, "bookmakers": BOOKMAKER, "oddsFormat": "decimal"}
        if event_ids:
            params["eventIds"] = ",".join(event_ids)
        return self.get(f"/sports/{sport}/odds", **params)


def _http_get(url: str, params: dict) -> tuple[bytes, dict]:
    with urllib.request.urlopen(f"{url}?{urllib.parse.urlencode(params)}", timeout=30) as r:
        return r.read(), dict(r.headers)


def sportsbet_1x2(event: dict) -> list[float] | None:
    """(home, draw, away) decimal odds from Sportsbet's h2h market, or None."""
    for bm in event.get("bookmakers", []):
        if bm.get("key") != BOOKMAKER:
            continue
        for m in bm.get("markets", []):
            if m.get("key") != MARKET:
                continue
            prices = {o["name"]: float(o["price"]) for o in m.get("outcomes", [])}
            try:
                return [prices[event["home_team"]], prices["Draw"], prices[event["away_team"]]]
            except KeyError:
                return None
    return None


def load_ratings() -> dict[str, float]:
    from worldcup.db import connect

    with connect(read_only=True) as con:
        return dict(con.execute("SELECT team, rating FROM elo_current").fetchall())


def active_keys(client: OddsClient) -> list[str]:
    return sorted(s["key"] for s in client.sports() if s.get("active") and s["key"] in INTL_KEYS)


def run_bet(
    led: Ledger, client: OddsClient, ratings: dict[str, float], *, rule: str = "flat",
    min_edge: float = engine.DEFAULT_MIN_EDGE, now: datetime | None = None, dry: bool = False,
) -> list[str]:
    now = now or datetime.now(timezone.utc)
    log = []
    for sport in active_keys(client):
        decided = led.decided_events()
        raw, events = client.odds(sport)
        due = [e for e in events
               if e["id"] not in decided and now < _parse_ts(e["commence_time"]) <= now + BET_WINDOW]
        if not due:
            log.append(f"{sport}: {len(events)} events, none due")
            continue
        snap = None if dry else led.save_snapshot(raw, f"odds-{sport}")
        for e in due:
            log.append(_decide(led, e, sport, ratings, rule, min_edge, now, snap, dry))
    log.append(f"credits used this run: {client.used_this_run}, remaining: {client.remaining}")
    return log


def _decide(led, e, sport, ratings, rule, min_edge, now, snap, dry) -> str:
    home, away = canonical(e["home_team"]), canonical(e["away_team"])
    base = dict(event_id=e["id"], kickoff=e["commence_time"], home=home, away=away)
    tag = f"{home} v {away} ({e['commence_time']})"
    odds = sportsbet_1x2(e)
    if odds is None:
        return f"skip {tag}: no Sportsbet 1X2 yet"
    missing = [t for t in (home, away) if t not in ratings]
    if missing:
        if not dry:
            led.record_pass(**base, reason=f"unknown team {missing}", odds=odds, snapshot=snap)
        return f"pass {tag}: unknown team {missing}"
    p = probs_from_ratings(ratings[home], ratings[away], neutral=sport in FINALS_KEYS)
    sel = engine.pick_selection(p, odds, min_edge)
    if sel is None:
        if not dry:
            led.record_pass(**base, reason="no edge", odds=odds, model_p=p, snapshot=snap)
        return f"pass {tag}: no edge"
    stake = engine.stake_for(rule, p[sel], odds[sel], led.bankroll())
    if stake < 0.01:
        if not dry:
            led.record_pass(**base, reason="zero stake", odds=odds, model_p=p, snapshot=snap)
        return f"pass {tag}: zero stake"
    if not dry:
        led.place_bet(**base, selection=sel, odds=odds, stake=stake, model_p=p,
                      placed_at=now.isoformat(timespec="microseconds"), snapshot=snap,
                      extra={"sport_key": sport, "rule": rule, "min_edge": min_edge,
                             "feed_home": e["home_team"], "feed_away": e["away_team"]})
    return f"BET {tag}: {['home', 'draw', 'away'][sel]} @ {odds[sel]} ${stake:.2f} edge {engine.edge(p[sel], odds[sel]):+.3f}"


def run_close(led: Ledger, client: OddsClient, *, now: datetime | None = None, dry: bool = False) -> list[str]:
    now = now or datetime.now(timezone.utc)
    captured = {c["event_id"] for c in led.closing()}
    due = [b for b in led.open_bets()
           if b["event_id"] not in captured and now < _parse_ts(b["kickoff"]) <= now + CLOSE_WINDOW]
    if not due:
        return ["close: no open bets kicking off within 90 min"]
    log = []
    by_sport: dict[str, list[dict]] = {}
    for b in due:
        by_sport.setdefault(b.get("extra", {}).get("sport_key", ""), []).append(b)
    for sport, bets in by_sport.items():
        ids = [b["event_id"] for b in bets]
        live = {e["id"]: e for e in client.events(sport)}  # free endpoint
        ids = [i for i in ids if i in live]
        if not ids:
            log.append(f"{sport}: due events no longer listed")
            continue
        raw, events = client.odds(sport, ids)
        snap = None if dry else led.save_snapshot(raw, f"close-{sport}")
        for e in events:
            odds = sportsbet_1x2(e)
            if odds is None:
                log.append(f"{e['id']}: no Sportsbet price at close")
                continue
            if not dry:
                led.record_closing(event_id=e["id"], kickoff=e["commence_time"], odds=odds,
                                   captured_at=now.isoformat(timespec="microseconds"), snapshot=snap)
            log.append(f"closing {e['home_team']} v {e['away_team']}: {odds}")
    log.append(f"credits used this run: {client.used_this_run}, remaining: {client.remaining}")
    return log


def find_result(con, bet: dict) -> int | None:
    """Outcome index from the home side's view of the bet, or None if not found."""
    ko = _parse_ts(bet["kickoff"]).date()
    lo, hi = ko - timedelta(days=1), ko + timedelta(days=1)
    rows = con.execute(
        """
        SELECT m.home_team, m.home_score, m.away_score,
               EXISTS (SELECT 1 FROM shootouts s WHERE s.date = m.date
                       AND s.home_team = m.home_team AND s.away_team = m.away_team) AS pens
        FROM matches m
        WHERE m.date BETWEEN ? AND ? AND m.home_score IS NOT NULL
          AND ((m.home_team = ? AND m.away_team = ?) OR (m.home_team = ? AND m.away_team = ?))
        ORDER BY m.date
        """,
        [lo, hi, bet["home"], bet["away"], bet["away"], bet["home"]],
    ).fetchall()
    if not rows:
        return None
    home, hg, ag, pens = rows[0]
    out = 1 if pens else outcome_idx(int(hg), int(ag))
    return out if home == bet["home"] or out == 1 else 2 - out


def run_settle(led: Ledger, con, *, now: datetime | None = None, dry: bool = False) -> list[str]:
    now = now or datetime.now(timezone.utc)
    log = []
    for b in led.open_bets():
        ko = _parse_ts(b["kickoff"])
        if now < ko + SETTLE_AFTER:
            continue
        tag = f"{b['home']} v {b['away']} ({b['kickoff']})"
        out = find_result(con, b)
        if out is None and now < ko + VOID_AFTER:
            log.append(f"waiting {tag}: no result yet")
            continue
        close = led.closing_odds(b["event_id"])
        if not dry:
            led.settle(b["hash"], out, closing_odds=close["odds_1x2"] if close else None,
                       settled_at=now.isoformat(timespec="microseconds"),
                       snapshot=close.get("snapshot") if close else None)
        log.append(f"settled {tag}: {'void' if out is None else ['home', 'draw', 'away'][out]}")
    return log or ["settle: nothing to settle"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Paper-money Sportsbet 1X2 bot.")
    ap.add_argument("task", choices=["bet", "close", "settle"])
    ap.add_argument("--ledger", default="ledger")
    ap.add_argument("--rule", choices=["flat", "kelly"], default="flat")
    ap.add_argument("--min-edge", type=float, default=engine.DEFAULT_MIN_EDGE)
    ap.add_argument("--min-credits", type=int, default=MIN_CREDITS)
    ap.add_argument("--dry", action="store_true", help="price and log, write nothing")
    args = ap.parse_args(argv)
    led = Ledger(args.ledger)

    if args.task == "settle":
        from worldcup.db import connect

        with connect(read_only=True) as con:
            log = run_settle(led, con, dry=args.dry)
    else:
        key = os.environ.get("ODDS_API_KEY")
        if not key:
            print("ODDS_API_KEY not set", file=sys.stderr)
            return 2
        client = OddsClient(key, min_credits=args.min_credits)
        if args.task == "bet":
            log = run_bet(led, client, load_ratings(), rule=args.rule, min_edge=args.min_edge, dry=args.dry)
        else:
            log = run_close(led, client, dry=args.dry)
    print(f"[{now_utc()}] {args.task}")
    print("\n".join(log))
    return 0


if __name__ == "__main__":
    sys.exit(main())
