"""Export the data files for the v2 site (Bankroll + 2026 tabs).

Usage: uv run python scripts/export_site_v2.py OUTDIR

Writes OUTDIR/bankroll.json (from ledger/), OUTDIR/capsule.json,
OUTDIR/elo.json (from the DuckDB elo_current table; skipped if the DB is
missing), OUTDIR/tournaments.json (from data/forecast_<key>.json, written by
`worldcup-forecast <key>`) and OUTDIR/odds.json (copied from data/title_odds.json, written
weekly by `python -m worldcup.title_odds`). With no bets yet, bankroll.json is an untouched $1,000 with
`source: not-started`. The synthetic backtest stays internal
(`uv run worldcup-betting-backtest`).
"""

import json
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from worldcup.betting import engine
from worldcup.betting.ledger import Ledger
from worldcup.registry import REGISTRY

ROOT = Path(__file__).resolve().parents[1]
SCORECARD = ROOT / "data" / "capsule" / "scorecard_2026.json"
FROZEN_SIM = ROOT / "data" / "simulation-prediction.json"
TITLE_ODDS = ROOT / "data" / "title_odds.json"
FIXTURES = ROOT / "data" / "fixtures.json"

N_RECENT = 30

RULE, MIN_EDGE = "flat", engine.DEFAULT_MIN_EDGE
STAGE_SHORT = {"champion": "Won", "final": "Final", "semi_finals": "SF", "quarter_finals": "QF",
               "round_of_16": "R16", "round_of_32": "R32", "group": "Group"}


def _game(row: dict) -> dict:
    """One priced game, bet or pass: model vs Sportsbet 1X2, all as fractions."""
    odds, p = row.get("odds_1x2"), row.get("model_p_1x2")
    g = {"kickoff": row["kickoff"][:10], "home": row["home"], "away": row["away"],
         "model_p": [round(x, 4) for x in p] if p else None, "odds": odds,
         "market_p": [round(x, 4) for x in engine.devig_proportional(odds)[0]] if odds else None}
    if row["type"] == "bet":
        g.update(decision="bet", pick=row["selection"], reason=None)
    else:
        g.update(decision="pass", pick=None, reason=row["reason"])
    return g


def games(led: Ledger, now: datetime) -> dict:
    """Upcoming = every game the bot has decided on that has not kicked off.
    The bot only prices games inside 48h, so nothing further out exists."""
    rows = led.bets() + led.passes()
    future = sorted((r for r in rows if r["kickoff"] > now.isoformat()), key=lambda r: r["kickoff"])
    passes = sorted(led.passes(), key=lambda r: r["decided_at"])[-10:][::-1]
    return {"upcoming": [_game(r) for r in future], "recent_passes": [_game(r) for r in passes]}


def fixtures(led: Ledger, now: datetime, path: Path = FIXTURES, elo_by_team: dict | None = None) -> dict | None:
    """Every game the last bet run saw, not yet kicked off, with the bot's decision if it made one.
    Sportsbet odds appear only once Sportsbet prices the game; model_p and home/away_elo are null for unknown teams."""
    from worldcup.betting.bot import FINALS_KEYS, canonical

    elo_by_team = elo_by_team or {}
    rate = lambda t: round(elo_by_team[canonical(t)]) if canonical(t) in elo_by_team else None
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    decided = {r["event_id"]: r for r in led.bets() + led.passes()}
    games = []
    for f in data["games"]:
        if f["kickoff"] <= now.isoformat():
            continue
        d = decided.get(f["event_id"])
        odds, p = f["odds"], f["model_p"]
        games.append({"kickoff": f["kickoff"][:10], "home": f["home"], "away": f["away"],
                      "neutral": f["sport"] in FINALS_KEYS,
                      "home_elo": rate(f["home"]), "away_elo": rate(f["away"]),
                      "model_p": [round(x, 4) for x in p] if p else None, "odds": odds,
                      "market_p": [round(x, 4) for x in engine.devig_proportional(odds)[0]] if odds else None,
                      "decision": d["type"] if d else None,
                      "pick": d.get("selection") if d and d["type"] == "bet" else None})
    return {"generated_at": now.date().isoformat(), "fetched_at": data["fetched_at"], "games": games}


def window_start(dates: list[date], gap_days: int = 10) -> date:
    """First day of the latest run of matches: walk back from the newest date
    until two consecutive match days are `gap_days` or more apart."""
    ds = sorted(set(dates))
    i = len(ds) - 1
    while i > 0 and (ds[i] - ds[i - 1]).days < gap_days:
        i -= 1
    return ds[i]


def recent_matches(rows: list[tuple], current: dict[str, float], n: int = N_RECENT) -> list[dict]:
    """Last `n` of `rows` (date, tournament, home, away, home pre, away pre, home goals, away goals; oldest first).
    A match's rating change is the team's next pre-match rating, or its current one, minus its pre-match rating."""
    after, out = dict(current), []
    for d, comp, home, away, home_pre, away_pre, hg, ag in reversed(rows):
        out.append({"date": d.date().isoformat(), "competition": comp, "home": home, "away": away, "score": [hg, ag],
                    "home_delta": round(after[home] - home_pre, 1), "away_delta": round(after[away] - away_pre, 1)})
        after[home], after[away] = home_pre, away_pre
    return out[:n]


def elo(active_years: int = 4) -> dict | None:
    """Every team that played in the last `active_years` by current Elo, each with
    its rank and rating from before the latest international window, plus the latest matches."""
    from worldcup.db import DB_PATH, connect

    if not Path(DB_PATH).exists():
        return None
    with connect(read_only=True) as con:
        as_of = con.execute("SELECT max(date) FROM matches").fetchone()[0]
        since = as_of - timedelta(days=365 * active_years)
        active = con.execute(
            "SELECT team, rating, last_match FROM elo_current WHERE last_match >= ? ORDER BY rating DESC",
            [since]).fetchall()
        start = window_start([d for (d,) in con.execute(
            "SELECT DISTINCT date FROM matches WHERE date >= ?", [as_of - timedelta(days=120)]).fetchall()])
        # A team's pre-window rating is its rating going into its first match of the window.
        pre = dict(con.execute("""
            SELECT team, arg_min(r, match_date) FROM (
              SELECT home_team AS team, rating_home_pre AS r, match_date FROM predictions WHERE match_date >= ?
              UNION ALL
              SELECT away_team, rating_away_pre, match_date FROM predictions WHERE match_date >= ?)
            GROUP BY team""", [start, start]).fetchall())
        rows = con.execute("""
            SELECT match_date, tournament, home_team, away_team, rating_home_pre, rating_away_pre,
                   actual_home_goals, actual_away_goals
            FROM predictions WHERE match_date >= ? ORDER BY match_date, home_team""", [start]).fetchall()
        current = dict(con.execute("SELECT team, rating FROM elo_current").fetchall())
    prev = sorted(((pre.get(t, r), t) for t, r, _ in active), reverse=True)
    prev_rank = {t: i + 1 for i, (_, t) in enumerate(prev)}
    return {"generated_at": date.today().isoformat(), "as_of": as_of.isoformat(), "window_start": start.isoformat(),
            "top": [{"rank": i + 1, "team": t, "rating": round(r), "last_match": lm.date().isoformat(),
                     "prev_rank": prev_rank[t], "prev_rating": round(pre.get(t, r))}
                    for i, (t, r, lm) in enumerate(active)],
            "recent": recent_matches(rows, current)}


def ratings() -> dict | None:
    """Current Elo for every team, keyed by name, for fixtures.json."""
    from worldcup.db import DB_PATH, connect

    if not Path(DB_PATH).exists():
        return None
    with connect(read_only=True) as con:
        return dict(con.execute("SELECT team, rating FROM elo_current").fetchall())


def bankroll(ledger_dir: Path = ROOT / "ledger", now: datetime | None = None) -> dict:
    """Live numbers from the hash-chained ledger. Until the first bet the page
    shows an untouched $1,000; synthetic backtests never reach the page."""
    led = Ledger(ledger_dir)
    start, bets, sets = led.start_bankroll, led.bets(), led.settlements()
    base = {
        "generated_at": date.today().isoformat(),
        "rule": {"staking": RULE, "stake": engine.FLAT_UNIT, "min_edge": MIN_EDGE, "max_odds": engine.MAX_ODDS},
        "start": start,
        **games(led, now or datetime.now(UTC)),
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


def tournaments(data_dir: Path = ROOT / "data") -> dict | None:
    """Title odds for upcoming tournaments, skipping any with no forecast file yet."""
    files = [data_dir / f"forecast_{k}.json" for k, t in REGISTRY.items() if t["draw"] == "simulated"]
    found = [json.loads(f.read_text()) for f in files if f.exists()]
    return {"generated_at": date.today().isoformat(), "tournaments": found} if found else None


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
        "accuracy_plain_elo": round(allm["plain_elo"]["accuracy"], 4),
        "top": [{"team": t["team"], "p_champion": t["p_champion"],
                 "stage": STAGE_SHORT.get(t["stage_reached"], t["stage_reached"])} for t in teams],
    }


def main() -> None:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    odds = json.loads(TITLE_ODDS.read_text()) if TITLE_ODDS.exists() else None
    fx = fixtures(Ledger(ROOT / "ledger"), datetime.now(UTC), elo_by_team=ratings())
    for name, data in (("bankroll", bankroll()), ("capsule", capsule()), ("elo", elo()), ("odds", odds),
                       ("fixtures", fx), ("tournaments", tournaments())):
        if data is None:
            print(f"skip {name}.json (no source data)")
            continue
        (out / f"{name}.json").write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")
        print(f"wrote {out / name}.json")


if __name__ == "__main__":
    main()
