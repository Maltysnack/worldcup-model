"""Tests for the betting bot against a fake Odds API and an in-memory DB.

Run: uv run python -m unittest discover tests
"""

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import date, datetime, timedelta, timezone

import duckdb

from worldcup.betting import bot
from worldcup.betting.ledger import Ledger, verify

NOW = datetime(2026, 11, 12, 9, 0, tzinfo=timezone.utc)
SPORT = "soccer_fifa_world_cup_qualifiers_europe"
RATINGS = {"Spain": 2176.0, "Georgia": 1700.0, "Turkey": 1790.0, "Italy": 1898.0, "Wales": 1750.0}


def _event(eid, home, away, hours, prices=None):
    ko = (NOW + timedelta(hours=hours)).isoformat().replace("+00:00", "Z")
    e = {"id": eid, "sport_key": SPORT, "commence_time": ko, "home_team": home, "away_team": away,
         "bookmakers": []}
    if prices:
        e["bookmakers"].append({"key": "sportsbet", "markets": [{"key": "h2h", "outcomes": [
            {"name": home, "price": prices[0]}, {"name": "Draw", "price": prices[1]},
            {"name": away, "price": prices[2]}]}]})
    return e


class FakeApi:
    def __init__(self, events):
        self.events = events
        self.calls = []
        self.remaining = 500

    def __call__(self, url, params):
        path = url.removeprefix(bot.API)
        self.calls.append((path, params))
        cost = 0
        if path == "/sports":
            body = [{"key": SPORT, "active": True}, {"key": "soccer_epl", "active": True},
                    {"key": "soccer_fifa_world_cup", "active": False}]
        elif path.endswith("/events"):
            body = [{k: v for k, v in e.items() if k != "bookmakers"} for e in self.events]
        else:
            ids = params.get("eventIds")
            body = [e for e in self.events if not ids or e["id"] in ids.split(",")]
            cost = 1
        self.remaining -= cost
        return json.dumps(body).encode(), {"x-requests-remaining": str(self.remaining), "x-requests-last": str(cost)}


class BotTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.led = Ledger(self.tmp.name)
        self.api = FakeApi([
            _event("a", "Spain", "Georgia", 30, [1.60, 4.0, 6.0]),    # Spain huge edge at 1.60
            _event("b", "Wales", "Italy", 20, [3.4, 3.3, 2.2]),       # roughly fair
            _event("c", "Türkiye", "Italy", 60, [2.9, 3.2, 2.5]),     # outside 48h
            _event("d", "Spain", "Narnia", 10, [1.1, 9.0, 21.0]),     # unknown team
            _event("e", "Spain", "Italy", 5),                         # no Sportsbet price
        ])
        self.client = bot.OddsClient("k", fetch=self.api)

    def tearDown(self):
        self.tmp.cleanup()

    def test_bet_decides_once_per_event(self):
        bot.run_bet(self.led, self.client, RATINGS, now=NOW)
        bets = {b["event_id"]: b for b in self.led.bets()}
        passes = {p["event_id"]: p for p in self.led.passes()}
        self.assertEqual(bets["a"]["selection"], "home")
        self.assertEqual(bets["a"]["stake"], 10.0)
        self.assertIn("unknown team", passes["d"]["reason"])
        self.assertNotIn("c", bets.keys() | passes.keys())
        self.assertNotIn("e", bets.keys() | passes.keys())
        self.assertEqual(len(bets) + len(passes), 3)
        # Only the one active intl key is pulled; epl and inactive keys skipped.
        self.assertEqual([c[0] for c in self.api.calls], ["/sports", f"/sports/{SPORT}/odds", f"/sports/{SPORT}/events"])
        self.assertEqual(self.client.used_this_run, 1)
        n = len(self.led.bets()) + len(self.led.passes())
        bot.run_bet(self.led, self.client, RATINGS, now=NOW + timedelta(hours=1))
        self.assertEqual(len(self.led.bets()) + len(self.led.passes()), n)
        self.assertTrue(verify(self.tmp.name).ok, verify(self.tmp.name).errors)

    def test_alias(self):
        self.assertEqual(bot.canonical("Türkiye"), "Turkey")
        self.assertEqual(bot.canonical("Spain"), "Spain")

    def test_dry_writes_nothing(self):
        bot.run_bet(self.led, self.client, RATINGS, now=NOW, dry=True)
        self.assertEqual(self.led.bets() + self.led.passes(), [])

    def test_close_is_free_when_nothing_due(self):
        bot.run_bet(self.led, self.client, RATINGS, now=NOW)
        self.api.calls.clear()
        bot.run_close(self.led, self.client, now=NOW)  # bet "a" kicks off in 30h
        self.assertEqual(self.api.calls, [])
        log = bot.run_close(self.led, self.client, now=NOW + timedelta(hours=29))
        self.assertEqual([c[0] for c in self.api.calls], [f"/sports/{SPORT}/events", f"/sports/{SPORT}/odds"])
        self.assertEqual(self.api.calls[-1][1]["eventIds"], "a")
        self.assertEqual(self.led.closing_odds("a")["odds_1x2"], [1.6, 4.0, 6.0])
        self.assertEqual(self.led.closing_odds("a")["minutes_before"], 60)
        self.api.calls.clear()
        bot.run_close(self.led, self.client, now=NOW + timedelta(hours=29, minutes=30))
        self.assertEqual(self.api.calls, [], log)

    def test_close_due_gate_matches_run_close(self):
        # bot.yml skips the close job when scripts/close_due.py says nothing is due.
        import importlib.util, pathlib
        p = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "close_due.py"
        spec = importlib.util.spec_from_file_location("close_due", p)
        cd = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cd)
        led_dir = pathlib.Path(self.tmp.name)
        self.assertFalse(cd.due(led_dir, NOW))  # empty ledger
        bot.run_bet(self.led, self.client, RATINGS, now=NOW)  # bet "a" kicks off in 30h
        self.assertFalse(cd.due(led_dir, NOW))
        self.assertFalse(cd.due(led_dir, NOW + timedelta(hours=26, minutes=50)))  # 3h10m out
        self.assertTrue(cd.due(led_dir, NOW + timedelta(hours=27, minutes=5)))  # 2h55m out
        self.assertEqual(bot.run_close(self.led, self.client, now=NOW + timedelta(hours=26, minutes=50))[0],
                         "close: no open bets kicking off within 3 h")
        self.assertTrue(cd.due(led_dir, NOW + timedelta(hours=29)))
        bot.run_close(self.led, self.client, now=NOW + timedelta(hours=29))
        self.assertFalse(cd.due(led_dir, NOW + timedelta(hours=29, minutes=30)))

    def test_export_games(self):
        # scripts/export_site_v2.py: upcoming = decided games not yet kicked off.
        import importlib.util, pathlib
        p = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "export_site_v2.py"
        spec = importlib.util.spec_from_file_location("export_site_v2", p)
        ex = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ex)
        bot.run_bet(self.led, self.client, RATINGS, now=NOW)
        g = ex.games(self.led, NOW)
        up = {(x["home"], x["decision"]) for x in g["upcoming"]}
        self.assertIn(("Spain", "bet"), up)
        self.assertEqual(len(g["upcoming"]), 3)
        self.assertEqual(len(g["recent_passes"]), 2)
        spain = next(x for x in g["upcoming"] if x["decision"] == "bet")
        self.assertEqual(spain["pick"], "home")
        self.assertAlmostEqual(sum(spain["market_p"]), 1.0, places=3)
        self.assertAlmostEqual(sum(spain["model_p"]), 1.0, places=3)
        unknown = next(x for x in g["recent_passes"] if x["reason"].startswith("unknown"))
        self.assertIsNone(unknown["model_p"])
        self.assertEqual(ex.games(self.led, NOW + timedelta(hours=31))["upcoming"], [])

    def test_fixtures_cover_every_listed_game(self):
        import importlib.util, pathlib
        p = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "export_site_v2.py"
        spec = importlib.util.spec_from_file_location("export_site_v2", p)
        ex = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ex)
        path = pathlib.Path(self.tmp.name) / "fixtures.json"
        bot.run_bet(self.led, self.client, RATINGS, now=NOW, fixtures_path=path)
        self.assertEqual(self.client.used_this_run, 1)  # no extra credits
        g = {x["home"] + x["away"]: x for x in ex.fixtures(self.led, NOW, path, elo_by_team=RATINGS)["games"]}
        self.assertEqual(len(g), 5)  # includes "c", outside the 48h bet window
        self.assertIsNone(g["TurkeyItaly"]["decision"])
        self.assertAlmostEqual(sum(g["TurkeyItaly"]["model_p"]), 1.0, places=3)
        self.assertEqual(g["SpainGeorgia"]["pick"], "home")
        self.assertIsNone(g["SpainItaly"]["odds"])
        self.assertIsNone(g["SpainNarnia"]["model_p"])
        self.assertEqual(g["TurkeyItaly"]["home_elo"], round(RATINGS["Turkey"]))
        self.assertIsNone(g["SpainNarnia"]["away_elo"])
        # Elo window: the latest run of match days after a 10+ day gap.
        d = lambda m, x: date(2026, m, x)
        self.assertEqual(ex.window_start([d(8, 26), d(9, 18), d(9, 24), d(9, 30), d(10, 6)]), d(9, 18))
        self.assertEqual(ex.window_start([d(10, 6)]), d(10, 6))
        self.assertEqual(len(ex.fixtures(self.led, NOW + timedelta(hours=61), path)["games"]), 0)
        bot.run_bet(self.led, self.client, RATINGS, now=NOW, dry=True, fixtures_path=path / "x")
        self.assertFalse((path / "x").exists())
        # Free /events adds games the odds pull left out; fixtures task decides nothing.
        self.api.events.append(_event("f", "Wales", "Spain", 24 * 30))
        n = len(self.led.bets()) + len(self.led.passes())
        bot.run_fixtures(self.client, RATINGS, path, now=NOW)
        self.assertEqual(len(ex.fixtures(self.led, NOW, path)["games"]), 6)
        self.assertEqual(len(self.led.bets()) + len(self.led.passes()), n)

    def test_settle_with_closing_and_reversed_fixture(self):
        bot.run_bet(self.led, self.client, RATINGS, now=NOW)
        bot.run_close(self.led, self.client, now=NOW + timedelta(hours=29))
        con = duckdb.connect()
        con.execute("CREATE TABLE matches (date DATE, home_team VARCHAR, away_team VARCHAR, home_score BIGINT, away_score BIGINT)")
        con.execute("CREATE TABLE shootouts (date DATE, home_team VARCHAR, away_team VARCHAR)")
        # Feed listed Spain at home; DB stores the fixture reversed, Georgia 0-2 Spain.
        con.execute("INSERT INTO matches VALUES ('2026-11-13', 'Georgia', 'Spain', 0, 2)")
        bot.run_settle(self.led, con, now=NOW + timedelta(hours=40))
        s = {x["event_id"]: x for x in self.led.settlements()}
        self.assertEqual(s["a"]["result"], "won")
        self.assertEqual(s["a"]["outcome"], "home")
        self.assertAlmostEqual(s["a"]["clv"], 1.6 * (1 / 1.6) / (1 / 1.6 + 1 / 4 + 1 / 6) - 1, places=5)
        self.assertTrue(verify(self.tmp.name).ok, verify(self.tmp.name).errors)

    def test_settle_voids_after_two_weeks(self):
        bot.run_bet(self.led, self.client, RATINGS, now=NOW)
        con = duckdb.connect()
        con.execute("CREATE TABLE matches (date DATE, home_team VARCHAR, away_team VARCHAR, home_score BIGINT, away_score BIGINT)")
        con.execute("CREATE TABLE shootouts (date DATE, home_team VARCHAR, away_team VARCHAR)")
        bot.run_settle(self.led, con, now=NOW + timedelta(days=3))
        self.assertEqual(self.led.settlements(), [])
        bot.run_settle(self.led, con, now=NOW + timedelta(days=16))
        self.assertTrue(all(x["result"] == "void" for x in self.led.settlements()))
        self.assertEqual(self.led.bankroll(), 1000.0)

    def test_credit_guard(self):
        self.api.remaining = 10
        client = bot.OddsClient("k", fetch=self.api, min_credits=50)
        with self.assertRaises(RuntimeError):
            bot.run_bet(self.led, client, RATINGS, now=NOW)


if __name__ == "__main__":
    unittest.main()
