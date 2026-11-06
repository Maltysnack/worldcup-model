"""Registry and overlay checks for worldcup.forecast (no DB)."""

import unittest
from datetime import date
from unittest import mock

from worldcup import forecast as fc
from worldcup.altitude import altitude_bonus
from worldcup.host import host_bonus
from worldcup.registry import REGISTRY
from worldcup.trophies import nt_trophy_bonus

RAW = {"Mexico": 1900.0, "United States": 1900.0, "Argentina": 1900.0, "Spain": 1900.0}
OFF = {k: False for k in fc.LAYERS}
WC26 = REGISTRY["wc2026"]
ASOF = date(2026, 6, 10)


class RegistryTest(unittest.TestCase):
    def test_seeded_tournaments(self):
        self.assertEqual(set(REGISTRY), {"wc2026", "euro2028", "copa2028", "wc2030"})

    def test_formats_are_consistent(self):
        for k, t in REGISTRY.items():
            self.assertEqual(t["key"], k)
            self.assertIn(t["draw"], ("fixtures", "simulated"))
            n = 4 * t["groups"]
            self.assertTrue(t["year"] == t["kickoff"].year)
            if t["draw"] == "simulated":
                self.assertEqual(sum(t["slots"].values()), n, k)
            # Knockout field must be a power of two.
            ko = 2 * t["groups"] + t["best_thirds"]
            self.assertEqual(ko & (ko - 1), 0, k)

    def test_hosts_and_altitude_match_lookups(self):
        for t in REGISTRY.values():
            if t["key"] in ("euro2028", "wc2026", "wc2030"):
                self.assertTrue(any(host_bonus(h, t["competition"], t["year"]) > 0 for h in t["hosts"]), t["key"])
            has_alt = any(altitude_bonus(h, t["competition"], t["year"]) > 0 for h in t["hosts"])
            self.assertEqual(has_alt, bool(t["altitude_venues"]), t["key"])


class OverlayTest(unittest.TestCase):
    def test_all_off_is_plain_elo(self):
        self.assertEqual(fc.apply_overlay(RAW, WC26, ASOF, OFF), RAW)

    def test_each_layer_adds_its_bonus(self):
        r = fc.apply_overlay(RAW, WC26, ASOF, {**OFF, "nt": True})
        self.assertAlmostEqual(r["Argentina"] - 1900, nt_trophy_bonus("Argentina", 2026))
        self.assertGreater(r["Argentina"], 1900)
        r = fc.apply_overlay(RAW, WC26, ASOF, {**OFF, "host": True})
        self.assertEqual(r["United States"] - 1900, host_bonus("United States", "FIFA World Cup", 2026))
        self.assertEqual(r["Spain"], 1900)
        r = fc.apply_overlay(RAW, WC26, ASOF, {**OFF, "altitude": True})
        self.assertEqual((r["Mexico"] - 1900, r["Spain"]), (50.0, 1900))

    def test_future_tournament_has_no_altitude(self):
        r = fc.apply_overlay(RAW, REGISTRY["euro2028"], ASOF, {**OFF, "altitude": True})
        self.assertEqual(r, RAW)

    def test_form_gets_as_of_date(self):
        with mock.patch.object(fc, "apply_form", side_effect=lambda r, as_of_date: r) as m:
            fc.apply_overlay(RAW, WC26, ASOF, {**OFF, "form": True})
        m.assert_called_once()
        self.assertEqual(m.call_args.kwargs["as_of_date"], ASOF)

    def test_defaults_follow_evidence(self):
        self.assertEqual(fc.LAYERS, {"form": False, "nt": True, "host": True, "altitude": True})

    def test_layer_deltas_name_each_effect(self):
        d = fc.layer_deltas(RAW, WC26, ASOF, {**OFF, "host": True, "altitude": True})
        self.assertEqual(d["Mexico"]["altitude"], 50)
        self.assertEqual(d["United States"]["host"], host_bonus("United States", "FIFA World Cup", 2026))
        self.assertEqual(d["Argentina"], {})

    def test_forecast_joins_elo_only_odds(self):
        sims = iter([[{"team": "Spain", "p_champion": 0.4}, {"team": "Mexico", "p_champion": 0.1}],
                     [{"team": "Mexico", "p_champion": 0.3}, {"team": "Argentina", "p_champion": 0.2}]])
        with (mock.patch.object(fc, "raw_elo", return_value=RAW),
              mock.patch.object(fc, "simulate_fixtures", side_effect=lambda *a: next(sims))):
            out = fc.forecast("wc2026", ASOF, {**OFF, "altitude": True})
        mex, arg = out["teams"]
        self.assertEqual((mex["team"], mex["p_champion_elo"], mex["elo"], mex["layers"]),
                         ("Mexico", 0.1, 1900, {"altitude": 50}))
        self.assertEqual((arg["team"], arg["p_champion_elo"]), ("Argentina", 0.0))

    def test_unknown_layer_rejected(self):
        with self.assertRaises(AssertionError):
            fc.apply_overlay(RAW, WC26, ASOF, {"squad": True})


if __name__ == "__main__":
    unittest.main()
