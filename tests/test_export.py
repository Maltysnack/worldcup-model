"""Site export helpers that need no DB."""

import importlib.util
import json
import pathlib
import tempfile
import unittest
from datetime import datetime

p = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "export_site_v2.py"
spec = importlib.util.spec_from_file_location("export_site_v2", p)
ex = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ex)


class RecentMatchesTest(unittest.TestCase):
    ROWS = [(datetime(2026, 10, 1), "Friendly", "Spain", "Italy", 2000.0, 1900.0, 2, 0),
            (datetime(2026, 10, 4), "Friendly", "Italy", "Spain", 1890.0, 2010.0, 1, 1)]

    def test_deltas_come_from_next_rating(self):
        newest, oldest = ex.recent_matches(self.ROWS, {"Spain": 2008.0, "Italy": 1892.0})
        self.assertEqual((newest["home"], newest["home_delta"], newest["away_delta"]), ("Italy", 2.0, -2.0))
        self.assertEqual((oldest["home_delta"], oldest["away_delta"], oldest["score"]), (10.0, -10.0, [2, 0]))
        self.assertEqual(oldest["date"], "2026-10-01")

    def test_keeps_newest_n(self):
        out = ex.recent_matches(self.ROWS, {"Spain": 2008.0, "Italy": 1892.0}, n=1)
        self.assertEqual([m["date"] for m in out], ["2026-10-04"])


class TournamentsTest(unittest.TestCase):
    def test_only_simulated_tournaments_with_a_file(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(ex.tournaments(pathlib.Path(d)))
            for k in ("wc2026", "wc2030"):
                (pathlib.Path(d) / f"forecast_{k}.json").write_text(json.dumps({"tournament": k}))
            self.assertEqual(ex.tournaments(pathlib.Path(d))["tournaments"], [{"tournament": "wc2030"}])


if __name__ == "__main__":
    unittest.main()
