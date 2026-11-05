"""Smoke test: scripts/weekly_report.py renders from the real ledger."""

import importlib.util
import pathlib
import unittest
from datetime import UTC, datetime

P = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "weekly_report.py"


class WeeklyReportTest(unittest.TestCase):
    def test_renders(self):
        spec = importlib.util.spec_from_file_location("weekly_report", P)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        now = datetime(2026, 11, 3, tzinfo=UTC)
        runs = [{"conclusion": "failure", "createdAt": "2026-11-01T06:00:00Z", "url": "u", "event": "schedule"}]
        out = m.report(now, runs, 123)
        self.assertIn("**Bankroll $", out)
        self.assertIn("failed: 1", out)
        self.assertIn("123", out)


if __name__ == "__main__":
    unittest.main()
