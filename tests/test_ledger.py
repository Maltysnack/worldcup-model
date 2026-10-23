"""Tests for the hash-chained ledger.

Run: uv run python -m unittest discover tests
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from worldcup.betting import ledger
from worldcup.betting.ledger import Ledger, verify

ODDS = [2.10, 3.40, 3.60]
MODEL = [0.52, 0.26, 0.22]


def _bet(led: Ledger, n: int, **kw) -> dict:
    args = dict(
        event_id=f"ev{n}",
        kickoff=f"2026-11-{10 + n:02d}T18:00:00+00:00",
        home=f"Home{n}",
        away=f"Away{n}",
        selection=0,
        odds=ODDS,
        stake=10.0,
        model_p=MODEL,
        placed_at=f"2026-11-{9 + n:02d}T09:00:00+00:00",
    )
    args.update(kw)
    return led.place_bet(**args)


class LedgerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.led = Ledger(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def _rewrite(self, name: str, rows: list[dict]) -> None:
        (self.root / name).write_text("".join(ledger.canonical_json(r) + "\n" for r in rows))

    def test_empty_ledger_verifies(self):
        rep = verify(self.root)
        self.assertTrue(rep.ok, rep.errors)
        self.assertEqual(rep.bankroll, 1000.0)

    def test_bet_and_settle_round_trip(self):
        b1 = _bet(self.led, 1)
        self.assertEqual(b1["prev_hash"], ledger.GENESIS_HASH)
        self.assertEqual(b1["bankroll_before"], 1000.0)
        self.assertAlmostEqual(b1["edge"], 0.52 * 2.10 - 1, places=6)
        b2 = _bet(self.led, 2, selection=2)
        self.assertEqual(b2["prev_hash"], b1["hash"])
        self.assertEqual(b2["bankroll_before"], 990.0)
        s1 = self.led.settle(b1["hash"], 0, closing_odds=[2.00, 3.50, 3.80], settled_at="2026-11-11T20:00:00+00:00")
        self.assertEqual(s1["result"], "won")
        self.assertEqual(s1["returns"], 21.0)
        self.assertEqual(s1["bankroll_after"], 1001.0)
        self.assertGreater(s1["clv"], 0)
        s2 = self.led.settle(b2["hash"], 1, settled_at="2026-11-12T20:00:00+00:00")
        self.assertEqual(s2["result"], "lost")
        self.assertEqual(s2["bankroll_after"], 1001.0)
        rep = verify(self.root)
        self.assertTrue(rep.ok, rep.errors)
        self.assertEqual((rep.n_bets, rep.n_settled, rep.bankroll), (2, 2, 1001.0))
        self.assertAlmostEqual(rep.pnl, 1.0)

    def test_void_refunds_stake(self):
        b = _bet(self.led, 1)
        s = self.led.settle(b["hash"], None, settled_at="2026-11-11T20:00:00+00:00")
        self.assertEqual((s["result"], s["pnl"], s["bankroll_after"]), ("void", 0.0, 1000.0))
        self.assertTrue(verify(self.root).ok)

    def test_rejects_bad_bets(self):
        _bet(self.led, 1)
        with self.assertRaises(ValueError):
            _bet(self.led, 1)  # same event
        with self.assertRaises(ValueError):
            _bet(self.led, 2, placed_at="2026-11-12T18:00:00+00:00")  # at kickoff
        with self.assertRaises(ValueError):
            _bet(self.led, 3, stake=5000)
        with self.assertRaises(ValueError):
            self.led.settle("f" * 64, 0)

    def test_double_settlement_rejected(self):
        b = _bet(self.led, 1)
        self.led.settle(b["hash"], 0, settled_at="2026-11-11T20:00:00+00:00")
        with self.assertRaises(ValueError):
            self.led.settle(b["hash"], 1)

    def test_edited_row_detected(self):
        _bet(self.led, 1)
        _bet(self.led, 2)
        rows = self.led.bets()
        rows[0]["odds"] = 5.0
        self._rewrite(ledger.BETS_FILE, rows)
        rep = verify(self.root)
        self.assertFalse(rep.ok)
        self.assertTrue(any("hash mismatch" in e for e in rep.errors))

    def test_rehashed_edit_breaks_chain(self):
        _bet(self.led, 1)
        _bet(self.led, 2)
        rows = self.led.bets()
        rows[0]["odds"] = 5.0
        rows[0]["hash"] = ledger.row_hash(rows[0])
        self._rewrite(ledger.BETS_FILE, rows)
        rep = verify(self.root)
        self.assertTrue(any("prev_hash" in e for e in rep.errors), rep.errors)

    def test_deleted_row_detected(self):
        for n in (1, 2, 3):
            _bet(self.led, n)
        rows = self.led.bets()
        self._rewrite(ledger.BETS_FILE, [rows[0], rows[2]])
        rep = verify(self.root)
        self.assertTrue(any("prev_hash" in e for e in rep.errors), rep.errors)

    def test_deleted_losing_settlement_detected_by_replay(self):
        b1 = _bet(self.led, 1)
        self.led.settle(b1["hash"], 2, settled_at="2026-11-11T20:00:00+00:00")
        _bet(self.led, 2)
        # Dropping the whole settlements file leaves both chains intact, but the
        # second bet's bankroll_before no longer replays.
        (self.root / ledger.SETTLEMENTS_FILE).unlink()
        rep = verify(self.root)
        self.assertFalse(rep.ok)
        self.assertEqual(rep.n_settled, 0)

    def test_snapshot_evidence_checked(self):
        snap = self.led.save_snapshot(json.dumps({"bookmaker": "sportsbet"}), "odds")
        _bet(self.led, 1, snapshot=snap)
        self.assertTrue(verify(self.root).ok)
        (self.led.snapshot_dir / snap["file"]).write_text("{}")
        rep = verify(self.root)
        self.assertTrue(any("snapshot sha256" in e for e in rep.errors), rep.errors)

    def test_cli_exit_codes(self):
        _bet(self.led, 1)
        self.assertEqual(ledger.main(["--dir", str(self.root)]), 0)
        with (self.root / ledger.BETS_FILE).open("a") as f:
            f.write('{"type":"bet"}\n')
        self.assertEqual(ledger.main(["--dir", str(self.root)]), 1)


if __name__ == "__main__":
    unittest.main()
