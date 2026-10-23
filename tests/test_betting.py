"""Unit tests for the betting engine on synthetic odds.

Run: uv run python -m unittest discover tests
"""

from __future__ import annotations

import unittest

import numpy as np

from worldcup.betting import engine, synthetic
from worldcup.betting.backtest import bootstrap_roi_ci, roi, run_bets


class DevigTest(unittest.TestCase):
    def test_fair_book_has_no_margin(self):
        probs, over = engine.devig_proportional([2.0, 4.0, 4.0])
        self.assertAlmostEqual(over, 1.0)
        self.assertEqual([round(p, 9) for p in probs], [0.5, 0.25, 0.25])

    def test_margin_is_removed_proportionally(self):
        odds = synthetic.quote([0.5, 0.3, 0.2], margin=1.06)
        probs, over = engine.devig_proportional(odds)
        self.assertAlmostEqual(over, 1.06)
        for got, want in zip(probs, [0.5, 0.3, 0.2]):
            self.assertAlmostEqual(got, want)

    def test_rejects_bad_odds(self):
        with self.assertRaises(ValueError):
            engine.devig_proportional([1.0, 3.0, 5.0])


class EdgeAndStakeTest(unittest.TestCase):
    def test_edge(self):
        self.assertAlmostEqual(engine.edge(0.5, 2.2), 0.1)
        self.assertAlmostEqual(engine.edge(0.5, 2.0), 0.0)

    def test_pick_selection_threshold(self):
        p = [0.5, 0.3, 0.2]
        self.assertEqual(engine.pick_selection(p, [2.2, 3.0, 4.0], 0.03), 0)
        self.assertIsNone(engine.pick_selection(p, [1.9, 3.2, 5.0], 0.03))
        self.assertEqual(engine.pick_selection(p, [1.9, 3.6, 5.0], 0.03), 1)

    def test_flat_never_exceeds_bankroll(self):
        self.assertEqual(engine.stake_flat(1000.0), 10.0)
        self.assertEqual(engine.stake_flat(4.0), 4.0)
        self.assertEqual(engine.stake_flat(0.0), 0.0)

    def test_quarter_kelly(self):
        # full Kelly at p=0.5, odds 2.2: 0.1 / 1.2 = 0.0833; quarter = 0.0208
        self.assertAlmostEqual(engine.stake_kelly(0.5, 2.2, 1000.0), 1000 * 0.1 / 1.2 / 4)

    def test_kelly_cap(self):
        # full Kelly 0.6 / 1 = 0.6; quarter = 0.15; capped at 5%
        self.assertAlmostEqual(engine.stake_kelly(0.8, 2.0, 1000.0), 50.0)

    def test_kelly_zero_without_edge(self):
        self.assertEqual(engine.stake_kelly(0.4, 2.0, 1000.0), 0.0)


class SettleAndClvTest(unittest.TestCase):
    def test_profit(self):
        self.assertAlmostEqual(engine.profit(10, 2.5, True), 15.0)
        self.assertAlmostEqual(engine.profit(10, 2.5, False), -10.0)

    def test_clv_against_fair_close(self):
        close = synthetic.quote([0.5, 0.3, 0.2], margin=1.06)
        self.assertAlmostEqual(engine.clv(2.2, close, 0), 0.1)
        self.assertLess(engine.clv(2.2, close, 0), engine.clv_raw(2.2, close, 0))

    def test_bet_settle(self):
        b = engine.Bet("m", "2026-06-11", "A", "B", 2, 4.0, 10.0, 0.3, 0.24, 0.2, 1000.0)
        self.assertAlmostEqual(b.settle(2, [2.0, 3.5, 4.0]), 30.0)
        self.assertIsNotNone(b.clv)


def _world(n: int, seed: int, scenario: str) -> list[dict]:
    """Synthetic matches whose results are drawn from the true probabilities."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        truth = rng.dirichlet([4, 2.5, 3])
        if scenario == "edge":
            p_bettor = list(truth)
            o_open, o_close = synthetic.book_prices(truth, rng)
        else:
            p_bettor = list(synthetic.perturb(truth, 0.15, rng))
            o_open, o_close = synthetic.book_prices(truth, rng, open_noise=0.0)
        out.append({
            "match_id": str(i), "kickoff": str(i), "home": "H", "away": "A",
            "p_bettor": p_bettor, "odds_open": o_open, "odds_close": o_close,
            "outcome": int(rng.choice(3, p=truth)),
        })
    return out


class BacktestTest(unittest.TestCase):
    def test_true_model_beats_noisy_book(self):
        bets = run_bets(_world(4000, 7, "edge"), "flat", 0.03)
        self.assertGreater(len(bets), 500)
        self.assertGreater(roi(bets), 0.0)

    def test_noisy_model_loses_to_sharp_book(self):
        bets = run_bets(_world(4000, 7, "no_edge"), "flat", 0.03)
        self.assertLess(roi(bets), 0.0)

    def test_kelly_bankroll_stays_positive(self):
        bets = run_bets(_world(2000, 3, "no_edge"), "kelly", 0.0)
        self.assertTrue(all(b.bankroll_before > 0 for b in bets))
        self.assertTrue(all(b.stake <= 0.05 * b.bankroll_before + 1e-9 for b in bets))

    def test_bootstrap_ci_brackets_point_estimate(self):
        bets = run_bets(_world(1000, 11, "edge"), "flat", 0.03)
        lo, hi = bootstrap_roi_ci(bets)
        self.assertLess(lo, roi(bets))
        self.assertGreater(hi, roi(bets))


if __name__ == "__main__":
    unittest.main()
