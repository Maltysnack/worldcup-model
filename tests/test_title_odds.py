"""Field selection and one-tournament sim for worldcup.title_odds (no DB)."""

import unittest

import numpy as np

from worldcup import title_odds as to

T = {t["key"]: t for t in to.TOURNAMENTS}


def world(n_per=30):
    """Synthetic teams per confederation, plus every named host."""
    conf, ratings = {}, {}
    for c, base in (("UEFA", 1900), ("CAF", 1700), ("AFC", 1650), ("CONCACAF", 1700), ("OFC", 1300)):
        for i in range(n_per):
            conf[f"{c}{i}"], ratings[f"{c}{i}"] = c, base - 10 * i
    for t in to.CONMEBOL:
        conf[t], ratings[t] = "CONMEBOL", 1800
    for h, c in (("England", "UEFA"), ("Scotland", "UEFA"), ("Wales", "UEFA"), ("Republic of Ireland", "UEFA"),
                 ("Northern Ireland", "UEFA"), ("Spain", "UEFA"), ("Portugal", "UEFA"), ("Morocco", "CAF")):
        conf[h], ratings[h] = c, 1000  # hosts rated too low to qualify on merit
    return conf, ratings


class TitleOddsTest(unittest.TestCase):
    def test_euro_field_has_two_hosts(self):
        conf, r = world()
        f = to.pick_field(T["euro2028"], r, conf)
        self.assertEqual(len(f), 24)
        self.assertEqual(len(set(f)), 24)
        self.assertEqual(sum(h in f for h in T["euro2028"]["hosts"]), 2)

    def test_wc_field_hosts_and_slots(self):
        conf, r = world()
        f = to.pick_field(T["wc2030"], r, conf)
        self.assertEqual(len(f), 48)
        self.assertEqual(len(set(f)), 48)
        self.assertTrue(set(T["wc2030"]["hosts"]) <= set(f))
        self.assertEqual(sum(conf[x] == "UEFA" for x in f), 16)
        self.assertGreaterEqual(sum(conf[x] == "CONMEBOL" for x in f), 6)  # plus any play-off winners
        self.assertIn("OFC0", f)

    def test_copa_field(self):
        conf, r = world()
        f = to.pick_field(T["copa2028"], r, conf)
        self.assertEqual(len(f), 16)
        self.assertTrue(to.CONMEBOL <= set(f))

    def test_play_returns_two_finalists_from_field(self):
        conf, r = world()
        rng = np.random.default_rng(0)
        for key in T:
            t = T[key]
            f = to.pick_field(t, r, conf)
            w, ru = to.play(f, r, t["groups"], t["best_thirds"], rng)
            self.assertIn(w, f)
            self.assertIn(ru, f)
            self.assertNotEqual(w, ru)


if __name__ == "__main__":
    unittest.main()
