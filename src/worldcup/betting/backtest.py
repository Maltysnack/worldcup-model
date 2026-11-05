"""Walk-forward betting backtest on WC 2022, Euro 2024, Copa América 2024, WC 2026.

Model probabilities come from plain Elo (v4a-flat-mild) using the rating each
team had strictly before kickoff (the `predictions` table records pre-match
ratings as the Elo walk goes), mapped through the Poisson goals model.

Until historical Sportsbet odds exist, odds are synthetic
(see betting.synthetic). Two scenarios bracket what the engine does:

  - edge:    the model is the truth; the book quotes noisy prices
  - no_edge: the book quotes near-true prices; the bettor's model is the noisy one

Both settle against real results. Synthetic results are a pipeline check,
not evidence of skill. Knockout scores in the source data include extra
time, so a 90-minute draw won in ET settles as a win here (real 1X2 markets
settle on 90 minutes); shootouts count as draws.

Run: uv run worldcup-betting-backtest
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from worldcup.betting import engine, synthetic
from worldcup.db import connect
from worldcup.elo import HOME_ADVANTAGE
from worldcup.goals import expected_goals

ROOT = Path(__file__).resolve().parents[3]
SUMMARY_PATH = ROOT / "data" / "raw" / "betting_backtest_summary.md"
START_BANKROLL = 1000.0
TOURNAMENTS = [
    ("FIFA World Cup", 2022),
    ("UEFA Euro", 2024),
    ("Copa América", 2024),
    ("FIFA World Cup", 2026),
]
SEEDS = 200
BOOTSTRAP = 2000
EPS = 1e-12


def poisson_wdl(lh: float, la: float, max_k: int = 15) -> list[float]:
    k = np.arange(max_k + 1)
    fact = np.array([math.factorial(int(i)) for i in k], dtype=float)
    h = np.exp(-lh) * lh ** k / fact
    a = np.exp(-la) * la ** k / fact
    j = np.outer(h, a)
    j /= j.sum()
    return [float(np.tril(j, -1).sum()), float(np.trace(j)), float(np.triu(j, 1).sum())]


def probs_from_ratings(r_home: float, r_away: float, neutral: bool) -> list[float]:
    eff_h = r_home + (0.0 if neutral else HOME_ADVANTAGE)
    lh, la = expected_goals(eff_h - r_away, neutral=neutral)
    return poisson_wdl(float(lh), float(la))


def outcome_idx(hg: int, ag: int) -> int:
    return 0 if hg > ag else (1 if hg == ag else 2)


def load_matches() -> list[dict]:
    """Every match in the target tournaments, chronological, with pre-match ratings."""
    clauses = " OR ".join(
        "(tournament = ? AND EXTRACT(YEAR FROM match_date) = ?)" for _ in TOURNAMENTS
    )
    params = [x for t in TOURNAMENTS for x in t]
    with connect(read_only=True) as con:
        df = con.execute(
            f"""
            SELECT match_date, tournament, home_team, away_team, neutral,
                   rating_home_pre, rating_away_pre,
                   actual_home_goals, actual_away_goals
            FROM predictions
            WHERE {clauses}
            ORDER BY match_date, home_team
            """,
            params,
        ).fetchdf()
    rows = []
    for r in df.itertuples(index=False):
        rows.append({
            "match_id": f"{r.match_date}_{r.home_team}_{r.away_team}",
            "kickoff": str(r.match_date),
            "tournament": f"{r.tournament} {r.match_date.year}",
            "home": r.home_team,
            "away": r.away_team,
            "p_model": probs_from_ratings(r.rating_home_pre, r.rating_away_pre, bool(r.neutral)),
            "outcome": outcome_idx(int(r.actual_home_goals), int(r.actual_away_goals)),
        })
    return rows


def attach_odds(matches: list[dict], scenario: str, seed: int) -> list[dict]:
    """Return copies of matches with synthetic odds and the bettor's probabilities."""
    rng = np.random.default_rng(seed)
    out = []
    for m in matches:
        truth = m["p_model"]
        if scenario == "edge":
            p_bettor = list(truth)
            odds_open, odds_close = synthetic.book_prices(truth, rng)
        elif scenario == "no_edge":
            p_bettor = list(synthetic.perturb(truth, synthetic.OPEN_NOISE, rng))
            odds_open, odds_close = synthetic.book_prices(truth, rng, open_noise=0.03)
        else:
            raise ValueError(scenario)
        out.append({**m, "p_bettor": p_bettor, "odds_open": odds_open, "odds_close": odds_close})
    return out


def run_bets(matches: list[dict], rule: str, min_edge: float) -> list[engine.Bet]:
    """Place and settle bets in kickoff order. Stakes use the bankroll at the time."""
    bankroll = START_BANKROLL
    bets = []
    for m in matches:
        sel = engine.pick_selection(m["p_bettor"], m["odds_open"], min_edge)
        if sel is None:
            continue
        o = m["odds_open"][sel]
        p = m["p_bettor"][sel]
        stake = engine.stake_for(rule, p, o, bankroll)
        if stake <= 0:
            continue
        fair, _ = engine.devig_proportional(m["odds_open"])
        bet = engine.Bet(
            match_id=m["match_id"], kickoff=m["kickoff"], home=m["home"], away=m["away"],
            selection=sel, odds=o, stake=stake, model_p=p, implied_p=fair[sel],
            edge=engine.edge(p, o), bankroll_before=bankroll,
        )
        bankroll += bet.settle(m["outcome"], m["odds_close"])
        bets.append(bet)
    return bets


def roi(bets: list[engine.Bet]) -> float:
    staked = sum(b.stake for b in bets)
    return sum(b.pnl for b in bets) / staked if staked else 0.0


def bootstrap_roi_ci(bets: list[engine.Bet], n: int = BOOTSTRAP, seed: int = 0) -> tuple[float, float]:
    if not bets:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    stakes = np.array([b.stake for b in bets])
    pnls = np.array([b.pnl for b in bets])
    idx = rng.integers(0, len(bets), size=(n, len(bets)))
    rois = pnls[idx].sum(axis=1) / stakes[idx].sum(axis=1)
    lo, hi = np.percentile(rois, [2.5, 97.5])
    return float(lo), float(hi)


def score_probs(probs: list[list[float]], outcomes: list[int]) -> tuple[float, float]:
    """(mean log-loss, mean Brier) over 3-way probabilities."""
    p = np.asarray(probs)
    o = np.asarray(outcomes)
    ll = float(-np.log(np.clip(p[np.arange(len(o)), o], EPS, 1.0)).mean())
    onehot = np.zeros_like(p)
    onehot[np.arange(len(o)), o] = 1.0
    return ll, float(((p - onehot) ** 2).sum(axis=1).mean())


def summarise_run(bets: list[engine.Bet]) -> dict:
    curve = [START_BANKROLL] + [b.bankroll_before + b.pnl for b in bets]
    peak, max_dd = curve[0], 0.0
    for v in curve:
        peak = max(peak, v)
        max_dd = max(max_dd, (peak - v) / peak if peak > 0 else 0.0)
    lo, hi = bootstrap_roi_ci(bets)
    return {
        "n_bets": len(bets),
        "staked": sum(b.stake for b in bets),
        "pnl": sum(b.pnl for b in bets),
        "roi": roi(bets),
        "roi_ci95": (lo, hi),
        "final_bankroll": curve[-1],
        "max_drawdown": max_dd,
        "mean_clv": float(np.mean([b.clv for b in bets])) if bets else 0.0,
        "curve": curve,
    }


def run_all() -> dict:
    matches = load_matches()
    outcomes = [m["outcome"] for m in matches]
    results: dict = {"n_matches": len(matches), "by_tournament": {}, "scenarios": {}}
    for m in matches:
        results["by_tournament"][m["tournament"]] = results["by_tournament"].get(m["tournament"], 0) + 1
    results["model_score"] = score_probs([m["p_model"] for m in matches], outcomes)

    for scenario in ("edge", "no_edge"):
        sc: dict = {}
        priced = attach_odds(matches, scenario, seed=1)
        sc["book_open_score"] = score_probs(
            [engine.devig_proportional(m["odds_open"])[0] for m in priced], outcomes)
        sc["book_close_score"] = score_probs(
            [engine.devig_proportional(m["odds_close"])[0] for m in priced], outcomes)
        sc["bettor_score"] = score_probs([m["p_bettor"] for m in priced], outcomes)
        sc["runs"] = {}
        for rule in ("flat", "kelly"):
            for min_edge in (0.0, 0.03, 0.05, 0.10):
                seed1 = summarise_run(run_bets(priced, rule, min_edge))
                # Average over many synthetic-odds draws so one lucky seed does not decide.
                multi = [roi(run_bets(attach_odds(matches, scenario, s), rule, min_edge))
                         for s in range(2, SEEDS + 2)]
                seed1["roi_seed_mean"] = float(np.mean(multi))
                seed1["roi_seed_p_positive"] = float(np.mean(np.array(multi) > 0))
                sc["runs"][(rule, min_edge)] = seed1
        results["scenarios"][scenario] = sc
    return results


def render_md(r: dict) -> str:
    L = ["# Betting backtest summary (synthetic odds)", "",
         "Engine: proportional devig, edge = p x odds - 1, one bet per match "
         "on the largest edge above a threshold, flat $10 or quarter-Kelly capped at 5% of "
         "bankroll, $1000 start. Model: walk-forward plain Elo (v4a-flat-mild, pre-match "
         "ratings) through the Poisson goals model.", "",
         "**Odds are synthetic.** Real Sportsbet history needs the paid Odds API month "
         "(item 6, blocked). These numbers test the plumbing; they are not evidence of edge.", "",
         f"Matches: {r['n_matches']} ("
         + ", ".join(f"{k} {v}" for k, v in r["by_tournament"].items()) + ")", "",
         f"Model 3-way log-loss {r['model_score'][0]:.3f}, Brier {r['model_score'][1]:.3f}.", "",
         "Scenarios (both settle on real results):", "",
         "- **edge**: model probabilities are the truth; book prices them with log-normal "
         f"noise (sd {synthetic.OPEN_NOISE}) and a {synthetic.SPORTSBET_MARGIN - 1:.0%} margin.",
         "- **no_edge**: book prices near-true (sd 0.03); the bettor's model carries the noise "
         f"(sd {synthetic.OPEN_NOISE}). Expect to lose about the margin.", "",
         f"Closing price = bet price with zero-mean drift (sd {synthetic.CLOSE_DRIFT}), so "
         "prices do not move toward the truth and synthetic CLV against the devigged close sits "
         "near minus the margin by construction.", ""]
    for name, sc in r["scenarios"].items():
        L += [f"## Scenario: {name}", "",
              f"Log-loss / Brier: bettor {sc['bettor_score'][0]:.3f} / {sc['bettor_score'][1]:.3f}, "
              f"book open (devigged) {sc['book_open_score'][0]:.3f} / {sc['book_open_score'][1]:.3f}, "
              f"book close {sc['book_close_score'][0]:.3f} / {sc['book_close_score'][1]:.3f}.", "",
              "| Staking | Min edge | Bets | Staked | P&L | ROI | ROI 95% CI | Final bankroll "
              f"| Max DD | Mean CLV | Mean ROI over {SEEDS} odds draws | P(ROI>0) |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for (rule, me), s in sc["runs"].items():
            lo, hi = s["roi_ci95"]
            L.append(
                f"| {rule} | {me:.2f} | {s['n_bets']} | ${s['staked']:.0f} | ${s['pnl']:+.0f} | "
                f"{s['roi']:+.1%} | {lo:+.1%} to {hi:+.1%} | ${s['final_bankroll']:.0f} | "
                f"{s['max_drawdown']:.0%} | {s['mean_clv']:+.1%} | {s['roi_seed_mean']:+.1%} | "
                f"{s['roi_seed_p_positive']:.0%} |")
        L.append("")
    L += ["## Recommended staking rule (provisional)", "",
          "**Flat $10, min edge 0.03.** Reasons: quarter-Kelly sizes bets by the model's own "
          "edge estimate, so a miscalibrated model bets biggest exactly where it is most wrong; "
          "flat stakes keep ROI and CLV readable per bet and drawdowns small; with roughly 100 "
          "bets a year the ROI CI is wide either way, so variance control matters more than "
          "growth rate. Revisit once item 6 supplies real Sportsbet prices: if real CLV is "
          "reliably positive, quarter-Kelly becomes defensible.", "",
          "**The rule is locked before live betting.**", "",
          "Limitations: synthetic odds; one bet per match; stakes use bankroll at placement in "
          "kickoff order (same-day matches settle sequentially); knockout results include extra "
          "time, unlike real 1X2 settlement.", ""]
    return "\n".join(L)


def main() -> None:
    r = run_all()
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(render_md(r))
    print(f"wrote {SUMMARY_PATH}")
    for name, sc in r["scenarios"].items():
        for (rule, me), s in sc["runs"].items():
            print(f"  {name:8s} {rule:5s} edge>={me:.2f}  bets={s['n_bets']:3d}  "
                  f"roi={s['roi']:+.1%}  mean_roi={s['roi_seed_mean']:+.1%}  "
                  f"bank=${s['final_bankroll']:.0f}")


if __name__ == "__main__":
    main()
