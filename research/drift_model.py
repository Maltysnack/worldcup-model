"""Queue item 13: Elo drift model for title odds years out.

Fits how far a team's Elo moves over a horizon of h years, as a function
of its starting rating, then hindcasts World Cup title odds from ratings
taken ~4 years before kickoff: static (freeze the old ratings) vs drift
(sample each team's future rating from the fitted distribution). Scored
against what happened, with the real pre-tournament ratings as a ceiling.

Each fit uses only rating moves that finished before the forecast origin.
The field and groups are taken as known (the 32 teams that qualified);
a real 2030 forecast would also have to model qualifying.

Run: uv run python research/drift_model.py   (writes data/raw/drift_model_summary.md)
"""

from __future__ import annotations

from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np

from worldcup.db import connect
from worldcup.tournament_backtest import (
    _PRED_TABLE, fetch_actual_results, fetch_pre_wc_ratings, fetch_wc_teams_and_groups, simulate_tournament,
)

OUT = Path(__file__).resolve().parents[1] / "data" / "raw" / "drift_model_summary.md"
# (World Cup year, forecast origin): origin is the December after the previous WC.
CASES = [(2010, date(2006, 12, 31)), (2014, date(2010, 12, 31)), (2018, date(2014, 12, 31)), (2022, date(2018, 12, 31))]
FIT_FROM = 1990
STAGES = ["group_stage", "round_of_16", "quarter_finals", "semi_finals", "runner_up", "champion"]
N_DRAWS, SIMS_PER_DRAW = 200, 50  # drift: 200 rating draws x 50 tournaments = 10k
N_STATIC = 10_000


def rating_series() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per team: match dates (as ordinal days) and the pre-match rating."""
    with connect(read_only=True) as con:
        rows = con.execute(f"""
            SELECT match_date, home_team AS team, rating_home_pre AS r FROM {_PRED_TABLE}
            UNION ALL
            SELECT match_date, away_team, rating_away_pre FROM {_PRED_TABLE}
            ORDER BY 2, 1""").fetchall()
    out: dict[str, tuple[list, list]] = {}
    for d, t, r in rows:
        ds, rs = out.setdefault(t, ([], []))
        ds.append(d.toordinal())
        rs.append(r)
    return {t: (np.array(ds), np.array(rs)) for t, (ds, rs) in out.items()}


def rating_at(series, team: str, d: int) -> float | None:
    """Rating on day d = pre-match rating of the team's first match after d."""
    ds, rs = series[team]
    i = np.searchsorted(ds, d, side="right")
    return float(rs[i]) if i < len(ds) else None


def fit_drift(series, h_years: float, origin: date, min_rating: float = 1600) -> dict:
    """Delta = a + b*r + N(0, sd), fitted on 6-monthly start dates whose end is before origin.
    Teams need a match within a year before the start and after the end (active both times),
    and a starting rating of at least `min_rating` (the World Cup field's range)."""
    h = int(round(h_years * 365.25))
    starts = range(date(FIT_FROM, 1, 1).toordinal(), origin.toordinal() - h, 182)
    r0s, deltas = [], []
    for t, (ds, _) in series.items():
        for s in starts:
            e = s + h
            if not (np.any((ds > s - 365) & (ds <= s)) and np.any((ds > e - 365) & (ds <= e))):
                continue
            a, b = rating_at(series, t, s), rating_at(series, t, e)
            if a is not None and b is not None and a >= min_rating:
                r0s.append(a)
                deltas.append(b - a)
    r0s, deltas = np.array(r0s), np.array(deltas)
    b, a = np.polyfit(r0s, deltas, 1)
    resid = deltas - (a + b * r0s)
    bands = [(lo, hi, float(resid[(r0s >= lo) & (r0s < hi)].std()), int(((r0s >= lo) & (r0s < hi)).sum()))
             for lo, hi in ((1600, 1750), (1750, 1900), (1900, 2000), (2000, 9999))]
    return {"a": float(a), "b": float(b), "sd": float(resid.std()), "n": len(deltas), "bands": bands}


def stage_probs(counts: dict[str, Counter], teams: list[str]) -> dict[str, np.ndarray]:
    # Laplace smoothing so a stage never seen in sims does not give infinite log-loss.
    return {t: (np.array([counts[t][s] for s in STAGES]) + 0.5) / (sum(counts[t].values()) + 0.5 * len(STAGES))
            for t in teams}


def actual_stage(year: int, teams: list[str]) -> dict[str, str]:
    """Furthest stage each team reached, from the WC's knockout matches."""
    res = fetch_actual_results(year)
    with connect(read_only=True) as con:
        ko = con.execute("""SELECT home_team, away_team FROM matches WHERE tournament = 'FIFA World Cup'
                            AND EXTRACT(YEAR FROM date) = ? AND home_score IS NOT NULL ORDER BY date""",
                         [year]).fetchall()[48:]
    r16 = {x for m in ko[:8] for x in m}
    qf = {x for m in ko[8:12] for x in m}
    out = {}
    for t in teams:
        if t == res["champion"]:
            out[t] = "champion"
        elif t == res["runner_up"]:
            out[t] = "runner_up"
        elif t in res["semifinalists"]:
            out[t] = "semi_finals"
        elif t in qf:
            out[t] = "quarter_finals"
        elif t in r16:
            out[t] = "round_of_16"
        else:
            out[t] = "group_stage"
    return out


def score(probs: dict[str, np.ndarray], actual: dict[str, str], champion: str) -> dict:
    stage_ll = np.mean([-np.log(probs[t][STAGES.index(s)]) for t, s in actual.items()])
    p_champ = {t: p[-1] for t, p in probs.items()}
    total = sum(p_champ.values())
    return {"champ_ll": float(-np.log(p_champ[champion] / total)), "stage_ll": float(stage_ll),
            "p_champ": p_champ[champion] / total,
            "rank": 1 + sum(v > p_champ[champion] for v in p_champ.values())}


def run_case(series, year: int, origin: date) -> dict:
    groups = fetch_wc_teams_and_groups(year)
    teams = [t for g in groups.values() for t in g]
    actual = actual_stage(year, teams)
    champion = fetch_actual_results(year)["champion"]
    with connect(read_only=True) as con:
        kickoff = con.execute("SELECT min(date) FROM matches WHERE tournament = 'FIFA World Cup' AND EXTRACT(YEAR FROM date) = ?",
                              [year]).fetchone()[0].toordinal()
    h_years = (kickoff - origin.toordinal()) / 365.25
    fit = fit_drift(series, h_years, origin)
    r0 = {t: rating_at(series, t, origin.toordinal()) for t in teams}

    static = stage_probs(simulate_tournament(r0, groups, N_STATIC, seed=1)[0], teams)
    oracle = stage_probs(simulate_tournament(fetch_pre_wc_ratings(year, teams), groups, N_STATIC, seed=1)[0], teams)
    rng = np.random.default_rng(7)
    drift_counts: dict[str, Counter] = {t: Counter() for t in teams}
    for i in range(N_DRAWS):
        draw = {t: r + fit["a"] + fit["b"] * r + fit["sd"] * rng.standard_normal() for t, r in r0.items()}
        for t, c in simulate_tournament(draw, groups, SIMS_PER_DRAW, seed=100 + i)[0].items():
            drift_counts[t].update(c)
    drift = stage_probs(drift_counts, teams)
    uniform = {t: np.array([16, 8, 4, 2, 1, 1]) / 32 for t in teams}
    return {"year": year, "origin": origin, "h": h_years, "fit": fit, "champion": champion,
            "scores": {k: score(v, actual, champion) for k, v in
                       (("static", static), ("drift", drift), ("oracle", oracle), ("uniform", uniform))},
            "top": sorted(teams, key=lambda t: -drift[t][-1])[:5], "drift": drift, "static": static}


def report(cases: list[dict]) -> str:
    L = ["# Elo drift model (queue item 13)", "",
         f"_Generated {date.today().isoformat()} by `research/drift_model.py`._", "",
         "Question: when forecasting a World Cup years out, does modelling how Elo ratings drift beat freezing today's ratings?", "",
         "Method: for each team, Elo change over h years (start dates every 6 months from 1990, active teams rated 1600+ at the start, moves ending before the forecast origin) fitted as `delta = a + b*rating + N(0, sd)`. "
         "Forecast origin is 31 Dec after the previous World Cup (about 3.5 years out). Static = simulate with origin ratings. Drift = 200 draws of future ratings x 50 tournaments each. "
         "Oracle = real pre-tournament ratings (the ceiling). Field and groups taken as known. Group letters follow discovery order, so knockout cross-pairings are approximate (same as `worldcup-backtest-tournament`).", "",
         "## Fitted drift", "", "| WC | origin | h (yr) | a | b | sd | n | mean move at 1700 | at 2000 |", "|---|---|---|---|---|---|---|---|---|"]
    for c in cases:
        f = c["fit"]
        L.append(f"| {c['year']} | {c['origin']} | {c['h']:.2f} | {f['a']:.0f} | {f['b']:.3f} | {f['sd']:.0f} | {f['n']} | {f['a'] + f['b'] * 1700:+.0f} | {f['a'] + f['b'] * 2000:+.0f} |")
    L += ["", "Residual sd by starting rating (latest fit):", ""]
    L += [f"- {lo}-{hi if hi < 9999 else '+'}: sd {sd:.0f} (n {n})" for lo, hi, sd, n in cases[-1]["fit"]["bands"]]
    L += ["", "## Hindcast scores", "",
          "Champion log-loss: -ln P(actual champion). Stage log-loss: mean over 32 teams of -ln P(stage actually reached). Lower is better.", "",
          "| WC | champion | static P / rank | drift P / rank | oracle P / rank | champ LL static | drift | oracle | uniform | stage LL static | drift | oracle | uniform |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in cases:
        s = c["scores"]
        L.append(f"| {c['year']} | {c['champion']} | " + " | ".join(f"{s[k]['p_champ'] * 100:.1f}% / {s[k]['rank']}" for k in ("static", "drift", "oracle"))
                 + " | " + " | ".join(f"{s[k]['champ_ll']:.2f}" for k in ("static", "drift", "oracle", "uniform"))
                 + " | " + " | ".join(f"{s[k]['stage_ll']:.3f}" for k in ("static", "drift", "oracle", "uniform")) + " |")
    mean = {k: (np.mean([c["scores"][k]["champ_ll"] for c in cases]), np.mean([c["scores"][k]["stage_ll"] for c in cases]))
            for k in ("static", "drift", "oracle", "uniform")}
    L += ["", "Means over the four tournaments:", ""]
    L += [f"- {k}: champion LL {v[0]:.3f}, stage LL {v[1]:.3f}" for k, v in mean.items()]
    last = cases[-1]
    L += ["", f"WC {last['year']} drift top 5 (from {last['origin']}): " + ", ".join(
        f"{t} {last['drift'][t][-1] * 100:.1f}% (static {last['static'][t][-1] * 100:.1f}%)" for t in last["top"]), ""]
    better = mean["drift"][1] < mean["static"][1]
    L += ["## Verdict", "",
          f"Drift {'beats' if better else 'does not beat'} static on mean stage log-loss ({mean['drift'][1]:.3f} vs {mean['static'][1]:.3f}); "
          f"champion log-loss {mean['drift'][0]:.3f} vs {mean['static'][0]:.3f}. Four tournaments is a small sample: the champion score is one event each, so lean on stage log-loss (32 teams each).", ""]
    return "\n".join(L)


def main() -> None:
    series = rating_series()
    cases = []
    for year, origin in CASES:
        c = run_case(series, year, origin)
        print(year, {k: round(v["stage_ll"], 3) for k, v in c["scores"].items()}, flush=True)
        cases.append(c)
    OUT.write_text(report(cases))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
