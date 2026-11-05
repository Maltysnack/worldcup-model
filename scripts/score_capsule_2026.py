"""Score the frozen 2026 prediction (tag wc2026-final) against actual results.

Writes data/capsule/scorecard_2026.json and data/capsule/scorecard_2026.md.

Model probabilities:
  - Group matches (72): taken verbatim from data/simulation-prediction.json.
  - Knockout matches (32): the frozen files only hold the predicted bracket,
    so W/D/L is recomputed from the frozen adjusted ratings (team_stats
    "rating": Elo blended with squad + form, host and altitude bonuses)
    using the same independent-Poisson goals model the frozen file used.
Baselines:
  - Uniform 1/3.
  - Plain Elo: raw v4a-flat-mild Elo built only on matches up to 2026-06-10,
    same goals model, no squad/form/host adjustments.

Knockout scores in the source data include extra time; a shootout counts
as a draw for 3-way scoring. A separate "advance" score covers knockouts.

Run: uv run python scripts/score_capsule_2026.py
"""

from __future__ import annotations

import json
import math
from datetime import date
from pathlib import Path

import numpy as np

from worldcup.db import connect
from worldcup.elo import HOME_ADVANTAGE
from worldcup.elo import run as elo_run
from worldcup.goals import PEN_ELO_DIVISOR, expected_goals

ROOT = Path(__file__).resolve().parents[1]
FROZEN_SIM = ROOT / "data" / "simulation-prediction.json"
OUT_DIR = ROOT / "data" / "capsule"
CUTOFF = date(2026, 6, 10)
KO_ROUNDS = [("round_of_32", 16), ("round_of_16", 8), ("quarter_finals", 4),
             ("semi_finals", 2), ("third_place", 1), ("final", 1)]
STAGE_ORDER = ["group_stage", "round_of_32", "round_of_16", "quarter_finals",
               "semi_finals", "final", "champion"]
# team_stats key for "reached at least this stage"
STAGE_KEY = {"round_of_32": "p_advance_from_groups", "round_of_16": "p_round_of_16",
             "quarter_finals": "p_quarters", "semi_finals": "p_semis",
             "final": "p_final", "champion": "p_champion"}


def poisson_wdl(lh: float, la: float, max_k: int = 15) -> tuple[float, float, float]:
    k = np.arange(max_k + 1)
    fact = np.array([math.factorial(int(i)) for i in k], dtype=float)
    h = np.exp(-lh) * lh ** k / fact
    a = np.exp(-la) * la ** k / fact
    j = np.outer(h, a)
    j /= j.sum()
    return float(np.tril(j, -1).sum()), float(np.trace(j)), float(np.triu(j, 1).sum())


def probs_from_ratings(r_home: float, r_away: float, neutral: bool) -> tuple[float, float, float]:
    eff_h = r_home + (0.0 if neutral else HOME_ADVANTAGE)
    lh, la = expected_goals(eff_h - r_away, neutral=neutral)
    return poisson_wdl(float(lh), float(la))


def p_pen(r_a: float, r_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((r_b - r_a) / PEN_ELO_DIVISOR))


def outcome_idx(hg: int, ag: int) -> int:
    return 0 if hg > ag else (1 if hg == ag else 2)


def log_loss(p: list[float], o: int) -> float:
    return -math.log(max(p[o], 1e-12))


def brier(p: list[float], o: int) -> float:
    return sum((p[i] - (1.0 if i == o else 0.0)) ** 2 for i in range(3))


def summarise(rows: list[dict], key: str) -> dict:
    ll = [log_loss(r[key], r["outcome"]) for r in rows]
    br = [brier(r[key], r["outcome"]) for r in rows]
    acc = [int(np.argmax(r[key]) == r["outcome"]) for r in rows]
    return {"n": len(rows), "log_loss": round(float(np.mean(ll)), 4),
            "brier": round(float(np.mean(br)), 4), "accuracy": round(float(np.mean(acc)), 4)}


def reliability(rows: list[dict], key: str, n_bins: int = 10) -> list[dict]:
    ps, ys = [], []
    for r in rows:
        for i in range(3):
            ps.append(r[key][i])
            ys.append(1.0 if i == r["outcome"] else 0.0)
    ps, ys = np.array(ps), np.array(ys)
    out = []
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        m = (ps >= lo) & ((ps < hi) if b < n_bins - 1 else (ps <= hi))
        if m.sum() == 0:
            continue
        out.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": int(m.sum()),
                    "mean_p": round(float(ps[m].mean()), 3),
                    "observed": round(float(ys[m].mean()), 3)})
    ece = sum(x["n"] * abs(x["mean_p"] - x["observed"]) for x in out) / len(ps)
    return {"bins": out, "ece": round(float(ece), 4)}


def main() -> None:
    frozen = json.loads(FROZEN_SIM.read_text())
    adj = {t["team"]: float(t["rating"]) for t in frozen["team_stats"]}
    team_stats = {t["team"]: t for t in frozen["team_stats"]}
    frozen_group = {frozenset((m["home_team"], m["away_team"])): m
                    for m in frozen["match_predictions"] if m["tournament"] == "FIFA World Cup"}

    raw_df, _ = elo_run(max_date=CUTOFF, quiet=True)
    raw = dict(zip(raw_df["team"], raw_df["rating"]))

    with connect(read_only=True) as con:
        matches = con.execute(
            "SELECT date, home_team, away_team, home_score, away_score, neutral "
            "FROM matches WHERE tournament = 'FIFA World Cup' AND date >= '2026-06-01' "
            "ORDER BY date, home_team"
        ).fetchdf()
        shoot = con.execute(
            "SELECT home_team, away_team, winner FROM shootouts "
            "WHERE date BETWEEN '2026-06-11' AND '2026-07-31'"
        ).fetchdf()
    assert len(matches) == 104, len(matches)
    pens = {frozenset((r.home_team, r.away_team)): r.winner for r in shoot.itertuples()}

    stages = ["group"] * 72
    for name, n in KO_ROUNDS:
        stages += [name] * n

    rows, recheck = [], []
    reached = {t: "group_stage" for t in adj}
    for i, m in enumerate(matches.itertuples(index=False)):
        h, a, neutral = m.home_team, m.away_team, bool(m.neutral)
        hg, ag = int(m.home_score), int(m.away_score)
        stage = stages[i]
        recomputed = list(probs_from_ratings(adj[h], adj[a], neutral))
        if stage == "group":
            f = frozen_group[frozenset((h, a))]
            p = [f["p_home_win"], f["p_draw"], f["p_away_win"]]
            if f["home_team"] != h:
                p = p[::-1]
            recheck.append(max(abs(x - y) for x, y in zip(p, recomputed)))
            src = "frozen"
        else:
            p, src = recomputed, "recomputed_from_frozen_ratings"
        p_elo = list(probs_from_ratings(raw[h], raw[a], neutral))
        row = {"n": i + 1, "date": str(m.date)[:10], "stage": stage, "home": h, "away": a,
               "neutral": neutral, "score": [hg, ag], "outcome": outcome_idx(hg, ag),
               "model": [round(x, 4) for x in p], "model_source": src,
               "plain_elo": [round(x, 4) for x in p_elo], "uniform": [1 / 3] * 3}
        if stage != "group":
            winner = h if hg > ag else a if ag > hg else pens[frozenset((h, a))]
            row["shootout"] = hg == ag
            row["advanced"] = winner
            row["p_advance_model"] = round(p[0] + p[1] * p_pen(adj[h], adj[a]) if winner == h
                                           else p[2] + p[1] * p_pen(adj[a], adj[h]), 4)
            row["p_advance_plain_elo"] = round(p_elo[0] + p_elo[1] * p_pen(raw[h], raw[a]) if winner == h
                                               else p_elo[2] + p_elo[1] * p_pen(raw[a], raw[h]), 4)
            if stage == "round_of_32":
                reached[h] = reached[a] = "round_of_32"
            elif stage not in ("third_place",):
                reached[h] = reached[a] = stage
            if stage == "final":
                reached[winner] = "champion"
        for k in ("model", "plain_elo", "uniform"):
            row[f"log_loss_{k}"] = round(log_loss(row[k], row["outcome"]), 4)
            row[f"brier_{k}"] = round(brier(row[k], row["outcome"]), 4)
        rows.append(row)

    group_rows = [r for r in rows if r["stage"] == "group"]
    ko_rows = [r for r in rows if r["stage"] != "group"]
    summary = {seg: {k: summarise(rs, k) for k in ("model", "plain_elo", "uniform")}
               for seg, rs in (("all", rows), ("group", group_rows), ("knockout", ko_rows))}
    ko_adv = {k: {"n": len(ko_rows),
                  "log_loss": round(float(np.mean([-math.log(r[f"p_advance_{k}"]) for r in ko_rows])), 4),
                  "brier": round(float(np.mean([(1 - r[f"p_advance_{k}"]) ** 2 * 2 for r in ko_rows])), 4),
                  "accuracy": round(float(np.mean([r[f"p_advance_{k}"] > 0.5 for r in ko_rows])), 4)}
              for k in ("model", "plain_elo")}
    ko_adv["coin_flip"] = {"n": len(ko_rows), "log_loss": round(math.log(2), 4), "brier": 0.5, "accuracy": 0.5}

    teams = []
    for t, st in reached.items():
        ts = team_stats[t]
        idx = STAGE_ORDER.index(st)
        p_reach = 1.0 if idx == 0 else ts[STAGE_KEY[st]] / 100
        p_next = ts[STAGE_KEY[STAGE_ORDER[idx + 1]]] / 100 if idx + 1 < len(STAGE_ORDER) else 0.0
        teams.append({"team": t, "p_champion": ts["p_champion"], "stage_reached": st,
                      "p_reach_that_stage": round(p_reach, 4),
                      "p_exactly_that_stage": round(p_reach - p_next, 4)})
    teams.sort(key=lambda x: -x["p_champion"])
    champion = next(t for t, s in reached.items() if s == "champion")
    champ_rank = [t["team"] for t in teams].index(champion) + 1
    stage_brier = {}
    for st, key in STAGE_KEY.items():
        i = STAGE_ORDER.index(st)
        stage_brier[st] = round(float(np.mean([
            (team_stats[t][key] / 100 - (1.0 if STAGE_ORDER.index(reached[t]) >= i else 0.0)) ** 2
            for t in reached])), 4)

    result = {
        "generated_at": date.today().isoformat(),
        "frozen_tag": "wc2026-final",
        "frozen_model": frozen["model_version"],
        "notes": [
            "Group probabilities verbatim from frozen file; knockout probabilities recomputed "
            "from frozen adjusted ratings with the same Poisson goals model.",
            "Knockout scores include extra time; shootouts count as draws for 3-way scoring.",
            "Plain Elo: raw v4a-flat-mild ratings from matches up to 2026-06-10, same goals model.",
            f"Recompute check on group matches: max abs diff vs frozen = {max(recheck):.4f} "
            "(frozen used 5000-sample Monte Carlo and rounded ratings).",
        ],
        "summary": summary,
        "knockout_advance": ko_adv,
        "reliability": {k: reliability(rows, k) for k in ("model", "plain_elo")},
        "champion": {"team": champion, "p_champion": team_stats[champion]["p_champion"],
                     "rank_by_p_champion": champ_rank,
                     "log_loss": round(-math.log(team_stats[champion]["p_champion"] / 100), 4)},
        "stage_brier_48_teams": stage_brier,
        "teams": teams,
        "matches": rows,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "scorecard_2026.json").write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n")
    (OUT_DIR / "scorecard_2026.md").write_text(render_md(result))
    print(render_md(result))


def render_md(r: dict) -> str:
    L = ["# 2026 World Cup scorecard", "",
         f"Frozen prediction `{r['frozen_tag']}` ({r['frozen_model']}) scored against all 104 results. "
         f"Generated {r['generated_at']} by `scripts/score_capsule_2026.py`.", ""]
    L += [f"- {n}" for n in r["notes"]] + [""]
    L += ["## Match-level, 3-way (lower log-loss and Brier are better)", "",
          "| Segment | Model | n | Log-loss | Brier | Accuracy |", "|---|---|---|---|---|---|"]
    for seg, d in r["summary"].items():
        for k, v in d.items():
            L.append(f"| {seg} | {k} | {v['n']} | {v['log_loss']:.4f} | {v['brier']:.4f} | {v['accuracy']:.1%} |")
    L += ["", "## Knockouts: who advanced (binary)", "",
          "| Model | Log-loss | Brier | Accuracy |", "|---|---|---|---|"]
    for k, v in r["knockout_advance"].items():
        L.append(f"| {k} | {v['log_loss']:.4f} | {v['brier']:.4f} | {v['accuracy']:.1%} |")
    L += ["", "## Reliability (all 312 outcome probabilities)", ""]
    for k, d in r["reliability"].items():
        L += [f"**{k}** (ECE {d['ece']:.4f})", "", "| Bin | n | Mean p | Observed |", "|---|---|---|---|"]
        L += [f"| {b['bin']} | {b['n']} | {b['mean_p']:.3f} | {b['observed']:.3f} |" for b in d["bins"]]
        L.append("")
    c = r["champion"]
    L += ["## Champion", "",
          f"{c['team']} won. Pre-tournament p_champion {c['p_champion']:.2f}% "
          f"(rank {c['rank_by_p_champion']} of 48), log-loss {c['log_loss']:.3f} "
          f"(uniform 1/48 would be {math.log(48):.3f}).", "",
          "Stage Brier across 48 teams (\"reached at least this stage\"): "
          + ", ".join(f"{k} {v:.4f}" for k, v in r["stage_brier_48_teams"].items()) + ".", "",
          "## Teams: champion odds vs stage reached", "",
          "| Team | p_champion % | Reached | P(reach that stage) | P(exactly that stage) |",
          "|---|---|---|---|---|"]
    L += [f"| {t['team']} | {t['p_champion']:.2f} | {t['stage_reached']} | "
          f"{t['p_reach_that_stage']:.1%} | {t['p_exactly_that_stage']:.1%} |" for t in r["teams"]]
    L += ["", "## Biggest model misses (3-way log-loss)", "",
          "| # | Stage | Match | Score | Model H/D/A | Log-loss |", "|---|---|---|---|---|---|"]
    for m in sorted(r["matches"], key=lambda x: -x["log_loss_model"])[:10]:
        p = m["model"]
        L.append(f"| {m['n']} | {m['stage']} | {m['home']} v {m['away']} | {m['score'][0]}-{m['score'][1]} | "
                 f"{p[0]:.2f}/{p[1]:.2f}/{p[2]:.2f} | {m['log_loss_model']:.3f} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
