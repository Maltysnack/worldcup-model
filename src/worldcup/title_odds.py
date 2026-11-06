"""Title odds for future tournaments: Euro 2028, Copa America 2028, WC 2030.

Each simulation draws every team's rating at kickoff from the Elo drift
model (worldcup.drift), picks the field from those ratings (qualifying),
draws groups by pots, and plays the tournament with the goals model.
All matches neutral (no host bonus, same as the calibration variant).

shortcut: qualifying = best-rated teams per confederation slot, not real
qualifying groups; knockout = seeded bracket (winners meet thirds), not the
official pairing tables. Upgrade if a published bracket or qualifying draw
makes the difference worth it.

Run: uv run python -m worldcup.title_odds   (writes data/title_odds.json)
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from worldcup.db import connect
from worldcup.drift import fit_drift, rating_series
from worldcup.goals import sim_match
from worldcup.registry import REGISTRY

OUT = Path(__file__).resolve().parents[2] / "data" / "title_odds.json"
N_DRAWS, SIMS_PER_DRAW = 400, 25

CONMEBOL = {"Argentina", "Bolivia", "Brazil", "Chile", "Colombia", "Ecuador", "Paraguay", "Peru", "Uruguay", "Venezuela"}
# Tournaments only one confederation's members play; a team's confederation is where most of its recent games are.
MARKERS = {
    "UEFA": ("UEFA Nations League", "UEFA Euro qualification", "UEFA Euro"),
    "CAF": ("African Cup of Nations qualification", "African Cup of Nations", "COSAFA Cup", "CECAFA Cup"),
    "AFC": ("AFC Asian Cup qualification", "AFC Asian Cup", "AFF Championship", "ASEAN Championship", "Gulf Cup",
            "SAFF Cup", "EAFF Championship", "WAFF Championship"),
    "CONCACAF": ("CONCACAF Nations League", "CONCACAF Nations League qualification", "Gold Cup",
                 "Gold Cup qualification", "CFU Caribbean Cup qualification"),
    "OFC": ("Oceania Nations Cup", "OFC Nations Cup"),
}

# Formats live in the registry; only tournaments without a real draw are simulated here.
TOURNAMENTS = [t for t in REGISTRY.values() if t["draw"] == "simulated"]


def confederations(since: date) -> dict[str, str]:
    """Active teams (played since `since`) mapped to a confederation."""
    with connect(read_only=True) as con:
        rows = con.execute("""
            SELECT team, tournament, count(*) FROM (
              SELECT home_team AS team, tournament FROM matches WHERE date >= ?
              UNION ALL SELECT away_team, tournament FROM matches WHERE date >= ?)
            GROUP BY 1, 2""", [since, since]).fetchall()
    tally: dict[str, Counter] = {}
    for team, tour, n in rows:
        for conf, names in MARKERS.items():
            if tour in names:
                tally.setdefault(team, Counter())[conf] += n
    out = {t: c.most_common(1)[0][0] for t, c in tally.items()}
    out.update({t: "CONMEBOL" for t in CONMEBOL})
    return out


def pick_field(t: dict, ratings: dict[str, float], conf: dict[str, str]) -> list[str]:
    """Best-rated teams per confederation slot; hosts take places inside their confederation."""
    field: list[str] = []
    by_conf: dict[str, list[str]] = {}
    for team, c in conf.items():
        if team in ratings:
            by_conf.setdefault(c, []).append(team)
    hosts = sorted(t["hosts"], key=lambda h: -ratings[h])
    if t["key"] == "euro2028":
        field = sorted(by_conf["UEFA"], key=lambda x: -ratings[x])[:24]
        # Fewer than host_places hosts got in: the best missing hosts replace the weakest non-hosts.
        need = max(0, t["host_places"] - sum(h in field for h in hosts))
        if need:
            out = [x for x in field if x not in hosts][-need:]
            field = [x for x in field if x not in out] + [h for h in hosts if h not in field][:need]
        return field
    rest: list[str] = []
    for c, n in t["slots"].items():
        if c == "PLAYOFF":
            continue
        auto = [h for h in hosts if conf.get(h) == c]
        others = sorted((x for x in by_conf.get(c, []) if x not in auto), key=lambda x: -ratings[x])
        take = max(0, n - len(auto))
        field += auto + others[:take]
        if c != "UEFA":
            rest += others[take:]
    field += sorted(rest, key=lambda x: -ratings[x])[: t["slots"].get("PLAYOFF", 0)]
    return field


def play(field: list[str], ratings: dict[str, float], n_groups: int, best_thirds: int,
         rng: np.random.Generator) -> tuple[str, str]:
    """One tournament. Returns (champion, runner-up)."""
    assert len(field) == 4 * n_groups, (len(field), n_groups)
    pots = sorted(field, key=lambda x: -ratings[x])
    groups = [[] for _ in range(n_groups)]
    for p in range(4):
        pot = pots[p * n_groups:(p + 1) * n_groups]
        for g, team in zip(groups, rng.permutation(pot)):
            g.append(team)
    places = []  # (finish position, -pts, -gd, -gf, team)
    for g in groups:
        pts, gd, gf = Counter(), Counter(), Counter()
        for i in range(4):
            for j in range(i + 1, 4):
                a, b = g[i], g[j]
                sa, sb, w = sim_match(ratings[a], ratings[b], neutral=True, ko=False, rng=rng)
                pts[a] += 3 if w == 0 else 1 if w == -1 else 0
                pts[b] += 3 if w == 1 else 1 if w == -1 else 0
                gd[a] += sa - sb; gd[b] += sb - sa; gf[a] += sa; gf[b] += sb
        order = sorted(g, key=lambda x: (pts[x], gd[x], gf[x], rng.random()), reverse=True)
        places += [(k, -pts[x], -gd[x], -gf[x], x) for k, x in enumerate(order[:3])]
    places.sort()
    seeded = [p[-1] for p in places if p[0] < 2] + [p[-1] for p in places if p[0] == 2][:best_thirds]
    # Seeded bracket: 1 v N, 2 v N-1, ... laid out so top seeds meet late.
    n = len(seeded)
    slots = [0]
    while len(slots) < n:
        m = len(slots) * 2
        slots = [s for x in slots for s in (x, m - 1 - x)]
    alive = [seeded[s] for s in slots]
    while len(alive) > 2:
        alive = [a if sim_match(ratings[a], ratings[b], neutral=True, ko=True, rng=rng)[2] == 0 else b
                 for a, b in zip(alive[::2], alive[1::2])]
    a, b = alive
    return (a, b) if sim_match(ratings[a], ratings[b], neutral=True, ko=True, rng=rng)[2] == 0 else (b, a)


def odds_for(t: dict, series, base: dict[str, float], conf: dict[str, str], as_of: date, seed: int) -> dict:
    fit = fit_drift(series, (t["kickoff"] - as_of).days / 365.25, as_of)
    rng = np.random.default_rng(seed)
    champ, final, qual = Counter(), Counter(), Counter()
    teams = list(base)
    r0 = np.array([base[x] for x in teams])
    for _ in range(N_DRAWS):
        drawn = dict(zip(teams, r0 + fit["a"] + fit["b"] * r0 + fit["sd"] * rng.standard_normal(len(teams))))
        field = pick_field(t, drawn, conf)
        qual.update(field)
        for _ in range(SIMS_PER_DRAW):
            w, r = play(field, drawn, t["groups"], t["best_thirds"], rng)
            champ[w] += 1
            final[w] += 1
            final[r] += 1
    n = N_DRAWS * SIMS_PER_DRAW
    top = sorted(champ, key=lambda x: -champ[x])[:12]
    return {"key": t["key"], "name": t["name"], "kickoff": t["kickoff"].isoformat(),
            "years_out": round((t["kickoff"] - as_of).days / 365.25, 1), "n_teams": len(field),
            "assumptions": t["assumptions"], "n_sims": n,
            "top": [{"team": x, "p_champion": round(champ[x] / n, 4), "p_final": round(final[x] / n, 4),
                     "p_qualify": round(qual[x] / N_DRAWS, 3), "rating_now": round(base[x])} for x in top]}


def main() -> None:
    with connect(read_only=True) as con:
        as_of = con.execute("SELECT max(date) FROM matches").fetchone()[0]
        base = dict(con.execute("SELECT team, rating FROM elo_current WHERE last_match >= ?",
                                [as_of - timedelta(days=4 * 365)]).fetchall())
    conf = {t: c for t, c in confederations(as_of - timedelta(days=4 * 365)).items() if t in base}
    series = rating_series()
    out = {"generated_at": date.today().isoformat(), "as_of": as_of.isoformat(), "model": "elo-v4a-flat-mild + drift",
           "tournaments": [odds_for(t, series, base, conf, as_of, seed=i) for i, t in enumerate(TOURNAMENTS)]}
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    for t in out["tournaments"]:
        print(t["name"], ", ".join(f"{x['team']} {x['p_champion'] * 100:.1f}%" for x in t["top"][:5]))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
