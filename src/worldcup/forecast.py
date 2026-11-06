"""One-command tournament forecast: current Elo plus a small validated overlay.

Run: uv run worldcup-forecast <tournament> [--as-of YYYY-MM-DD] [--sims N]
                                           [--on LAYER ...] [--off LAYER ...]
Tournaments come from worldcup.registry. Writes data/forecast_<key>.json.

Every layer is computed from the match database (or the public honours list
in worldcup.trophies); nothing is hand-keyed per tournament except host and
altitude lookups. The squad market-value blend and club-trophy score are NOT
in the live path: they were hand-keyed for 2026 and never backtested.

Default layer flags (LAYERS below) follow the evidence:
  nt        ON.  Helped most in the 16-tournament backtest; in the 2026
            ablation, removing it cost the most log-loss (+0.0102).
  altitude  ON.  Backed by an n=867 sweep; only ever nonzero for a host
            playing at altitude, so it is a no-op elsewhere.
  host      ON.  Lifts the eventual champion's rank in the backtest but costs a
            little match log-loss; this tool forecasts champions, and on 2026
            it was neutral (leave-one-out change -0.0004).
  form      OFF. Mixed in the backtest, and on 2026 removing it improved
            log-loss (0.8481 to 0.8443).
Caution: on the 2026 ablation (n=104) every layer interval includes zero, so
none of this is proven. Flip flags with --on / --off as new tournaments score.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import date, timedelta

import numpy as np

from worldcup.altitude import apply_altitude_bonus
from worldcup.bracket import derive_groups_from_fixtures, label_groups_by_fifa
from worldcup.db import PROJECT_ROOT, connect
from worldcup.elo import run as elo_run
from worldcup.form import apply_form
from worldcup.host import apply_host_bonus
from worldcup.registry import REGISTRY
from worldcup.trophies import apply_nt_bonus

LAYERS = {"form": False, "nt": True, "host": True, "altitude": True}
ACTIVE_YEARS = 4  # teams with no match in this window are dropped


def apply_overlay(raw: dict[str, float], t: dict, as_of: date, layers: dict[str, bool]) -> dict[str, float]:
    """Apply the enabled layers in the simulator's fixed order. Pure given the DB."""
    unknown = set(layers) - set(LAYERS)
    assert not unknown, f"unknown layers {unknown}"
    r, comp, year = dict(raw), t["competition"], t["year"]
    if layers.get("form"):
        r = apply_form(r, as_of_date=as_of)  # as_of stops later results leaking in
    if layers.get("nt"):
        r = apply_nt_bonus(r, as_of_year=as_of.year)
    if layers.get("host"):
        r = apply_host_bonus(r, comp, year)
    if layers.get("altitude"):
        r = apply_altitude_bonus(r, comp, year)
    return r


def raw_elo(as_of: date) -> dict[str, float]:
    df, _ = elo_run(max_date=as_of, quiet=True)
    with connect(read_only=True) as con:
        recent = {t for (t,) in con.execute(
            "SELECT home_team FROM matches WHERE date BETWEEN ? AND ? UNION "
            "SELECT away_team FROM matches WHERE date BETWEEN ? AND ?",
            [as_of - timedelta(days=ACTIVE_YEARS * 365), as_of] * 2).fetchall()}
    return {t: r for t, r in zip(df["team"], df["rating"]) if t in recent}


def simulate_fixtures(t: dict, ratings: dict[str, float], n_sims: int, seed: int) -> list[dict]:
    """WC 2026: real draw and official bracket, via the frozen simulator's engine."""
    from worldcup.simulator import simulate_one
    with connect(read_only=True) as con:
        fx = con.execute(
            "SELECT date, home_team, away_team, neutral FROM matches "
            "WHERE tournament = ? AND date >= ? ORDER BY date, home_team LIMIT ?",
            [t["competition"], t["kickoff"], t["groups"] * 6]).fetchdf()  # group games come first; knockouts would merge groups
    groups = label_groups_by_fifa(derive_groups_from_fixtures(fx))
    by_group: dict[str, list] = defaultdict(list)
    for row in fx.itertuples(index=False):
        for letter, members in groups.items():
            if row.home_team in members and row.away_team in members:
                by_group[letter].append({"home_team": row.home_team, "away_team": row.away_team,
                                         "neutral": bool(row.neutral)})
                break
    rng = np.random.default_rng(seed)
    counts: dict[str, Counter] = defaultdict(Counter)
    for _ in range(n_sims):
        rounds, _ = simulate_one(ratings, groups, by_group, rng)
        for team, rnd in rounds.items():
            counts[team][rnd] += 1
    out = []
    for team, c in counts.items():
        n = sum(c.values())
        out.append({"team": team, "p_champion": round(c["champion"] / n, 4),
                    "p_final": round((c["champion"] + c["runner_up"]) / n, 4),
                    "rating_now": round(ratings[team])})
    return sorted(out, key=lambda x: -x["p_champion"])


def simulate_future(t: dict, ratings: dict[str, float], as_of: date, seed: int) -> list[dict]:
    """Every team that won a simulated tournament, so an Elo-only run can be joined on the overlay's top teams."""
    from worldcup import title_odds as to
    from worldcup.drift import rating_series
    conf = {k: c for k, c in to.confederations(as_of - timedelta(days=ACTIVE_YEARS * 365)).items() if k in ratings}
    # shortcut: overlay is added before the drift draw, so drift sees overlay-shifted ratings; fine while bonuses are small.
    return to.odds_for(t, rating_series(), ratings, conf, as_of, seed, n_top=len(ratings))["top"]


def layer_deltas(raw: dict[str, float], t: dict, as_of: date, layers: dict[str, bool]) -> dict[str, dict[str, float]]:
    """Each enabled layer's own Elo effect per team, zero effects left out."""
    off = {k: False for k in LAYERS}
    out: dict[str, dict[str, float]] = {team: {} for team in raw}
    for k in (k for k, on in layers.items() if on):
        for team, r in apply_overlay(raw, t, as_of, {**off, k: True}).items():
            if round(r - raw[team]):
                out[team][k] = round(r - raw[team])
    return out


def forecast(key: str, as_of: date | None = None, layers: dict[str, bool] | None = None,
             n_sims: int = 10_000, seed: int = 42) -> dict:
    t = REGISTRY[key]
    if as_of is None:
        as_of = t["kickoff"] - timedelta(days=1) if t["draw"] == "fixtures" else date.today()
    layers = {**LAYERS, **(layers or {})}
    raw = raw_elo(as_of)
    simulate = (lambda r: simulate_fixtures(t, r, n_sims, seed)) if t["draw"] == "fixtures" \
        else (lambda r: simulate_future(t, r, as_of, seed))
    elo_only = {x["team"]: x["p_champion"] for x in simulate(raw)}
    deltas = layer_deltas(raw, t, as_of, layers)
    teams = simulate(apply_overlay(raw, t, as_of, layers))[:12]
    for x in teams:
        x.update(p_champion_elo=elo_only.get(x["team"], 0.0), elo=round(raw[x["team"]]), layers=deltas[x["team"]])
    return {"tournament": key, "name": t["name"], "as_of": as_of.isoformat(), "layers": layers,
            "assumptions": t["assumptions"], "teams": teams}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("tournament", choices=sorted(REGISTRY))
    ap.add_argument("--as-of", type=date.fromisoformat)
    ap.add_argument("--sims", type=int, default=10_000)
    ap.add_argument("--on", nargs="*", default=[], choices=sorted(LAYERS))
    ap.add_argument("--off", nargs="*", default=[], choices=sorted(LAYERS))
    a = ap.parse_args()
    layers = {**{k: True for k in a.on}, **{k: False for k in a.off}}
    out = forecast(a.tournament, a.as_of, layers, a.sims)
    path = PROJECT_ROOT / "data" / f"forecast_{a.tournament}.json"
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print(out["name"], "as of", out["as_of"], "layers", {k: v for k, v in out["layers"].items() if v})
    for x in out["teams"][:8]:
        print(f"  {x['team']:22s} {x['p_champion'] * 100:5.1f}%  Elo only {x['p_champion_elo'] * 100:5.1f}%")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
