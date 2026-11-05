"""Elo drift: how far a team's rating moves over h years, by starting rating.

Fitted as delta = a + b*rating + N(0, sd) on rating moves since 1990.
Validated in research/drift_model.py: beats frozen ratings
on 3 of 4 World Cup hindcasts from ~3.5 years out. Used by title_odds.py.
"""

from __future__ import annotations

from datetime import date

import numpy as np

from worldcup.db import connect
from worldcup.tournament_backtest import _PRED_TABLE

FIT_FROM = 1990


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
