"""Layer sets: which boosts get applied on top of raw Elo.

Two layer sets are defined:

* ``full`` (default) — adds form, NT trophy, and host bonuses on top of
  the squad-value-blended Elo. Optimised for *picking the champion*. The
  host bonus moves the headline pick toward the host nation, which is
  empirically right more often than not but slightly overconfident on
  per-match probabilities for host games.

* ``calibration`` — drops the host bonus only. Form and NT trophies stay.
  Optimised for *honest per-match probabilities*. Validated overnight on
  16 historical tournaments: lower per-match log-loss (0.9797 vs 0.9828)
  and lower ECE (0.091 vs 0.093) than ``full``, at the cost of one fewer
  top-pick hit (4/16 vs 5/16) and a marginally worse champion rank.

Select via the ``WC_LAYERS`` env var:

    WC_LAYERS=calibration uv run worldcup-bracket
    WC_LAYERS=calibration uv run worldcup-simulate

When a non-default layer set is active, output filenames get a suffix so
the headline ``bracket.json`` / ``simulation.json`` files (read by the
website's main bracket / champion-finder views) stay untouched.
"""

from __future__ import annotations

from dataclasses import dataclass
from os import environ


@dataclass(frozen=True)
class LayerSet:
    name: str
    form: bool       # 12-month residual goal-difference bump
    nt: bool         # national-team trophy density bonus
    host: bool       # host-nation bump


FULL = LayerSet(name="full", form=True, nt=True, host=True)
CALIBRATION = LayerSet(name="calibration", form=True, nt=True, host=False)

LAYER_SETS: dict[str, LayerSet] = {ls.name: ls for ls in [FULL, CALIBRATION]}

DEFAULT_LAYER_SET_NAME = "full"


def current_layer_set() -> LayerSet:
    """Resolve the active layer set from WC_LAYERS env var (default = full)."""
    name = environ.get("WC_LAYERS", DEFAULT_LAYER_SET_NAME)
    if name not in LAYER_SETS:
        raise ValueError(
            f"Unknown WC_LAYERS: {name!r}. Available: {sorted(LAYER_SETS.keys())}"
        )
    return LAYER_SETS[name]


def file_suffix(layer_set: LayerSet | None = None) -> str:
    """Filename suffix. Empty for the default; ``-calibration`` otherwise."""
    ls = layer_set or current_layer_set()
    return "" if ls.name == DEFAULT_LAYER_SET_NAME else f"-{ls.name}"


def table_suffix(layer_set: LayerSet | None = None) -> str:
    """SQL identifier suffix. Hyphens become underscores."""
    ls = layer_set or current_layer_set()
    return "" if ls.name == DEFAULT_LAYER_SET_NAME else f"_{ls.name.replace('-', '_')}"


def apply_layers(
    ratings: dict[str, float],
    *,
    tournament: str,
    year: int,
    layer_set: LayerSet | None = None,
) -> dict[str, float]:
    """Apply the active (or specified) layer set on top of raw ratings.

    Form / NT / host bonuses are toggled by the layer set's flags. Inputs
    are *post*-squad-blend ratings — the squad-value blend is treated as
    a separate signal upstream and is not part of the layer-set machinery
    (it would leak future info if applied historically anyway).
    """
    ls = layer_set or current_layer_set()
    boosted = dict(ratings)
    if ls.form:
        from worldcup.form import compute_form_bumps
        bumps = compute_form_bumps()
        for t in boosted:
            boosted[t] += bumps.get(t, 0.0)
    if ls.nt:
        from worldcup.trophies import nt_trophy_bonus
        for t in boosted:
            boosted[t] += nt_trophy_bonus(t, as_of_year=year)
    if ls.host:
        from worldcup.host import host_bonus
        for t in boosted:
            boosted[t] += host_bonus(t, tournament, year)
    return boosted
