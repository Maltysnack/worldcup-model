"""Model profiles: named K-factor + parameter sets that can be swapped
without breaking the v1 baseline outputs.

Select a profile via the WC_PROFILE environment variable:

    WC_PROFILE=v2-continental uv run worldcup-elo
    WC_PROFILE=v2-continental uv run worldcup-bracket
    WC_PROFILE=v2-continental uv run worldcup-simulate
    WC_PROFILE=v2-continental uv run worldcup-backtest 2018

When a non-default profile is active, outputs are suffixed so the
v1 files (data/bracket.json, predictions table, etc.) stay untouched.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from os import environ


@dataclass(frozen=True)
class Profile:
    name: str
    k_factors: dict[str, float] = field(default_factory=dict)
    default_k: float = 25.0
    # If set, the Elo walk only includes matches within this many days
    # before the prediction date. For backtests this is dynamic (relative
    # to the WC's kickoff). For forward predictions it's relative to today.
    # None = use the static historical cutoff (1990-01-01).
    min_date_offset_days: int | None = None


# Current production model. K-factors weighted from
#   World Cup final stage (60) down to friendlies (20).
V1_BASELINE = Profile(
    name="v1",
    k_factors={
        "FIFA World Cup": 60,
        "UEFA Euro": 50,
        "Copa América": 50,
        "African Cup of Nations": 50,
        "AFC Asian Cup": 50,
        "Gold Cup": 50,
        "Confederations Cup": 50,
        "Oceania Nations Cup": 45,
        "FIFA World Cup qualification": 40,
        "UEFA Nations League": 35,
        "CONCACAF Nations League": 35,
        "UEFA Euro qualification": 30,
        "Copa América qualification": 30,
        "African Cup of Nations qualification": 30,
        "AFC Asian Cup qualification": 30,
        "Friendly": 20,
    },
    default_k=25,
)

# Experimental: trust continental tournaments as much as World Cups,
# discount friendlies and qualifiers. The thinking: high-stakes finals
# stage matches are the cleanest signal of a team's true level, while
# friendlies are noisy (rotated XIs, no jeopardy) and qualifiers vary
# wildly in opponent quality.
V2_CONTINENTAL = Profile(
    name="v2-continental",
    k_factors={
        "FIFA World Cup": 75,
        "UEFA Euro": 75,
        "Copa América": 75,
        "African Cup of Nations": 75,
        "AFC Asian Cup": 75,
        "Gold Cup": 75,
        "Confederations Cup": 75,
        "Oceania Nations Cup": 60,
        "FIFA World Cup qualification": 20,
        "UEFA Nations League": 20,
        "CONCACAF Nations League": 20,
        "UEFA Euro qualification": 20,
        "Copa América qualification": 20,
        "African Cup of Nations qualification": 20,
        "AFC Asian Cup qualification": 20,
        "Friendly": 10,
    },
    default_k=15,
)

# Idea v3: tournaments matter more (3 magnitudes)
V3A_TOURNAMENT_MILD = Profile(
    name="v3a-tournament-mild",
    k_factors={
        "FIFA World Cup": 70, "UEFA Euro": 65, "Copa América": 65,
        "African Cup of Nations": 65, "AFC Asian Cup": 65, "Gold Cup": 65,
        "Confederations Cup": 65, "Oceania Nations Cup": 55,
        "FIFA World Cup qualification": 30,
        "UEFA Nations League": 30, "CONCACAF Nations League": 30,
        "UEFA Euro qualification": 25, "Copa América qualification": 25,
        "African Cup of Nations qualification": 25,
        "AFC Asian Cup qualification": 25,
        "Friendly": 15,
    },
    default_k=22,
)
V3B_TOURNAMENT_MOD = Profile(
    name="v3b-tournament-mod",
    k_factors={
        "FIFA World Cup": 80, "UEFA Euro": 75, "Copa América": 75,
        "African Cup of Nations": 75, "AFC Asian Cup": 75, "Gold Cup": 75,
        "Confederations Cup": 75, "Oceania Nations Cup": 60,
        "FIFA World Cup qualification": 20,
        "UEFA Nations League": 25, "CONCACAF Nations League": 25,
        "UEFA Euro qualification": 20, "Copa América qualification": 20,
        "African Cup of Nations qualification": 20,
        "AFC Asian Cup qualification": 20,
        "Friendly": 10,
    },
    default_k=18,
)
V3C_TOURNAMENT_EXTREME = Profile(
    name="v3c-tournament-extreme",
    k_factors={
        "FIFA World Cup": 100, "UEFA Euro": 90, "Copa América": 90,
        "African Cup of Nations": 90, "AFC Asian Cup": 90, "Gold Cup": 90,
        "Confederations Cup": 90, "Oceania Nations Cup": 70,
        "FIFA World Cup qualification": 10,
        "UEFA Nations League": 15, "CONCACAF Nations League": 15,
        "UEFA Euro qualification": 8, "Copa América qualification": 8,
        "African Cup of Nations qualification": 8,
        "AFC Asian Cup qualification": 8,
        "Friendly": 5,
    },
    default_k=10,
)

# Idea v4: flatten match-type weights (3 magnitudes)
V4A_FLAT_MILD = Profile(
    name="v4a-flat-mild",
    k_factors={
        "FIFA World Cup": 50, "UEFA Euro": 45, "Copa América": 45,
        "African Cup of Nations": 45, "AFC Asian Cup": 45, "Gold Cup": 45,
        "Confederations Cup": 45, "Oceania Nations Cup": 42,
        "FIFA World Cup qualification": 40,
        "UEFA Nations League": 40, "CONCACAF Nations League": 40,
        "UEFA Euro qualification": 38, "Copa América qualification": 38,
        "African Cup of Nations qualification": 38,
        "AFC Asian Cup qualification": 38,
        "Friendly": 30,
    },
    default_k=35,
)
V4B_FLAT_MOD = Profile(
    name="v4b-flat-mod",
    k_factors={
        "FIFA World Cup": 40, "UEFA Euro": 38, "Copa América": 38,
        "African Cup of Nations": 38, "AFC Asian Cup": 38, "Gold Cup": 38,
        "Confederations Cup": 38, "Oceania Nations Cup": 36,
        "FIFA World Cup qualification": 35,
        "UEFA Nations League": 35, "CONCACAF Nations League": 35,
        "UEFA Euro qualification": 33, "Copa América qualification": 33,
        "African Cup of Nations qualification": 33,
        "AFC Asian Cup qualification": 33,
        "Friendly": 30,
    },
    default_k=33,
)
V4C_FLAT_UNIFORM = Profile(
    name="v4c-flat-uniform",
    k_factors={
        "FIFA World Cup": 35, "UEFA Euro": 35, "Copa América": 35,
        "African Cup of Nations": 35, "AFC Asian Cup": 35, "Gold Cup": 35,
        "Confederations Cup": 35, "Oceania Nations Cup": 35,
        "FIFA World Cup qualification": 35,
        "UEFA Nations League": 35, "CONCACAF Nations League": 35,
        "UEFA Euro qualification": 35, "Copa América qualification": 35,
        "African Cup of Nations qualification": 35,
        "AFC Asian Cup qualification": 35,
        "Friendly": 35,
    },
    default_k=35,
)

# Idea v5: only recent matches count (3 cutoffs). K-factors stay at v1
# baseline to isolate the recency effect.
def _v5_kfactors():
    return dict(V1_BASELINE.k_factors)

V5A_RECENT_12 = Profile(
    name="v5a-recent-12",
    k_factors=_v5_kfactors(),
    default_k=25,
    min_date_offset_days=365,
)
V5B_RECENT_18 = Profile(
    name="v5b-recent-18",
    k_factors=_v5_kfactors(),
    default_k=25,
    min_date_offset_days=int(365 * 1.5),
)
V5C_RECENT_24 = Profile(
    name="v5c-recent-24",
    k_factors=_v5_kfactors(),
    default_k=25,
    min_date_offset_days=730,
)

PROFILES: dict[str, Profile] = {p.name: p for p in [
    V1_BASELINE, V2_CONTINENTAL,
    V3A_TOURNAMENT_MILD, V3B_TOURNAMENT_MOD, V3C_TOURNAMENT_EXTREME,
    V4A_FLAT_MILD, V4B_FLAT_MOD, V4C_FLAT_UNIFORM,
    V5A_RECENT_12, V5B_RECENT_18, V5C_RECENT_24,
]}


# The default profile gets the unsuffixed file/table names. It's the
# "production" model — what the website reads. Validated via the 20-
# iteration backtest (WC 2010-2018 + EUROs/Copa/AFCON/Asian Cup/Gold
# Cup since 2010, ~1334 matches): v4a-flat-mild scored 0.9567 log-loss
# vs v1's 0.9575. Tiny but real improvement, with much wider iteration
# confirming the low-flat-K region is genuinely better than alternatives.
DEFAULT_PROFILE_NAME = "v4a-flat-mild"


def current_profile() -> Profile:
    """Resolve the active profile from WC_PROFILE env var (default = v4a-flat-mild)."""
    name = environ.get("WC_PROFILE", DEFAULT_PROFILE_NAME)
    if name not in PROFILES:
        raise ValueError(
            f"Unknown WC_PROFILE: {name!r}. "
            f"Available: {sorted(PROFILES.keys())}"
        )
    return PROFILES[name]


def file_suffix(profile: Profile | None = None) -> str:
    """Suffix to append to JSON output filenames. Empty for the default."""
    p = profile or current_profile()
    return "" if p.name == DEFAULT_PROFILE_NAME else f"-{p.name}"


def table_suffix(profile: Profile | None = None) -> str:
    """Suffix to append to DB table names. Empty for the default.
    Hyphens in profile name become underscores for SQL identifier safety."""
    p = profile or current_profile()
    return "" if p.name == DEFAULT_PROFILE_NAME else f"_{p.name.replace('-', '_')}"
