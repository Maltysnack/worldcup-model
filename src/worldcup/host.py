"""Host-nation advantage layer.

Continental tournament hosts win their own tournament far more often
than a neutral-venue model predicts. Examples in our backtest where
the eventual champion ranked outside our top 5:

  AFCON 2024  Ivory Coast (host) - ranked #6
  AFC Asian Cup 2024  Qatar (host) - ranked #7
  AFC Asian Cup 2015  Australia (host) - ranked #7
  AFCON 2015  Ivory Coast (won; not host but co-confederation)

The pattern is consistent enough to address: hosts of continental
tournaments get a modest Elo bump.

For the 2026 World Cup, there are three hosts (USA, Mexico, Canada).
USA hosts ~78/104 matches; Mexico and Canada each host ~13. We give
USA a larger bonus than the co-hosts.
"""

from __future__ import annotations

# Bonus magnitudes (halved from hand-set +60/+50/+30 after the issue #5
# empirical sweep, scripts/sweep_host_bonus.py, found scale 0.5
# minimises pooled log-loss and Brier across the 16-tournament audit set.
# The previous +60 was slightly over-aggressive given the calibration
# regime introduced in issues #3/#10 (T=1.31 + neutral lambdas).
HOST_BONUS_SOLO = 30.0
HOST_BONUS_PRIMARY = 25.0
HOST_BONUS_SECONDARY = 15.0

# Tournament -> year -> {team: bonus}
TOURNAMENT_HOSTS: dict[tuple[str, int], dict[str, float]] = {
    # Continental hosts (verified facts, single-country except where noted)
    ("UEFA Euro", 2012): {"Poland": HOST_BONUS_PRIMARY, "Ukraine": HOST_BONUS_PRIMARY},
    ("UEFA Euro", 2016): {"France": HOST_BONUS_SOLO},
    ("UEFA Euro", 2021): {},  # Multi-city across Europe; no single host
    ("UEFA Euro", 2024): {"Germany": HOST_BONUS_SOLO},
    # Euro 2028: five UK and Ireland hosts; England stages most matches.
    ("UEFA Euro", 2028): {"England": HOST_BONUS_PRIMARY, "Scotland": HOST_BONUS_SECONDARY,
                          "Wales": HOST_BONUS_SECONDARY, "Republic of Ireland": HOST_BONUS_SECONDARY,
                          "Northern Ireland": HOST_BONUS_SECONDARY},

    ("Copa América", 2011): {"Argentina": HOST_BONUS_SOLO},
    ("Copa América", 2015): {"Chile": HOST_BONUS_SOLO},
    ("Copa América", 2016): {"United States": HOST_BONUS_SOLO},  # Centenario
    ("Copa América", 2019): {"Brazil": HOST_BONUS_SOLO},
    ("Copa América", 2021): {"Brazil": HOST_BONUS_SOLO},
    ("Copa América", 2024): {"United States": HOST_BONUS_SOLO},

    ("African Cup of Nations", 2010): {"Angola": HOST_BONUS_SOLO},
    ("African Cup of Nations", 2012): {
        "Equatorial Guinea": HOST_BONUS_PRIMARY,
        "Gabon": HOST_BONUS_PRIMARY,
    },
    ("African Cup of Nations", 2013): {"South Africa": HOST_BONUS_SOLO},
    ("African Cup of Nations", 2015): {"Equatorial Guinea": HOST_BONUS_SOLO},
    ("African Cup of Nations", 2017): {"Gabon": HOST_BONUS_SOLO},
    ("African Cup of Nations", 2019): {"Egypt": HOST_BONUS_SOLO},
    ("African Cup of Nations", 2022): {"Cameroon": HOST_BONUS_SOLO},
    ("African Cup of Nations", 2024): {"Ivory Coast": HOST_BONUS_SOLO},
    ("African Cup of Nations", 2026): {"Morocco": HOST_BONUS_SOLO},

    ("AFC Asian Cup", 2011): {"Qatar": HOST_BONUS_SOLO},
    ("AFC Asian Cup", 2015): {"Australia": HOST_BONUS_SOLO},
    ("AFC Asian Cup", 2019): {"United Arab Emirates": HOST_BONUS_SOLO},
    ("AFC Asian Cup", 2024): {"Qatar": HOST_BONUS_SOLO},

    # Gold Cup is normally USA (sometimes co-hosted with Mexico/Canada)
    ("Gold Cup", 2011): {"United States": HOST_BONUS_SOLO},
    ("Gold Cup", 2013): {"United States": HOST_BONUS_SOLO},
    ("Gold Cup", 2015): {"United States": HOST_BONUS_PRIMARY,
                         "Canada": HOST_BONUS_SECONDARY},
    ("Gold Cup", 2017): {"United States": HOST_BONUS_SOLO},
    ("Gold Cup", 2019): {"United States": HOST_BONUS_PRIMARY,
                         "Costa Rica": HOST_BONUS_SECONDARY,
                         "Jamaica": HOST_BONUS_SECONDARY},
    ("Gold Cup", 2021): {"United States": HOST_BONUS_SOLO},
    ("Gold Cup", 2023): {"United States": HOST_BONUS_PRIMARY,
                         "Canada": HOST_BONUS_SECONDARY},
    ("Gold Cup", 2025): {"United States": HOST_BONUS_PRIMARY,
                         "Canada": HOST_BONUS_SECONDARY},

    # World Cups
    ("FIFA World Cup", 2002): {"South Korea": HOST_BONUS_PRIMARY,
                                "Japan": HOST_BONUS_PRIMARY},
    ("FIFA World Cup", 2006): {"Germany": HOST_BONUS_SOLO},
    ("FIFA World Cup", 2010): {"South Africa": HOST_BONUS_SOLO},
    ("FIFA World Cup", 2014): {"Brazil": HOST_BONUS_SOLO},
    ("FIFA World Cup", 2018): {"Russia": HOST_BONUS_SOLO},
    # 2030: Spain, Portugal, Morocco host the tournament; Argentina, Uruguay
    # and Paraguay stage one centenary match each, so get no bonus.
    ("FIFA World Cup", 2030): {"Spain": HOST_BONUS_PRIMARY, "Portugal": HOST_BONUS_PRIMARY,
                                "Morocco": HOST_BONUS_PRIMARY},
    ("FIFA World Cup", 2026): {
        "United States": HOST_BONUS_PRIMARY,
        "Mexico": HOST_BONUS_SECONDARY,
        "Canada": HOST_BONUS_SECONDARY,
    },
}


def host_bonus(team: str, tournament: str, year: int) -> float:
    """Return the Elo bonus for `team` if they're a host of (tournament, year).
    Returns 0.0 if not a host or if the (tournament, year) isn't known."""
    return TOURNAMENT_HOSTS.get((tournament, year), {}).get(team, 0.0)


def apply_host_bonus(
    ratings: dict[str, float],
    tournament: str,
    year: int,
) -> dict[str, float]:
    """Add the host bonus to each team's rating. Non-hosts unchanged."""
    return {t: r + host_bonus(t, tournament, year) for t, r in ratings.items()}
