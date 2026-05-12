"""Manually-curated list of pre-WC 2026 friendlies between qualified teams.

martj42's dataset doesn't include scheduled-but-not-played friendlies.
We compile the public schedule here so the website's "Next match" card
shows them in the run-up to kickoff.

Only includes matches where BOTH teams are in the WC 2026 field; other
warm-up friendlies (vs non-qualified opponents) are skipped.

Source: public news + FIFA + national team announcements as of mid
May 2026. Update as more get scheduled.
"""

from __future__ import annotations

from datetime import date

# Each entry mirrors the shape of a row from the matches table that
# the rest of the pipeline expects.
UPCOMING_FRIENDLIES: list[dict] = [
    {
        "date": date(2026, 5, 22),
        "home_team": "Mexico",
        "away_team": "Ghana",
        "neutral": False,
        "city": "Puebla",
        "country": "Mexico",
    },
    {
        "date": date(2026, 5, 30),
        "home_team": "Mexico",
        "away_team": "Australia",
        "neutral": True,
        "city": "Pasadena",
        "country": "United States",
    },
    {
        "date": date(2026, 5, 31),
        "home_team": "United States",
        "away_team": "Senegal",
        "neutral": False,
        "city": "Charlotte",
        "country": "United States",
    },
    {
        "date": date(2026, 6, 6),
        "home_team": "United States",
        "away_team": "Germany",
        "neutral": False,
        "city": "Chicago",
        "country": "United States",
    },
    {
        "date": date(2026, 6, 6),
        "home_team": "England",
        "away_team": "New Zealand",
        "neutral": True,
        "city": "Miami",
        "country": "United States",
    },
    {
        "date": date(2026, 6, 6),
        "home_team": "Brazil",
        "away_team": "Egypt",
        "neutral": True,
        "city": None,
        "country": None,
    },
]
