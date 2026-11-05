"""Geocoded host venues for the 2026 FIFA World Cup.

A fixed dict, public coordinates from each stadium's Wikipedia page,
elevation rounded to the nearest 10 m. Used downstream to derive
altitude and travel-distance covariates (sub-issues of the deleted
match-state #6).

Each entry is keyed on city name as it appears in `data/raw/wc2026_matches.csv`
(check upstream to keep keys in sync). Values are (lat, lon, elev_m, country,
stadium).

Haversine helper provided for travel distance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Venue:
    city: str
    country: str
    stadium: str
    lat: float
    lon: float
    elevation_m: int


# 16 host cities for WC 2026 (USA × 11, Mexico × 3, Canada × 2).
VENUES: dict[str, Venue] = {
    # ── United States ────────────────────────────────────────────────
    "Atlanta":        Venue("Atlanta",        "United States", "Mercedes-Benz Stadium",       33.7553,  -84.4006,  320),
    "Boston":         Venue("Boston",         "United States", "Gillette Stadium",            42.0909,  -71.2643,   40),  # Foxborough
    "Dallas":         Venue("Dallas",         "United States", "AT&T Stadium",                32.7473,  -97.0945,  170),  # Arlington
    "Houston":        Venue("Houston",        "United States", "NRG Stadium",                 29.6847,  -95.4107,   10),
    "Kansas City":    Venue("Kansas City",    "United States", "Arrowhead Stadium",           39.0489,  -94.4839,  270),
    "Los Angeles":    Venue("Los Angeles",    "United States", "SoFi Stadium",                33.9535, -118.3392,   30),  # Inglewood
    "Miami":          Venue("Miami",          "United States", "Hard Rock Stadium",           25.9580,  -80.2389,    0),  # Miami Gardens
    "New York":       Venue("New York",       "United States", "MetLife Stadium",             40.8135,  -74.0744,   10),  # East Rutherford NJ
    "Philadelphia":   Venue("Philadelphia",   "United States", "Lincoln Financial Field",     39.9008,  -75.1675,    0),
    "San Francisco":  Venue("San Francisco",  "United States", "Levi's Stadium",              37.4030, -121.9698,    0),  # Santa Clara
    "Seattle":        Venue("Seattle",        "United States", "Lumen Field",                 47.5952, -122.3316,   10),
    # ── Mexico ───────────────────────────────────────────────────────
    "Guadalajara":    Venue("Guadalajara",    "Mexico",        "Estadio Akron",               20.6818, -103.4625, 1560),
    "Mexico City":    Venue("Mexico City",    "Mexico",        "Estadio Azteca",              19.3029,  -99.1505, 2240),
    "Monterrey":      Venue("Monterrey",      "Mexico",        "Estadio BBVA",                25.6692, -100.2447,  490),
    # ── Canada ───────────────────────────────────────────────────────
    "Toronto":        Venue("Toronto",        "Canada",        "BMO Field",                   43.6332,  -79.4185,   80),
    "Vancouver":      Venue("Vancouver",      "Canada",        "BC Place",                    49.2768, -123.1119,    0),
}


_EARTH_R_KM = 6371.0


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two points."""
    p1 = math.radians(lat1); p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1); dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * _EARTH_R_KM * math.asin(math.sqrt(a))


def distance_between(city_a: str, city_b: str) -> float | None:
    """Great-circle km between two host cities. Returns None if either is unknown."""
    a = VENUES.get(city_a); b = VENUES.get(city_b)
    if a is None or b is None:
        return None
    return haversine_km(a.lat, a.lon, b.lat, b.lon)


def elevation_m(city: str) -> int | None:
    v = VENUES.get(city)
    return v.elevation_m if v else None
