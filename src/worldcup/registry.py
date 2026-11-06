"""Tournament registry: format, hosts and altitude venues, one entry per tournament.

`draw` says where the field comes from: "fixtures" (the real draw is in the
match database, used for WC 2026) or "simulated" (qualifying and draw are
modelled from ratings, see worldcup.title_odds). Host bonuses and altitude
bonuses are looked up by (competition, year) in worldcup.host and
worldcup.altitude; `altitude_venues` here only records which venues are high.

Add a tournament by adding an entry; when its real draw is made, switch
`draw` to "fixtures" and make sure its group fixtures are in `matches`.
"""

from __future__ import annotations

from datetime import date

REGISTRY: dict[str, dict] = {t["key"]: t for t in [
    {"key": "wc2026", "name": "World Cup 2026", "competition": "FIFA World Cup", "year": 2026,
     "kickoff": date(2026, 6, 11), "draw": "fixtures", "groups": 12, "best_thirds": 8,
     "slots": {}, "hosts": ["United States", "Mexico", "Canada"], "host_places": 3,
     "altitude_venues": ["Mexico City", "Guadalajara"],
     "assumptions": "48 teams, 12 groups of 4, top 2 and 8 best thirds to the last 32, official FIFA bracket."},
    {"key": "euro2028", "name": "Euro 2028", "competition": "UEFA Euro", "year": 2028,
     "kickoff": date(2028, 6, 9), "draw": "simulated", "groups": 6, "best_thirds": 4,
     "slots": {"UEFA": 24}, "hosts": ["England", "Scotland", "Wales", "Republic of Ireland", "Northern Ireland"],
     "host_places": 2, "altitude_venues": [],
     "assumptions": "24 teams, 6 groups of 4, top 2 and 4 best thirds to the last 16. Two places for the best UK and Ireland hosts that miss out."},
    {"key": "copa2028", "name": "Copa America 2028", "competition": "Copa América", "year": 2028,
     "kickoff": date(2028, 6, 15), "draw": "simulated", "groups": 4, "best_thirds": 0,
     "slots": {"CONMEBOL": 10, "CONCACAF": 6}, "hosts": [], "host_places": 0, "altitude_venues": [],
     "assumptions": "Host and format not confirmed. Assumed 16 teams (all 10 CONMEBOL plus 6 CONCACAF), 4 groups of 4, top 2 to the quarter-finals."},
    {"key": "wc2030", "name": "World Cup 2030", "competition": "FIFA World Cup", "year": 2030,
     "kickoff": date(2030, 6, 13), "draw": "simulated", "groups": 12, "best_thirds": 8,
     "slots": {"UEFA": 16, "CAF": 9, "AFC": 8, "CONMEBOL": 6, "CONCACAF": 6, "OFC": 1, "PLAYOFF": 2},
     "hosts": ["Spain", "Portugal", "Morocco", "Argentina", "Uruguay", "Paraguay"], "host_places": 6,
     "altitude_venues": [],
     "assumptions": "48 teams, 12 groups of 4, top 2 and 8 best thirds to the last 32. Six hosts qualify automatically inside their confederation's places; 2026 place allocation assumed; 2 play-off places go to the best remaining non-UEFA teams."},
]}
