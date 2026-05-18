"""Fetch club crest URLs via the Wikipedia REST summary API.

For each club appearing in the WC rosters we resolve the club's main
Wikipedia article (not the per-season subpage) and pull the thumbnail
URL — for football club articles this is reliably the club crest.

Output cached to data/raw/club_badges.json so re-runs are cheap. The
top-scorer module reads this file and surfaces badges in the UI in
place of the country flag.

URL: https://en.wikipedia.org/api/rest_v1/page/summary/{PageName}
Returns JSON with `thumbnail.source` pointing to a hosted PNG.

Run: `uv run worldcup-fetch-club-badges`
     `uv run worldcup-fetch-club-badges --refresh`
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
from datetime import datetime, timezone

import requests

from worldcup.db import PROJECT_ROOT
from worldcup.wiki_rosters import fetch_and_parse as fetch_rosters

USER_AGENT = "maltysnack-worldcup-hobby/0.1 (https://github.com/Maltysnack/worldcup)"
CACHE_PATH = PROJECT_ROOT / "data" / "raw" / "club_badges.json"
FETCH_DELAY_S = 0.4


# Wikipedia article name for each WC-roster club name. Add more as
# new clubs appear. Keys are the club names produced by wiki_rosters.py;
# values are the Wikipedia page slug (URL-encoded if needed).
CLUB_PAGE_NAME: dict[str, str] = {
    # Premier League
    "Manchester City": "Manchester_City_F.C.",
    "Manchester United": "Manchester_United_F.C.",
    "Liverpool": "Liverpool_F.C.",
    "Arsenal": "Arsenal_F.C.",
    "Chelsea": "Chelsea_F.C.",
    "Tottenham Hotspur": "Tottenham_Hotspur_F.C.",
    "Aston Villa": "Aston_Villa_F.C.",
    "Newcastle United": "Newcastle_United_F.C.",
    "Nottingham Forest": "Nottingham_Forest_F.C.",
    "Brighton & Hove Albion": "Brighton_%26_Hove_Albion_F.C.",
    "Brentford": "Brentford_F.C.",
    "Crystal Palace": "Crystal_Palace_F.C.",
    "Bournemouth": "AFC_Bournemouth",
    "Everton": "Everton_F.C.",
    "Fulham": "Fulham_F.C.",
    "West Ham United": "West_Ham_United_F.C.",
    "Wolverhampton Wanderers": "Wolverhampton_Wanderers_F.C.",
    "Leicester City": "Leicester_City_F.C.",
    "Sunderland": "Sunderland_A.F.C.",
    "Burnley": "Burnley_F.C.",
    "Ipswich Town": "Ipswich_Town_F.C.",
    "Leeds United": "Leeds_United_F.C.",
    "Southampton": "Southampton_F.C.",
    # La Liga
    "Barcelona": "FC_Barcelona",
    "Real Madrid": "Real_Madrid_CF",
    "Atlético Madrid": "Atl%C3%A9tico_Madrid",
    "Sevilla": "Sevilla_FC",
    "Valencia": "Valencia_CF",
    "Real Betis": "Real_Betis",
    "Athletic Bilbao": "Athletic_Bilbao",
    "Real Sociedad": "Real_Sociedad",
    "Villarreal": "Villarreal_CF",
    "Girona": "Girona_FC",
    "Getafe": "Getafe_CF",
    "Osasuna": "CA_Osasuna",
    "Celta Vigo": "RC_Celta_de_Vigo",
    "Mallorca": "RCD_Mallorca",
    # Serie A
    "Inter Milan": "Inter_Milan",
    "Internazionale": "Inter_Milan",
    "Juventus": "Juventus_FC",
    "AC Milan": "AC_Milan",
    "Napoli": "SSC_Napoli",
    "Roma": "AS_Roma",
    "AS Roma": "AS_Roma",
    "Lazio": "S.S._Lazio",
    "Atalanta": "Atalanta_BC",
    "Fiorentina": "ACF_Fiorentina",
    "Bologna": "Bologna_F.C._1909",
    "Torino": "Torino_FC",
    "Udinese": "Udinese_Calcio",
    "Como": "Como_1907",
    "Parma": "Parma_Calcio_1913",
    "Genoa": "Genoa_CFC",
    # Bundesliga
    "Bayern Munich": "FC_Bayern_Munich",
    "Borussia Dortmund": "Borussia_Dortmund",
    "RB Leipzig": "RB_Leipzig",
    "Bayer Leverkusen": "Bayer_04_Leverkusen",
    "VfB Stuttgart": "VfB_Stuttgart",
    "Eintracht Frankfurt": "Eintracht_Frankfurt",
    "Borussia Mönchengladbach": "Borussia_M%C3%B6nchengladbach",
    "VfL Wolfsburg": "VfL_Wolfsburg",
    "Hoffenheim": "TSG_1899_Hoffenheim",
    "Werder Bremen": "SV_Werder_Bremen",
    "Mainz 05": "1._FSV_Mainz_05",
    "Freiburg": "SC_Freiburg",
    "St. Pauli": "FC_St._Pauli",
    "Schalke 04": "FC_Schalke_04",
    # Ligue 1
    "Paris Saint-Germain": "Paris_Saint-Germain_F.C.",
    "PSG": "Paris_Saint-Germain_F.C.",
    "Marseille": "Olympique_de_Marseille",
    "Lyon": "Olympique_Lyonnais",
    "Olympique Lyonnais": "Olympique_Lyonnais",
    "Monaco": "AS_Monaco_FC",
    "Nice": "OGC_Nice",
    "Lille": "Lille_OSC",
    "Rennes": "Stade_Rennais_F.C.",
    "Lens": "RC_Lens",
    "Strasbourg": "RC_Strasbourg_Alsace",
    "Toulouse": "Toulouse_FC",
    "Nantes": "FC_Nantes",
    # Primeira Liga
    "Benfica": "S.L._Benfica",
    "Porto": "FC_Porto",
    "Sporting CP": "Sporting_CP",
    "Sporting": "Sporting_CP",
    "Braga": "S.C._Braga",
    "Vitória de Guimarães": "Vit%C3%B3ria_S.C.",
    # Eredivisie
    "Ajax": "AFC_Ajax",
    "PSV": "PSV_Eindhoven",
    "PSV Eindhoven": "PSV_Eindhoven",
    "Feyenoord": "Feyenoord",
    "AZ": "AZ_Alkmaar",
    "Twente": "FC_Twente",
    # Belgian Pro League
    "Club Brugge": "Club_Brugge_KV",
    "Anderlecht": "R.S.C._Anderlecht",
    "Genk": "K.R.C._Genk",
    "Union Saint-Gilloise": "Union_Saint-Gilloise",
    # Scottish Premiership
    "Celtic": "Celtic_F.C.",
    "Rangers": "Rangers_F.C.",
    # Turkish Super Lig
    "Galatasaray": "Galatasaray_S.K._(football)",
    "Fenerbahçe": "Fenerbah%C3%A7e_S.K._(football)",
    "Beşiktaş": "Be%C5%9Fikta%C5%9F_J.K._(football)",
    "Trabzonspor": "Trabzonspor",
    "İstanbul Başakşehir": "%C4%B0stanbul_Ba%C5%9Faksehir_F.K.",
    # Saudi Pro League
    "Al-Nassr": "Al-Nassr_FC",
    "Al-Hilal": "Al-Hilal_SFC",
    "Al-Ittihad": "Al-Ittihad_Club_(Jeddah)",
    "Al Ittihad": "Al-Ittihad_Club_(Jeddah)",
    "Al-Ahli": "Al-Ahli_Saudi_FC",
    "Al-Qadsiah": "Al-Qadsiah_FC",
    "Al Qadsiah": "Al-Qadsiah_FC",
    "Al-Ettifaq": "Ettifaq_FC",
    "Al-Shabab": "Al-Shabab_FC_(Riyadh)",
    "Al-Taawoun": "Al-Taawoun_FC",
    "Al-Fateh": "Al-Fateh_SC",
    # MLS
    "Inter Miami": "Inter_Miami_CF",
    "Inter Miami CF": "Inter_Miami_CF",
    "LA Galaxy": "LA_Galaxy",
    "LAFC": "Los_Angeles_FC",
    "Los Angeles FC": "Los_Angeles_FC",
    "Atlanta United": "Atlanta_United_FC",
    "Seattle Sounders": "Seattle_Sounders_FC",
    "Toronto FC": "Toronto_FC",
    "Vancouver Whitecaps": "Vancouver_Whitecaps_FC",
    "CF Montréal": "CF_Montr%C3%A9al",
    "Columbus Crew": "Columbus_Crew",
    "Philadelphia Union": "Philadelphia_Union",
    "FC Cincinnati": "FC_Cincinnati",
    "New York City FC": "New_York_City_FC",
    "New York Red Bulls": "New_York_Red_Bulls",
    "Orlando City": "Orlando_City_SC",
    # Brazilian Serie A
    "Flamengo": "CR_Flamengo",
    "Palmeiras": "Sociedade_Esportiva_Palmeiras",
    "Corinthians": "Sport_Club_Corinthians_Paulista",
    "São Paulo": "S%C3%A3o_Paulo_FC",
    "Santos": "Santos_FC",
    "Botafogo": "Botafogo_de_Futebol_e_Regatas",
    "Fluminense": "Fluminense_FC",
    "Internacional": "Sport_Club_Internacional",
    "Grêmio": "Gr%C3%AAmio_Foot-Ball_Porto_Alegrense",
    "Atlético Mineiro": "Clube_Atl%C3%A9tico_Mineiro",
    "Cruzeiro": "Cruzeiro_Esporte_Clube",
    # Argentine Primera
    "River Plate": "Club_Atl%C3%A9tico_River_Plate",
    "Boca Juniors": "Boca_Juniors",
    "Racing": "Racing_Club_de_Avellaneda",
    "Independiente": "Club_Atl%C3%A9tico_Independiente",
    "San Lorenzo": "San_Lorenzo_de_Almagro",
    "Vélez Sarsfield": "Club_Atl%C3%A9tico_V%C3%A9lez_Sarsfield",
    "Estudiantes": "Estudiantes_de_La_Plata",
    # Liga MX
    "Tigres": "Tigres_UANL",
    "Tigres UANL": "Tigres_UANL",
    "América": "Club_Am%C3%A9rica",
    "Club América": "Club_Am%C3%A9rica",
    "Cruz Azul": "Cruz_Azul",
    "Monterrey": "C.F._Monterrey",
    "Pumas": "Club_Universidad_Nacional",
    "Chivas": "C.D._Guadalajara",
    "Guadalajara": "C.D._Guadalajara",
    # Other notable
    "Red Bull Salzburg": "FC_Red_Bull_Salzburg",
    "Sturm Graz": "SK_Sturm_Graz",
    "Pakhtakor": "Pakhtakor_Tashkent_FK",
    "Astana": "FC_Astana",
    "Pafos": "Pafos_FC",
    "Jagiellonia Białystok": "Jagiellonia_Bia%C5%82ystok",
    "Slavia Prague": "SK_Slavia_Prague",
    "Sparta Prague": "AC_Sparta_Prague",
    "Viktoria Plzeň": "FC_Viktoria_Plze%C5%88",
    "Dinamo Zagreb": "GNK_Dinamo_Zagreb",
    "Hajduk Split": "HNK_Hajduk_Split",
    "Young Boys": "BSC_Young_Boys",
    "Olympiacos": "Olympiacos_F.C.",
}


def fetch_badge(club: str) -> str | None:
    """Resolve a club name to a Wikipedia thumbnail URL, or None if
    unknown / no thumbnail available."""
    page = CLUB_PAGE_NAME.get(club)
    if page is None:
        return None
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{page}"
    try:
        time.sleep(FETCH_DELAY_S)
        r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=15)
    except requests.RequestException:
        return None
    if r.status_code != 200:
        return None
    try:
        data = r.json()
    except ValueError:
        return None
    thumb = data.get("thumbnail") or {}
    return thumb.get("source")


def load_cache() -> dict[str, str]:
    if not CACHE_PATH.exists():
        return {}
    return json.loads(CACHE_PATH.read_text()).get("badges", {})


def save_cache(badges: dict[str, str]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps({
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_badges": len(badges),
        "badges": badges,
    }, indent=2, ensure_ascii=False))


def main() -> None:
    p = argparse.ArgumentParser(description="Fetch club crest URLs via Wikipedia.")
    p.add_argument("--refresh", action="store_true", help="Re-fetch even if cached")
    args = p.parse_args()

    rosters = fetch_rosters(refresh=False)
    clubs = sorted({pl.club for r in rosters for pl in r.players})
    print(f"Resolving badges for {len(clubs)} unique clubs in current rosters")

    cache = {} if args.refresh else load_cache()
    n_new = n_skip = n_miss = 0
    for c in clubs:
        if c in cache:
            n_skip += 1
            continue
        url = fetch_badge(c)
        if url:
            cache[c] = url
            print(f"  ✓ {c}: {url[:80]}")
            n_new += 1
        else:
            print(f"  ✗ {c}: no badge (no Wikipedia page mapping or fetch failed)")
            n_miss += 1

    save_cache(cache)
    print(f"\n{n_new} fetched, {n_skip} cached, {n_miss} missing  ({len(cache)} total)")


if __name__ == "__main__":
    sys.exit(main())
