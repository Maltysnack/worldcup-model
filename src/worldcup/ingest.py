"""Data ingestion: download raw CSVs and load them into DuckDB.

Source: https://github.com/martj42/international_results
This dataset has every international men's match since 1872, plus
goalscorers and shootouts. It's MIT licensed and updated weekly. It's
the canonical source most public World Cup predictors start from.

Run with: `uv run worldcup-ingest`
"""

from __future__ import annotations

import sys
from pathlib import Path

import requests

from worldcup.db import PROJECT_ROOT, connect

RAW_DIR = PROJECT_ROOT / "data" / "raw"

# Each entry: (table name, source URL)
SOURCES: list[tuple[str, str]] = [
    (
        "matches",
        "https://raw.githubusercontent.com/martj42/international_results/master/results.csv",
    ),
    (
        "goalscorers",
        "https://raw.githubusercontent.com/martj42/international_results/master/goalscorers.csv",
    ),
    (
        "shootouts",
        "https://raw.githubusercontent.com/martj42/international_results/master/shootouts.csv",
    ),
]


def download(url: str, dest: Path) -> None:
    """Download a file to dest. No-op if already cached.

    We keep raw downloads in data/raw/ and never edit them by hand. If
    the file ever needs refreshing, delete it and rerun ingest. This
    keeps a clear separation between 'data we got from outside' and
    'data we derived'.
    """
    if dest.exists():
        print(f"  cached: {dest.name}")
        return
    print(f"  downloading: {url}")
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    dest.write_bytes(response.content)
    print(f"  saved: {dest.name} ({len(response.content):,} bytes)")


def load_into_duckdb(table_name: str, csv_path: Path) -> None:
    """(Re)create a table in DuckDB by reading the CSV directly.

    DuckDB can read CSVs natively with type inference: read_csv_auto.
    DROP + CREATE keeps things idempotent: rerunning ingest gives a
    clean state. For a tiny dataset like this (~50k rows) that's fine.
    """
    with connect() as con:
        con.execute(f"DROP TABLE IF EXISTS {table_name}")
        # nullstr=['NA', ''] — the dataset uses literal "NA" for unplayed
        # future fixtures (e.g. 2026 World Cup matches that haven't kicked
        # off yet). Without this, DuckDB tries to parse "NA" as a number
        # and crashes. Treating both 'NA' and '' as NULL is safer.
        con.execute(
            f"CREATE TABLE {table_name} AS "
            f"SELECT * FROM read_csv_auto(?, nullstr=['NA', ''])",
            [str(csv_path)],
        )
        # Cast date column if present (CSVs come in as VARCHAR by default
        # for date-shaped strings; DuckDB usually infers DATE but we make
        # it explicit so downstream code can rely on it).
        cols = [
            row[0]
            for row in con.execute(f"DESCRIBE {table_name}").fetchall()
        ]
        if "date" in cols:
            con.execute(
                f"ALTER TABLE {table_name} ALTER date TYPE DATE USING CAST(date AS DATE)"
            )
        count = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        print(f"  loaded {table_name}: {count:,} rows")


def create_views() -> None:
    """Create convenience views for later querying.

    Views are saved SQL queries that look like tables. Cheap to create,
    cheap to query, and they keep the underlying tables clean. We add:
    - matches_recent: last 15 years of matches (training window)
    - matches_competitive: excludes friendlies (more predictive of WC perf)
    """
    with connect() as con:
        con.execute("""
            CREATE OR REPLACE VIEW matches_recent AS
            SELECT *
            FROM matches
            WHERE date >= (CURRENT_DATE - INTERVAL 15 YEAR)
        """)
        con.execute("""
            CREATE OR REPLACE VIEW matches_competitive AS
            SELECT *
            FROM matches
            WHERE tournament != 'Friendly'
        """)
        print("  created views: matches_recent, matches_competitive")


def main() -> None:
    print("World Cup ingest starting")
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[1/3] Downloading raw CSVs")
    for table_name, url in SOURCES:
        download(url, RAW_DIR / f"{table_name}.csv")

    print("\n[2/3] Loading into DuckDB")
    for table_name, _ in SOURCES:
        load_into_duckdb(table_name, RAW_DIR / f"{table_name}.csv")

    print("\n[3/3] Creating views")
    create_views()

    print("\nDone. Database at:", PROJECT_ROOT / "data" / "worldcup.duckdb")


if __name__ == "__main__":
    sys.exit(main())
