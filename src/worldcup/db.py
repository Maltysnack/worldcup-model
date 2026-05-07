"""DuckDB connection helper.

DuckDB is an embedded analytical database, like SQLite but optimized for
analytical queries (joins, aggregations, window functions). It reads CSVs
and Parquet files natively, so we don't need to round-trip through pandas
for most things. The whole DB lives in a single file under data/.
"""

from __future__ import annotations

from pathlib import Path

import duckdb

# Project root is two parents up from this file:
#   src/worldcup/db.py -> src/worldcup -> src -> <project root>
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "data" / "worldcup.duckdb"


def connect(read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """Open a DuckDB connection to the project database.

    Set read_only=True for read-only access (e.g. from notebooks) so
    accidental writes can't corrupt the file.
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(DB_PATH), read_only=read_only)
