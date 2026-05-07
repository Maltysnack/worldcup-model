# worldcup

A Monte Carlo simulator for the 2026 FIFA World Cup. A learning project, not a betting tool.

Public progress log: [maltysnack.github.io/projects/worldcup.html](https://maltysnack.github.io/projects/worldcup.html)

## Status

Week 1: data acquisition. Loaded ~49k historical international match results (1872 to present) plus the embedded 2026 World Cup fixture list into a local DuckDB.

## Setup

Requires [uv](https://github.com/astral-sh/uv).

```bash
uv sync                # install dependencies into .venv
uv run worldcup-ingest # download raw CSVs and build the local DB
```

The database lives at `data/worldcup.duckdb`. Both `data/raw/` and the DB file are gitignored: regenerate them with `worldcup-ingest`.

## Layout

```
src/worldcup/   # the package (importable)
  db.py         # DuckDB connection helper
  ingest.py     # download + load raw data
data/           # gitignored: raw CSVs and DuckDB file
notebooks/      # exploration
scripts/        # one-off CLI scripts (e.g. inspect_data.py)
```

## Data source

International match results from [martj42/international_results](https://github.com/martj42/international_results) (MIT licensed, weekly updates).
