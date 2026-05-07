"""Quick sanity check on the loaded data.

After loading any new dataset, the very first thing to do is poke at it:
- How many rows?
- What's the date range?
- Any obvious nulls or weirdness?
- Does the schema match expectations?

If you skip this step you build models on top of broken data and discover
it three weeks later. Run with: `uv run python scripts/inspect_data.py`.
"""

from __future__ import annotations

from worldcup.db import connect


def main() -> None:
    with connect(read_only=True) as con:
        print("=" * 60)
        print("matches table")
        print("=" * 60)

        schema = con.execute("DESCRIBE matches").fetchdf()
        print("\nSchema:")
        print(schema.to_string(index=False))

        stats = con.execute("""
            SELECT
                COUNT(*)                        AS total_matches,
                MIN(date)                       AS earliest,
                MAX(date)                       AS latest,
                COUNT(DISTINCT home_team)       AS distinct_home_teams,
                COUNT(DISTINCT tournament)      AS distinct_tournaments,
                SUM(CASE WHEN home_score IS NULL THEN 1 ELSE 0 END)
                                                AS unplayed_fixtures
            FROM matches
        """).fetchdf()
        print("\nOverall stats:")
        print(stats.to_string(index=False))

        print("\nMost common tournaments (top 10):")
        top_tournaments = con.execute("""
            SELECT tournament, COUNT(*) AS n
            FROM matches
            GROUP BY tournament
            ORDER BY n DESC
            LIMIT 10
        """).fetchdf()
        print(top_tournaments.to_string(index=False))

        print("\nUpcoming 2026 World Cup fixtures already in dataset:")
        upcoming = con.execute("""
            SELECT date, home_team, away_team, city, country
            FROM matches
            WHERE date >= CURRENT_DATE
              AND tournament = 'FIFA World Cup'
            ORDER BY date
            LIMIT 10
        """).fetchdf()
        print(upcoming.to_string(index=False))

        print("\nSample of recent (last week) played matches:")
        recent = con.execute("""
            SELECT date, home_team, home_score, away_score, away_team, tournament
            FROM matches
            WHERE date < CURRENT_DATE
              AND home_score IS NOT NULL
            ORDER BY date DESC
            LIMIT 10
        """).fetchdf()
        print(recent.to_string(index=False))


if __name__ == "__main__":
    main()
