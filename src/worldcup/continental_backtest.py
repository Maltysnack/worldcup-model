"""Lightweight continental-tournament backtest.

For each (tournament, year) we want to test:
  1. Identify the participating teams from match data
  2. Identify the actual top 4 finishers (champion, runner-up,
     and the two semifinal losers) from the historical KO matches
  3. Compute each team's pre-tournament effective rating using our
     full model (raw Elo + form + NT trophy bonus, all as-of the
     tournament's start date)
  4. Rank all teams by effective rating
  5. Report where the actual top 4 finished in that ranking

This is a *ranking* backtest, not a Monte Carlo simulation of the
bracket. We're asking: "did our model identify the eventual top
finishers as among the strongest teams pre-tournament?" not "did our
simulator pick the exact bracket outcome?"

Run: `uv run worldcup-continental-backtest`
"""

from __future__ import annotations

from datetime import date

from worldcup.db import connect
from worldcup.form import compute_form_bumps
from worldcup.host import host_bonus
from worldcup.trophies import nt_trophy_bonus

# Tournaments to backtest. Pre-2010 not interesting because trophy bonus
# data is sparse. WC 2022 excluded (locked hold-out).
TOURNAMENTS_TO_TEST = [
    ("UEFA Euro", 2012),
    ("UEFA Euro", 2016),
    ("UEFA Euro", 2021),  # EURO 2020 played in 2021
    ("UEFA Euro", 2024),
    ("Copa América", 2011),
    ("Copa América", 2015),
    ("Copa América", 2016),
    ("Copa América", 2019),
    ("Copa América", 2021),
    ("Copa América", 2024),
    ("African Cup of Nations", 2012),
    ("African Cup of Nations", 2015),
    ("African Cup of Nations", 2017),
    ("African Cup of Nations", 2019),
    ("African Cup of Nations", 2022),
    ("African Cup of Nations", 2024),
    ("AFC Asian Cup", 2015),
    ("AFC Asian Cup", 2019),
    ("AFC Asian Cup", 2024),
    ("Gold Cup", 2015),
    ("Gold Cup", 2017),
    ("Gold Cup", 2019),
    ("Gold Cup", 2021),
    ("Gold Cup", 2023),
]


def fetch_teams_and_dates(tournament: str, year: int):
    with connect(read_only=True) as con:
        df = con.execute(
            """
            SELECT date, home_team, away_team, home_score, away_score
            FROM matches
            WHERE tournament = ?
              AND EXTRACT(YEAR FROM date) = ?
              AND home_score IS NOT NULL
            ORDER BY date
            """,
            [tournament, year],
        ).fetchdf()
    if df.empty:
        return None
    teams = sorted(set(df["home_team"]) | set(df["away_team"]))
    start_date = df["date"].min()
    return df, teams, start_date


def fetch_actual_top4(tournament: str, year: int):
    """Identify top 4 from the historical KO matches.

    The last match is the final → champion + runner-up. The two matches
    before the final at the highest-numbered date that had two distinct
    pairings are the SFs (in EURO/AFCON there's no 3rd-place playoff;
    in Copa/Gold Cup there usually is, played around the same time).

    We infer SF participants as: any team that lost in the final or in
    one of the matches immediately preceding the final.
    """
    df, teams, _ = fetch_teams_and_dates(tournament, year)
    if df is None or df.empty:
        return None
    final_row = df.iloc[-1]
    if final_row.home_score > final_row.away_score:
        champion, runner_up = final_row.home_team, final_row.away_team
    elif final_row.home_score < final_row.away_score:
        champion, runner_up = final_row.away_team, final_row.home_team
    else:
        # Drawn final — penalty result not in DB; use home as champ for now
        champion, runner_up = final_row.home_team, final_row.away_team

    # SF losers: teams that played within ~7 days before the final but
    # didn't make the final themselves
    final_date = final_row.date
    finalist_set = {champion, runner_up}
    sf_window = df[
        (df["date"] >= final_date - pd_timedelta(days=10))
        & (df["date"] < final_date)
    ]
    sf_losers = set()
    for r in sf_window.itertuples(index=False):
        for t in [r.home_team, r.away_team]:
            if t not in finalist_set:
                sf_losers.add(t)
    return {
        "champion": champion,
        "runner_up": runner_up,
        "semifinalists": finalist_set | sf_losers,
    }


def pd_timedelta(days):
    import pandas as pd
    return pd.Timedelta(days=days)


def fetch_pre_tournament_ratings(tournament: str, year: int, teams: list[str]):
    """Pre-tournament Elo for each participating team."""
    with connect(read_only=True) as con:
        sql = """
        WITH first_match AS (
          SELECT team, MIN(match_date) AS d FROM (
            SELECT match_date, home_team AS team FROM predictions
            WHERE tournament = ? AND EXTRACT(YEAR FROM match_date) = ?
            UNION ALL
            SELECT match_date, away_team AS team FROM predictions
            WHERE tournament = ? AND EXTRACT(YEAR FROM match_date) = ?
          ) GROUP BY team
        )
        SELECT t.team,
          CASE WHEN p.home_team = t.team
               THEN p.rating_home_pre ELSE p.rating_away_pre END AS rating
        FROM first_match t
        JOIN predictions p ON p.match_date = t.d
          AND (p.home_team = t.team OR p.away_team = t.team)
        """
        df = con.execute(sql, [tournament, year, tournament, year]).fetchdf()
    return {row.team: float(row.rating) for row in df.itertuples()}


def effective_rating(team, raw_ratings, form, tournament, year):
    return (
        raw_ratings.get(team, 1500)
        + form.get(team, 0.0)
        + nt_trophy_bonus(team, as_of_year=year)
        + host_bonus(team, tournament, year)
    )


def run_backtest(tournament: str, year: int):
    matches_data = fetch_teams_and_dates(tournament, year)
    if matches_data is None:
        return None
    df, teams, start_date = matches_data

    raw = fetch_pre_tournament_ratings(tournament, year, teams)
    form = compute_form_bumps(as_of_date=start_date)
    actual = fetch_actual_top4(tournament, year)
    if actual is None:
        return None

    # Rank teams by effective rating
    ranked = sorted(
        teams, key=lambda t: -effective_rating(t, raw, form, tournament, year)
    )

    # Where did the actual top 4 finish?
    top4_ranks = {}
    for team in actual["semifinalists"]:
        try:
            top4_ranks[team] = ranked.index(team) + 1
        except ValueError:
            top4_ranks[team] = None

    return {
        "tournament": tournament,
        "year": year,
        "n_teams": len(teams),
        "actual_champion": actual["champion"],
        "actual_runner_up": actual["runner_up"],
        "champion_rank": ranked.index(actual["champion"]) + 1,
        "runner_up_rank": ranked.index(actual["runner_up"]) + 1,
        "top4_ranks": top4_ranks,
        "ranking": ranked,
    }


def main() -> None:
    results = []
    print("Continental tournament backtest:\n")
    for tournament, year in TOURNAMENTS_TO_TEST:
        try:
            r = run_backtest(tournament, year)
        except Exception as e:
            print(f"  {tournament} {year}: ERROR {e}")
            continue
        if r is None:
            print(f"  {tournament} {year}: no data")
            continue
        results.append(r)
        # Per-tournament one-liner
        print(
            f"  {r['tournament']:25s} {r['year']}  n={r['n_teams']:>2}  "
            f"champ={r['actual_champion']:14s} ranked #{r['champion_rank']:>2}  "
            f"RU={r['actual_runner_up']:14s} #{r['runner_up_rank']:>2}"
        )

    # Summary table
    print(f"\n{'=' * 82}")
    print("SUMMARY")
    print(f"{'=' * 82}\n")
    champ_ranks = [r["champion_rank"] for r in results]
    ru_ranks = [r["runner_up_rank"] for r in results]
    n_teams_avg = sum(r["n_teams"] for r in results) / len(results)
    n_total = len(results)

    print(f"  Tournaments backtested: {n_total}")
    print(f"  Average tournament size: {n_teams_avg:.1f} teams")
    print(f"\n  Average champion rank:    {sum(champ_ranks)/n_total:.2f}")
    print(f"  Average runner-up rank:   {sum(ru_ranks)/n_total:.2f}")
    print(
        f"  Champion in top 3:        "
        f"{sum(1 for r in champ_ranks if r <= 3)}/{n_total}  "
        f"({100*sum(1 for r in champ_ranks if r <= 3)/n_total:.0f}%)"
    )
    print(
        f"  Champion in top 5:        "
        f"{sum(1 for r in champ_ranks if r <= 5)}/{n_total}  "
        f"({100*sum(1 for r in champ_ranks if r <= 5)/n_total:.0f}%)"
    )
    print(
        f"  Champion outside top 8:   "
        f"{sum(1 for r in champ_ranks if r > 8)}/{n_total}  "
        f"({100*sum(1 for r in champ_ranks if r > 8)/n_total:.0f}%)"
    )

    # SFs in top 6
    sf_in_top_6_total = 0
    sf_total = 0
    for r in results:
        for team, rank in r["top4_ranks"].items():
            if rank is None:
                continue
            sf_total += 1
            if rank <= 6:
                sf_in_top_6_total += 1
    print(
        f"  Semifinalists in top 6:   "
        f"{sf_in_top_6_total}/{sf_total}  "
        f"({100*sf_in_top_6_total/sf_total:.0f}%)"
    )


if __name__ == "__main__":
    main()
