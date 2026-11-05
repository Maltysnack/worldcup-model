# worldcup

An international football prediction model. It predicted the 2026 World Cup before kickoff and was scored on all 104 matches. It now tests its Elo ratings against bookmaker prices with paper money.

Live page: [maltysnack.github.io/worldcup-model/](https://maltysnack.github.io/worldcup-model/)

## Results

The prediction was frozen before kickoff under the tag `wc2026-final`. Spain were the favourite at 18.3% and won the tournament.

| Model | Log-loss | Accuracy |
|---|---|---|
| This model | 0.851 | 62.5% |
| Plain Elo | 0.861 | 64.4% |
| Uniform | 1.099 | 47.1% |

The model beat plain Elo on log-loss overall and in the group stage, but lost to it in the knockouts. Full breakdown: [data/capsule/scorecard_2026.md](data/capsule/scorecard_2026.md).

## How it works

- **Ratings:** Elo from about 49,000 internationals since 1872.
- **Match model:** Poisson goals with a Dixon-Coles correction.
- **Tournament:** Monte Carlo simulation of the full bracket.
- **Paper bot:** compares Elo probabilities with Sportsbet 1X2 prices through The Odds API. Rules were locked on 2026-10-26: flat $10, minimum edge 3%, maximum odds 6.0. Decisions go in an append-only, hash-chained ledger in `ledger/`.

No real money is staked, and nothing here is betting advice.

## Setup

Requires [uv](https://github.com/astral-sh/uv).

```bash
uv sync
uv run worldcup-ingest
uv run worldcup-elo
uv run worldcup-simulate
uv run python -m unittest discover tests
```

`worldcup-ingest` downloads match data and builds a local DuckDB file. The bot also needs `ODDS_API_KEY` in `.env`:

```bash
uv run worldcup-bot bet --dry
uv run worldcup-verify-ledger
```

## Layout

```
src/worldcup/          model pipeline
src/worldcup/betting/  paper bot, ledger, backtest
scripts/               exports and scoring
research/              archived experiments, not installed
tests/
data/capsule/          2026 scorecard
ledger/                bot decisions and settlements
site_data/             JSON for the live page
index.html             the live page, served by GitHub Pages
```

## Data

- Results: [martj42/international_results](https://github.com/martj42/international_results) (MIT).
- Prices: [The Odds API](https://the-odds-api.com/).
- Squads and trophies: Wikipedia.

Club player stats used for the 2026 squad and top-scorer inputs are not included.

## Contributing

Issues and pull requests are welcome. Changing the bot's rules means a new dated rules version, never a retroactive edit, and ledger rows are never edited.

Built with heavy use of Claude Code.

## Licence

MIT for the code. Data sources keep their own terms.
