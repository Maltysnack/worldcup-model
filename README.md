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

### Ratings

Elo over every international since 1990, in date order, from the [international results dataset](https://github.com/martj42/international_results). All teams start at 1500.

```
E = 1 / (1 + 10^(-(R_home + H - R_away) / 400))
R' = R + K * (S - E)        S = 1 / 0.5 / 0
```

- `H = 25` at non-neutral venues, 0 otherwise.
- `K` by competition: World Cup 50; continental finals 45; qualifiers and Nations League 35 to 40; friendlies 30. The flat profile `v4a-flat-mild` was chosen over 10 alternatives by log-loss on 16 tournaments.
- No time decay. Recency comes from sequential updates; recent-only windows scored worse.

### Match model

The rating gap `d` (with `H` at non-neutral venues) maps to expected goals:

```
lambda_a = 1.24 * exp(+0.0023 * d)
lambda_b = 1.24 * exp(-0.0023 * d)
```

Scorelines are independent Poisson up to 10 goals, and win/draw/loss is summed from the joint distribution. A Dixon-Coles correction is implemented but its fitted `rho` is 0. Knockout draws go to a penalty coin flip weighted by `d / 600`.

### Tournament forecast

`worldcup-forecast <tournament>` adds a small overlay to current Elo, then runs a Monte Carlo over the full bracket (groups, tie-breaks, best third-placed teams, knockout structure):

- National-team honours: time-decayed bonus for past continental and World Cup finishes. On.
- Host bonus: +15 to +30. On.
- Altitude: +50 for a host playing at altitude (Mexico, 2026). On.
- Recent form: off. It hurt log-loss on 2026.

On the 2026 ablation (104 matches) every layer's interval includes zero, so the overlay is unproven; see [data/capsule/layer_ablation_2026.md](data/capsule/layer_ablation_2026.md). Tournaments are defined in `src/worldcup/registry.py`.

### Paper bot

Plain Elo (no overlay, no temperature scaling) through the match model, against de-vigged Sportsbet 1X2 prices from The Odds API.

- Bet when `p_model * odds - 1 >= 0.03` and `odds <= 6.0`. Flat $10 on a $1,000 starting bankroll, one decision per event at the first daily pull within 48 hours of kickoff.
- Rules v1 were locked on 2026-10-26. Changes need a new dated version, never a retroactive edit.
- Decisions and settlements are append-only and hash-chained in `ledger/`; `worldcup-verify-ledger` checks the chain.
- Known bias: draws are under-predicted (26% against 31% observed), so bets skew towards draws. The odds cap limits longshot edges from favourite-longshot bias.

No real money is staked, and nothing here is betting advice.

## Setup

Requires [uv](https://github.com/astral-sh/uv).

```bash
uv sync
uv run worldcup-ingest
uv run worldcup-elo
uv run worldcup-simulate
uv run worldcup-forecast wc2030
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
- National-team honours: public Wikipedia lists, encoded in `src/worldcup/trophies.py`.

## Contributing

Issues and pull requests are welcome. Changing the bot's rules means a new dated rules version, never a retroactive edit, and ledger rows are never edited.

Built with heavy use of Claude Code.

## Licence

MIT for the code. Data sources keep their own terms.
