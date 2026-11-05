# Elo drift model

_Generated 2026-10-26 by `research/drift_model.py`._

Question: when forecasting a World Cup years out, does modelling how Elo ratings drift beat freezing today's ratings?

Method: for each team, Elo change over h years (start dates every 6 months from 1990, active teams rated 1600+ at the start, moves ending before the forecast origin) fitted as `delta = a + b*rating + N(0, sd)`. Forecast origin is 31 Dec after the previous World Cup (about 3.5 years out). Static = simulate with origin ratings. Drift = 200 draws of future ratings x 50 tournaments each. Oracle = real pre-tournament ratings (the ceiling). Field and groups taken as known. Group letters follow discovery order, so knockout cross-pairings are approximate (same as `worldcup-backtest-tournament`).

## Fitted drift

| WC | origin | h (yr) | a | b | sd | n | mean move at 1700 | at 2000 |
|---|---|---|---|---|---|---|---|---|
| 2010 | 2006-12-31 | 3.44 | 237 | -0.133 | 67 | 940 | +10 | -30 |
| 2014 | 2010-12-31 | 3.45 | 168 | -0.094 | 66 | 1426 | +8 | -20 |
| 2018 | 2014-12-31 | 3.45 | 150 | -0.085 | 67 | 1950 | +6 | -20 |
| 2022 | 2018-12-31 | 3.89 | 119 | -0.067 | 70 | 2452 | +6 | -14 |

Residual sd by starting rating (latest fit):

- 1600-1750: sd 70 (n 1854)
- 1750-1900: sd 69 (n 535)
- 1900-2000: sd 82 (n 49)
- 2000-+: sd 58 (n 14)

## Hindcast scores

Champion log-loss: -ln P(actual champion). Stage log-loss: mean over 32 teams of -ln P(stage actually reached). Lower is better.

| WC | champion | static P / rank | drift P / rank | oracle P / rank | champ LL static | drift | oracle | uniform | stage LL static | drift | oracle | uniform |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2010 | Spain | 2.9% / 8 | 3.5% / 8 | 31.1% / 1 | 3.54 | 3.34 | 1.17 | 3.47 | 1.271 | 1.261 | 1.094 | 1.343 |
| 2014 | Germany | 10.9% / 4 | 10.0% / 4 | 9.7% / 3 | 2.21 | 2.30 | 2.33 | 3.47 | 1.330 | 1.321 | 1.217 | 1.343 |
| 2018 | France | 5.4% / 7 | 6.4% / 5 | 6.7% / 5 | 2.91 | 2.75 | 2.71 | 3.47 | 1.161 | 1.165 | 1.212 | 1.343 |
| 2022 | Argentina | 5.0% / 6 | 5.1% / 6 | 19.4% / 2 | 3.00 | 2.98 | 1.64 | 3.47 | 1.200 | 1.197 | 1.193 | 1.343 |

Means over the four tournaments:

- static: champion LL 2.916, stage LL 1.241
- drift: champion LL 2.844, stage LL 1.236
- oracle: champion LL 1.961, stage LL 1.179
- uniform: champion LL 3.466, stage LL 1.343

WC 2022 drift top 5 (from 2018-12-31): Brazil 19.1% (static 23.9%), France 15.4% (static 16.0%), Belgium 10.8% (static 11.0%), Spain 7.0% (static 8.0%), England 6.6% (static 6.4%)

## Verdict

Drift beats static on mean stage log-loss (1.236 vs 1.241); champion log-loss 2.844 vs 2.916. Four tournaments is a small sample: the champion score is one event each, so lean on stage log-loss (32 teams each).
