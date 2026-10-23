# 2026 World Cup scorecard

Frozen prediction `wc2026-final` (elo-v4a-flat-mild+squad35+form) scored against all 104 results. Generated 2026-10-23 by `scripts/score_capsule_2026.py`.

- Group probabilities verbatim from frozen file; knockout probabilities recomputed from frozen adjusted ratings with the same Poisson goals model.
- Knockout scores include extra time; shootouts count as draws for 3-way scoring.
- Plain Elo: raw v4a-flat-mild ratings from matches up to 2026-06-10, same goals model.
- Recompute check on group matches: max abs diff vs frozen = 0.0231 (frozen used 5000-sample Monte Carlo and rounded ratings).

## Match-level, 3-way (lower log-loss and Brier are better)

| Segment | Model | n | Log-loss | Brier | Accuracy |
|---|---|---|---|---|---|
| all | model | 104 | 0.8513 | 0.4963 | 62.5% |
| all | plain_elo | 104 | 0.8608 | 0.5030 | 64.4% |
| all | uniform | 104 | 1.0986 | 0.6667 | 47.1% |
| group | model | 72 | 0.8806 | 0.5199 | 58.3% |
| group | plain_elo | 72 | 0.9075 | 0.5371 | 59.7% |
| group | uniform | 72 | 1.0986 | 0.6667 | 45.8% |
| knockout | model | 32 | 0.7852 | 0.4432 | 71.9% |
| knockout | plain_elo | 32 | 0.7558 | 0.4261 | 75.0% |
| knockout | uniform | 32 | 1.0986 | 0.6667 | 50.0% |

## Knockouts: who advanced (binary)

| Model | Log-loss | Brier | Accuracy |
|---|---|---|---|
| model | 0.5052 | 0.3332 | 75.0% |
| plain_elo | 0.4886 | 0.3224 | 78.1% |
| coin_flip | 0.6931 | 0.5000 | 50.0% |

## Reliability (all 312 outcome probabilities)

**model** (ECE 0.0589)

| Bin | n | Mean p | Observed |
|---|---|---|---|
| 0.0-0.1 | 16 | 0.071 | 0.000 |
| 0.1-0.2 | 56 | 0.152 | 0.107 |
| 0.2-0.3 | 118 | 0.249 | 0.237 |
| 0.3-0.4 | 28 | 0.348 | 0.214 |
| 0.4-0.5 | 31 | 0.457 | 0.484 |
| 0.5-0.6 | 18 | 0.540 | 0.722 |
| 0.6-0.7 | 27 | 0.644 | 0.815 |
| 0.7-0.8 | 14 | 0.754 | 0.786 |
| 0.8-0.9 | 4 | 0.846 | 0.750 |

**plain_elo** (ECE 0.0585)

| Bin | n | Mean p | Observed |
|---|---|---|---|
| 0.0-0.1 | 23 | 0.065 | 0.043 |
| 0.1-0.2 | 62 | 0.153 | 0.145 |
| 0.2-0.3 | 103 | 0.250 | 0.214 |
| 0.3-0.4 | 33 | 0.354 | 0.273 |
| 0.4-0.5 | 24 | 0.449 | 0.500 |
| 0.5-0.6 | 19 | 0.542 | 0.789 |
| 0.6-0.7 | 24 | 0.651 | 0.750 |
| 0.7-0.8 | 15 | 0.744 | 0.800 |
| 0.8-0.9 | 7 | 0.848 | 0.714 |
| 0.9-1.0 | 2 | 0.903 | 0.500 |

## Champion

Spain won. Pre-tournament p_champion 18.27% (rank 1 of 48), log-loss 1.700 (uniform 1/48 would be 3.871).

Stage Brier across 48 teams ("reached at least this stage"): round_of_32 0.1580, round_of_16 0.1322, quarter_finals 0.1000, semi_finals 0.0505, final 0.0284, champion 0.0148.

## Teams: champion odds vs stage reached

| Team | p_champion % | Reached | P(reach that stage) | P(exactly that stage) |
|---|---|---|---|---|
| Spain | 18.27 | champion | 18.3% | 18.3% |
| France | 11.53 | semi_finals | 29.4% | 10.6% |
| Argentina | 8.52 | final | 15.5% | 6.9% |
| Portugal | 8.32 | round_of_16 | 66.2% | 23.3% |
| Brazil | 5.53 | round_of_16 | 57.3% | 23.6% |
| England | 4.55 | semi_finals | 16.4% | 7.3% |
| Colombia | 4.08 | round_of_16 | 54.1% | 23.8% |
| Mexico | 3.64 | round_of_16 | 55.4% | 25.2% |
| Morocco | 3.37 | quarter_finals | 27.7% | 13.0% |
| Ecuador | 3.06 | round_of_32 | 87.2% | 39.8% |
| Germany | 2.90 | round_of_32 | 86.4% | 40.2% |
| Senegal | 2.86 | round_of_32 | 78.8% | 33.4% |
| Netherlands | 2.75 | round_of_32 | 83.7% | 40.0% |
| Japan | 2.42 | round_of_32 | 83.3% | 39.6% |
| Turkey | 2.32 | group_stage | 100.0% | 22.8% |
| Switzerland | 2.10 | quarter_finals | 25.7% | 14.1% |
| Belgium | 1.75 | quarter_finals | 25.8% | 15.5% |
| Norway | 1.60 | quarter_finals | 18.5% | 9.7% |
| Uruguay | 1.37 | group_stage | 100.0% | 18.9% |
| South Korea | 1.33 | group_stage | 100.0% | 20.0% |
| Croatia | 1.29 | round_of_32 | 81.5% | 40.5% |
| Canada | 1.17 | round_of_16 | 50.7% | 29.1% |
| United States | 1.03 | round_of_16 | 43.5% | 22.7% |
| Ivory Coast | 0.70 | round_of_32 | 72.7% | 40.9% |
| Paraguay | 0.53 | round_of_16 | 30.5% | 17.2% |
| Australia | 0.52 | round_of_32 | 55.1% | 28.3% |
| Austria | 0.41 | round_of_32 | 65.6% | 42.1% |
| Algeria | 0.40 | round_of_32 | 62.3% | 40.7% |
| Egypt | 0.31 | round_of_16 | 29.6% | 19.0% |
| Uzbekistan | 0.25 | group_stage | 100.0% | 51.9% |
| Scotland | 0.21 | group_stage | 100.0% | 47.0% |
| DR Congo | 0.16 | round_of_32 | 46.1% | 27.0% |
| Iran | 0.15 | group_stage | 100.0% | 35.2% |
| Panama | 0.12 | group_stage | 100.0% | 43.4% |
| Sweden | 0.12 | round_of_32 | 54.2% | 36.4% |
| Czech Republic | 0.07 | group_stage | 100.0% | 47.6% |
| Bosnia and Herzegovina | 0.06 | round_of_32 | 57.8% | 37.3% |
| Tunisia | 0.06 | group_stage | 100.0% | 51.8% |
| New Zealand | 0.04 | group_stage | 100.0% | 52.5% |
| Saudi Arabia | 0.04 | group_stage | 100.0% | 63.7% |
| Cape Verde | 0.03 | round_of_32 | 39.8% | 28.7% |
| Jordan | 0.02 | group_stage | 100.0% | 56.8% |
| South Africa | 0.02 | round_of_32 | 35.5% | 25.0% |
| Ghana | 0.01 | round_of_32 | 36.8% | 27.7% |
| Haiti | 0.01 | group_stage | 100.0% | 66.0% |
| Curaçao | 0.00 | group_stage | 100.0% | 75.6% |
| Iraq | 0.00 | group_stage | 100.0% | 73.5% |
| Qatar | 0.00 | group_stage | 100.0% | 71.0% |

## Biggest model misses (3-way log-loss)

| # | Stage | Match | Score | Model H/D/A | Log-loss |
|---|---|---|---|---|---|
| 16 | group | Spain v Cape Verde | 0-0 | 0.86/0.10/0.03 | 2.267 |
| 33 | group | Ecuador v Curaçao | 0-0 | 0.79/0.14/0.07 | 1.950 |
| 54 | group | South Africa v South Korea | 1-0 | 0.14/0.21/0.65 | 1.938 |
| 8 | group | Qatar v Switzerland | 1-1 | 0.09/0.15/0.76 | 1.885 |
| 46 | group | England v Ghana | 0-0 | 0.76/0.16/0.08 | 1.851 |
| 23 | group | Portugal v DR Congo | 1-1 | 0.69/0.19/0.12 | 1.660 |
| 15 | group | Saudi Arabia v Uruguay | 1-1 | 0.14/0.22/0.65 | 1.532 |
| 40 | group | Uruguay v Cape Verde | 2-2 | 0.63/0.22/0.15 | 1.510 |
| 3 | group | Canada v Bosnia and Herzegovina | 1-1 | 0.63/0.23/0.14 | 1.483 |
| 57 | group | Japan v Sweden | 1-1 | 0.57/0.24/0.19 | 1.421 |
