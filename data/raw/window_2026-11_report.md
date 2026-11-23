# November 2026 window report (12 to 17 Nov, UEFA Nations League)

Ledger verified with `worldcup-verify-ledger`: hash chains and bankroll replay OK. All 11 bets settled. Rules v1 in force (flat $10, min edge 3%, max odds 6.0). Results refreshed from the martj42 ingest.

## Bets

| Measure | Value |
|---|---|
| Bets | 11 (5 won, 6 lost, 0 void) |
| Staked | $110.00 |
| P&L | -$6.60 |
| ROI | -6.0% |
| Bankroll | $993.40 |
| Mean edge at placement | 10.4% (median 6.8%) |
| Hit rate | 45.5% (5 of 11) |
| Mean model chance on the chosen side | 52.4% |
| Mean margin-free Sportsbet chance on the chosen side | 45.1% |

At 11 bets the P&L and ROI are noise. A single $29 win (Italy v France at 3.9) outweighs five losses.

## CLV: not available

Mean and median CLV are n/a. Every settlement has `closing_odds: null` and there is no `ledger/closing.jsonl`, so no closing prices were captured in this window. The closing-minutes_before distribution is therefore also empty. CLV is the only number worth reading at this sample size, so this window gave us no usable signal on it. The workflow fix to keep closing prices was committed on 20 Nov, after the window, so the next window is the first that can show CLV. Check `closing.jsonl` fills before trusting it.

## By selection

| Selection | Bets | Won | P&L |
|---|---|---|---|
| home | 8 | 4 | +$11.70 |
| draw | 1 | 0 | -$10.00 |
| away | 2 | 1 | -$8.30 |

CLV by selection: n/a (see above). One bet was a draw (Greece v Serbia at 4.8, lost). None of the 16 priced games ended in a draw.

## Passes

5 passes, all with reason "no edge" (England v Croatia, Serbia v Germany, Norway v Wales, France v Turkey, Wales v Portugal). All five were heavy favourites at 1.2 to 1.4, and the favourite won each time.

## Biggest edges

| Game | Pick | Odds | Edge | Result |
|---|---|---|---|---|
| Turkey v Belgium | home | 4.0 | 33.6% | lost (Belgium won) |
| Spain v England (listed Spain home) | home | 1.70 | 15.2% | lost (England won) |
| Germany v Netherlands | away | 3.6 | 13.1% | lost (Germany won) |
| Italy v France | home | 3.9 | 11.7% | won (+$29.00) |
| Croatia v Czech Republic | home | 1.56 | 9.9% | won (+$5.60) |

The four largest edges went 2 from 5 on the list above. A 33.6% edge is far more likely to be model error than a real mispricing.

## Model v margin-free Sportsbet, log-loss against results

Market chances are Sportsbet 1X2 odds at decision time, normalised proportionally to remove the margin. 16 priced games (11 bets, 5 passes). Lower is better.

| Set | Games | Model | Market |
|---|---|---|---|
| All priced | 16 | 0.782 | 0.690 |
| Bets | 11 | 0.973 | 0.872 |
| Passes | 5 | 0.361 | 0.289 |

The market beat the model on both subsets. The model is more confident than the market on the sides it chose (52.4% v 45.1%), while the actual hit rate was 45.5%, in line with the market. That is consistent with the model being overconfident where it disagrees with the price, but 16 games cannot separate this from luck.

## Recommendation

Keep rules v1. 11 bets and no CLV cannot justify a change. Revisit a dated rules v2 only once CLV is being captured and there are roughly 100 or more settled bets. If the market keeps beating the model on log-loss then, the candidates are a higher minimum edge or shrinking model chances toward the market, to be tested on that data, not this window.
