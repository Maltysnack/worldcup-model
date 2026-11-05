# Betting backtest summary (synthetic odds)

Engine: proportional devig, edge = p x odds - 1, one bet per match on the largest edge above a threshold, flat $10 or quarter-Kelly capped at 5% of bankroll, $1000 start. Model: walk-forward plain Elo (v4a-flat-mild, pre-match ratings) through the Poisson goals model.

**Odds are synthetic.** Real Sportsbet history needs the paid Odds API month (item 6, blocked). These numbers test the plumbing; they are not evidence of edge.

Matches: 251 (FIFA World Cup 2022 64, UEFA Euro 2024 51, Copa América 2024 32, FIFA World Cup 2026 104)

Model 3-way log-loss 0.947, Brier 0.557.

Scenarios (both settle on real results):

- **edge**: model probabilities are the truth; book prices them with log-normal noise (sd 0.15) and a 6% margin.
- **no_edge**: book prices near-true (sd 0.03); the bettor's model carries the noise (sd 0.15). Expect to lose about the margin.

Closing price = bet price with zero-mean drift (sd 0.05), so prices do not move toward the truth and synthetic CLV against the devigged close sits near minus the margin by construction.

## Scenario: edge

Log-loss / Brier: bettor 0.947 / 0.557, book open (devigged) 0.950 / 0.558, book close 0.949 / 0.559.

| Staking | Min edge | Bets | Staked | P&L | ROI | ROI 95% CI | Final bankroll | Max DD | Mean CLV | Mean ROI over 200 odds draws | P(ROI>0) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| flat | 0.00 | 205 | $2050 | $+60 | +2.9% | -25.0% to +35.6% | $1060 | 15% | -6.1% | +15.7% | 80% |
| flat | 0.03 | 155 | $1550 | $+74 | +4.8% | -28.1% to +40.0% | $1074 | 18% | -6.1% | +19.0% | 82% |
| flat | 0.05 | 133 | $1330 | $+107 | +8.1% | -28.9% to +50.5% | $1107 | 18% | -6.0% | +21.2% | 78% |
| flat | 0.10 | 92 | $920 | $+73 | +7.9% | -34.9% to +57.1% | $1073 | 14% | -6.0% | +26.0% | 77% |
| kelly | 0.00 | 205 | $1942 | $+89 | +4.6% | -22.5% to +32.7% | $1089 | 17% | -6.1% | +12.3% | 80% |
| kelly | 0.03 | 155 | $1827 | $+137 | +7.5% | -20.5% to +38.1% | $1137 | 17% | -6.1% | +13.0% | 80% |
| kelly | 0.05 | 133 | $1684 | $+128 | +7.6% | -22.3% to +38.0% | $1128 | 19% | -6.0% | +13.7% | 80% |
| kelly | 0.10 | 92 | $1308 | $+114 | +8.7% | -28.4% to +47.5% | $1114 | 18% | -6.0% | +15.0% | 78% |

## Scenario: no_edge

Log-loss / Brier: bettor 0.945 / 0.554, book open (devigged) 0.947 / 0.557, book close 0.948 / 0.558.

| Staking | Min edge | Bets | Staked | P&L | ROI | ROI 95% CI | Final bankroll | Max DD | Mean CLV | Mean ROI over 200 odds draws | P(ROI>0) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| flat | 0.00 | 192 | $1920 | $+87 | +4.5% | -21.4% to +36.4% | $1087 | 20% | -5.8% | +0.3% | 45% |
| flat | 0.03 | 146 | $1460 | $+120 | +8.2% | -27.0% to +46.7% | $1120 | 16% | -6.1% | -0.0% | 46% |
| flat | 0.05 | 120 | $1200 | $+122 | +10.1% | -27.6% to +56.4% | $1122 | 14% | -6.1% | -0.1% | 46% |
| flat | 0.10 | 78 | $780 | $-9 | -1.1% | -36.9% to +39.0% | $991 | 9% | -6.1% | +0.8% | 42% |
| kelly | 0.00 | 192 | $1948 | $+247 | +12.7% | -15.3% to +42.0% | $1247 | 13% | -5.8% | -5.7% | 32% |
| kelly | 0.03 | 146 | $1742 | $+219 | +12.6% | -21.3% to +43.9% | $1219 | 13% | -6.1% | -6.0% | 34% |
| kelly | 0.05 | 120 | $1568 | $+200 | +12.8% | -21.3% to +48.1% | $1200 | 13% | -6.1% | -6.2% | 35% |
| kelly | 0.10 | 78 | $1206 | $+177 | +14.7% | -26.1% to +55.7% | $1177 | 12% | -6.1% | -7.8% | 33% |

## Recommended staking rule (provisional)

**Flat $10, min edge 0.03.** Reasons: quarter-Kelly sizes bets by the model's own edge estimate, so a miscalibrated model bets biggest exactly where it is most wrong; flat stakes keep ROI and CLV readable per bet and drawdowns small; with roughly 100 bets a year the ROI CI is wide either way, so variance control matters more than growth rate. Revisit once item 6 supplies real Sportsbet prices: if real CLV is reliably positive, quarter-Kelly becomes defensible.

**The rule is locked before live betting.**

Limitations: synthetic odds; one bet per match; stakes use bankroll at placement in kickoff order (same-day matches settle sequentially); knockout results include extra time, unlike real 1X2 settlement.
