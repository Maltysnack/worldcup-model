"""Single source of truth for the goals-model calibration constants.

Lives in its own tiny module so simulator.py, form.py, and backtest.py
can all import it without creating circular dependencies.

Slopes have a 0.85 multiplier baked in vs the raw binned-mean fit,
chosen by sweeping log-loss against WC 2010, 2014, and 2018.

DIXON_COLES_RHO is a small joint-distribution correction on low-scoring
results. Pure independent Poisson under-predicts 0-0 / 1-0 / 0-1 / 1-1,
so we re-weight those four cells in the joint PMF (Dixon & Coles 1997).
A negative rho boosts 0-0 and 1-1, a positive rho boosts 1-0 and 0-1.
The literature suggests roughly -0.1 for football; we tune empirically.
"""

LAMBDA_HOME_AT_PARITY = 1.41
LAMBDA_AWAY_AT_PARITY = 1.07
SLOPE_HOME = 0.00204
SLOPE_AWAY = -0.00255

# Neutral-venue constants. The home/away lambdas above bake in real home
# advantage; applying them to neutral matches biases the first-listed team
# by ~9pp (see audit, issue #10). For neutral matches use the average of
# the two: same expected total goals, no first-listed bias.
LAMBDA_NEUTRAL_AT_PARITY = (LAMBDA_HOME_AT_PARITY + LAMBDA_AWAY_AT_PARITY) / 2
SLOPE_NEUTRAL = (abs(SLOPE_HOME) + abs(SLOPE_AWAY)) / 2
# Dixon-Coles rho: -0.10 is the literature value for English club football,
# but a backtest sweep across WC 2010 / 2014 / 2018 found rho ~= 0 was best
# on our international-tournament data (years split between slightly negative
# and slightly positive). The DC machinery is in place for future tuning;
# currently zeroed out so we get pure independent-Poisson behaviour.
DIXON_COLES_RHO = 0.0
MAX_GOALS = 10

# Post-hoc temperature scaling on W/D/L probabilities. Fitted on 762
# matches across 16 tournaments (4-fold CV by tournament), best T = 1.31
# minimises pooled log-loss and ECE. Compresses the model's tail
# confidence: an 80% prediction becomes ~70%, matching observed win rate.
# Applied inside wld_probs(); the simulator's joint-PMF sampler is
# untouched. Disable with WC_TEMPERATURE=0.
TEMPERATURE = 1.31
