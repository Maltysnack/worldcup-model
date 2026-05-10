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
# Dixon-Coles rho: -0.10 is the literature value for English club football,
# but a backtest sweep across WC 2010 / 2014 / 2018 found rho ~= 0 was best
# on our international-tournament data (years split between slightly negative
# and slightly positive). The DC machinery is in place for future tuning;
# currently zeroed out so we get pure independent-Poisson behaviour.
DIXON_COLES_RHO = 0.0
MAX_GOALS = 10
