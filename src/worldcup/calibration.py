"""Single source of truth for the goals-model calibration constants.

Lives in its own tiny module so simulator.py, form.py, and backtest.py
can all import it without creating circular dependencies.

Slopes have a 0.85 multiplier baked in vs the raw binned-mean fit,
chosen by sweeping log-loss against WC 2010, 2014, and 2018.
"""

LAMBDA_HOME_AT_PARITY = 1.41
LAMBDA_AWAY_AT_PARITY = 1.07
SLOPE_HOME = 0.00204
SLOPE_AWAY = -0.00255
