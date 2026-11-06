# 2026 layer ablation

Ratings at 2026-06-10 (raw Elo, profile v4a-flat-mild), 104 matches, 3-way log-loss, lower is better.
Negative "vs plain Elo" means the layers helped. Intervals resample matches; with n = 104 they are wide.
Approximate rebuild: the frozen file was generated 2026-06-06 from slightly earlier inputs, so rebuilt ratings differ from it by up to 58 Elo points (mean 19). Read the layer effects as indicative, not an exact replay.

## Cumulative (layers added in simulator order)

| Config | Log-loss | vs plain Elo | 95% CI (paired bootstrap) |
|---|---:|---:|---|
| Elo only | 0.8608 | +0.0000 | [+0.0000, +0.0000] |
| + squad | 0.8514 | -0.0094 | [-0.0301, +0.0100] |
| + form | 0.8571 | -0.0037 | [-0.0337, +0.0245] |
| + nt | 0.8498 | -0.0111 | [-0.0388, +0.0151] |
| + host | 0.8499 | -0.0109 | [-0.0384, +0.0149] |
| + altitude | 0.8481 | -0.0128 | [-0.0406, +0.0138] |

## Leave-one-out

| Config | Log-loss | vs plain Elo | 95% CI (paired bootstrap) |
|---|---:|---:|---|
| All layers | 0.8481 | -0.0128 | [-0.0406, +0.0138] |
| All minus squad | 0.8596 | -0.0012 | [-0.0208, +0.0195] |
| All minus form | 0.8443 | -0.0165 | [-0.0373, +0.0031] |
| All minus nt | 0.8545 | -0.0063 | [-0.0353, +0.0208] |
| All minus host | 0.8477 | -0.0131 | [-0.0409, +0.0136] |
| All minus altitude | 0.8499 | -0.0109 | [-0.0384, +0.0149] |
