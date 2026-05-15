"""Run the bracket + simulator pipeline twice, once per layer set.

Produces both:

  * ``data/bracket.json``                – champion-finder (full layers)
  * ``data/simulation.json``             – champion-finder (full layers)
  * ``data/bracket-calibration.json``    – calibration (no host)
  * ``data/simulation-calibration.json`` – calibration (no host)

The website's headline bracket / champion view reads the unsuffixed files;
per-match displays (live tab, market overlay, betting strip) should read
the ``-calibration`` variant for honestly-calibrated probabilities.

Run: ``uv run worldcup-build-dual``

This is a thin orchestrator: each layer set spawns a fresh subprocess so
the module-level ``WC_LAYERS`` resolution at import time picks up the
right value. Faster than reloading modules in-process and matches how
``WC_PROFILE`` is already used.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path


LAYER_SETS = ["full", "calibration"]
COMMANDS = ["worldcup-bracket", "worldcup-simulate"]


def run_one(layer_set: str) -> None:
    print(f"\n{'='*70}\nLayer set: {layer_set}\n{'='*70}")
    env = os.environ.copy()
    env["WC_LAYERS"] = layer_set
    for cmd in COMMANDS:
        t0 = time.time()
        print(f"\n[{layer_set}] -> {cmd}")
        result = subprocess.run([cmd], env=env, check=False)
        if result.returncode != 0:
            print(f"\n!!! {cmd} (layers={layer_set}) failed with exit {result.returncode}")
            sys.exit(result.returncode)
        print(f"[{layer_set}] {cmd} done in {time.time() - t0:.1f}s")


def main() -> None:
    t0 = time.time()
    for ls in LAYER_SETS:
        run_one(ls)
    print(f"\n{'='*70}\nBoth layer sets built in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    sys.exit(main())
