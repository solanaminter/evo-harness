#!/usr/bin/env python3
"""exp1_core runner: load config.yaml, run all (seed, arm) evolutions, print summary.

Usage (from repo root):
    python3 -m evoharness.run_exp1            # or:
    PYTHONPATH=src python3 experiments/exp1_core/run.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from evoharness.config import Config  # noqa: E402
from evoharness.evolve import run_all  # noqa: E402


def summarize(log_paths):
    print("\n=== exp1_core summary (final generation) ===")
    for lp in sorted(log_paths):
        with open(lp) as f:
            rows = list(csv.DictReader(f))
        last = rows[-1]
        print(f"{lp.parent.parent.name}/{lp.parent.name}: "
              f"best={last['best']} mean={last['mean']} "
              f"tools=[{last['best_tools']}]")


def main():
    cfg = Config.from_yaml(HERE / "config.yaml")
    cfg.results_dir = HERE / cfg.results_dir
    logs = run_all(cfg)
    summarize(logs)


if __name__ == "__main__":
    main()
