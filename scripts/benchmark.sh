#!/usr/bin/env bash
set -euo pipefail

uv sync --locked

TIMESTEPS="${TIMESTEPS:-50000}"
for variant in baseline domain-rand; do
  uv run python train.py \
    --variant "$variant" \
    --seed 0 \
    --timesteps "$TIMESTEPS" \
    --run-dir "runs/benchmark/${variant}"
done

printf '\nBenchmark complete. Check elapsed time and steps/s in runs/benchmark/*/config.json before selecting the full-run budget.\n'
