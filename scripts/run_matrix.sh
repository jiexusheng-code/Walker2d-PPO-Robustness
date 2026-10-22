#!/usr/bin/env bash
set -euo pipefail

uv sync --locked

TIMESTEPS="${TIMESTEPS:-1000000}"
SEEDS="${SEEDS:-0 1 2}"
EPISODES="${EPISODES:-100}"

for variant in baseline domain-rand; do
  for seed in $SEEDS; do
    run_dir="runs/main/${variant}/seed_${seed}"
    uv run python train.py \
      --variant "$variant" \
      --seed "$seed" \
      --timesteps "$TIMESTEPS" \
      --run-dir "$run_dir"
    uv run python evaluate.py \
      --model "$run_dir/final_model.zip" \
      --episodes "$EPISODES" \
      --seed 40000 \
      --output-dir "$run_dir/evaluation"
  done
done
