#!/usr/bin/env bash
set -euo pipefail

uv sync --locked
uv run python -m unittest discover -s tests -v

for variant in baseline domain-rand; do
  run_dir="runs/smoke/${variant}"
  uv run python train.py \
    --variant "$variant" \
    --seed 0 \
    --timesteps 4096 \
    --run-dir "$run_dir"
  uv run python evaluate.py \
    --model "$run_dir/final_model.zip" \
    --episodes 2 \
    --seed 40000 \
    --output-dir "$run_dir/evaluation"
done

printf '\nCloud smoke test passed. Inspect runs/smoke/ before launching the full matrix.\n'
