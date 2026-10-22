#!/usr/bin/env bash
set -euo pipefail

uv sync --locked

TIMESTEPS="${TIMESTEPS:-1500000}"
SEEDS="${SEEDS:-0 1 2}"
EPISODES="${EPISODES:-100}"
VARIANTS="${VARIANTS:-baseline ppo-normalized truly-ppo}"
RESUME="${RESUME:-0}"

resume_flag=()
if [ "$RESUME" = "1" ]; then
  resume_flag=(--resume)
fi

for variant in $VARIANTS; do
  for seed in $SEEDS; do
    run_dir="runs/main/${variant}/seed_${seed}"
    if [ -f "$run_dir/final_model.zip" ] && [ -f "$run_dir/evaluation/summary.json" ]; then
      echo "Skipping completed run: $run_dir"
      continue
    fi
    uv run python train.py \
      --variant "$variant" \
      --seed "$seed" \
      --timesteps "$TIMESTEPS" \
      --run-dir "$run_dir" \
      ${resume_flag[@]+"${resume_flag[@]}"}
    uv run python evaluate.py \
      --model "$run_dir/final_model.zip" \
      --variant "$variant" \
      --episodes "$EPISODES" \
      --seed 40000 \
      --output-dir "$run_dir/evaluation"
  done
done
