#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
cd "$REPO_ROOT"

uv sync --locked

TIMESTEPS="${TIMESTEPS:-1500000}"
SEEDS="${SEEDS:-0 1 2}"
EPISODES="${EPISODES:-100}"
VARIANTS="${VARIANTS:-baseline ppo-normalized truly-ppo}"
RESUME="${RESUME:-0}"

for variant in $VARIANTS; do
  for seed in $SEEDS; do
    run_dir="runs/main/${variant}/seed_${seed}"
    plan_args=(
      --run-dir "$run_dir"
      --variant "$variant"
      --seed "$seed"
      --budget "$TIMESTEPS"
    )
    if [ "$RESUME" = "1" ]; then
      plan_args+=(--resume)
    fi

    if ! plan="$(uv run python scripts/run_matrix_lib.py plan "${plan_args[@]}")"; then
      printf 'ERROR [%s seed %s]: %s\n' "$variant" "$seed" "${plan#*|}" >&2
      exit 1
    fi
    action="${plan%%|*}"
    reason="${plan#*|}"
    printf '[%s seed %s] %s: %s\n' "$variant" "$seed" "$action" "$reason"

    case "$action" in
      skip)
        continue
        ;;
      evaluate)
        ;;
      train|resume)
        train_args=(
          --variant "$variant"
          --seed "$seed"
          --timesteps "$TIMESTEPS"
          --run-dir "$run_dir"
        )
        if [ "$action" = "resume" ]; then
          train_args+=(--resume)
        fi
        uv run python train.py "${train_args[@]}"
        ;;
      *)
        printf 'ERROR [%s seed %s]: unexpected plan action: %s\n' \
          "$variant" "$seed" "$action" >&2
        exit 1
        ;;
    esac

    uv run python evaluate.py \
      --model "$run_dir/final_model.zip" \
      --variant "$variant" \
      --episodes "$EPISODES" \
      --seed 40000 \
      --output-dir "$run_dir/evaluation"
  done
done
