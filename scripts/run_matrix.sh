#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
cd "$REPO_ROOT"

# Formal two-arm matrix. Both arms use the same shared reference profile; only
# the actor objective differs. Defaults to one 1,000,000-step pilot per arm at
# seed 0. Use SEEDS="0 1 2" only after reviewing seed 0.
TIMESTEPS="${TIMESTEPS:-1000000}"
SEEDS="${SEEDS:-0}"
EPISODES="${EPISODES:-100}"
VARIANTS="${VARIANTS:-ppo-reference truly-ppo}"
RESUME="${RESUME:-0}"
EVAL_EVERY="${EVAL_EVERY:-50000}"
EVAL_EPISODES="${EVAL_EPISODES:-10}"
EVAL_SEED="${EVAL_SEED:-50000}"
FINAL_EVAL_SEED="${FINAL_EVAL_SEED:-40000}"
# New root for the formal two-arm comparison. Do not reuse runs/main, which
# holds the retired three-group history.
RUNS_ROOT="${RUNS_ROOT:-runs/two_arm}"

case "$RESUME" in
  0|1) ;;
  *)
    printf 'error: RESUME must be 0 or 1 (got %s)\n' "$RESUME" >&2
    exit 1
    ;;
esac

if [ -z "$SEEDS" ]; then
  printf 'error: SEEDS must not be empty\n' >&2
  exit 1
fi
if [ -z "$VARIANTS" ]; then
  printf 'error: VARIANTS must not be empty\n' >&2
  exit 1
fi
if [ -z "$RUNS_ROOT" ]; then
  printf 'error: RUNS_ROOT must not be empty\n' >&2
  exit 1
fi
for variant in $VARIANTS; do
  case "$variant" in
    ppo-reference|truly-ppo) ;;
    *)
      printf 'error: unsupported training variant: %s\n' "$variant" >&2
      printf '%s\n' \
        'The two-arm matrix trains only ppo-reference and truly-ppo. Historical baseline models are evaluation-only.' >&2
      exit 1
      ;;
  esac
done

printf 'two-arm matrix: shared_profile=ppo-reference variants=%s seeds=%s timesteps=%s episodes=%s final_eval_seed=%s runs_root=%s\n' \
  "$VARIANTS" "$SEEDS" "$TIMESTEPS" "$EPISODES" "$FINAL_EVAL_SEED" "$RUNS_ROOT"

uv sync --locked

evaluate_run() {
  local variant="$1"
  local run_dir="$2"
  uv run python evaluate.py \
    --model "$run_dir/final_model.zip" \
    --variant "$variant" \
    --episodes "$EPISODES" \
    --seed "$FINAL_EVAL_SEED" \
    --output-dir "$run_dir/evaluation"

  # best_model is diagnostic only; the final_model evaluation above remains
  # the primary result.
  if [ -f "$run_dir/best/best_model.zip" ] \
    && [ -f "$run_dir/best/vecnormalize.pkl" ]; then
    uv run python evaluate.py \
      --model "$run_dir/best/best_model.zip" \
      --variant "$variant" \
      --vecnormalize "$run_dir/best/vecnormalize.pkl" \
      --episodes "$EPISODES" \
      --seed "$FINAL_EVAL_SEED" \
      --output-dir "$run_dir/best/evaluation"
  fi
}

for variant in $VARIANTS; do
  for seed in $SEEDS; do
    run_dir="${RUNS_ROOT%/}/${variant}/seed_${seed}"
    plan_args=(
      --run-dir "$run_dir"
      --variant "$variant"
      --seed "$seed"
      --budget "$TIMESTEPS"
      --episodes "$EPISODES"
      --eval-seed "$FINAL_EVAL_SEED"
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
    printf '[%s seed %s] variant=%s shared_profile=ppo-reference run_dir=%s\n' \
      "$variant" "$seed" "$variant" "$run_dir"
    printf '[%s seed %s] %s: %s\n' "$variant" "$seed" "$action" "$reason"

    case "$action" in
      skip)
        continue
        ;;
      evaluate)
        evaluate_run "$variant" "$run_dir"
        ;;
      train|resume)
        train_args=(
          --variant "$variant"
          --seed "$seed"
          --timesteps "$TIMESTEPS"
          --run-dir "$run_dir"
          --eval-every "$EVAL_EVERY"
          --eval-episodes "$EVAL_EPISODES"
          --eval-seed "$EVAL_SEED"
        )
        if [ "$action" = "resume" ]; then
          train_args+=(--resume)
        fi
        uv run python train.py "${train_args[@]}"
        evaluate_run "$variant" "$run_dir"
        ;;
      *)
        printf 'ERROR [%s seed %s]: unexpected plan action: %s\n' \
          "$variant" "$seed" "$action" >&2
        exit 1
        ;;
    esac
  done
done

printf '%s\n' ''
printf '%s\n' \
  'Two-arm matrix complete. final_model evaluation is the primary result; best/ is diagnostic only.'
