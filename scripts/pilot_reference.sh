#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
cd "$REPO_ROOT"

TIMESTEPS="${TIMESTEPS:-1000000}"
SEED="${SEED:-0}"
EVAL_EVERY="${EVAL_EVERY:-50000}"
EVAL_EPISODES="${EVAL_EPISODES:-10}"
FINAL_EPISODES="${FINAL_EPISODES:-100}"
EVAL_SEED="${EVAL_SEED:-50000}"
FINAL_EVAL_SEED="${FINAL_EVAL_SEED:-40000}"
RESUME="${RESUME:-0}"
RUNS_ROOT="${RUNS_ROOT:-$REPO_ROOT/runs/reference_pilot}"
RUNS_ROOT="$(realpath -m -- "$RUNS_ROOT")"

case "$RESUME" in
  0|1) ;;
  *)
    printf 'error: RESUME must be 0 or 1 (got %s)\n' "$RESUME" >&2
    exit 1
    ;;
esac

if [ "$RESUME" = "1" ] && [ -z "${RUN_DIR:-}" ]; then
  printf 'error: RESUME=1 requires an explicit RUN_DIR; refusing to create a new directory and call it a resume\n' >&2
  exit 1
fi

if [ -n "${RUN_DIR:-}" ]; then
  RUN_DIR_ABS="$(realpath -m -- "$RUN_DIR")"
else
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  RUN_DIR_ABS="$RUNS_ROOT/${stamp}_seed_${SEED}"
  suffix=1
  while [ -e "$RUN_DIR_ABS" ]; do
    RUN_DIR_ABS="$RUNS_ROOT/${stamp}_seed_${SEED}_${suffix}"
    suffix=$((suffix + 1))
  done
fi

case "$RUN_DIR_ABS" in
  "$REPO_ROOT/runs/main"|"$REPO_ROOT/runs/main/"*)
    printf 'error: refusing to write under runs/main: %s\n' "$RUN_DIR_ABS" >&2
    exit 1
    ;;
esac

if [ -e "$RUN_DIR_ABS" ] && [ ! -d "$RUN_DIR_ABS" ]; then
  printf 'error: RUN_DIR exists and is not a directory: %s\n' "$RUN_DIR_ABS" >&2
  exit 1
fi
if [ -d "$RUN_DIR_ABS" ] && [ "$RESUME" != "1" ] \
  && [ -n "$(find "$RUN_DIR_ABS" -mindepth 1 -maxdepth 1 -print -quit)" ]; then
  printf 'error: RUN_DIR is not empty: %s\n' "$RUN_DIR_ABS" >&2
  printf 'Set RESUME=1 to continue it, or use a new RUN_DIR.\n' >&2
  exit 1
fi

mkdir -p -- "$RUN_DIR_ABS"
LOG_FILE="$RUN_DIR_ABS/console.log"

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$LOG_FILE"
}

# Keep the command's exit status instead of tee's status; set -e then aborts
# with the real failure code.
run_logged() {
  local status
  set +e
  "$@" 2>&1 | tee -a "$LOG_FILE"
  status=${PIPESTATUS[0]}
  set -e
  return "$status"
}

trap 'code=$?; log "pilot_reference.sh exited with code $code"' EXIT

log "reference pilot: variant=ppo-reference seed=$SEED timesteps=$TIMESTEPS"
log "eval_every=$EVAL_EVERY eval_episodes=$EVAL_EPISODES eval_seed=$EVAL_SEED"
log "final_episodes=$FINAL_EPISODES final_eval_seed=$FINAL_EVAL_SEED resume=$RESUME"
log "run directory: $RUN_DIR_ABS"

run_logged uv sync --locked

train_args=(
  --variant ppo-reference
  --seed "$SEED"
  --timesteps "$TIMESTEPS"
  --run-dir "$RUN_DIR_ABS"
  --eval-every "$EVAL_EVERY"
  --eval-episodes "$EVAL_EPISODES"
  --eval-seed "$EVAL_SEED"
)
if [ "$RESUME" = "1" ]; then
  train_args+=(--resume)
fi

log "training"
run_logged uv run python train.py "${train_args[@]}"

if [ ! -f "$RUN_DIR_ABS/final_model.zip" ]; then
  log "error: training finished without final_model.zip"
  exit 1
fi

log "final evaluation: 3 conditions, $FINAL_EPISODES episodes, hold-out seed $FINAL_EVAL_SEED"
run_logged uv run python evaluate.py \
  --model "$RUN_DIR_ABS/final_model.zip" \
  --variant ppo-reference \
  --vecnormalize "$RUN_DIR_ABS/vecnormalize.pkl" \
  --conditions nominal heldout_low heldout_high \
  --episodes "$FINAL_EPISODES" \
  --seed "$FINAL_EVAL_SEED" \
  --output-dir "$RUN_DIR_ABS/evaluation"

BEST_MODEL="$RUN_DIR_ABS/best/best_model.zip"
BEST_STATS="$RUN_DIR_ABS/best/vecnormalize.pkl"
if [ -f "$BEST_MODEL" ] && [ -f "$BEST_STATS" ]; then
  log "best-model evaluation (diagnostic only; same hold-out final seeds)"
  run_logged uv run python evaluate.py \
    --model "$BEST_MODEL" \
    --variant ppo-reference \
    --vecnormalize "$BEST_STATS" \
    --conditions nominal heldout_low heldout_high \
    --episodes "$FINAL_EPISODES" \
    --seed "$FINAL_EVAL_SEED" \
    --output-dir "$RUN_DIR_ABS/best/evaluation"
elif [ -f "$BEST_MODEL" ]; then
  log "warning: $BEST_MODEL exists but $BEST_STATS is missing; skipping best evaluation"
else
  log "no best/best_model.zip was produced; skipping best-model evaluation"
fi

run_logged uv run python - "$RUN_DIR_ABS/evaluation/summary.json" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1], encoding="utf-8"))
print("Final deterministic evaluation (hold-out per-episode seeds):")
for name in ("nominal", "heldout_low", "heldout_high"):
    condition = summary["conditions"][name]
    print(
        f"  {name}: return={condition['mean_return']:.1f} "
        f"+/- {condition['std_return']:.1f}; "
        f"length={condition['mean_episode_length']:.1f}; "
        f"fall_rate={condition['fall_rate']:.1%}; "
        f"healthy_time_limit_rate={condition['healthy_time_limit_rate']:.1%}"
    )
PY

log "artifacts: $RUN_DIR_ABS"
log "final_model is the primary result; best_model is diagnostic only."
log "A healthy 1000-step episode rate is a project target, not a guarantee."
log "Download the whole run directory (console, Monitor CSV, metrics, periodic npz, final evaluation)."
