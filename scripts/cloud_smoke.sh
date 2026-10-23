#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
cd "$REPO_ROOT"

VARIANTS="${VARIANTS:-ppo-reference truly-ppo}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUNS_ROOT="${RUNS_ROOT:-$REPO_ROOT/runs/smoke/$STAMP}"

if [ -z "$VARIANTS" ]; then
  printf 'error: VARIANTS must not be empty\n' >&2
  exit 1
fi
for variant in $VARIANTS; do
  case "$variant" in
    ppo-reference|truly-ppo|rpo) ;;
    *)
      printf 'error: unsupported training variant: %s\n' "$variant" >&2
      exit 1
      ;;
  esac
done

uv sync --locked
PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
  uv run python -m unittest discover -s tests -v

for variant in $VARIANTS; do
  run_dir="$RUNS_ROOT/$variant"
  uv run python train.py \
    --variant "$variant" \
    --seed 0 \
    --timesteps 4096 \
    --run-dir "$run_dir"
  if [ ! -f "$run_dir/final_model.zip" ]; then
    printf 'error: %s finished without final_model.zip\n' "$variant" >&2
    exit 1
  fi
  uv run python evaluate.py \
    --model "$run_dir/final_model.zip" \
    --variant "$variant" \
    --episodes 2 \
    --seed 40000 \
    --output-dir "$run_dir/evaluation"
done

printf '\nCloud smoke test passed. Artifacts: %s\n' "$RUNS_ROOT"
printf '%s\n' 'Inspect them before launching the full two-arm matrix.'
