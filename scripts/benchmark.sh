#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
cd "$REPO_ROOT"

TIMESTEPS="${TIMESTEPS:-50000}"
VARIANTS="${VARIANTS:-ppo-reference truly-ppo}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUNS_ROOT="${RUNS_ROOT:-$REPO_ROOT/runs/benchmark/$STAMP}"

if [ -z "$VARIANTS" ]; then
  printf 'error: VARIANTS must not be empty\n' >&2
  exit 1
fi
for variant in $VARIANTS; do
  case "$variant" in
    ppo-reference|truly-ppo) ;;
    *)
      printf 'error: unsupported training variant: %s\n' "$variant" >&2
      exit 1
      ;;
  esac
done

uv sync --locked

for variant in $VARIANTS; do
  uv run python train.py \
    --variant "$variant" \
    --seed 0 \
    --timesteps "$TIMESTEPS" \
    --run-dir "$RUNS_ROOT/$variant"
done

printf '\nBenchmark complete. Artifacts: %s\n' "$RUNS_ROOT"
printf '%s\n' 'Compare elapsed time and steps/s in each config.json before choosing the full-run budget.'
