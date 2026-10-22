"""Small, dependency-free decision helper for scripts/run_matrix.sh.

The matrix must not mistake an older, shorter completed run for a run that
satisfies the current budget.  It also must not silently overwrite an existing
run when the budget grows without RESUME=1.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

Action = Literal["skip", "evaluate", "train", "resume", "error"]
REQUIRED_CONDITIONS = {"nominal", "heldout_low", "heldout_high"}
REFERENCE_VARIANT = "ppo-reference"
INCOMPLETE_STATUSES = {"running", "interrupted", "failed"}
REFERENCE_ARTIFACTS = (
    "final_model.zip",
    "vecnormalize.pkl",
    "config.json",
    "metrics.jsonl",
    "monitor.csv",
    "evaluations.npz",
)


@dataclass(frozen=True)
class RunPlan:
    action: Action
    reason: str
    completed_steps: int | None


def read_json(path: Path) -> dict | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _as_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def completed_steps(config: dict | None) -> int | None:
    """Return `actual_timesteps`, which train.py writes on completion."""
    if not isinstance(config, dict):
        return None
    return _as_int(config.get("actual_timesteps"))


def status_is_incomplete(config: dict | None) -> bool:
    if not isinstance(config, dict):
        return False
    status = config.get("status")
    if not isinstance(status, str):
        return False
    return status.strip().lower() in INCOMPLETE_STATUSES


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def existing_artifacts(run_dir: Path, variant: str) -> list[Path]:
    names = (
        REFERENCE_ARTIFACTS
        if variant == REFERENCE_VARIANT
        else ("final_model.zip", "config.json")
    )
    found = [run_dir / name for name in names if (run_dir / name).exists()]
    directories = ("checkpoints",)
    if variant == REFERENCE_VARIANT:
        directories += ("best", "tensorboard")
    for name in directories:
        directory = run_dir / name
        if directory.is_dir() and any(directory.iterdir()):
            found.append(directory)
    return found


def _resolved_model_path(summary: dict, run_dir: Path) -> set[Path]:
    raw = summary.get("model")
    if not isinstance(raw, str):
        return set()
    path = Path(raw)
    candidates = [path]
    if not path.is_absolute():
        candidates.append(Path.cwd() / path)
        candidates.append(run_dir / path)
    return {candidate.resolve() for candidate in candidates}


def evaluation_is_current(run_dir: Path, variant: str) -> bool:
    summary_path = run_dir / "evaluation" / "summary.json"
    summary = read_json(summary_path)
    if summary is None:
        return False
    if summary.get("variant") not in (None, variant):
        return False
    expected_model = (run_dir / "final_model.zip").resolve()
    if not expected_model.is_file():
        return False
    if expected_model not in _resolved_model_path(summary, run_dir):
        return False
    conditions = summary.get("conditions")
    if not isinstance(conditions, dict):
        return False
    if not REQUIRED_CONDITIONS.issubset(conditions):
        return False

    summary_hash = summary.get("model_sha256")
    if isinstance(summary_hash, str) and summary_hash:
        return file_sha256(expected_model) == summary_hash
    # Legacy summaries predate model hashes. A summary written after the
    # model was saved is the best available evidence that it matches.
    return summary_path.stat().st_mtime_ns >= expected_model.stat().st_mtime_ns


def plan_run(
    run_dir: Path,
    *,
    variant: str,
    seed: int,
    budget: int,
    resume: bool,
) -> RunPlan:
    run_dir = Path(run_dir)
    config_path = run_dir / "config.json"
    config = read_json(config_path)
    artifacts = existing_artifacts(run_dir, variant)
    if config_path.exists() and config is None:
        return RunPlan(
            "error",
            f"{config_path} is invalid or unreadable and existing artifacts "
            f"must not be overwritten; repair or remove them explicitly",
            None,
        )
    config_matches = (
        isinstance(config, dict)
        and config.get("variant") == variant
        and config.get("seed") == seed
    )
    if config is not None and not config_matches:
        return RunPlan(
            "error",
            f"{config_path} does not match variant={variant} "
            f"seed={seed}; refusing to reuse or resume it",
            completed_steps(config),
        )
    if variant == REFERENCE_VARIANT and config is None and artifacts:
        listed = ", ".join(str(path) for path in artifacts)
        return RunPlan(
            "error",
            f"cannot resume {REFERENCE_VARIANT} without {config_path}; "
            f"existing artifacts: {listed}. Refusing to start fresh over them",
            None,
        )

    steps = completed_steps(config)
    final_model = run_dir / "final_model.zip"
    complete = (
        final_model.is_file()
        and config_matches
        and steps is not None
        and steps >= budget
        and not status_is_incomplete(config)
    )
    if complete:
        if evaluation_is_current(run_dir, variant):
            return RunPlan(
                "skip",
                f"completed {steps} steps >= budget {budget} and evaluation "
                f"matches {final_model}",
                steps,
            )
        return RunPlan(
            "evaluate",
            f"completed {steps} steps >= budget {budget}, but the evaluation "
            f"is missing or does not match {final_model}",
            steps,
        )

    if resume:
        return RunPlan(
            "resume",
            f"resuming {run_dir} toward budget {budget} "
            f"(recorded {steps if steps is not None else 'unknown'} steps)",
            steps,
        )

    if artifacts:
        recorded = f"{steps} steps" if steps is not None else "an unverified step count"
        listed = ", ".join(str(path) for path in artifacts)
        return RunPlan(
            "error",
            f"existing run {run_dir} has {recorded} and artifacts: {listed}; "
            f"below budget {budget}. Set RESUME=1 to continue it, or choose a "
            f"new run directory. Refusing to restart from scratch",
            steps,
        )
    return RunPlan(
        "train",
        f"no existing artifacts in {run_dir}; starting fresh for budget {budget}",
        None,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan_parser = subparsers.add_parser("plan", help="Decide what to do for one run.")
    plan_parser.add_argument("--run-dir", type=Path, required=True)
    plan_parser.add_argument("--variant", required=True)
    plan_parser.add_argument("--seed", type=int, required=True)
    plan_parser.add_argument("--budget", type=int, required=True)
    plan_parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command != "plan":
        raise AssertionError(f"unknown command: {args.command}")
    plan = plan_run(
        args.run_dir,
        variant=args.variant,
        seed=args.seed,
        budget=args.budget,
        resume=args.resume,
    )
    print(f"{plan.action}|{plan.reason}")
    return 3 if plan.action == "error" else 0


if __name__ == "__main__":
    sys.exit(main())
