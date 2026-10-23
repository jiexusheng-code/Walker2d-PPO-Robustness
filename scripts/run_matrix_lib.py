"""Small, dependency-free decision helper for scripts/run_matrix.sh.

The matrix must not mistake an older, shorter completed run for a run that
satisfies the current budget. It also must not silently overwrite an existing
run when the budget grows without RESUME=1, and it must not treat a stale
evaluation (another model, statistics file, episode count, or base seed) as
current.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

Action = Literal["skip", "evaluate", "train", "resume", "error"]
TRAINING_VARIANTS = ("ppo-reference", "truly-ppo", "rpo")
REFERENCE_PROFILE = "ppo-reference"
ACTOR_OBJECTIVES = {
    "ppo-reference": "ppo_clip",
    "truly-ppo": "truly_ppo_kl_triggered_rollback",
    "rpo": "rpo_uniform_mean_perturbation",
}
REQUIRED_CONDITIONS = {"nominal", "heldout_low", "heldout_high"}
INCOMPLETE_STATUSES = {"running", "interrupted", "failed"}
RUN_ARTIFACTS = (
    "final_model.zip",
    "vecnormalize.pkl",
    "config.json",
    "metrics.jsonl",
    "monitor.csv",
    "evaluations.npz",
)
DIRECTORY_ARTIFACTS = ("checkpoints", "best", "tensorboard", "evaluation")


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


def existing_artifacts(run_dir: Path) -> list[Path]:
    found = [
        run_dir / name for name in RUN_ARTIFACTS
        if (run_dir / name).exists()
    ]
    for name in DIRECTORY_ARTIFACTS:
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


def evaluation_current(
    run_dir: Path, variant: str, episodes: int, eval_seed: int
) -> tuple[bool, str]:
    """Whether evaluation/summary.json describes this exact final result."""
    summary_path = run_dir / "evaluation" / "summary.json"
    summary = read_json(summary_path)
    if summary is None:
        return False, "evaluation/summary.json is missing or invalid"
    if summary.get("variant") != variant:
        return False, (
            f"evaluation variant {summary.get('variant')!r} != {variant!r}"
        )
    expected_model = (run_dir / "final_model.zip").resolve()
    if not expected_model.is_file():
        return False, "final_model.zip is missing"
    if expected_model not in _resolved_model_path(summary, run_dir):
        return False, f"evaluation does not reference {expected_model}"
    if summary.get("deterministic") is not True:
        return False, "evaluation is not marked deterministic"

    summary_model_hash = summary.get("model_sha256")
    if not isinstance(summary_model_hash, str) or not summary_model_hash:
        return False, "evaluation has no model SHA-256"
    if file_sha256(expected_model) != summary_model_hash:
        return False, "evaluation model SHA-256 does not match final_model.zip"

    stats = run_dir / "vecnormalize.pkl"
    if not stats.is_file():
        return False, "vecnormalize.pkl is missing"
    summary_stats_hash = summary.get("vecnormalize_sha256")
    if not isinstance(summary_stats_hash, str) or not summary_stats_hash:
        return False, "evaluation has no VecNormalize SHA-256"
    if file_sha256(stats) != summary_stats_hash:
        return False, (
            "evaluation VecNormalize SHA-256 does not match vecnormalize.pkl"
        )

    if summary.get("episodes_per_condition") != episodes:
        return False, (
            f"evaluation has "
            f"{summary.get('episodes_per_condition')!r} episodes per condition "
            f"!= {episodes!r}"
        )
    if summary.get("base_seed") != eval_seed:
        return False, (
            f"evaluation base seed {summary.get('base_seed')!r} != "
            f"{eval_seed!r}"
        )
    conditions = summary.get("conditions")
    if not isinstance(conditions, dict):
        return False, "evaluation has no per-condition results"
    if not REQUIRED_CONDITIONS.issubset(conditions):
        missing = sorted(REQUIRED_CONDITIONS - set(conditions))
        return False, f"evaluation is missing conditions: {', '.join(missing)}"
    return True, "evaluation matches the final model, statistics, and protocol"


def _config_matches(
    config: dict, variant: str, seed: int, rpo_alpha: float | None
) -> tuple[bool, str]:
    if config.get("variant") != variant:
        return False, (
            f"variant {config.get('variant')!r} != {variant!r}"
        )
    if config.get("seed") != seed:
        return False, f"seed {config.get('seed')!r} != {seed!r}"
    if config.get("profile") != REFERENCE_PROFILE:
        return False, (
            f"profile {config.get('profile')!r} != {REFERENCE_PROFILE!r}"
        )
    if config.get("environment") != "Walker2d-v5":
        return False, (
            f"environment {config.get('environment')!r} != 'Walker2d-v5'"
        )
    actor = config.get("actor")
    actor_type = actor.get("objective") if isinstance(actor, dict) else None
    expected_actor = ACTOR_OBJECTIVES[variant]
    if actor_type is None:
        # Reference runs written before the explicit actor record used the
        # same PPO objective and remain compatible. A truly-ppo run without
        # its actor record is an old, unsupported base.
        if variant != "ppo-reference":
            return False, "actor record is missing"
    elif actor_type != expected_actor:
        return False, (
            f"actor objective {actor_type!r} != {expected_actor!r}"
        )
    if variant == "rpo":
        rpo = actor.get("rpo") if isinstance(actor, dict) else None
        alpha = rpo.get("alpha") if isinstance(rpo, dict) else None
        if not isinstance(alpha, (int, float)) or isinstance(alpha, bool):
            return False, "actor rpo alpha is missing"
        if rpo_alpha is not None and not math.isclose(
            float(alpha), rpo_alpha, rel_tol=0.0, abs_tol=1e-12
        ):
            return False, (
                f"actor rpo alpha {alpha!r} != requested RPO_ALPHA "
                f"{rpo_alpha!r}"
            )
    return True, "configuration matches"


def _has_resume_point(run_dir: Path) -> bool:
    if (run_dir / "final_model.zip").is_file():
        return True
    checkpoint_dir = run_dir / "checkpoints"
    return checkpoint_dir.is_dir() and any(
        checkpoint_dir.glob("*_*_steps.zip")
    )


def parse_requested_alpha(value: object) -> tuple[float | None, str | None]:
    """Return ``(alpha, error)`` for an optional requested RPO alpha."""
    if value is None:
        return None, None
    try:
        alpha = float(value)
    except (TypeError, ValueError):
        return None, (
            f"RPO_ALPHA must be a finite, non-negative number, got {value!r}"
        )
    if not math.isfinite(alpha) or alpha < 0.0:
        return None, (
            f"RPO_ALPHA must be a finite, non-negative number, got {value!r}"
        )
    return alpha, None


def plan_run(
    run_dir: Path,
    *,
    variant: str,
    seed: int,
    budget: int,
    episodes: int,
    eval_seed: int,
    resume: bool,
    rpo_alpha: object = None,
) -> RunPlan:
    run_dir = Path(run_dir)
    if variant not in TRAINING_VARIANTS:
        return RunPlan(
            "error",
            f"unsupported training variant {variant!r}; this project trains "
            f"only {', '.join(TRAINING_VARIANTS)}. Historical baseline models "
            f"are supported only by evaluate.py",
            None,
        )
    alpha, alpha_error = parse_requested_alpha(rpo_alpha)
    if alpha_error is not None:
        return RunPlan("error", alpha_error, None)
    if alpha is not None and variant != "rpo":
        return RunPlan(
            "error",
            f"RPO_ALPHA is only valid for the rpo variant, not {variant!r}",
            None,
        )

    config_path = run_dir / "config.json"
    config = read_json(config_path)
    artifacts = existing_artifacts(run_dir)
    if config_path.exists() and config is None:
        return RunPlan(
            "error",
            f"{config_path} is invalid or unreadable and existing artifacts "
            f"must not be overwritten; repair or remove them explicitly",
            None,
        )
    if config is not None:
        config_ok, config_reason = _config_matches(
            config, variant, seed, alpha
        )
        if not config_ok:
            return RunPlan(
                "error",
                f"{config_path} does not match the current shared profile "
                f"({config_reason}); refusing to reuse or resume it",
                completed_steps(config),
            )
    else:
        config_ok = False
    if config is None and artifacts:
        listed = ", ".join(str(path) for path in artifacts)
        return RunPlan(
            "error",
            f"cannot resume {variant} without {config_path}; existing "
            f"artifacts: {listed}. Refusing to start fresh over them",
            None,
        )

    steps = completed_steps(config)
    final_model = run_dir / "final_model.zip"
    complete = (
        final_model.is_file()
        and config_ok
        and steps is not None
        and steps >= budget
        and not status_is_incomplete(config)
    )
    if complete and not (run_dir / "vecnormalize.pkl").is_file():
        return RunPlan(
            "error",
            f"completed run {run_dir} has no vecnormalize.pkl; refusing to "
            f"evaluate with missing statistics. Restore it or use a new run "
            f"directory",
            steps,
        )
    if complete:
        current, reason = evaluation_current(
            run_dir, variant, episodes, eval_seed
        )
        if current:
            return RunPlan(
                "skip",
                f"completed {steps} steps >= budget {budget} and {reason}",
                steps,
            )
        return RunPlan(
            "evaluate",
            f"completed {steps} steps >= budget {budget}, but {reason}",
            steps,
        )

    if resume:
        if not _has_resume_point(run_dir):
            if config is None and not artifacts:
                # RESUME=1 is a matrix-wide mode. An untouched later arm is a
                # fresh start, not a failure to resume.
                return RunPlan(
                    "train",
                    f"no existing artifacts in {run_dir}; starting fresh "
                    f"for budget {budget}",
                    None,
                )
            return RunPlan(
                "error",
                f"no final_model.zip or checkpoint in {run_dir}; there is "
                f"nothing to resume, and restarting over existing artifacts "
                f"is refused",
                steps,
            )
        return RunPlan(
            "resume",
            f"resuming {run_dir} toward budget {budget} "
            f"(recorded {steps if steps is not None else 'unknown'} steps)",
            steps,
        )

    if artifacts:
        recorded = (
            f"{steps} steps" if steps is not None else
            "an unverified step count"
        )
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
    plan_parser.add_argument("--episodes", type=int, required=True)
    plan_parser.add_argument("--eval-seed", type=int, required=True)
    plan_parser.add_argument("--resume", action="store_true")
    plan_parser.add_argument(
        "--rpo-alpha",
        help="Requested RPO alpha; must match a completed run's config.",
    )
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
        episodes=args.episodes,
        eval_seed=args.eval_seed,
        resume=args.resume,
        rpo_alpha=args.rpo_alpha,
    )
    print(f"{plan.action}|{plan.reason}")
    return 3 if plan.action == "error" else 0


if __name__ == "__main__":
    sys.exit(main())
