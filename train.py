from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path

import gymnasium
import mujoco
import numpy
import stable_baselines3
import torch
from stable_baselines3 import PPO
from stable_baselines3.common import utils as sb3_utils
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from diagnostics import (
    HealthyEvalCallback,
    JsonlMetricsWriter,
    SaveVecNormalizeOnBest,
    TrainingMetricsCallback,
    git_metadata,
    restore_eval_history,
    runtime_metadata,
    source_hashes,
)
from envs import make_env
from truly_ppo import TrulyPPO


REFERENCE_VARIANT = "ppo-reference"
TRULY_VARIANT = "truly-ppo"
VARIANTS = (REFERENCE_VARIANT, TRULY_VARIANT)
DEFAULT_VARIANT = REFERENCE_VARIANT
REFERENCE_PROFILE = REFERENCE_VARIANT
ACTOR_OBJECTIVES = {
    REFERENCE_VARIANT: "ppo_clip",
    TRULY_VARIANT: "truly_ppo_kl_triggered_rollback",
}
CHECKPOINT_PREFIX = "ppo_walker2d"
CHECKPOINT_PATTERN = re.compile(r"^(?P<prefix>.+)_(?P<steps>\d+)_steps\.zip$")

REFERENCE_LEARNING_RATE = 5.05041e-5
REFERENCE_SOURCE_URL = (
    "https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/ppo.yml"
)
TRULY_PPO_SOURCE = (
    "Y. Wang, H. He, C. Wen, X. Tan, Truly Proximal Policy Optimization "
    "(arXiv:1903.07940, 2019; extended 2020 version)"
)
REFERENCE_PPO_KWARGS = {
    "n_steps": 512,
    "batch_size": 32,
    "n_epochs": 20,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.1,
    "ent_coef": 0.000585045,
    "vf_coef": 0.871923,
    "max_grad_norm": 1.0,
    "policy_kwargs": {"net_arch": {"pi": [64, 64], "vf": [64, 64]}},
    "device": "cpu",
}
POLICY_DEFAULTS = {
    "activation_fn": "Tanh",
    "ortho_init": True,
    "net_arch": {"pi": [64, 64], "vf": [64, 64]},
    "log_std_init": 0.0,
    "use_sde": False,
    "squash_output": False,
}
RUN_ARTIFACTS = (
    "final_model.zip",
    "vecnormalize.pkl",
    "config.json",
    "metrics.jsonl",
    "monitor.csv",
    "evaluations.npz",
)


def _require_supported_variant(variant: str) -> None:
    if variant not in VARIANTS:
        raise SystemExit(
            f"error: unsupported variant {variant!r}; train.py only trains "
            f"{', '.join(VARIANTS)}"
        )


def make_training_env(
    stats_path: Path | None = None,
    *,
    monitor_path: Path | None = None,
    monitor_append: bool = False,
):
    """Create the single normalized training environment used by both arms."""
    monitor_kwargs = {}
    if monitor_path is not None:
        monitor_kwargs = {
            "filename": str(monitor_path),
            "override_existing": not monitor_append,
        }
    vector_env = DummyVecEnv([lambda: Monitor(make_env(), **monitor_kwargs)])
    if stats_path is not None:
        env = VecNormalize.load(str(stats_path), vector_env)
        # Resuming must keep updating the loaded statistics.
        env.training = True
        env.norm_reward = True
        return env
    return VecNormalize(
        vector_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=REFERENCE_PPO_KWARGS["gamma"],
    )


def make_eval_env():
    """Independent nominal environment used only for periodic evaluation."""
    vector_env = DummyVecEnv([lambda: Monitor(make_env())])
    env = VecNormalize(
        vector_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=REFERENCE_PPO_KWARGS["gamma"],
    )
    env.training = False
    env.norm_reward = False
    return env


def model_class_for_variant(variant: str):
    _require_supported_variant(variant)
    return TrulyPPO if variant == TRULY_VARIANT else PPO


def model_kwargs_for_variant(variant: str) -> dict:
    _require_supported_variant(variant)
    return REFERENCE_PPO_KWARGS


def learning_rate_for_variant(variant: str) -> float:
    _require_supported_variant(variant)
    return REFERENCE_LEARNING_RATE


def actor_config_for_variant(variant: str) -> dict:
    _require_supported_variant(variant)
    if variant == TRULY_VARIANT:
        return {
            "objective": ACTOR_OBJECTIVES[variant],
            "clip_range_used": False,
            "kl_rollback": {
                "alpha": TrulyPPO.truly_alpha,
                "delta": TrulyPPO.truly_delta,
                "trigger": (
                    "kl(old||new) >= delta and "
                    "(ratio - 1) * advantage > 0"
                ),
                "penalty": "ratio * advantage - alpha * kl(old||new)",
                "note": (
                    "KL penalty inside the actor objective; no parameter or "
                    "optimizer state is restored."
                ),
            },
        }
    return {
        "objective": ACTOR_OBJECTIVES[variant],
        "clip_range_used": True,
        "kl_rollback": None,
    }


def default_run_dir(variant: str, seed: int) -> Path:
    _require_supported_variant(variant)
    return Path("runs") / variant / f"seed_{seed}"


def existing_run_artifacts(run_dir: Path) -> list[Path]:
    found = [
        run_dir / name for name in RUN_ARTIFACTS
        if (run_dir / name).exists()
    ]
    for name in ("checkpoints", "best", "tensorboard", "evaluation"):
        directory = run_dir / name
        if directory.is_dir() and any(directory.iterdir()):
            found.append(directory)
    return found


def ensure_run_dir_available(
    run_dir: Path, *, allow_existing_from: Path | None = None
) -> None:
    """Refuse to write into a directory that already holds another run.

    A resume source inside ``run_dir`` is expected to leave artifacts behind.
    An external resume source is allowed only when ``run_dir`` itself is new.
    """
    if allow_existing_from is not None:
        source = Path(allow_existing_from).resolve()
        if source.is_relative_to(run_dir.resolve()):
            return
    existing = existing_run_artifacts(run_dir)
    if existing:
        listed = ", ".join(str(path) for path in existing)
        raise SystemExit(
            f"error: refusing to overwrite existing run artifacts in "
            f"{run_dir}: {listed}. Use --resume pointing into this directory "
            f"or a different --run-dir."
        )


def _load_config(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _same_number(first, second) -> bool:
    try:
        return math.isclose(
            float(first), float(second), rel_tol=1e-9, abs_tol=0.0
        )
    except (TypeError, ValueError):
        return False


def _schedule_value(value) -> float:
    return float(value(1.0)) if callable(value) else float(value)


def validate_resume(
    run_dir: Path, args: argparse.Namespace, model
) -> None:
    """Reject a model whose saved configuration does not match this run.

    Both arms use the same strict reference profile. The only permitted
    difference is the recorded actor objective. This also rejects a legacy
    ``truly-ppo`` run that used the old 2048-step recipe, rather than silently
    resuming it into the new shared profile.
    """
    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise SystemExit(
            f"error: cannot resume {args.variant} without {config_path}"
        )
    config = _load_config(config_path)
    if not config:
        raise SystemExit(
            f"error: {config_path} is invalid or unreadable; refusing to "
            f"resume from it"
        )
    mismatches: list[str] = []
    variant = args.variant
    expected_algorithm = "TrulyPPO" if variant == TRULY_VARIANT else "PPO"

    if config.get("variant") != variant:
        mismatches.append(
            f"variant {config.get('variant')!r} != {variant!r}"
        )
    if config.get("seed") != args.seed:
        mismatches.append(f"seed {config.get('seed')!r} != {args.seed!r}")
    if config.get("profile") != REFERENCE_PROFILE:
        mismatches.append(
            f"profile {config.get('profile')!r} != {REFERENCE_PROFILE!r}"
        )
    if config.get("environment") != "Walker2d-v5":
        mismatches.append(
            f"environment {config.get('environment')!r} != 'Walker2d-v5'"
        )
    if config.get("algorithm") != expected_algorithm:
        mismatches.append(
            f"algorithm {config.get('algorithm')!r} != "
            f"{expected_algorithm!r}"
        )

    actual = {
        "n_steps": model.n_steps,
        "batch_size": model.batch_size,
        "n_epochs": model.n_epochs,
        "gamma": model.gamma,
        "gae_lambda": model.gae_lambda,
        "ent_coef": model.ent_coef,
        "vf_coef": model.vf_coef,
        "max_grad_norm": model.max_grad_norm,
    }
    for key, expected in actual.items():
        if not _same_number(expected, REFERENCE_PPO_KWARGS[key]):
            mismatches.append(
                f"{key} {expected!r} != {REFERENCE_PPO_KWARGS[key]!r}"
            )
    if not _same_number(
        _schedule_value(model.learning_rate), REFERENCE_LEARNING_RATE
    ):
        mismatches.append(
            f"learning_rate {_schedule_value(model.learning_rate)!r} != "
            f"{REFERENCE_LEARNING_RATE!r}"
        )
    if not _same_number(
        _schedule_value(model.clip_range),
        REFERENCE_PPO_KWARGS["clip_range"],
    ):
        mismatches.append(
            f"clip_range {_schedule_value(model.clip_range)!r} != "
            f"{REFERENCE_PPO_KWARGS['clip_range']!r}"
        )
    net_arch = (getattr(model, "policy_kwargs", None) or {}).get("net_arch")
    if net_arch != REFERENCE_PPO_KWARGS["policy_kwargs"]["net_arch"]:
        mismatches.append(f"net_arch {net_arch!r} != [64, 64] pi/vf")
    if getattr(model, "n_envs", None) != 1:
        mismatches.append(f"n_envs {getattr(model, 'n_envs', None)!r} != 1")

    saved_ppo = config.get("ppo") or {}
    for key in (
        "n_steps",
        "batch_size",
        "n_epochs",
        "gamma",
        "gae_lambda",
        "clip_range",
        "ent_coef",
        "vf_coef",
        "max_grad_norm",
    ):
        if not _same_number(saved_ppo.get(key), REFERENCE_PPO_KWARGS[key]):
            mismatches.append(
                f"config ppo.{key} {saved_ppo.get(key)!r} != "
                f"{REFERENCE_PPO_KWARGS[key]!r}"
            )
    if not _same_number(
        saved_ppo.get("learning_rate_initial"), REFERENCE_LEARNING_RATE
    ):
        mismatches.append(
            "config ppo.learning_rate_initial "
            f"{saved_ppo.get('learning_rate_initial')!r} != "
            f"{REFERENCE_LEARNING_RATE!r}"
        )
    if saved_ppo.get("learning_rate_schedule") != "constant":
        mismatches.append(
            "config ppo.learning_rate_schedule "
            f"{saved_ppo.get('learning_rate_schedule')!r} != 'constant'"
        )
    saved_net = (saved_ppo.get("policy_kwargs") or {}).get("net_arch")
    if saved_net != REFERENCE_PPO_KWARGS["policy_kwargs"]["net_arch"]:
        mismatches.append(f"config ppo.net_arch {saved_net!r} != [64, 64] pi/vf")

    normalization = config.get("normalization") or {}
    if not (
        normalization.get("observations")
        and normalization.get("rewards")
    ):
        mismatches.append("config normalization obs/reward is not enabled")

    actor = config.get("actor")
    if not isinstance(actor, dict):
        if variant == REFERENCE_VARIANT:
            # Reference runs written before the explicit actor record used
            # exactly the same PPO objective, so they remain resumable.
            actor = {"objective": ACTOR_OBJECTIVES[variant]}
        else:
            mismatches.append("config actor record is missing")
            actor = {}
    if actor.get("objective") != ACTOR_OBJECTIVES[variant]:
        mismatches.append(
            f"actor objective {actor.get('objective')!r} != "
            f"{ACTOR_OBJECTIVES[variant]!r}"
        )
    if variant == TRULY_VARIANT:
        rollback = actor.get("kl_rollback") or {}
        if not _same_number(rollback.get("alpha"), TrulyPPO.truly_alpha):
            mismatches.append(
                f"actor alpha {rollback.get('alpha')!r} != "
                f"{TrulyPPO.truly_alpha!r}"
            )
        if not _same_number(rollback.get("delta"), TrulyPPO.truly_delta):
            mismatches.append(
                f"actor delta {rollback.get('delta')!r} != "
                f"{TrulyPPO.truly_delta!r}"
            )
        if not _same_number(
            getattr(model, "truly_alpha", None), TrulyPPO.truly_alpha
        ):
            mismatches.append(
                f"model truly_alpha {getattr(model, 'truly_alpha', None)!r} "
                f"!= {TrulyPPO.truly_alpha!r}"
            )
        if not _same_number(
            getattr(model, "truly_delta", None), TrulyPPO.truly_delta
        ):
            mismatches.append(
                f"model truly_delta {getattr(model, 'truly_delta', None)!r} "
                f"!= {TrulyPPO.truly_delta!r}"
            )

    if mismatches:
        raise SystemExit(
            "error: refusing to resume with mismatched configuration: "
            + "; ".join(mismatches)
        )



def write_config(path: Path, config: dict) -> None:
    path.write_text(
        json.dumps(config, indent=2, default=str), encoding="utf-8"
    )


def build_run_config(
    args: argparse.Namespace,
    run_dir: Path,
    *,
    previous_config: dict,
    resumed_from: Path | None,
    start_timesteps: int,
    resume_history: list[dict],
    segments: list[dict],
    eval_enabled: bool,
    versions: dict,
    runtime: dict,
    git: dict,
    hashes: dict,
) -> dict:
    variant = args.variant
    ppo_kwargs = model_kwargs_for_variant(variant)
    learning_rate = learning_rate_for_variant(variant)
    config = {
        "environment": "Walker2d-v5",
        "variant": variant,
        "profile": REFERENCE_PROFILE,
        "seed": args.seed,
        "requested_timesteps": args.timesteps,
        "algorithm": "TrulyPPO" if variant == TRULY_VARIANT else "PPO",
        "actor": actor_config_for_variant(variant),
        "resumed_from": str(resumed_from) if resumed_from is not None else None,
        "resume_start_timesteps": start_timesteps,
        "resume_history": resume_history,
        "segments": segments,
        "status": "running",
        "last_observed_timesteps": start_timesteps,
        "error_type": None,
        "actual_timesteps": None,
        "elapsed_seconds": round(
            float(previous_config.get("elapsed_seconds", 0.0)), 2
        ),
        "segment_seconds": 0.0,
        "steps_per_second": None,
        "metrics_jsonl": str(run_dir / "metrics.jsonl"),
        "monitor_csv": str(run_dir / "monitor.csv"),
        "ppo": {
            "policy": "MlpPolicy",
            **ppo_kwargs,
            "learning_rate_initial": learning_rate,
            "learning_rate_schedule": "constant",
            "n_envs": 1,
            "policy_defaults": POLICY_DEFAULTS,
            "torch_threads": args.torch_threads,
        },
        "normalization": {
            "observations": True,
            "rewards": True,
            "clip_obs": 10.0,
            "clip_reward": 10.0,
            "gamma": ppo_kwargs["gamma"],
        },
        "recipe": {
            "profile": REFERENCE_PROFILE,
            "source_url": REFERENCE_SOURCE_URL,
            "source_environment": "Walker2d-v4",
            "environment_label": (
                "Walker2d-v5 (Walker2d-v4 reference hyperparameters "
                "ported unchanged to v5)"
            ),
            "truly_ppo_source": TRULY_PPO_SOURCE,
            "shared_note": (
                "Both arms share environment, normalization, network, "
                "learning rate and schedule, rollout settings, value and "
                "entropy coefficients, seed, evaluation, and budget. Only "
                "the actor objective differs."
            ),
        },
        "eval": {
            "enabled": eval_enabled,
            "eval_every": args.eval_every,
            "eval_episodes": args.eval_episodes,
            "eval_seed": args.eval_seed,
            "deterministic": True,
            "best_seed": args.eval_seed,
            "final_evaluate_default_seed": 40000,
            "protocol": (
                "EvalCallback on an independent nominal Walker2d-v5 env; "
                "VecNormalize statistics are synchronized from training and "
                "eval normalization is frozen (training=False, "
                "norm_reward=False). eval_env.seed(eval_seed) is applied at "
                "the start of every evaluation, so evaluations share the "
                "same RNG stream (not per-episode eval_seed+i). Best is "
                "selected by mean raw eval return; best/vecnormalize.pkl is "
                "saved with best_model.zip."
            ),
        },
        "versions": versions,
        "runtime": runtime,
        "git": git,
        "source_sha256": hashes,
    }
    return config


class ConfigCheckpointCallback(CheckpointCallback):
    """CheckpointCallback that also persists the latest observed step."""

    def __init__(
        self, *, config_path: Path, state: dict, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.config_path = config_path
        self.state = state

    def _on_step(self) -> bool:
        keep_training = super()._on_step()
        if self.n_calls % self.save_freq == 0:
            self.state["last_observed_timesteps"] = int(
                self.model.num_timesteps
            )
            write_config(self.config_path, self.state)
        return keep_training


def latest_checkpoint(checkpoint_dir: Path) -> Path | None:
    if not checkpoint_dir.is_dir():
        return None
    candidates: list[tuple[int, Path]] = []
    for path in checkpoint_dir.glob(f"{CHECKPOINT_PREFIX}_*_steps.zip"):
        match = CHECKPOINT_PATTERN.match(path.name)
        if match is None or "_vecnormalize_" in path.name:
            continue
        candidates.append((int(match.group("steps")), path))
    if not candidates:
        return None
    return max(candidates)[1]


def vecnormalize_path_for(model_path: Path) -> Path:
    match = CHECKPOINT_PATTERN.match(model_path.name)
    if match is not None:
        return model_path.parent / (
            f"{match.group('prefix')}_vecnormalize_{match.group('steps')}_steps.pkl"
        )
    return model_path.parent / "vecnormalize.pkl"


def find_resume_point(run_dir: Path) -> tuple[Path, Path] | None:
    """Pick `final_model.zip` if present, otherwise the latest checkpoint."""
    final_model = run_dir / "final_model.zip"
    model_path = final_model if final_model.exists() else latest_checkpoint(
        run_dir / "checkpoints"
    )
    if model_path is None:
        return None
    return model_path, _require_stats(model_path)


def _require_stats(model_path: Path) -> Path:
    stats_path = vecnormalize_path_for(model_path)
    if not stats_path.exists():
        raise FileNotFoundError(
            f"VecNormalize statistics for {model_path.name} not found at "
            f"{stats_path}; cannot resume a normalized run without them"
        )
    return stats_path


def resolve_resume(spec: str, run_dir: Path) -> tuple[Path, Path] | None:
    """Resolve `--resume` into (model path, vecnormalize path)."""
    if spec == "auto":
        point = find_resume_point(run_dir)
        if point is None:
            print(f"No final model or checkpoint in {run_dir}; cannot resume.")
        return point

    path = Path(spec)
    if path.is_dir():
        point = find_resume_point(path)
        if point is None:
            raise FileNotFoundError(
                f"No final_model.zip or checkpoint found under {path}"
            )
        return point
    if not path.exists():
        raise FileNotFoundError(f"Resume path not found: {path}")
    if path.suffix != ".zip":
        raise ValueError(
            "--resume expects a run directory, a model .zip file, or 'auto'"
        )
    return path, _require_stats(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train one of the two shared-profile Walker2d-v5 arms: the "
            "standard PPO-Clip reference or Truly PPO."
        )
    )
    parser.add_argument(
        "--variant",
        choices=VARIANTS,
        default=DEFAULT_VARIANT,
        help=(
            "ppo-reference trains standard PPO-Clip; truly-ppo replaces only "
            "the actor objective with Truly PPO's KL-triggered rollback."
        ),
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--timesteps",
        type=int,
        default=1_000_000,
        help="Absolute total step budget for the run. When resuming, only the "
             "remaining steps are trained.",
    )
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        default=None,
        help="Continue an interrupted run. Without a value, resume from "
             "final_model.zip or the latest checkpoint in --run-dir; with a "
             "value, use that run directory or model .zip as the starting point.",
    )
    parser.add_argument("--checkpoint-every", type=int, default=250_000)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument(
        "--eval-every",
        type=int,
        default=0,
        help="Evaluate every N environment steps. 0 disables periodic eval.",
    )
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument(
        "--eval-seed",
        type=int,
        default=50_000,
        help="RNG stream seed used at the start of every periodic eval.",
    )
    args = parser.parse_args()
    if args.eval_every < 0:
        parser.error("--eval-every must be >= 0")
    if args.eval_episodes < 1:
        parser.error("--eval-episodes must be at least 1")
    return args


def main() -> None:
    args = parse_args()
    _require_supported_variant(args.variant)
    run_dir = args.run_dir or default_run_dir(args.variant, args.seed)
    eval_enabled = args.eval_every > 0
    print(
        f"variant={args.variant} shared_profile={REFERENCE_PROFILE} "
        f"run_dir={run_dir}"
    )

    torch.set_num_threads(args.torch_threads)
    model_class = model_class_for_variant(args.variant)
    learning_rate = learning_rate_for_variant(args.variant)
    model_kwargs = model_kwargs_for_variant(args.variant)

    try:
        resume_point = (
            resolve_resume(args.resume, run_dir)
            if args.resume is not None
            else None
        )
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error

    if args.resume is not None and resume_point is None:
        raise SystemExit(
            f"error: --resume found no final_model.zip or checkpoint in "
            f"{run_dir}; refusing to start a new run under a resume flag"
        )

    # Both arms own their directory. A resume from inside it may reuse its
    # artifacts; a fresh start or an external resume may not overwrite them.
    ensure_run_dir_available(
        run_dir,
        allow_existing_from=(
            resume_point[0] if resume_point is not None else None
        ),
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    env = None
    eval_env = None
    model = None
    metrics_writer: JsonlMetricsWriter | None = None
    state: dict | None = None
    config_path = run_dir / "config.json"
    previous_config: dict = {}
    resumed_from: Path | None = None
    start_timesteps = 0
    segments: list[dict] = []

    try:
        if resume_point is None:
            monitor_path = run_dir / "monitor.csv"
            env = make_training_env(monitor_path=monitor_path)
            model = model_class(
                "MlpPolicy",
                env,
                seed=args.seed,
                learning_rate=learning_rate,
                tensorboard_log=str(run_dir / "tensorboard"),
                verbose=1,
                **model_kwargs,
            )
        else:
            resumed_from, stats_path = resume_point
            monitor_path = run_dir / "monitor.csv"
            env = make_training_env(
                stats_path,
                monitor_path=monitor_path,
                monitor_append=monitor_path.exists(),
            )
            model = model_class.load(
                resumed_from, env=env, device="cpu", print_system_info=False
            )
            # Validate the saved model before overriding its schedules.
            validate_resume(run_dir, args, model)
            # `load` restores the pickled schedule; rebuild it from this module
            # so a resumed run keeps the same constant reference rate.
            model.learning_rate = learning_rate
            model._setup_lr_schedule()
            model.tensorboard_log = str(run_dir / "tensorboard")
            model.verbose = 1
            start_timesteps = int(model.num_timesteps)
            print(f"Resuming from {resumed_from} at {start_timesteps} steps")

        # Only a resume carries previous history and timing forward.
        if resumed_from is not None:
            previous_config = _load_config(config_path)
        remaining = args.timesteps - model.num_timesteps
        resume_history = list(previous_config.get("resume_history", []))
        if resumed_from is not None and remaining > 0:
            resume_history.append(
                {"from": str(resumed_from), "start_timesteps": start_timesteps}
            )
        segments = list(previous_config.get("segments", []))

        versions = {
            "gymnasium": gymnasium.__version__,
            "mujoco": mujoco.__version__,
            "numpy": numpy.__version__,
            "stable_baselines3": stable_baselines3.__version__,
            "torch": torch.__version__,
        }
        repo_root = Path(__file__).resolve().parent
        state = build_run_config(
            args,
            run_dir,
            previous_config=previous_config,
            resumed_from=resumed_from,
            start_timesteps=start_timesteps,
            resume_history=resume_history,
            segments=segments,
            eval_enabled=eval_enabled,
            versions=versions,
            runtime=runtime_metadata(args.torch_threads),
            git=git_metadata(repo_root),
            hashes=source_hashes([
                repo_root / "train.py",
                repo_root / "diagnostics.py",
                repo_root / "envs.py",
                repo_root / "truly_ppo.py",
            ]),
        )
        write_config(config_path, state)

        metrics_writer = JsonlMetricsWriter(
            run_dir / "metrics.jsonl",
            append=resumed_from is not None,
        )
        logger = sb3_utils.configure_logger(
            verbose=1,
            tensorboard_log=str(run_dir / "tensorboard"),
            tb_log_name=f"{args.variant}_seed_{args.seed}",
            reset_num_timesteps=resumed_from is None,
        )
        logger.output_formats.append(metrics_writer)
        model.set_logger(logger)

        callbacks = [
            ConfigCheckpointCallback(
                config_path=config_path,
                state=state,
                save_freq=args.checkpoint_every,
                save_path=str(checkpoint_dir),
                name_prefix=CHECKPOINT_PREFIX,
                save_replay_buffer=False,
                save_vecnormalize=True,
            ),
            TrainingMetricsCallback(),
        ]
        if start_timesteps > 0:
            # Keep checkpoint filenames on absolute step multiples after resume.
            callbacks[0].n_calls = start_timesteps

        if eval_enabled:
            eval_env = make_eval_env()
            eval_callback = HealthyEvalCallback(
                eval_env=eval_env,
                callback_on_new_best=SaveVecNormalizeOnBest(
                    run_dir / "best" / "vecnormalize.pkl"
                ),
                n_eval_episodes=args.eval_episodes,
                eval_freq=args.eval_every,
                log_path=str(run_dir),
                best_model_save_path=str(run_dir / "best"),
                deterministic=True,
                verbose=1,
                eval_seed=args.eval_seed,
                metrics_writer=metrics_writer,
            )
            if resumed_from is not None:
                if restore_eval_history(eval_callback, run_dir):
                    print("Restored periodic evaluation history.")
                eval_callback.n_calls = start_timesteps
            callbacks.append(eval_callback)

        start = time.perf_counter()
        segment_seconds = 0.0
        try:
            if remaining <= 0:
                print(
                    f"Model already has {model.num_timesteps} steps, which "
                    f"meets the requested {args.timesteps}; nothing left to "
                    f"train."
                )
            else:
                model.learn(
                    total_timesteps=remaining,
                    callback=callbacks,
                    tb_log_name=f"{args.variant}_seed_{args.seed}",
                    reset_num_timesteps=resumed_from is None,
                )
                # SB3 dumps before each update, so the last train metrics would
                # otherwise never reach the logger.
                model.logger.dump(model.num_timesteps)
            segment_seconds = time.perf_counter() - start
            model.save(run_dir / "final_model")
            env.save(run_dir / "vecnormalize.pkl")
        except BaseException as error:
            segment_seconds = time.perf_counter() - start
            state["status"] = (
                "interrupted"
                if isinstance(error, KeyboardInterrupt)
                else "failed"
            )
            state["error_type"] = type(error).__name__
            state["last_observed_timesteps"] = int(model.num_timesteps)
            state["segment_seconds"] = round(segment_seconds, 2)
            total_elapsed = (
                float(previous_config.get("elapsed_seconds", 0.0))
                + segment_seconds
            )
            state["elapsed_seconds"] = round(total_elapsed, 2)
            state["steps_per_second"] = (
                round(model.num_timesteps / total_elapsed, 2)
                if total_elapsed > 0 else None
            )
            segments.append({
                "started_utc": state["runtime"]["utc_started"],
                "start_timesteps": start_timesteps,
                "end_timesteps": int(model.num_timesteps),
                "seconds": round(segment_seconds, 2),
                "status": state["status"],
            })
            state["segments"] = segments
            write_config(config_path, state)
            raise
        else:
            segment_seconds = time.perf_counter() - start
            state["status"] = "completed"
            state["error_type"] = None
            state["actual_timesteps"] = int(model.num_timesteps)
            state["last_observed_timesteps"] = int(model.num_timesteps)
            state["segment_seconds"] = round(segment_seconds, 2)
            total_elapsed = (
                float(previous_config.get("elapsed_seconds", 0.0))
                + segment_seconds
            )
            state["elapsed_seconds"] = round(total_elapsed, 2)
            state["steps_per_second"] = (
                round(model.num_timesteps / total_elapsed, 2)
                if total_elapsed > 0 else None
            )
            segments.append({
                "started_utc": state["runtime"]["utc_started"],
                "start_timesteps": start_timesteps,
                "end_timesteps": int(model.num_timesteps),
                "seconds": round(segment_seconds, 2),
                "status": "completed",
            })
            state["segments"] = segments
            write_config(config_path, state)
            print(
                f"Training complete: steps={model.num_timesteps}, "
                f"segment={segment_seconds:.1f}s, "
                f"steps/s={state['steps_per_second']}"
            )
            print(f"Saved model: {run_dir / 'final_model.zip'}")
    except BaseException as error:
        # Setup failures (before the training try/except above) still need a
        # persisted status. Training failures already updated the state.
        if state is not None and state.get("status") == "running":
            state["status"] = (
                "interrupted"
                if isinstance(error, KeyboardInterrupt)
                else "failed"
            )
            state["error_type"] = type(error).__name__
            if model is not None:
                state["last_observed_timesteps"] = int(model.num_timesteps)
            write_config(config_path, state)
        raise
    finally:
        if env is not None:
            env.close()
        if eval_env is not None:
            eval_env.close()
        if metrics_writer is not None:
            metrics_writer.close()



if __name__ == "__main__":
    main()
