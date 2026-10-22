from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Callable

import gymnasium
import mujoco
import numpy
import stable_baselines3
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs import make_env
from truly_ppo import TrulyPPO


VARIANTS = ("baseline", "ppo-normalized", "truly-ppo")
CHECKPOINT_PREFIX = "ppo_walker2d"
CHECKPOINT_PATTERN = re.compile(r"^(?P<prefix>.+)_(?P<steps>\d+)_steps\.zip$")
TOTAL_PPO_KWARGS = {
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.0,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "policy_kwargs": {"net_arch": {"pi": [64, 64], "vf": [64, 64]}},
    "device": "cpu",
}
INITIAL_LEARNING_RATE = 3e-4


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    return lambda progress_remaining: progress_remaining * initial_value


def make_training_env(normalize: bool, stats_path: Path | None = None):
    if not normalize:
        return Monitor(make_env())

    vector_env = DummyVecEnv([lambda: Monitor(make_env())])
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
        gamma=TOTAL_PPO_KWARGS["gamma"],
    )


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


def clear_stale_artifacts(run_dir: Path, checkpoint_dir: Path) -> list[Path]:
    """Remove artifacts of a discarded run before training the same directory
    from scratch, so a later resume cannot pick up a mismatched checkpoint."""
    removed: list[Path] = []
    for path in (run_dir / "final_model.zip", run_dir / "vecnormalize.pkl"):
        if path.exists():
            path.unlink()
            removed.append(path)
    if checkpoint_dir.is_dir():
        for path in sorted(checkpoint_dir.iterdir()):
            if path.is_file():
                path.unlink()
                removed.append(path)
    return removed


def vecnormalize_path_for(model_path: Path) -> Path:
    match = CHECKPOINT_PATTERN.match(model_path.name)
    if match is not None:
        sibling = model_path.parent / (
            f"{match.group('prefix')}_vecnormalize_{match.group('steps')}_steps.pkl"
        )
        if sibling.exists():
            return sibling
    return model_path.parent / "vecnormalize.pkl"


def find_resume_point(
    run_dir: Path, *, normalized: bool
) -> tuple[Path, Path | None] | None:
    """Pick `final_model.zip` if present, otherwise the latest checkpoint."""
    final_model = run_dir / "final_model.zip"
    model_path = final_model if final_model.exists() else latest_checkpoint(
        run_dir / "checkpoints"
    )
    if model_path is None:
        return None
    return model_path, _require_stats(model_path, normalized=normalized)


def _require_stats(model_path: Path, *, normalized: bool) -> Path | None:
    if not normalized:
        return None
    stats_path = vecnormalize_path_for(model_path)
    if not stats_path.exists():
        raise FileNotFoundError(
            f"VecNormalize statistics for {model_path.name} not found at "
            f"{stats_path}; cannot resume a normalized run without them"
        )
    return stats_path


def resolve_resume(
    spec: str, run_dir: Path, *, normalized: bool
) -> tuple[Path, Path | None] | None:
    """Resolve `--resume` into (model path, vecnormalize path or None)."""
    if spec == "auto":
        point = find_resume_point(run_dir, normalized=normalized)
        if point is None:
            print(f"No final model or checkpoint in {run_dir}; starting fresh.")
        return point

    path = Path(spec)
    if path.is_dir():
        point = find_resume_point(path, normalized=normalized)
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
    return path, _require_stats(path, normalized=normalized)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO on Walker2d-v5.")
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--timesteps",
        type=int,
        default=1_500_000,
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir or Path("runs") / args.variant / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    torch.set_num_threads(args.torch_threads)
    normalized = args.variant != "baseline"
    model_class = TrulyPPO if args.variant == "truly-ppo" else PPO
    learning_rate = (
        linear_schedule(INITIAL_LEARNING_RATE)
        if normalized
        else INITIAL_LEARNING_RATE
    )

    try:
        resume_point = (
            resolve_resume(args.resume, run_dir, normalized=normalized)
            if args.resume is not None
            else None
        )
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error

    resumed_from: Path | None = None
    start_timesteps = 0
    if resume_point is None:
        discarded = clear_stale_artifacts(run_dir, checkpoint_dir)
        if discarded:
            print(
                f"Starting from scratch; removed {len(discarded)} artifact(s) of "
                f"the previous run in {run_dir}"
            )
        env = make_training_env(normalized)
        model = model_class(
            "MlpPolicy",
            env,
            seed=args.seed,
            learning_rate=learning_rate,
            tensorboard_log=str(run_dir / "tensorboard"),
            verbose=1,
            **TOTAL_PPO_KWARGS,
        )
    else:
        resumed_from, stats_path = resume_point
        env = make_training_env(normalized, stats_path)
        model = model_class.load(
            resumed_from, env=env, device="cpu", print_system_info=False
        )
        # `load` restores the pickled schedule; rebuild it from this module so a
        # resumed run keeps annealing against the absolute step budget.
        model.learning_rate = learning_rate
        model._setup_lr_schedule()
        model.tensorboard_log = str(run_dir / "tensorboard")
        model.verbose = 1
        start_timesteps = int(model.num_timesteps)
        print(f"Resuming from {resumed_from} at {start_timesteps} steps")

    config_path = run_dir / "config.json"
    previous_config: dict = {}
    # A fresh start discards the previous run, so its history and timing must
    # not be carried over.
    if resumed_from is not None and config_path.exists():
        try:
            previous_config = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            previous_config = {}
    remaining = args.timesteps - model.num_timesteps
    resume_history = list(previous_config.get("resume_history", []))
    if resumed_from is not None and remaining > 0:
        resume_history.append(
            {"from": str(resumed_from), "start_timesteps": start_timesteps}
        )

    config = {
        "environment": "Walker2d-v5",
        "variant": args.variant,
        "seed": args.seed,
        "requested_timesteps": args.timesteps,
        "algorithm": "TrulyPPO" if args.variant == "truly-ppo" else "PPO",
        "resumed_from": str(resumed_from) if resumed_from is not None else None,
        "resume_start_timesteps": start_timesteps,
        "resume_history": resume_history,
        "ppo": {
            "policy": "MlpPolicy",
            **TOTAL_PPO_KWARGS,
            "learning_rate_initial": INITIAL_LEARNING_RATE,
            "learning_rate_schedule": "linear_to_zero" if normalized else "constant",
            "clip_range_used": args.variant != "truly-ppo",
            "torch_threads": args.torch_threads,
        },
        "normalization": {
            "observations": True,
            "rewards": True,
            "clip_obs": 10.0,
            "clip_reward": 10.0,
            "gamma": TOTAL_PPO_KWARGS["gamma"],
        } if normalized else None,
        "truly_ppo": {
            "alpha": TrulyPPO.truly_alpha,
            "delta": TrulyPPO.truly_delta,
        } if args.variant == "truly-ppo" else None,
        "versions": {
            "gymnasium": gymnasium.__version__,
            "mujoco": mujoco.__version__,
            "numpy": numpy.__version__,
            "stable_baselines3": stable_baselines3.__version__,
            "torch": torch.__version__,
        },
    }
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    callback = CheckpointCallback(
        save_freq=args.checkpoint_every,
        save_path=str(checkpoint_dir),
        name_prefix=CHECKPOINT_PREFIX,
        save_replay_buffer=False,
        save_vecnormalize=normalized,
    )
    if start_timesteps > 0:
        # Keep checkpoint filenames on absolute step multiples after a resume.
        callback.n_calls = start_timesteps

    start = time.perf_counter()
    elapsed_seconds = 0.0
    try:
        if remaining <= 0:
            print(
                f"Model already has {model.num_timesteps} steps, which meets the "
                f"requested {args.timesteps}; nothing left to train."
            )
        else:
            model.learn(
                total_timesteps=remaining,
                callback=callback,
                tb_log_name=f"{args.variant}_seed_{args.seed}",
                reset_num_timesteps=resumed_from is None,
            )
        elapsed_seconds = time.perf_counter() - start
        model.save(run_dir / "final_model")
        if normalized:
            env.save(run_dir / "vecnormalize.pkl")
    finally:
        env.close()

    total_elapsed = float(previous_config.get("elapsed_seconds", 0.0)) + elapsed_seconds
    config["actual_timesteps"] = model.num_timesteps
    config["elapsed_seconds"] = round(total_elapsed, 2)
    config["steps_per_second"] = (
        round(model.num_timesteps / total_elapsed, 2) if total_elapsed > 0 else None
    )
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(
        f"Training complete: steps={model.num_timesteps}, "
        f"segment={elapsed_seconds:.1f}s, "
        f"steps/s={config['steps_per_second']}"
    )
    print(f"Saved model: {run_dir / 'final_model.zip'}")


if __name__ == "__main__":
    main()
