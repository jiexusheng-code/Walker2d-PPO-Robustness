from __future__ import annotations

import argparse
import json
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


def make_training_env(normalize: bool):
    if not normalize:
        return Monitor(make_env())

    vector_env = DummyVecEnv([lambda: Monitor(make_env())])
    return VecNormalize(
        vector_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=TOTAL_PPO_KWARGS["gamma"],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO on Walker2d-v5.")
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--run-dir", type=Path)
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
    env = make_training_env(normalized)
    model_class = TrulyPPO if args.variant == "truly-ppo" else PPO
    learning_rate = (
        linear_schedule(INITIAL_LEARNING_RATE)
        if normalized
        else INITIAL_LEARNING_RATE
    )
    callback = CheckpointCallback(
        save_freq=args.checkpoint_every,
        save_path=str(checkpoint_dir),
        name_prefix="ppo_walker2d",
        save_replay_buffer=False,
        save_vecnormalize=normalized,
    )

    config = {
        "environment": "Walker2d-v5",
        "variant": args.variant,
        "seed": args.seed,
        "requested_timesteps": args.timesteps,
        "algorithm": "TrulyPPO" if args.variant == "truly-ppo" else "PPO",
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
    (run_dir / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    model = model_class(
        "MlpPolicy",
        env,
        seed=args.seed,
        learning_rate=learning_rate,
        tensorboard_log=str(run_dir / "tensorboard"),
        verbose=1,
        **TOTAL_PPO_KWARGS,
    )
    start = time.perf_counter()
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=callback,
            tb_log_name=f"{args.variant}_seed_{args.seed}",
        )
        elapsed_seconds = time.perf_counter() - start
        model.save(run_dir / "final_model")
        if normalized:
            env.save(run_dir / "vecnormalize.pkl")
    finally:
        env.close()

    config["actual_timesteps"] = model.num_timesteps
    config["elapsed_seconds"] = round(elapsed_seconds, 2)
    config["steps_per_second"] = round(model.num_timesteps / elapsed_seconds, 2)
    (run_dir / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    print(
        f"Training complete: steps={model.num_timesteps}, "
        f"elapsed={elapsed_seconds:.1f}s, "
        f"steps/s={model.num_timesteps / elapsed_seconds:.1f}"
    )
    print(f"Saved model: {run_dir / 'final_model.zip'}")


if __name__ == "__main__":
    main()
