from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import gymnasium
import mujoco
import numpy
import stable_baselines3
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from envs import make_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO on Walker2d-v5.")
    parser.add_argument("--variant", choices=("baseline", "domain-rand"), required=True)
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
    env = make_env(randomize=args.variant == "domain-rand")
    callback = CheckpointCallback(
        save_freq=args.checkpoint_every,
        save_path=str(checkpoint_dir),
        name_prefix="ppo_walker2d",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    config = {
        "environment": "Walker2d-v5",
        "variant": args.variant,
        "seed": args.seed,
        "requested_timesteps": args.timesteps,
        "ppo": {
            "policy": "MlpPolicy",
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.0,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "net_arch": {"pi": [64, 64], "vf": [64, 64]},
            "device": "cpu",
            "torch_threads": args.torch_threads,
        },
        "domain_randomization": {
            "geom_friction_scale": list(env.friction_range),
            "actuator_gear_scale": list(env.actuator_range),
        } if args.variant == "domain-rand" else None,
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

    model = PPO(
        "MlpPolicy",
        env,
        seed=args.seed,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs={"net_arch": {"pi": [64, 64], "vf": [64, 64]}},
        device="cpu",
        tensorboard_log=str(run_dir / "tensorboard"),
        verbose=1,
    )
    start = time.perf_counter()
    model.learn(
        total_timesteps=args.timesteps,
        callback=callback,
        tb_log_name=f"{args.variant}_seed_{args.seed}",
    )
    elapsed_seconds = time.perf_counter() - start
    model.save(run_dir / "final_model")
    config["actual_timesteps"] = model.num_timesteps
    config["elapsed_seconds"] = round(elapsed_seconds, 2)
    config["steps_per_second"] = round(model.num_timesteps / elapsed_seconds, 2)
    (run_dir / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    env.close()
    print(
        f"Training complete: steps={model.num_timesteps}, "
        f"elapsed={elapsed_seconds:.1f}s, "
        f"steps/s={model.num_timesteps / elapsed_seconds:.1f}"
    )
    print(f"Saved model: {run_dir / 'final_model.zip'}")


if __name__ == "__main__":
    main()
