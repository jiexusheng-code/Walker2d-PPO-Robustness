from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs import make_env
from truly_ppo import TrulyPPO


CONDITIONS = {
    "nominal": (1.0, 1.0),
    "heldout_low": (0.7, 0.7),
    "heldout_high": (1.3, 1.3),
}
VARIANTS = ("baseline", "ppo-normalized", "truly-ppo")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a Walker2d PPO model.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=40_000)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--vecnormalize", type=Path,
        help="VecNormalize statistics for normalized variants "
             "(default: vecnormalize.pkl next to --model).",
    )
    parser.add_argument(
        "--conditions", nargs="+", choices=tuple(CONDITIONS),
        default=list(CONDITIONS),
    )
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be at least 1")
    if not args.model.exists():
        parser.error(f"model not found: {args.model}")
    if args.variant == "baseline":
        if args.vecnormalize is not None:
            parser.error("--vecnormalize is only valid for normalized variants")
    else:
        stats_path = args.vecnormalize or (args.model.parent / "vecnormalize.pkl")
        if not stats_path.exists():
            parser.error(f"VecNormalize statistics not found: {stats_path}")
    return args


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    model_class = TrulyPPO if args.variant == "truly-ppo" else PPO
    model = model_class.load(args.model, device="cpu")
    rows: list[dict[str, object]] = []
    summary: dict[str, dict[str, float]] = {}
    normalized = args.variant != "baseline"
    normalization_path = args.vecnormalize or (args.model.parent / "vecnormalize.pkl")

    for condition in args.conditions:
        friction_scale, actuator_scale = CONDITIONS[condition]
        env = make_env(
            friction_scale=friction_scale,
            actuator_scale=actuator_scale,
        )
        normalizer = None
        if normalized:
            vec_env = DummyVecEnv([lambda: make_env(
                friction_scale=friction_scale,
                actuator_scale=actuator_scale,
            )])
            normalizer = VecNormalize.load(normalization_path, vec_env)
            normalizer.training = False
            normalizer.norm_reward = False

        returns: list[float] = []
        lengths: list[int] = []
        for episode in range(args.episodes):
            obs, info = env.reset(seed=args.seed + episode)
            initial_x = float(info["x_position"])
            final_x = initial_x
            forward_velocities: list[float] = []
            episode_return = 0.0
            length = 0
            terminated = truncated = False
            while not (terminated or truncated):
                policy_obs = normalizer.normalize_obs(obs) if normalizer else obs
                action, _ = model.predict(policy_obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_return += float(reward)
                forward_velocities.append(float(info["x_velocity"]))
                final_x = float(info["x_position"])
                length += 1
            returns.append(episode_return)
            lengths.append(length)
            rows.append({
                "variant": args.variant,
                "condition": condition,
                "episode": episode,
                "seed": args.seed + episode,
                "return": episode_return,
                "length": length,
                "mean_forward_velocity": float(np.mean(forward_velocities)),
                "displacement": final_x - initial_x,
                "fell": bool(terminated),
                "time_limit_reached": bool(truncated),
                "friction_scale": friction_scale,
                "actuator_scale": actuator_scale,
            })
        env.close()
        if normalizer:
            normalizer.close()
        condition_rows = rows[-args.episodes:]
        summary[condition] = {
            "episodes": args.episodes,
            "mean_return": float(np.mean(returns)),
            "std_return": float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0,
            "mean_episode_length": float(np.mean(lengths)),
            "mean_forward_velocity": float(np.mean([
                row["mean_forward_velocity"] for row in condition_rows
            ])),
            "mean_displacement": float(np.mean([
                row["displacement"] for row in condition_rows
            ])),
            "fall_rate": float(np.mean([
                row["fell"] for row in condition_rows
            ])),
            "time_limit_rate": float(np.mean([
                row["time_limit_reached"] for row in condition_rows
            ])),
        }
        print(
            f"{condition}: return={summary[condition]['mean_return']:.1f} "
            f"+/- {summary[condition]['std_return']:.1f}; "
            f"speed={summary[condition]['mean_forward_velocity']:.3f} m/s; "
            f"displacement={summary[condition]['mean_displacement']:.2f} m; "
            f"fall_rate={summary[condition]['fall_rate']:.1%}"
        )

    with (args.output_dir / "episodes.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "model": str(args.model),
        "variant": args.variant,
        "normalization": {
            "enabled": normalized,
            "vecnormalize": str(normalization_path) if normalized else None,
        },
        "episodes_per_condition": args.episodes,
        "base_seed": args.seed,
        "conditions": summary,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
