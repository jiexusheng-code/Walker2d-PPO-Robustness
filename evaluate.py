from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import stable_baselines3
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
VARIANTS = ("baseline", "ppo-normalized", "truly-ppo", "ppo-reference")
EVALUATOR_SCHEMA_VERSION = 2
METRIC_NOTES = {
    "fall_rate": (
        "Fraction of episodes ending with terminated=True. For Walker2d this "
        "is the primary unhealthy/fall indicator."
    ),
    "time_limit_rate": (
        "Fraction of episodes with truncated=True. A step can be both "
        "terminated and truncated, so this is not a clean healthy-survival rate."
    ),
    "healthy_time_limit_rate": (
        "Fraction of episodes with truncated=True and terminated=False: the "
        "episode reached the time limit without a termination."
    ),
    "length_full_rate": (
        "Fraction of episodes whose length is at least max_episode_steps. "
        "Null when the environment does not declare a time limit."
    ),
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def training_metadata(model_path: Path, variant: str) -> dict[str, object]:
    """Read training config next to the model, if present.

    The evaluation seed must never be reported as the training seed, so the
    seed is accepted only from a config whose variant matches the model.
    """
    candidates = [model_path.parent / "config.json"]
    if model_path.parent.name in {"checkpoints", "best"}:
        candidates.append(model_path.parent.parent / "config.json")
    for config_path in candidates:
        config = _read_json(config_path)
        if config is None:
            continue
        if config.get("variant") not in (None, variant):
            continue
        return {
            "config": str(config_path),
            "seed": config.get("seed"),
            "requested_timesteps": config.get("requested_timesteps"),
            "actual_timesteps": config.get("actual_timesteps"),
        }
    return {
        "config": None,
        "seed": None,
        "requested_timesteps": None,
        "actual_timesteps": None,
    }


def summarize_condition(
    rows: list[dict[str, object]], max_episode_steps: int | None
) -> dict[str, object]:
    returns = [float(row["return"]) for row in rows]
    lengths = [int(row["length"]) for row in rows]
    count = len(rows)
    return {
        "episodes": count,
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns, ddof=1)) if count > 1 else 0.0,
        "mean_episode_length": float(np.mean(lengths)),
        "mean_forward_velocity": float(np.mean([
            row["mean_forward_velocity"] for row in rows
        ])),
        "mean_displacement": float(np.mean([
            row["displacement"] for row in rows
        ])),
        "fall_rate": float(np.mean([row["fell"] for row in rows])),
        "time_limit_rate": float(np.mean([
            row["time_limit_reached"] for row in rows
        ])),
        "healthy_time_limit_rate": float(np.mean([
            row["healthy_time_limit"] for row in rows
        ])),
        "max_episode_steps": max_episode_steps,
        "length_full_rate": (
            float(np.mean([length >= max_episode_steps for length in lengths]))
            if max_episode_steps is not None
            else None
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a Walker2d PPO model.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument(
        "--seed", type=int, default=40_000,
        help="Base evaluation seed (episode i uses seed+i). This is never "
             "recorded as the training seed.",
    )
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
    summary: dict[str, dict[str, object]] = {}
    normalized = args.variant != "baseline"
    normalization_path = args.vecnormalize or (args.model.parent / "vecnormalize.pkl")
    model_sha256 = file_sha256(args.model)
    vecnormalize_sha256 = file_sha256(normalization_path) if normalized else None
    training = training_metadata(args.model, args.variant)
    evaluated_at = datetime.now(timezone.utc).isoformat()

    for condition in args.conditions:
        friction_scale, actuator_scale = CONDITIONS[condition]
        env = make_env(
            friction_scale=friction_scale,
            actuator_scale=actuator_scale,
        )
        max_episode_steps = getattr(env.spec, "max_episode_steps", None)
        normalizer = None
        vec_env = None
        try:
            if normalized:
                vec_env = DummyVecEnv([lambda: make_env(
                    friction_scale=friction_scale,
                    actuator_scale=actuator_scale,
                )])
                normalizer = VecNormalize.load(normalization_path, vec_env)
                normalizer.training = False
                normalizer.norm_reward = False

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
                    "healthy_time_limit": bool(truncated and not terminated),
                    "friction_scale": friction_scale,
                    "actuator_scale": actuator_scale,
                })
        finally:
            if normalizer is not None:
                normalizer.close()
            elif vec_env is not None:
                vec_env.close()
            env.close()

        condition_rows = rows[-args.episodes:]
        summary[condition] = summarize_condition(
            condition_rows, max_episode_steps
        )
        print(
            f"{condition}: return={summary[condition]['mean_return']:.1f} "
            f"+/- {summary[condition]['std_return']:.1f}; "
            f"speed={summary[condition]['mean_forward_velocity']:.3f} m/s; "
            f"displacement={summary[condition]['mean_displacement']:.2f} m; "
            f"fall_rate={summary[condition]['fall_rate']:.1%}; "
            f"healthy_time_limit_rate="
            f"{summary[condition]['healthy_time_limit_rate']:.1%}"
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
        "eval_seed": args.seed,
        "deterministic": True,
        "evaluated_at": evaluated_at,
        "evaluator": {
            "schema_version": EVALUATOR_SCHEMA_VERSION,
            "python": platform.python_version(),
            "stable_baselines3": stable_baselines3.__version__,
            "torch": torch.__version__,
        },
        "model_sha256": model_sha256,
        "vecnormalize_sha256": vecnormalize_sha256,
        "training": training,
        "metric_notes": METRIC_NOTES,
        "conditions": summary,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
