from __future__ import annotations

import json
from importlib.metadata import distributions
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import mujoco
import numpy as np

from diagnostics import TrainingMetricsCallback, classify_episode_end, jsonable


class TrainingTraceCallback(TrainingMetricsCallback):
    """Persist the actual single-env transitions, without another env step.

    Callback rewards precede SB3's timeout value bootstrap. The saved update
    buffer contains the actual bootstrapped training rewards and GAE targets.
    Next observations/state on done steps follow DummyVecEnv's auto-reset;
    terminal observations are therefore stored separately.
    """

    def __init__(self, directory: Path) -> None:
        super().__init__()
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.rows: dict[str, list] = {}
        self.episode_id = 0

    def _on_rollout_start(self) -> None:
        self.rows = {}
        normalizer = self.model.get_vec_normalize_env()
        self.raw_previous = normalizer.get_original_obs().copy()
        self.rollout_start = int(self.model.num_timesteps)

    def _on_step(self) -> bool:
        keep_training = super()._on_step()
        normalizer = self.model.get_vec_normalize_env()
        infos = self.locals["infos"]
        next_obs = np.asarray(self.locals["new_obs"])
        terminal = np.full_like(next_obs, np.nan)
        terminal_raw = np.full_like(next_obs, np.nan)
        present = np.zeros(len(infos), dtype=bool)
        for i, info in enumerate(infos):
            if "terminal_observation" in info:
                terminal[i] = info["terminal_observation"]
                terminal_raw[i] = normalizer.unnormalize_obs(terminal[i])
                present[i] = True
            if self.locals["dones"][i]:
                event = {
                    "step": int(self.num_timesteps),
                    "episode_id_in_segment": self.episode_id,
                    "segment_start": self.segment_start,
                    "end": classify_episode_end(info),
                    "episode": info.get("episode"),
                    "info": {k: v for k, v in info.items()
                             if k not in ("terminal_observation", "episode")},
                }
                with (self.directory / "episodes.jsonl").open(
                    "a", encoding="utf-8"
                ) as file:
                    file.write(json.dumps(jsonable(event)) + "\n")
                self.episode_id += 1
        # The environment is single-env by the shared reference contract.
        base = normalizer.venv.envs[0].unwrapped
        data = {
            "step": int(self.num_timesteps),
            "observations": self.model._last_obs,
            "raw_observations": self.raw_previous,
            "next_observations_autoreset": next_obs,
            "raw_next_observations_autoreset": normalizer.get_original_obs(),
            "actions_raw_gaussian": self.locals["actions"],
            "actions_environment_clipped": self.locals["clipped_actions"],
            "old_values": self.locals["values"].detach().cpu().numpy(),
            "old_log_prob": self.locals["log_probs"].detach().cpu().numpy(),
            "normalized_reward_pre_timeout_bootstrap": self.locals["rewards"],
            "raw_reward": normalizer.get_original_reward(),
            "done": self.locals["dones"],
            "episode_start": self.model._last_episode_starts,
            "healthy_timeout": [bool(i.get("TimeLimit.truncated", False))
                                for i in infos],
            "terminal_present": present,
            "terminal_observations": terminal,
            "terminal_observations_unnormalized_from_clipped": terminal_raw,
            "qpos_after_step_or_autoreset": base.data.qpos,
            "qvel_after_step_or_autoreset": base.data.qvel,
            "raw_reward_components": [[info.get(key, np.nan) for key in (
                "reward_forward", "reward_ctrl", "reward_survive", "x_velocity"
            )] for info in infos],
        }
        for key, value in data.items():
            self.rows.setdefault(key, []).append(np.asarray(value).copy())
        self.raw_previous = normalizer.get_original_obs().copy()
        return keep_training

    def _on_training_start(self) -> None:
        self.segment_start = int(self.model.num_timesteps)
        base = self.model.get_vec_normalize_env().venv.envs[0].unwrapped
        prefix = self.directory / f"environment_{self.segment_start:012d}"
        with prefix.with_suffix(".npz").open("xb") as file:
            np.savez_compressed(
                file, geom_friction=base.model.geom_friction,
                actuator_gear=base.model.actuator_gear,
                body_mass=base.model.body_mass,
                body_inertia=base.model.body_inertia,
                dof_damping=base.model.dof_damping,
                gravity=base.model.opt.gravity,
                observation_mean=self.model.get_vec_normalize_env().obs_rms.mean,
                observation_variance=self.model.get_vec_normalize_env().obs_rms.var,
            )
        mujoco.mj_saveLastXML(str(prefix.with_suffix(".xml")), base.model)
        packages = sorted(
            f"{package.metadata['Name']}=={package.version}"
            for package in distributions()
        )
        with prefix.with_suffix(".json").open("x", encoding="utf-8") as file:
            json.dump({
                "utc": datetime.now(timezone.utc).isoformat(),
                "python": sys.version,
                "platform": platform.platform(),
                "packages": packages,
                "environment_spec": str(base.spec),
                "frame_skip": base.frame_skip,
                "dt": base.dt,
                "timestep": base.model.opt.timestep,
                "solver": int(base.model.opt.solver),
                "iterations": int(base.model.opt.iterations),
                "scope": "configuration evidence, not a full resumable simulator snapshot",
            }, file, indent=2)

    def _on_rollout_end(self) -> None:
        path = self.directory / f"rollout_{self.num_timesteps:012d}.npz"
        with path.open("xb") as file:
            np.savez_compressed(file, rollout_start=self.rollout_start,
                                **{k: np.asarray(v) for k, v in self.rows.items()})
        super()._on_rollout_end()
