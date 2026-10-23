from __future__ import annotations

import hashlib
import json
import os
import platform
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.logger import KVWriter


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def jsonable(value: Any) -> Any:
    """Convert the scalar/array types SB3 logs into JSON-compatible values."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, RuntimeError):
            pass
    return str(value)


class JsonlMetricsWriter(KVWriter):
    """Append one JSON object per SB3 logger dump.

    Values are converted from numpy scalars/arrays. A key is skipped when its
    SB3 exclude tuple contains ``jsonl``. The writer is added to the logger's
    existing output formats, so stdout and TensorBoard output are unchanged.
    """

    def __init__(self, path: str | Path, *, append: bool = False) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a" if append else "w", encoding="utf-8")

    def write(
        self,
        key_values: dict[str, Any],
        key_excluded: dict[str, tuple[str, ...]],
        step: int = 0,
    ) -> None:
        row: dict[str, Any] = {"step": int(step), "utc": utc_now()}
        for key, value in key_values.items():
            excluded = key_excluded.get(key, ())
            if "jsonl" in excluded:
                continue
            row[key] = jsonable(value)
        self._write(row)

    def write_event(self, step: int, values: dict[str, Any]) -> None:
        """Write a non-logger event, used for evaluation-only diagnostics."""
        row: dict[str, Any] = {"step": int(step), "utc": utc_now()}
        row.update({key: jsonable(value) for key, value in values.items()})
        self._write(row)

    def _write(self, row: dict[str, Any]) -> None:
        self._file.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()


def classify_episode_end(info: dict[str, Any]) -> str:
    """Classify a done step using SB3's DummyVecEnv semantics.

    DummyVecEnv sets ``TimeLimit.truncated = truncated and not terminated``.
    Therefore ``True`` is a healthy time-limit timeout. ``False`` means the
    episode was terminated (including the rare terminated+truncated case,
    which SB3 cannot distinguish from the info dict alone).
    """
    if info.get("TimeLimit.truncated", False):
        return "healthy_timeout"
    return "unhealthy_termination"


class TrainingMetricsCallback(BaseCallback):
    """Collect rollout-window diagnostics and hand them to the SB3 logger.

    Metrics are recorded at ``_on_rollout_end`` and are therefore dumped by
    SB3 before the following update; a dumped row pairs this rollout's
    diagnostics with the previous update's train/* metrics. Episode
    return/length come from the inner
    Monitor (raw environment reward, not VecNormalize reward). Health rates use
    ``classify_episode_end``. Action clipping is the fraction of individual
    action dimensions changed by PPO's action clipping (single env, per step).
    Observation clipping is the fraction of observation dimensions whose
    unclipped normalized value exceeded ``clip_obs``; it is computed from
    ``VecNormalize.get_original_obs()`` and the current ``obs_rms``. Reward
    component and x-velocity means are raw per-step values from the inner
    Walker2d ``info`` dict. ``train/length_1000_rate`` is a horizon proxy
    (episode length >= 1000); ``train/healthy_timeout_rate`` is the direct
    TimeLimit-based measure.
    """

    def __init__(self, *, clip_obs: float = 10.0) -> None:
        super().__init__(verbose=0)
        self.clip_obs = clip_obs
        self._reset_window()

    def _reset_window(self) -> None:
        self.episode_returns: list[float] = []
        self.episode_lengths: list[int] = []
        self.healthy_timeouts = 0
        self.unhealthy_terminations = 0
        self.action_clipped = 0
        self.action_elements = 0
        self.obs_clipped = 0
        self.obs_elements = 0
        self.reward_forward_sum = 0.0
        self.reward_ctrl_sum = 0.0
        self.reward_survive_sum = 0.0
        self.reward_samples = 0
        self.x_velocity_sum = 0.0
        self.x_velocity_samples = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        for done, info in zip(dones, infos):
            if "reward_forward" in info:
                self.reward_forward_sum += float(info["reward_forward"])
                self.reward_ctrl_sum += float(info["reward_ctrl"])
                self.reward_survive_sum += float(info["reward_survive"])
                self.reward_samples += 1
            if "x_velocity" in info:
                self.x_velocity_sum += float(info["x_velocity"])
                self.x_velocity_samples += 1
            if not done:
                continue
            if classify_episode_end(info) == "healthy_timeout":
                self.healthy_timeouts += 1
            else:
                self.unhealthy_terminations += 1
            episode = info.get("episode")
            if episode is not None:
                self.episode_returns.append(float(episode["r"]))
                self.episode_lengths.append(int(episode["l"]))

        actions = self.locals.get("actions")
        clipped_actions = self.locals.get("clipped_actions")
        if actions is not None and clipped_actions is not None:
            actions_array = np.asarray(actions)
            clipped_array = np.asarray(clipped_actions)
            if actions_array.shape == clipped_array.shape:
                self.action_clipped += int(
                    np.count_nonzero(actions_array != clipped_array)
                )
                self.action_elements += actions_array.size

        vec_normalize = self.model.get_vec_normalize_env()
        if vec_normalize is not None:
            raw_obs = vec_normalize.get_original_obs()
            obs_rms = getattr(vec_normalize, "obs_rms", None)
            if isinstance(raw_obs, np.ndarray) and hasattr(obs_rms, "var"):
                normalized = (raw_obs - obs_rms.mean) / np.sqrt(
                    obs_rms.var + vec_normalize.epsilon
                )
                self.obs_clipped += int(
                    np.count_nonzero(np.abs(normalized) > self.clip_obs)
                )
                self.obs_elements += normalized.size
        return True

    def _on_rollout_end(self) -> None:
        if self.episode_returns:
            self.logger.record("train/episodes", len(self.episode_returns))
            self.logger.record(
                "train/episode_return_mean",
                float(np.mean(self.episode_returns)),
            )
            self.logger.record(
                "train/episode_length_mean",
                float(np.mean(self.episode_lengths)),
            )
            self.logger.record(
                "train/length_1000_rate",
                float(np.mean(np.asarray(self.episode_lengths) >= 1000)),
            )

        completed = self.healthy_timeouts + self.unhealthy_terminations
        if completed:
            self.logger.record(
                "train/healthy_timeout_rate",
                self.healthy_timeouts / completed,
            )
            self.logger.record(
                "train/unhealthy_termination_rate",
                self.unhealthy_terminations / completed,
            )

        if self.action_elements:
            self.logger.record(
                "train/action_clip_fraction",
                self.action_clipped / self.action_elements,
            )
        if self.obs_elements:
            self.logger.record(
                "train/obs_clip_fraction",
                self.obs_clipped / self.obs_elements,
            )
        if self.reward_samples:
            self.logger.record(
                "train/reward_forward_mean",
                self.reward_forward_sum / self.reward_samples,
            )
            self.logger.record(
                "train/reward_ctrl_mean",
                self.reward_ctrl_sum / self.reward_samples,
            )
            self.logger.record(
                "train/reward_survive_mean",
                self.reward_survive_sum / self.reward_samples,
            )
        if self.x_velocity_samples:
            self.logger.record(
                "train/x_velocity_mean",
                self.x_velocity_sum / self.x_velocity_samples,
            )

        vec_normalize = self.model.get_vec_normalize_env()
        if vec_normalize is not None:
            obs_var = np.asarray(vec_normalize.obs_rms.var)
            self.logger.record("train/obs_var_min", float(obs_var.min()))
            self.logger.record("train/obs_var_mean", float(obs_var.mean()))
            self.logger.record("train/obs_var_max", float(obs_var.max()))
            self.logger.record(
                "train/ret_var", float(vec_normalize.ret_rms.var)
            )

        self._reset_window()


class SaveVecNormalizeOnBest(BaseCallback):
    """Save the training VecNormalize statistics when EvalCallback finds a best."""

    def __init__(self, save_path: str | Path) -> None:
        super().__init__(verbose=0)
        self.save_path = Path(save_path)

    def _on_step(self) -> bool:
        vec_normalize = self.model.get_vec_normalize_env()
        if vec_normalize is not None:
            self.save_path.parent.mkdir(parents=True, exist_ok=True)
            vec_normalize.save(str(self.save_path))
        return True


class HealthyEvalCallback(EvalCallback):
    """EvalCallback with fixed eval RNG and health/horizon diagnostics.

    ``eval_seed`` is applied at the start of every evaluation. DummyVecEnv
    passes that seed to the next reset and then advances the environment RNG,
    so each evaluation sees the same seed stream; it is not per-episode
    ``eval_seed + episode``. Health rates are written directly to metrics.jsonl
    (not through the logger) to avoid a second stdout/TensorBoard dump.
    ``eval/length_1000_rate`` is the fraction of eval episodes reaching the
    1000-step time limit; it is a horizon proxy, while
    ``eval/healthy_timeout_rate`` uses the TimeLimit flag and is the direct
    healthy-timeout measure.
    """

    def __init__(
        self,
        *args: Any,
        eval_seed: int,
        metrics_writer: JsonlMetricsWriter | None = None,
        debug_directory: Path | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.eval_seed = eval_seed
        self.metrics_writer = metrics_writer
        self.debug_directory = debug_directory
        self._eval_debug = None
        self._healthy_timeouts = 0
        self._unhealthy_terminations = 0

    def _log_success_callback(
        self, locals_: dict[str, Any], globals_: dict[str, Any]
    ) -> None:
        super()._log_success_callback(locals_, globals_)
        if not locals_.get("done"):
            return
        if self._eval_debug is not None:
            info = locals_["info"]
            self._eval_debug.write(json.dumps(jsonable({
                "event": "episode",
                "step": int(self.num_timesteps),
                "evaluation_index": len(self.evaluations_timesteps),
                "episode_index": int(np.asarray(locals_["episode_counts"])[locals_["i"]]),
                "stream_seed": self.eval_seed,
                "seed_protocol": "stream seeded once per evaluation, not seed+i",
                "episode": info.get("episode"),
                "end": classify_episode_end(info),
                "info": {k: v for k, v in info.items()
                         if k not in ("episode", "terminal_observation")},
            })) + "\n")
            self._eval_debug.flush()
        if classify_episode_end(locals_["info"]) == "healthy_timeout":
            self._healthy_timeouts += 1
        else:
            self._unhealthy_terminations += 1

    def _on_step(self) -> bool:
        will_evaluate = (
            self.eval_freq > 0 and self.n_calls % self.eval_freq == 0
        )
        if will_evaluate:
            self.eval_env.seed(self.eval_seed)
            if self.debug_directory is not None:
                directory = self.debug_directory / f"eval_{self.num_timesteps:012d}"
                directory.mkdir(parents=True, exist_ok=False)
                self.model.save(directory / "model.zip")
                self.model.get_vec_normalize_env().save(directory / "vecnormalize.pkl")
                self._eval_debug = (directory / "episodes.jsonl").open("x", encoding="utf-8")
                self._eval_debug.write(json.dumps({
                    "event": "evaluation_start",
                    "step": int(self.num_timesteps),
                    "n_updates": int(self.model._n_updates),
                    "stream_seed": self.eval_seed,
                    "phase": "during rollout, before the following optimizer update",
                    "deterministic": bool(self.deterministic),
                }) + "\n")
                self._eval_debug.flush()

        previous_evaluations = len(self.evaluations_timesteps)
        try:
            continue_training = super()._on_step()
        finally:
            if self._eval_debug is not None:
                self._eval_debug.close()
                self._eval_debug = None

        if len(self.evaluations_timesteps) > previous_evaluations:
            completed = self._healthy_timeouts + self._unhealthy_terminations
            lengths = self.evaluations_length[-1]
            values: dict[str, Any] = {
                "eval/episodes": len(lengths),
                "eval/length_1000_rate": float(
                    np.mean(np.asarray(lengths) >= 1000)
                ),
            }
            if completed:
                values["eval/healthy_timeout_rate"] = (
                    self._healthy_timeouts / completed
                )
                values["eval/unhealthy_termination_rate"] = (
                    self._unhealthy_terminations / completed
                )
            if self.metrics_writer is not None:
                self.metrics_writer.write_event(self.num_timesteps, values)
            self._healthy_timeouts = 0
            self._unhealthy_terminations = 0

        return continue_training


def restore_eval_history(
    callback: EvalCallback, log_path: str | Path
) -> bool:
    """Restore evaluations.npz so a resumed run does not discard history.

    Returns True when history was restored. Object arrays are not accepted
    (``allow_pickle=False``); malformed/legacy files are left untouched.
    """
    path = Path(log_path) / "evaluations.npz"
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            timesteps = data["timesteps"].tolist()
            results = data["results"].tolist()
            lengths = data["ep_lengths"].tolist()
    except (OSError, ValueError, KeyError):
        return False

    callback.evaluations_timesteps = timesteps
    callback.evaluations_results = results
    callback.evaluations_length = lengths
    if results:
        callback.best_mean_reward = max(
            float(np.mean(episode_rewards)) for episode_rewards in results
        )
    return True


def git_metadata(repo_root: Path) -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            completed = subprocess.run(
                ["git", *args],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError:
            return None
        return completed.stdout.strip() if completed.returncode == 0 else None

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {
        "commit": commit,
        "dirty": bool(status) if status is not None else None,
    }


def source_hashes(paths: list[Path]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in paths:
        if not path.is_file():
            continue
        digest = hashlib.sha256()
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
        hashes[path.name] = digest.hexdigest()
    return hashes


def runtime_metadata(torch_threads: int) -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "torch_threads": torch_threads,
        "command": shlex.join(sys.argv),
        "utc_started": utc_now(),
    }
