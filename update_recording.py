"""Durable per-update flight recorder for RPO (and PPO) training updates.

Every call records a self-contained snapshot *before* the policy update:

    <root>/update_000000/
        manifest.json          status history, hashes, sizes, timings, metadata
        model_pre.zip           SB3 model save (policy + Adam state)
        rollout_buffer.pkl      cloudpickle of the full RolloutBuffer
        rng.pkl                 python/numpy/torch RNG state at train() entry
        vecnormalize.pkl        VecNormalize statistics (when present)
        pre_probe.npz           full-buffer old unperturbed distribution
        minibatches.jsonl       one row per actual minibatch (streamed)
        diagnostics.json        update-level summary
        post_probe.npz          full-buffer updated distribution + KL
        model_post.zip          SB3 model save after the update

Replay contract: restore the pre model, the rollout buffer, and rng.pkl, then
call ``model.train()`` with no recorder attached (and a logger set, e.g.
``model.set_logger(Logger())`` because SB3 saves exclude the logger). Because
the RNG snapshot and buffer are taken at the same point the original update
started, the replay is bit-exact (same CPU kernels).

The ``.pkl`` files are ``cloudpickle`` payloads from the local training
process. Treat them as trusted-local data only; never load snapshots that came
from an untrusted source.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
import traceback
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import cloudpickle
import numpy as np
import torch as th
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.distributions import DiagGaussianDistribution


SCHEMA_VERSION = "rpo-update-recorder/1"
TRUSTED_LOCAL_WARNING = (
    "cloudpickle snapshots are trusted-local only; do not load them from an "
    "untrusted source"
)
_LOG_2PI = math.log(2.0 * math.pi)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ------------------------------------------------------------------ RNG / snapshots
def capture_rng_state() -> dict[str, Any]:
    """Capture python, numpy and torch RNG state (CUDA too when available)."""
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": th.get_rng_state().clone(),
        # Replay must use the same intra-op thread count: reductions can
        # differ in the last bits between 1 thread and N threads.
        "torch_num_threads": int(th.get_num_threads()),
        "torch_num_interop_threads": int(th.get_num_interop_threads()),
    }
    if th.cuda.is_available():
        state["torch_cuda"] = [item.clone() for item in th.cuda.get_rng_state_all()]
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    """Restore a state previously returned by :func:`capture_rng_state`."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    th.set_rng_state(state["torch_cpu"])
    threads = state.get("torch_num_threads")
    if threads is not None:
        th.set_num_threads(int(threads))
    interop = state.get("torch_num_interop_threads")
    if interop is not None and int(interop) != th.get_num_interop_threads():
        try:
            th.set_num_interop_threads(int(interop))
        except RuntimeError:
            # torch only allows setting this before parallel work starts.
            pass
    cuda_state = state.get("torch_cuda")
    if cuda_state is not None and th.cuda.is_available():
        th.cuda.set_rng_state_all(cuda_state)


def save_rng_state(path: str | Path, state: dict[str, Any] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        cloudpickle.dump(capture_rng_state() if state is None else state, handle)


def load_rng_state(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as handle:
        return cloudpickle.load(handle)


def save_rollout_buffer(path: str | Path, buffer: RolloutBuffer) -> None:
    """Serialize a full RolloutBuffer for exact replay (trusted-local only)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        cloudpickle.dump(buffer, handle)


def load_rollout_buffer(path: str | Path) -> RolloutBuffer:
    """Load a buffer saved by :func:`save_rollout_buffer` (trusted-local only)."""
    with Path(path).open("rb") as handle:
        return cloudpickle.load(handle)


# ------------------------------------------------------------------ small utilities
def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_hashes(paths: list[Path]) -> dict[str, str]:
    return {
        path.name: sha256_file(path) for path in paths if path.is_file()
    }


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

    status = run("status", "--porcelain")
    return {
        "commit": run("rev-parse", "HEAD"),
        "dirty": bool(status) if status is not None else None,
    }


def _clean(value: Any) -> Any:
    """Convert a value into strict JSON (non-finite floats become ``null``)."""
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return _clean(value.tolist())
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, th.Tensor):
        return _clean(value.detach().cpu().numpy())
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        # Diagnostics do not need full float64 text; the NPZ payloads keep
        # exact arrays. Six significant digits preserve tiny KL/ratio values.
        return float(f"{value:.6g}")
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(_clean(payload), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _zip_members(path: Path) -> list[str] | None:
    try:
        with zipfile.ZipFile(path) as archive:
            return sorted(archive.namelist())
    except (OSError, zipfile.BadZipFile):
        return None


def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, th.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _finite_stats(value: Any, nonfinite: dict[str, int], key: str) -> dict[str, Any]:
    array = np.asarray(_as_numpy(value)).reshape(-1)
    if array.dtype.kind not in "fc":
        array = array.astype(np.float64)
    mask = np.isfinite(array)
    bad = int(array.size - int(mask.sum()))
    nonfinite[key] = nonfinite.get(key, 0) + bad
    finite = array[mask].astype(np.float64)
    if finite.size:
        return {
            "n": int(array.size),
            "nonfinite": bad,
            "mean": float(finite.mean()),
            "std": float(finite.std()) if finite.size > 1 else 0.0,
            "min": float(finite.min()),
            "max": float(finite.max()),
            "abs_mean": float(np.abs(finite).mean()),
        }
    return {
        "n": int(array.size),
        "nonfinite": bad,
        "mean": None,
        "std": None,
        "min": None,
        "max": None,
        "abs_mean": None,
    }


def _finite_list(value: Any, nonfinite: dict[str, int], key: str) -> list[float | None]:
    array = np.asarray(_as_numpy(value)).reshape(-1)
    bad = int(np.count_nonzero(~np.isfinite(array)))
    nonfinite[key] = nonfinite.get(key, 0) + bad
    return [float(item) if math.isfinite(float(item)) else None for item in array]


def _scalar(value: Any, nonfinite: dict[str, int], key: str) -> float | None:
    number = float(value)
    if not math.isfinite(number):
        nonfinite[key] = nonfinite.get(key, 0) + 1
        return None
    return number


def _cosine(first: np.ndarray, second: np.ndarray) -> float | None:
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if not math.isfinite(denominator) or denominator == 0.0:
        return None
    return float(np.dot(first, second) / denominator)


def _invalid_count(value: Any, nonfinite: dict[str, int], key: str) -> dict[str, int]:
    array = np.asarray(_as_numpy(value)).reshape(-1)
    bad = int(np.count_nonzero(~np.isfinite(array)))
    nonfinite[key] = nonfinite.get(key, 0) + bad
    return {"n": int(array.size), "nonfinite": bad}


def _grad_vector(grads: list[th.Tensor | None], params: list[th.Tensor]) -> np.ndarray:
    pieces = []
    for grad, parameter in zip(grads, params, strict=True):
        if grad is None:
            pieces.append(np.zeros(parameter.numel(), dtype=np.float64))
        else:
            pieces.append(grad.detach().cpu().double().numpy().reshape(-1))
    return np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float64)


def _grad_norm_and_invalid(params: list[th.Tensor]) -> tuple[float, int, int]:
    total = 0.0
    invalid = 0
    count = 0
    for parameter in params:
        if parameter.grad is None:
            continue
        grad = parameter.grad.detach()
        invalid += int(np.count_nonzero(~th.isfinite(grad).cpu().numpy()))
        count += grad.numel()
        total += float(grad.double().pow(2).sum().cpu())
    return math.sqrt(total), invalid, count


def _parameter_norm_and_invalid(params: list[th.Tensor]) -> tuple[float, int]:
    total = 0.0
    invalid = 0
    for parameter in params:
        data = parameter.detach()
        invalid += int(np.count_nonzero(~th.isfinite(data).cpu().numpy()))
        total += float(data.double().pow(2).sum().cpu())
    return math.sqrt(total), invalid


def _forward_unperturbed(
    policy: Any, obs: th.Tensor
) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
    """Replicate RPO's feature path without adding noise (no RNG use)."""
    features = policy.extract_features(obs)
    if policy.share_features_extractor:
        latent_pi, latent_vf = policy.mlp_extractor(features)
    else:
        pi_features, vf_features = features
        latent_pi = policy.mlp_extractor.forward_actor(pi_features)
        latent_vf = policy.mlp_extractor.forward_critic(vf_features)
    mean_actions = policy.action_net(latent_pi)
    std = policy.log_std.exp().expand_as(mean_actions)
    values = policy.value_net(latent_vf)
    return mean_actions, std, values


def _unperturbed_log_prob(
    policy: Any, obs: th.Tensor, actions: th.Tensor
) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
    mean, std, values = _forward_unperturbed(policy, obs)
    policy.action_dist.proba_distribution(mean, policy.log_std)
    log_prob = policy.action_dist.log_prob(actions)
    return mean, std, log_prob, values


def gaussian_kl(
    old_mean: np.ndarray,
    old_std: np.ndarray,
    new_mean: np.ndarray,
    new_std: np.ndarray,
) -> np.ndarray:
    """Analytic ``KL(N(old_mean, old_std) || N(new_mean, new_std))`` per sample."""
    old_mean = np.asarray(old_mean, dtype=np.float64)
    old_std = np.asarray(old_std, dtype=np.float64)
    new_mean = np.asarray(new_mean, dtype=np.float64)
    new_std = np.asarray(new_std, dtype=np.float64)
    old_std = np.maximum(old_std, np.finfo(np.float64).tiny)
    new_std = np.maximum(new_std, np.finfo(np.float64).tiny)
    per_dim = (
        np.log(new_std / old_std)
        + (old_std**2 + (old_mean - new_mean) ** 2) / (2.0 * new_std**2)
        - 0.5
    )
    return per_dim.sum(axis=-1)


def _flat_buffer_array(buffer: RolloutBuffer, name: str) -> np.ndarray:
    array = getattr(buffer, name)
    if not buffer.generator_ready:
        array = buffer.swap_and_flatten(array)
    return np.ascontiguousarray(array)


class _Minibatch:
    """Mutable per-minibatch observation bag filled by the hooks."""

    def __init__(self, epoch: int, minibatch: int) -> None:
        self.epoch = epoch
        self.minibatch = minibatch
        self.rollout_data: Any = None
        self.batch_indices: np.ndarray | None = None
        self.actions: th.Tensor | None = None
        self.values: th.Tensor | None = None
        self.log_prob: th.Tensor | None = None
        self.entropy: th.Tensor | None = None
        self.unperturbed_mean: th.Tensor | None = None
        self.unperturbed_std: th.Tensor | None = None
        self.unperturbed_log_prob: th.Tensor | None = None
        self.noise: th.Tensor | None = None
        self.metrics: dict[str, Any] = {}
        self.gradient: dict[str, Any] | None = None
        self.gradient_probe: dict[str, Any] | None = None
        self.step: dict[str, Any] | None = None
        self.nonfinite: dict[str, int] = {}
        self.errors: list[str] = []
        self.optimizer_stepped = False


class UpdateRecorder:
    """Record every RPO/PPO update into ``root/update_XXXXXX``.

    Attach with ``recorder.attach(model)``; ``RPO.train`` then routes through
    :meth:`run_update`. The recorder is excluded from model saves, and the
    unattached path is unchanged.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        # --resume rebuilds the recorder on an existing run_dir: never reuse
        # an update_XXXXXX directory, including failed/abandoned packages.
        highest = -1
        for existing in self.root.iterdir():
            suffix = existing.name.removeprefix("update_")
            if (
                existing.is_dir()
                and existing.name.startswith("update_")
                and suffix.isdigit()
            ):
                highest = max(highest, int(suffix))
        self._next_index = highest + 1
        self._metadata_cache: dict[str, Any] | None = None
        self._reset_update_state()

    # -------------------------------------------------------------- attach
    def attach(self, model: Any) -> "UpdateRecorder":
        model._update_recorder = self
        return self

    def detach(self, model: Any) -> None:
        if getattr(model, "_update_recorder", None) is self:
            del model._update_recorder

    # -------------------------------------------------------------- metadata
    def _metadata(self) -> dict[str, Any]:
        if self._metadata_cache is None:
            repo_root = Path(__file__).resolve().parent
            names = (
                "update_recording.py",
                "rpo_ppo.py",
                "diagnostics.py",
                "train.py",
                "envs.py",
                "truly_ppo.py",
            )
            self._metadata_cache = {
                "schema": SCHEMA_VERSION,
                "versions": {
                    "python": sys.version.split()[0],
                    "platform": platform.platform(),
                    "numpy": np.__version__,
                    "torch": th.__version__,
                    "stable_baselines3": _sb3_version(),
                    "gymnasium": _optional_version("gymnasium"),
                },
                "runtime": {
                    "torch_num_threads": int(th.get_num_threads()),
                    "torch_num_interop_threads": int(
                        th.get_num_interop_threads()
                    ),
                },
                "git": git_metadata(repo_root),
                "source_sha256": source_hashes(
                    [repo_root / name for name in names]
                ),
                "trusted_local_warning": TRUSTED_LOCAL_WARNING,
            }
        return self._metadata_cache

    # -------------------------------------------------------------- lifecycle
    def _reset_update_state(self) -> None:
        self._model: Any = None
        self._current: _Minibatch | None = None
        self._pending_indices: np.ndarray | None = None
        self._pending_data: Any = None
        self._rows: list[dict[str, Any]] = []
        self._epoch_orders: list[list[np.ndarray]] = []
        self._errors: list[dict[str, str]] = []
        self._minibatch_counter = 0
        self._n_minibatches_per_epoch = 0
        self._batch_size = 0
        self._clip_range = 0.0
        self._clip_range_vf: float | None = None
        self._pre: dict[str, np.ndarray] = {}
        self._jsonl_handle: Any = None
        self._train_started = 0.0

    def run_update(
        self, model: Any, train_callable: Callable[[], None]
    ) -> dict[str, Any]:
        """Snapshot, run one update while observing it, and save verification data."""
        self._validate_model(model)
        update_index = self._next_index
        self._next_index += 1
        update_dir = self.root / f"update_{update_index:06d}"
        update_dir.mkdir(parents=True, exist_ok=False)
        manifest_path = update_dir / "manifest.json"
        self._reset_update_state()
        self._model = model
        started = utc_now()
        total_started = time.perf_counter()
        manifest: dict[str, Any] = {
            **self._metadata(),
            "update_index": update_index,
            "update_dir": str(update_dir),
            "status": "running",
            "status_history": [[started, "running"]],
            "utc_started": started,
            "utc_finished": None,
            "config": self._model_config(model),
            "files": {},
            "timings": {},
        }
        self._write_manifest(manifest_path, manifest)

        rng_at_entry = capture_rng_state()
        pre_started = time.perf_counter()
        try:
            save_rng_state(update_dir / "rng.pkl", rng_at_entry)
            model.save(update_dir / "model_pre.zip")
            manifest["pre_model_members"] = _zip_members(
                update_dir / "model_pre.zip"
            )
            vec_normalize = model.get_vec_normalize_env()
            if vec_normalize is not None:
                vec_normalize.save(str(update_dir / "vecnormalize.pkl"))
            save_rollout_buffer(update_dir / "rollout_buffer.pkl", model.rollout_buffer)
            pre_probe = self._pre_probe(model)
            np.savez_compressed(update_dir / "pre_probe.npz", **pre_probe)
        except BaseException:
            manifest["timings"]["pre_save_s"] = time.perf_counter() - pre_started
            self._fail(update_dir, manifest_path, manifest, "pre_save_failed")
            raise
        manifest["timings"]["pre_save_s"] = time.perf_counter() - pre_started
        manifest["status"] = "pre_saved"
        manifest["status_history"].append([utc_now(), "pre_saved"])
        manifest["files"] = self._hash_files(update_dir)
        self._write_manifest(manifest_path, manifest)

        # The snapshot is authoritative: restore it so save/probe side effects
        # cannot change the update's RNG consumption.
        restore_rng_state(rng_at_entry)

        self._prepare_metrics(model)
        self._jsonl_handle = (update_dir / "minibatches.jsonl").open(
            "w", encoding="utf-8"
        )
        try:
            hooks: list[tuple[Any, str, Any, bool]] = []
            self._install_hooks(model, hooks)
        except BaseException:
            cleanup_errors = self._cleanup(hooks)
            manifest["timings"]["total_s"] = time.perf_counter() - total_started
            for error in cleanup_errors:
                self._record_error("cleanup", error)
            self._fail(update_dir, manifest_path, manifest, "instrumentation_failed")
            restore_rng_state(rng_at_entry)
            raise
        self._train_started = time.perf_counter()
        try:
            train_callable()
        except BaseException:
            manifest["timings"]["train_s"] = time.perf_counter() - self._train_started
            manifest["timings"]["total_s"] = time.perf_counter() - total_started
            try:
                self._finalize_pending()
            except BaseException as error:
                # A full disk can make the final flush fail. Record it and
                # still restore hooks; the original training error wins.
                self._record_error("finalize_after_train_failure", error)
            for error in self._cleanup(hooks):
                self._record_error("cleanup", error)
            self._fail(update_dir, manifest_path, manifest, "train_failed")
            raise
        manifest["timings"]["train_s"] = time.perf_counter() - self._train_started
        try:
            self._finalize_pending()
            self._verify_recording_complete()
        except BaseException as error:
            self._record_error("finalize", error)
            for cleanup_error in self._cleanup(hooks):
                self._record_error("cleanup", cleanup_error)
            manifest["timings"]["total_s"] = time.perf_counter() - total_started
            self._fail(update_dir, manifest_path, manifest, "instrumentation_failed")
            raise
        cleanup_errors = self._cleanup(hooks)
        if cleanup_errors:
            for error in cleanup_errors:
                self._record_error("cleanup", error)
            manifest["timings"]["total_s"] = time.perf_counter() - total_started
            try:
                raise cleanup_errors[0]
            except BaseException:
                self._fail(
                    update_dir, manifest_path, manifest, "instrumentation_failed"
                )
            raise cleanup_errors[0]
        rng_after_train = capture_rng_state()

        post_started = time.perf_counter()
        try:
            model.save(update_dir / "model_post.zip")
            manifest["post_model_members"] = _zip_members(
                update_dir / "model_post.zip"
            )
            post_probe = self._post_probe(model, pre_probe)
            np.savez_compressed(update_dir / "post_probe.npz", **post_probe)
            diagnostics = self._diagnostics(
                update_index, model, pre_probe, post_probe
            )
            _write_json(update_dir / "diagnostics.json", diagnostics)
        except BaseException:
            manifest["timings"]["post_save_s"] = time.perf_counter() - post_started
            manifest["timings"]["total_s"] = time.perf_counter() - total_started
            self._fail(update_dir, manifest_path, manifest, "post_save_failed")
            restore_rng_state(rng_after_train)
            raise
        finally:
            restore_rng_state(rng_after_train)

        manifest["timings"]["post_save_s"] = time.perf_counter() - post_started
        manifest["timings"]["total_s"] = time.perf_counter() - total_started
        if self._errors:
            # The update has already run and its post evidence is on disk;
            # mark the package invalid, then stop training (fail closed).
            failure = RuntimeError(
                "RPO update recorder instrumentation failed for "
                f"{update_dir.name}: {len(self._errors)} error(s); first "
                f"error: {self._errors[0].get('error')}; the update was "
                "preserved as instrumentation_failed"
            )
            try:
                raise failure
            except RuntimeError:
                self._fail(
                    update_dir,
                    manifest_path,
                    manifest,
                    "instrumentation_failed",
                )
            raise failure
        manifest["status"] = "completed"
        manifest["status_history"].append([utc_now(), "completed"])
        manifest["utc_finished"] = utc_now()
        manifest["files"] = self._hash_files(update_dir)
        manifest["metrics"] = self._summary()
        self._write_manifest(manifest_path, manifest)
        self._model = None
        return {"update_dir": str(update_dir), "manifest": manifest}

    # -------------------------------------------------------------- validation
    def _validate_model(self, model: Any) -> None:
        policy = getattr(model, "policy", None)
        if policy is None:
            raise ValueError("update recorder requires a model with a policy")
        if not isinstance(policy.action_dist, DiagGaussianDistribution):
            raise ValueError(
                "update recorder supports diagonal Gaussian policies only, "
                f"got {type(policy.action_dist).__name__}"
            )
        if not hasattr(policy, "rpo_alpha"):
            raise ValueError("update recorder requires an RPO policy")
        optimizers = policy.optimizer
        if isinstance(optimizers, (list, tuple)) and len(optimizers) != 1:
            raise ValueError(
                "update recorder expects exactly one policy optimizer"
            )
        buffer = getattr(model, "rollout_buffer", None)
        if buffer is None or not isinstance(buffer, RolloutBuffer):
            raise ValueError("update recorder requires a RolloutBuffer")
        if not buffer.full:
            raise ValueError("update recorder requires a full rollout buffer")

    def _model_config(self, model: Any) -> dict[str, Any]:
        progress = float(model._current_progress_remaining)
        clip_range = float(model.clip_range(progress))
        clip_range_vf = (
            None
            if model.clip_range_vf is None
            else float(model.clip_range_vf(progress))
        )
        return {
            "algorithm": type(model).__name__,
            "policy": type(model.policy).__name__,
            "rpo_alpha": float(getattr(model.policy, "rpo_alpha", getattr(model, "rpo_alpha", 0.0))),
            "num_timesteps": int(model.num_timesteps),
            "n_updates_attribute": int(model._n_updates),
            "progress_remaining": progress,
            "num_envs": int(model.n_envs),
            "n_steps": int(model.n_steps),
            "batch_size": int(model.batch_size),
            "n_epochs": int(model.n_epochs),
            "gamma": float(model.gamma),
            "gae_lambda": float(model.gae_lambda),
            "ent_coef": float(model.ent_coef),
            "vf_coef": float(model.vf_coef),
            "max_grad_norm": float(model.max_grad_norm),
            "clip_range": clip_range,
            "clip_range_vf": clip_range_vf,
            "normalize_advantage": bool(model.normalize_advantage),
            "target_kl": None if model.target_kl is None else float(model.target_kl),
            "learning_rate": float(model.lr_schedule(progress)),
            "device": str(model.device),
            "save_objects": list(model._get_torch_save_params()[0]),
            "log_std": [float(item) for item in model.policy.log_std.detach().cpu()],
            "buffer": {
                "buffer_size": int(model.rollout_buffer.buffer_size),
                "n_envs": int(model.rollout_buffer.n_envs),
                "size": int(model.rollout_buffer.size()),
                "full": bool(model.rollout_buffer.full),
                "generator_ready": bool(model.rollout_buffer.generator_ready),
            },
        }

    # -------------------------------------------------------------- pre/post probe
    def _pre_probe(self, model: Any) -> dict[str, np.ndarray]:
        buffer = model.rollout_buffer
        policy = model.policy
        observations = _flat_buffer_array(buffer, "observations")
        actions = _flat_buffer_array(buffer, "actions")
        old_log_probs = _flat_buffer_array(buffer, "log_probs").reshape(-1)
        old_values = _flat_buffer_array(buffer, "values").reshape(-1)
        advantages = _flat_buffer_array(buffer, "advantages").reshape(-1)
        returns = _flat_buffer_array(buffer, "returns").reshape(-1)

        means = []
        stds = []
        log_probs = []
        values = []
        probe_batch = 256
        with th.no_grad():
            for start in range(0, len(observations), probe_batch):
                obs_batch = th.as_tensor(
                    observations[start : start + probe_batch], device=model.device
                )
                action_batch = th.as_tensor(
                    actions[start : start + probe_batch], device=model.device
                )
                mean, std, log_prob, value = _unperturbed_log_prob(
                    policy, obs_batch, action_batch
                )
                means.append(mean.detach().cpu().numpy())
                stds.append(std.detach().cpu().numpy())
                log_probs.append(log_prob.detach().cpu().numpy())
                values.append(value.flatten().detach().cpu().numpy())
        old_mean = np.concatenate(means, axis=0)
        old_std = np.concatenate(stds, axis=0)
        old_unperturbed_log_probs = np.concatenate(log_probs, axis=0)
        recomputed_values = np.concatenate(values, axis=0)

        advantages64 = advantages.astype(np.float64)
        normalized = (advantages64 - advantages64.mean()) / (
            advantages64.std() + 1e-8
        )
        self._pre = {
            "mean": old_mean,
            "std": old_std,
            "unperturbed_log_prob": old_unperturbed_log_probs,
            "global_normalized_advantages": normalized,
        }
        return {
            "observations": observations,
            "actions": actions,
            "old_log_probs": old_log_probs,
            "old_values": old_values,
            "advantages": advantages,
            "returns": returns,
            "old_mean": old_mean,
            "old_std": old_std,
            "old_unperturbed_log_probs": old_unperturbed_log_probs,
            "recomputed_values": recomputed_values,
            "global_normalized_advantages": normalized.astype(np.float32),
            # .numpy() aliases the CPU parameter's memory; training mutates it.
            "log_std": policy.log_std.detach().cpu().numpy().copy(),
        }

    def _post_probe(
        self, model: Any, pre_probe: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        policy = model.policy
        observations = pre_probe["observations"]
        actions = pre_probe["actions"]
        means = []
        stds = []
        log_probs = []
        values = []
        probe_batch = 256
        with th.no_grad():
            for start in range(0, len(observations), probe_batch):
                obs_batch = th.as_tensor(
                    observations[start : start + probe_batch], device=model.device
                )
                action_batch = th.as_tensor(
                    actions[start : start + probe_batch], device=model.device
                )
                mean, std, log_prob, value = _unperturbed_log_prob(
                    policy, obs_batch, action_batch
                )
                means.append(mean.detach().cpu().numpy())
                stds.append(std.detach().cpu().numpy())
                log_probs.append(log_prob.detach().cpu().numpy())
                values.append(value.flatten().detach().cpu().numpy())
        new_mean = np.concatenate(means, axis=0)
        new_std = np.concatenate(stds, axis=0)
        new_log_probs = np.concatenate(log_probs, axis=0)
        new_values = np.concatenate(values, axis=0)
        kl = gaussian_kl(
            pre_probe["old_mean"], pre_probe["old_std"], new_mean, new_std
        )
        return {
            "new_mean": new_mean,
            "new_std": new_std,
            "new_unperturbed_log_probs": new_log_probs,
            "new_values": new_values,
            "kl_old_new": kl,
            "log_std": policy.log_std.detach().cpu().numpy().copy(),
        }

    # -------------------------------------------------------------- hooks
    def _prepare_metrics(self, model: Any) -> None:
        buffer = model.rollout_buffer
        total = int(buffer.buffer_size) * int(buffer.n_envs)
        self._batch_size = int(model.batch_size)
        self._n_minibatches_per_epoch = math.ceil(total / self._batch_size)
        progress = float(model._current_progress_remaining)
        self._clip_range = float(model.clip_range(progress))
        self._clip_range_vf = (
            None
            if model.clip_range_vf is None
            else float(model.clip_range_vf(progress))
        )

    def _install_hooks(
        self,
        model: Any,
        hooks: list[tuple[Any, str, Any, bool]] | None = None,
    ) -> list[tuple[Any, str, Any, bool]]:
        policy = model.policy
        buffer = model.rollout_buffer
        if hooks is None:
            hooks = []

        def install(target: Any, name: str, replacement: Any) -> None:
            had_own = name in target.__dict__
            hooks.append((target, name, target.__dict__.get(name), had_own))
            setattr(target, name, replacement)

        original_get = buffer.get

        def hooked_get(batch_size: int | None = None):
            for data in original_get(batch_size):
                self._pending_data = data
                yield data

        install(buffer, "get", hooked_get)

        original_get_samples = buffer._get_samples

        def hooked_get_samples(batch_inds: np.ndarray, env: Any = None):
            self._pending_indices = np.asarray(batch_inds).copy()
            return original_get_samples(batch_inds, env=env)

        install(buffer, "_get_samples", hooked_get_samples)

        original_evaluate = policy.evaluate_actions

        def hooked_evaluate(obs: Any, actions: th.Tensor):
            try:
                self._begin_minibatch(obs, actions)
            except Exception as error:  # recorded; package fails after the update
                self._record_error("begin_minibatch", error)
                self._current = None
            result = original_evaluate(obs, actions)
            try:
                self._end_minibatch(result)
            except Exception as error:
                self._record_error("end_minibatch", error)
            return result

        install(policy, "evaluate_actions", hooked_evaluate)

        original_noise = policy._sample_rpo_noise

        def hooked_noise(mean_actions: th.Tensor) -> th.Tensor:
            record = self._current
            if record is not None:
                try:
                    distribution = policy.action_dist
                    normal = distribution.distribution
                    # Clone before RPO mutates the shared distribution object.
                    record.unperturbed_mean = normal.mean.detach().clone()
                    record.unperturbed_std = normal.stddev.detach().clone()
                    with th.no_grad():
                        record.unperturbed_log_prob = distribution.log_prob(
                            record.actions
                        ).detach().clone()
                except Exception as error:
                    self._record_error("sample_rpo_noise", error)
            noise = original_noise(mean_actions)
            if record is not None:
                record.noise = noise.detach().clone()
            return noise

        install(policy, "_sample_rpo_noise", hooked_noise)

        optimizer = policy.optimizer
        params = [
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
            if parameter.requires_grad
        ]
        original_step = optimizer.step

        def hooked_step(*args: Any, **kwargs: Any):
            before = [parameter.detach().clone() for parameter in params]
            result = original_step(*args, **kwargs)
            record = self._current
            if record is not None:
                try:
                    step_sq = 0.0
                    invalid = 0
                    for parameter, previous in zip(params, before, strict=True):
                        delta = parameter.detach() - previous
                        invalid += int(
                            np.count_nonzero(~th.isfinite(delta).cpu().numpy())
                        )
                        step_sq += float(delta.double().pow(2).sum().cpu())
                    record.step = {
                        "param_step_norm": math.sqrt(step_sq),
                        "params_with_nonfinite_delta": invalid,
                        "learning_rate": float(optimizer.param_groups[0]["lr"]),
                        "parameters": len(params),
                    }
                    record.optimizer_stepped = True
                    self._finalize_current()
                except Exception as error:
                    self._record_error("optimizer_step", error)
            return result

        install(optimizer, "step", hooked_step)

        import torch.nn.utils as torch_nn_utils

        original_clip = torch_nn_utils.clip_grad_norm_

        def hooked_clip(parameters: Any, max_norm: float, *args: Any, **kwargs: Any):
            parameter_list = list(parameters)
            pre_norm, pre_invalid, element_count = _grad_norm_and_invalid(
                parameter_list
            )
            result = original_clip(parameter_list, max_norm, *args, **kwargs)
            post_norm, post_invalid, _ = _grad_norm_and_invalid(parameter_list)
            record = self._current
            if record is not None:
                record.gradient = {
                    "pre_clip_norm": pre_norm,
                    "post_clip_norm": post_norm,
                    "returned_total_norm": float(result),
                    "max_grad_norm": float(max_norm),
                    "nonfinite_grad_elements_before": pre_invalid,
                    "nonfinite_grad_elements_after": post_invalid,
                    "gradient_elements": element_count,
                }
            return result

        original_clip_holder = torch_nn_utils.clip_grad_norm_
        torch_nn_utils.clip_grad_norm_ = hooked_clip
        hooks.append((torch_nn_utils, "clip_grad_norm_", original_clip_holder, True))
        return hooks

    def _remove_hooks(self, hooks: list[tuple[Any, str, Any, bool]]) -> None:
        for target, name, original, had_own in reversed(hooks):
            if had_own:
                setattr(target, name, original)
            else:
                try:
                    delattr(target, name)
                except AttributeError:
                    pass
        hooks.clear()

    def _cleanup(self, hooks: list[tuple[Any, str, Any, bool]]) -> list[BaseException]:
        """Restore hooks and close the JSONL even if one of those steps fails."""
        errors: list[BaseException] = []
        try:
            self._remove_hooks(hooks)
        except BaseException as error:
            errors.append(error)
        try:
            self._close_jsonl()
        except BaseException as error:
            errors.append(error)
        return errors

    # -------------------------------------------------------------- per minibatch
    def _begin_minibatch(self, obs: Any, actions: th.Tensor) -> None:
        self._finalize_current()
        index = self._minibatch_counter
        self._minibatch_counter += 1
        epoch = index // self._n_minibatches_per_epoch
        if epoch >= len(self._epoch_orders):
            self._epoch_orders.append([])
        record = _Minibatch(epoch, index % self._n_minibatches_per_epoch)
        record.rollout_data = self._pending_data
        if self._pending_indices is not None:
            record.batch_indices = self._pending_indices.copy()
            self._epoch_orders[epoch].append(self._pending_indices.copy())
        record.actions = actions.detach().clone()
        self._current = record

    def _end_minibatch(self, result: Any) -> None:
        record = self._current
        if record is None or record.rollout_data is None:
            return
        values, log_prob, entropy = result
        record.values = values.detach().clone()
        record.log_prob = log_prob.detach().clone()
        record.entropy = None if entropy is None else entropy.detach().clone()
        if record.unperturbed_mean is None:
            # alpha == 0 bypasses the noise hook; probe the same policy path.
            with th.no_grad():
                mean, std, unperturbed_log_prob, _ = _unperturbed_log_prob(
                    self._model.policy,
                    self._rollout_observations(record),
                    record.actions,
                )
            record.unperturbed_mean = mean.detach().clone()
            record.unperturbed_std = std.detach().clone()
            record.unperturbed_log_prob = unperturbed_log_prob.detach().clone()
        record.metrics = self._compute_metrics(record)
        if record.minibatch == 0:
            record.gradient_probe = self._gradient_probe(
                record, values, log_prob, entropy
            )

    def _rollout_observations(self, record: _Minibatch) -> th.Tensor:
        return record.rollout_data.observations

    def _compute_metrics(self, record: _Minibatch) -> dict[str, Any]:
        model = self._model
        policy = model.policy
        data = record.rollout_data
        nonfinite: dict[str, int] = {}
        metrics: dict[str, Any] = {}

        actions = data.actions
        old_log_prob = data.old_log_prob
        old_values = data.old_values
        returns = data.returns
        raw_advantages = data.advantages
        if model.normalize_advantage and len(raw_advantages) > 1:
            advantages = (raw_advantages - raw_advantages.mean()) / (
                raw_advantages.std() + 1e-8
            )
        else:
            advantages = raw_advantages

        ratio = th.exp(record.log_prob - old_log_prob)
        unperturbed_ratio = th.exp(record.unperturbed_log_prob - old_log_prob)
        log_ratio = record.log_prob - old_log_prob
        unperturbed_log_ratio = record.unperturbed_log_prob - old_log_prob
        clip_range = self._clip_range

        policy_loss_1 = advantages * ratio
        policy_loss_2 = advantages * th.clamp(
            ratio, 1.0 - clip_range, 1.0 + clip_range
        )
        surrogate_loss = -th.min(policy_loss_1, policy_loss_2).mean()
        values = record.values.flatten()
        if self._clip_range_vf is None:
            values_pred = values
        else:
            values_pred = old_values + th.clamp(
                values - old_values,
                -self._clip_range_vf,
                self._clip_range_vf,
            )
        value_loss = th.nn.functional.mse_loss(returns, values_pred)
        entropy_loss = (
            -th.mean(record.entropy)
            if record.entropy is not None
            else -th.mean(-record.log_prob)
        )
        total_loss = (
            surrogate_loss
            + float(model.ent_coef) * entropy_loss
            + float(model.vf_coef) * value_loss
        )

        with th.no_grad():
            ratio_np = ratio.detach().cpu().numpy().astype(np.float64)
            unperturbed_ratio_np = (
                unperturbed_ratio.detach().cpu().numpy().astype(np.float64)
            )
            advantages_np = advantages.detach().cpu().numpy().astype(np.float64)
            clip_active = np.abs(ratio_np - 1.0) > clip_range
            unperturbed_clip_active = (
                np.abs(unperturbed_ratio_np - 1.0) > clip_range
            )
            upper = ratio_np > 1.0 + clip_range
            lower = ratio_np < 1.0 - clip_range
            positive = advantages_np > 0.0
            negative = advantages_np < 0.0

            old_mean = self._pre["mean"][record.batch_indices]
            old_std = self._pre["std"][record.batch_indices]
            old_pre_log_prob = self._pre["unperturbed_log_prob"][
                record.batch_indices
            ]
            new_mean = record.unperturbed_mean.detach().cpu().numpy()
            new_std = record.unperturbed_std.detach().cpu().numpy()
            kl = gaussian_kl(old_mean, old_std, new_mean, new_std)
            global_advantages = self._pre["global_normalized_advantages"][
                record.batch_indices
            ]
            sign_flip = np.sign(advantages_np) != np.sign(global_advantages)
            comparable = (
                (advantages_np != 0.0)
                & (global_advantages != 0.0)
                & np.isfinite(advantages_np)
                & np.isfinite(global_advantages)
            )
            sign_flip_count = int(np.count_nonzero(sign_flip & comparable))
            comparable_count = int(np.count_nonzero(comparable))

            perturbed_log_ratio_np = (
                log_ratio.detach().cpu().numpy().astype(np.float64)
            )
            unperturbed_log_ratio_np = (
                unperturbed_log_ratio.detach().cpu().numpy().astype(np.float64)
            )
            perturbed_approx_kl = float(
                np.mean(
                    (np.exp(perturbed_log_ratio_np) - 1.0)
                    - perturbed_log_ratio_np
                )
            )
            unperturbed_approx_kl = float(
                np.mean(
                    (np.exp(unperturbed_log_ratio_np) - 1.0)
                    - unperturbed_log_ratio_np
                )
            )

            returns_np = returns.detach().cpu().numpy().astype(np.float64)
            values_np = values.detach().cpu().numpy().astype(np.float64)
            old_values_np = old_values.detach().cpu().numpy().astype(np.float64)
            value_error = values_np - returns_np
            old_value_error = old_values_np - returns_np
            return_variance = float(np.var(returns_np))
            explained_variance = (
                1.0 - float(np.var(returns_np - values_np)) / return_variance
                if return_variance > 0.0
                else None
            )

        metrics["batch"] = {
            "size": int(len(actions)),
            "indices_head": None
            if record.batch_indices is None
            else [int(item) for item in record.batch_indices[:8]],
            "indices_sha256": None
            if record.batch_indices is None
            else hashlib.sha256(
                record.batch_indices.astype(np.int64).tobytes()
            ).hexdigest(),
        }
        metrics["nonfinite"] = nonfinite
        # Full arrays for these are in pre_probe.npz / the rollout buffer; the
        # per-minibatch JSON only needs integrity counts.
        metrics["observations"] = _invalid_count(
            data.observations, nonfinite, "observations"
        )
        metrics["actions"] = _invalid_count(actions, nonfinite, "actions")
        metrics["advantages"] = {
            "raw": _finite_stats(raw_advantages, nonfinite, "advantages.raw"),
            "normalized": _finite_stats(
                advantages, nonfinite, "advantages.normalized"
            ),
            "global_normalized": _finite_stats(
                global_advantages, nonfinite, "advantages.global_normalized"
            ),
            "sign_flips_vs_global": {
                "count": sign_flip_count,
                "comparable": comparable_count,
                "fraction": float(
                    sign_flip_count / max(comparable_count, 1)
                ),
            },
        }
        metrics["returns"] = _invalid_count(returns, nonfinite, "returns")
        metrics["old_values"] = _invalid_count(
            old_values, nonfinite, "old_values"
        )
        metrics["values"] = _finite_stats(values, nonfinite, "values")
        metrics["value_error"] = {
            "new": _finite_stats(value_error, nonfinite, "value_error.new"),
            "old": _finite_stats(value_error, nonfinite, "value_error.old"),
            "explained_variance": _scalar(
                explained_variance if explained_variance is not None else float("nan"),
                nonfinite,
                "value_error.explained_variance",
            ),
        }
        metrics["old_distribution"] = {
            "mean": _finite_stats(old_mean, nonfinite, "old_distribution.mean"),
            "std": _finite_stats(old_std, nonfinite, "old_distribution.std"),
            "log_prob": _finite_stats(
                old_pre_log_prob, nonfinite, "old_distribution.log_prob"
            ),
            "log_prob_max_abs_diff_vs_buffer": _scalar(
                float(
                    np.max(
                        np.abs(
                            old_pre_log_prob
                            - old_log_prob.detach().cpu().numpy()
                        )
                    )
                ),
                nonfinite,
                "old_distribution.log_prob_max_abs_diff_vs_buffer",
            ),
        }
        metrics["new_unperturbed_distribution"] = {
            "mean": _finite_stats(new_mean, nonfinite, "new_distribution.mean"),
            "std_per_action": _finite_list(
                new_std[0] if len(new_std) else np.zeros(0),
                nonfinite,
                "new_distribution.std",
            ),
            "mean_shift_from_old_l2": _scalar(
                float(np.linalg.norm((new_mean - old_mean).astype(np.float64))),
                nonfinite,
                "new_distribution.mean_shift_l2",
            ),
        }
        metrics["log_prob"] = {
            "perturbed": _finite_stats(
                record.log_prob, nonfinite, "log_prob.perturbed"
            ),
            "unperturbed": _finite_stats(
                record.unperturbed_log_prob, nonfinite, "log_prob.unperturbed"
            ),
        }
        metrics["ratio"] = {
            "rpo_perturbed": _finite_stats(
                ratio, nonfinite, "ratio.rpo_perturbed"
            ),
            "unperturbed": _finite_stats(
                unperturbed_ratio, nonfinite, "ratio.unperturbed"
            ),
        }
        metrics["log_ratio"] = {
            "rpo_perturbed": _finite_stats(
                log_ratio, nonfinite, "log_ratio.rpo_perturbed"
            ),
            "unperturbed": _finite_stats(
                unperturbed_log_ratio, nonfinite, "log_ratio.unperturbed"
            ),
        }
        metrics["kl"] = {
            "analytic_old_new": {
                **_finite_stats(kl, nonfinite, "kl.analytic_old_new"),
                "max": _scalar(float(np.max(kl)), nonfinite, "kl.analytic_old_new.max"),
                "p95": _scalar(
                    float(np.quantile(kl, 0.95)),
                    nonfinite,
                    "kl.analytic_old_new.p95",
                ),
            },
            "ppo_approx_perturbed": _scalar(
                perturbed_approx_kl, nonfinite, "kl.ppo_approx_perturbed"
            ),
            "ppo_approx_unperturbed": _scalar(
                unperturbed_approx_kl, nonfinite, "kl.ppo_approx_unperturbed"
            ),
        }
        metrics["clip"] = {
            "range": clip_range,
            "fraction_abs_perturbed": float(np.mean(clip_active)),
            "fraction_abs_unperturbed": float(np.mean(unperturbed_clip_active)),
            "upper_perturbed": int(np.count_nonzero(upper)),
            "lower_perturbed": int(np.count_nonzero(lower)),
            "active_positive_advantage": int(
                np.count_nonzero(clip_active & positive)
            ),
            "active_negative_advantage": int(
                np.count_nonzero(clip_active & negative)
            ),
            "clipped_positive_advantage": int(
                np.count_nonzero(upper & positive)
            ),
            "clipped_negative_advantage": int(
                np.count_nonzero(lower & negative)
            ),
            "active_positive_buffer_indices": None
            if record.batch_indices is None
            else [
                int(item)
                for item in record.batch_indices[clip_active & positive]
            ],
            "active_negative_buffer_indices": None
            if record.batch_indices is None
            else [
                int(item)
                for item in record.batch_indices[clip_active & negative]
            ],
        }
        metrics["loss"] = {
            "surrogate": _scalar(surrogate_loss, nonfinite, "loss.surrogate"),
            "entropy": _scalar(entropy_loss, nonfinite, "loss.entropy"),
            "value": _scalar(value_loss, nonfinite, "loss.value"),
            "total": _scalar(total_loss, nonfinite, "loss.total"),
            "ent_coef": float(model.ent_coef),
            "vf_coef": float(model.vf_coef),
            "clip_range_vf": self._clip_range_vf,
        }
        metrics["entropy"] = {
            "mean": None
            if record.entropy is None
            else _scalar(
                record.entropy.mean(), nonfinite, "entropy.mean"
            ),
            "per_action": None
            if record.entropy is None
            else _finite_list(
                0.5
                * (
                    record.unperturbed_std[0].log() * 2.0
                    + 1.0
                    + _LOG_2PI
                ).detach().cpu().numpy(),
                nonfinite,
                "entropy.per_action",
            ),
            "log_std_per_action": _finite_list(
                policy.log_std.detach().cpu().numpy(),
                nonfinite,
                "log_std",
            ),
        }
        if record.noise is not None:
            noise_np = record.noise.detach().cpu().numpy()
            metrics["rpo"] = {
                "alpha": float(policy.rpo_alpha),
                "noise": _finite_stats(record.noise, nonfinite, "rpo.noise"),
                "noise_mean_per_action": _finite_list(
                    noise_np.mean(axis=0), nonfinite, "rpo.noise_mean"
                ),
                "noise_std_per_action": _finite_list(
                    noise_np.std(axis=0), nonfinite, "rpo.noise_std"
                ),
                "noise_max_abs": _scalar(
                    float(np.max(np.abs(noise_np))),
                    nonfinite,
                    "rpo.noise_max_abs",
                ),
                "action_std_per_action": _finite_list(
                    new_std[0] if len(new_std) else np.zeros(0),
                    nonfinite,
                    "rpo.action_std",
                ),
            }
        else:
            metrics["rpo"] = {
                "alpha": float(policy.rpo_alpha),
                "noise": None,
                "action_std_per_action": _finite_list(
                    new_std[0] if len(new_std) else np.zeros(0),
                    nonfinite,
                    "rpo.action_std",
                ),
            }
        metrics["early_stop"] = {
            "target_kl": None if model.target_kl is None else float(model.target_kl),
            "would_stop": bool(
                model.target_kl is not None
                and perturbed_approx_kl > 1.5 * model.target_kl
            ),
        }
        return metrics

    def _gradient_probe(
        self,
        record: _Minibatch,
        values_live: th.Tensor,
        log_prob_live: th.Tensor,
        entropy_live: th.Tensor | None,
    ) -> dict[str, Any] | None:
        """autograd.grad probes for the first minibatch of every epoch.

        The live tensors are passed explicitly so the probe traverses the same
        autograd graph PPO is about to backpropagate. ``retain_graph=True`` and
        ``autograd.grad`` leave every ``.grad`` field untouched.
        """
        try:
            policy = self._model.policy
            data = record.rollout_data
            raw_advantages = data.advantages
            if self._model.normalize_advantage and len(raw_advantages) > 1:
                advantages = (raw_advantages - raw_advantages.mean()) / (
                    raw_advantages.std() + 1e-8
                )
            else:
                advantages = raw_advantages
            ratio = th.exp(log_prob_live - data.old_log_prob)
            clip_range = self._clip_range
            surrogate = -th.min(
                advantages * ratio,
                advantages
                * th.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range),
            ).mean()
            values = values_live.flatten()
            if self._clip_range_vf is None:
                values_pred = values
            else:
                values_pred = data.old_values + th.clamp(
                    values - data.old_values,
                    -self._clip_range_vf,
                    self._clip_range_vf,
                )
            value_loss = th.nn.functional.mse_loss(data.returns, values_pred)
            entropy_loss = (
                -th.mean(entropy_live)
                if entropy_live is not None
                else -th.mean(-log_prob_live)
            )
            actor_loss = surrogate + float(self._model.ent_coef) * entropy_loss
            critic_loss = float(self._model.vf_coef) * value_loss
            total_loss = actor_loss + critic_loss

            params = [
                parameter
                for parameter in policy.parameters()
                if parameter.requires_grad
            ]
            names = [
                name for name, parameter in policy.named_parameters()
                if parameter.requires_grad
            ]

            def grads(loss: th.Tensor) -> list[th.Tensor | None]:
                return th.autograd.grad(
                    loss,
                    params,
                    retain_graph=True,
                    allow_unused=True,
                )

            actor_grads = grads(actor_loss)
            critic_grads = grads(critic_loss)
            total_grads = grads(total_loss)
            actor_vector = _grad_vector(actor_grads, params)
            critic_vector = _grad_vector(critic_grads, params)
            total_vector = _grad_vector(total_grads, params)

            actor_group = np.array(
                [
                    name.startswith("mlp_extractor.policy_net.")
                    or name.startswith("action_net.")
                    for name in names
                ]
            )
            critic_group = np.array(
                [
                    name.startswith("mlp_extractor.value_net.")
                    or name.startswith("value_net.")
                    for name in names
                ]
            )
            shared_group = ~(actor_group | critic_group)

            def group_norm(vector: np.ndarray, mask: np.ndarray) -> float | None:
                if vector.size == 0 or not mask.any():
                    return None
                offset = 0
                squared = 0.0
                for parameter, selected in zip(params, mask, strict=True):
                    size = parameter.numel()
                    if selected:
                        segment = vector[offset : offset + size]
                        squared += float(np.dot(segment, segment))
                    offset += size
                return math.sqrt(squared)

            result = {
                "epoch": int(record.epoch),
                "minibatch": int(record.minibatch),
                "actor_component_norm": float(np.linalg.norm(actor_vector)),
                "critic_component_norm": float(np.linalg.norm(critic_vector)),
                "total_component_norm": float(np.linalg.norm(total_vector)),
                "actor_group_norm": group_norm(actor_vector, actor_group),
                "critic_group_norm": group_norm(critic_vector, critic_group),
                "shared_group_norm": group_norm(total_vector, shared_group),
                "cosine_actor_critic": _cosine(actor_vector, critic_vector),
                "cosine_actor_total": _cosine(actor_vector, total_vector),
                "cosine_critic_total": _cosine(critic_vector, total_vector),
                "grad_none_parameters": int(
                    sum(grad is None for grad in actor_grads)
                ),
                "grad_probe_modified_dot_grad": False,
            }
            result["nonfinite"] = {
                key: 1
                for key, value in result.items()
                if isinstance(value, float) and not math.isfinite(value)
            }
            return result
        except Exception as error:
            self._record_error("gradient_probe", error)
            return {"error": repr(error)}

    # -------------------------------------------------------------- finalize
    def _finalize_current(self) -> None:
        record = self._current
        if record is None:
            return
        self._current = None
        row = {
            "update_index": self._next_index - 1,
            "epoch": int(record.epoch),
            "minibatch_in_epoch": int(record.minibatch),
            "optimizer_stepped": bool(record.optimizer_stepped),
            "metrics": record.metrics,
            "gradient": record.gradient,
            "gradient_probe": record.gradient_probe,
            "step": record.step,
            "errors": record.errors,
        }
        # The per-row nonfinite map is already in metrics; keep a single copy
        # and drop zero counts (the summary reports complete per-field totals).
        row["nonfinite"] = {
            key: int(count)
            for key, count in record.metrics.pop("nonfinite", {}).items()
            if int(count)
        }
        self._rows.append(row)
        if self._jsonl_handle is not None:
            self._jsonl_handle.write(
                json.dumps(
                    _clean(row), allow_nan=False, separators=(",", ":")
                )
                + "\n"
            )
            self._jsonl_handle.flush()

    def _finalize_pending(self) -> None:
        self._finalize_current()

    def _verify_recording_complete(self) -> None:
        """Mark the package invalid if the update was not fully observed.

        A recorder that silently misses a minibatch is worse than no recorder,
        so this converts missing evidence into the same failure path as an
        instrumentation exception.
        """
        required = ("batch", "loss", "kl", "ratio", "advantages", "clip")
        problems: list[str] = []
        if not self._rows:
            problems.append("no minibatches were recorded")
        for row in self._rows:
            missing = [
                key for key in required if key not in row.get("metrics", {})
            ]
            if missing:
                problems.append(
                    f"epoch {row['epoch']} minibatch "
                    f"{row['minibatch_in_epoch']} missing {missing}"
                )
        if problems:
            self._record_error(
                "completeness", RuntimeError("; ".join(problems[:5]))
            )

    def _record_error(self, stage: str, error: BaseException) -> None:
        entry = {
            "stage": stage,
            "error": repr(error),
            "traceback": traceback.format_exc(),
        }
        self._errors.append(entry)
        if self._current is not None:
            self._current.errors.append(f"{stage}: {error!r}")

    # -------------------------------------------------------------- output
    def _close_jsonl(self) -> None:
        if self._jsonl_handle is not None:
            self._jsonl_handle.close()
            self._jsonl_handle = None

    def _hash_files(self, update_dir: Path) -> dict[str, Any]:
        files: dict[str, Any] = {}
        for path in sorted(update_dir.iterdir()):
            if not path.is_file() or path.name == "manifest.json":
                continue
            files[path.name] = {
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        return files

    def _write_manifest(self, path: Path, manifest: dict[str, Any]) -> None:
        _write_json(path, manifest)

    def _fail(
        self,
        update_dir: Path,
        manifest_path: Path,
        manifest: dict[str, Any],
        status: str,
    ) -> None:
        error_text = traceback.format_exc()
        try:
            (update_dir / "traceback.txt").write_text(error_text, encoding="utf-8")
        except OSError:
            pass
        manifest["status"] = status
        manifest["status_history"].append([utc_now(), status])
        manifest["utc_finished"] = utc_now()
        manifest["error"] = error_text[-8000:]
        manifest["metrics"] = self._summary()
        try:
            manifest["files"] = self._hash_files(update_dir)
            self._write_manifest(manifest_path, manifest)
        except Exception:
            pass
        self._model = None

    def _summary(self) -> dict[str, Any]:
        rows = self._rows
        nonfinite: dict[str, int] = {}
        for row in rows:
            for key, count in row.get("nonfinite", {}).items():
                nonfinite[key] = nonfinite.get(key, 0) + int(count)

        def values(path: tuple[str, ...]) -> list[float]:
            collected: list[float] = []
            for row in rows:
                node: Any = row
                for part in path:
                    if not isinstance(node, dict) or part not in node:
                        node = None
                        break
                    node = node[part]
                if isinstance(node, (int, float)) and math.isfinite(float(node)):
                    collected.append(float(node))
            return collected

        def aggregate(path: tuple[str, ...]) -> dict[str, Any]:
            # Quantiles are taken across per-minibatch scalar statistics
            # (one value per optimizer step), NOT across pooled per-sample
            # tensors. Raw per-sample values live in pre_probe/post_probe.npz.
            items = values(path)
            if not items:
                return {"n": 0, "mean": None, "min": None, "max": None}
            array = np.asarray(items, dtype=np.float64)
            p50, p90, p99 = np.quantile(array, [0.5, 0.9, 0.99])
            return {
                "n": len(items),
                "mean": float(array.mean()),
                "min": float(array.min()),
                "max": float(array.max()),
                "p50": float(p50),
                "p90": float(p90),
                "p99": float(p99),
            }

        epochs = sorted({row["epoch"] for row in rows})
        cosine_rows = [
            row["gradient_probe"]
            for row in rows
            if isinstance(row.get("gradient_probe"), dict)
            and row["gradient_probe"].get("cosine_actor_critic") is not None
        ]
        cosines = (
            np.asarray(
                [
                    row["cosine_actor_critic"]
                    for row in cosine_rows
                ],
                dtype=np.float64,
            )
            if cosine_rows
            else None
        )
        return {
            "minibatches": len(rows),
            "epochs_with_minibatches": len(epochs),
            "last_epoch": epochs[-1] if epochs else None,
            "optimizer_steps": sum(
                1 for row in rows if row["optimizer_stepped"]
            ),
            "early_stop_minibatch_present": any(
                row.get("metrics", {})
                .get("early_stop", {})
                .get("would_stop", False)
                for row in rows
            ),
            # Name states what is measured: mean(|ratio|), not |mean(ratio)-1|.
            "ratio_mean": aggregate(
                ("metrics", "ratio", "rpo_perturbed", "abs_mean")
            ),
            "unperturbed_ratio_mean": aggregate(
                ("metrics", "ratio", "unperturbed", "abs_mean")
            ),
            "ppo_approx_kl_perturbed": aggregate(
                ("metrics", "kl", "ppo_approx_perturbed")
            ),
            "ppo_approx_kl_unperturbed": aggregate(
                ("metrics", "kl", "ppo_approx_unperturbed")
            ),
            "analytic_kl_old_new": aggregate(
                ("metrics", "kl", "analytic_old_new", "mean")
            ),
            "rpo_noise_abs_mean": aggregate(
                ("metrics", "rpo", "noise", "abs_mean")
            ),
            "rpo_noise_max_abs": aggregate(
                ("metrics", "rpo", "noise_max_abs")
            ),
            "clip_fraction_abs": aggregate(
                ("metrics", "clip", "fraction_abs_perturbed")
            ),
            "loss_surrogate": aggregate(("metrics", "loss", "surrogate")),
            "loss_entropy": aggregate(("metrics", "loss", "entropy")),
            "loss_value": aggregate(("metrics", "loss", "value")),
            "loss_total": aggregate(("metrics", "loss", "total")),
            "gradient_pre_clip_norm": aggregate(
                ("gradient", "pre_clip_norm")
            ),
            "gradient_post_clip_norm": aggregate(
                ("gradient", "post_clip_norm")
            ),
            "parameter_step_norm": aggregate(("step", "param_step_norm")),
            "gradient_probe_actor_component_norm": aggregate(
                ("gradient_probe", "actor_component_norm")
            ),
            "gradient_probe_critic_component_norm": aggregate(
                ("gradient_probe", "critic_component_norm")
            ),
            "gradient_probe_total_component_norm": aggregate(
                ("gradient_probe", "total_component_norm")
            ),
            "advantage_sign_flips_vs_global": aggregate(
                ("metrics", "advantages", "sign_flips_vs_global", "fraction")
            ),
            "gradient_probe_cosine_actor_critic": {
                "n": len(cosine_rows),
                "mean": float(cosines.mean()) if cosines is not None else None,
                "min": float(cosines.min()) if cosines is not None else None,
                "max": float(cosines.max()) if cosines is not None else None,
                "p50": float(np.quantile(cosines, 0.5))
                if cosines is not None
                else None,
                "p90": float(np.quantile(cosines, 0.9))
                if cosines is not None
                else None,
                "p99": float(np.quantile(cosines, 0.99))
                if cosines is not None
                else None,
            },
            "nonfinite_totals": nonfinite,
            "instrumentation_errors": self._errors,
        }

    def _diagnostics(
        self,
        update_index: int,
        model: Any,
        pre_probe: dict[str, np.ndarray],
        post_probe: dict[str, np.ndarray],
    ) -> dict[str, Any]:
        orders = [
            hashlib.sha256(
                np.concatenate(epoch).astype(np.int64).tobytes()
            ).hexdigest()
            if epoch
            else None
            for epoch in self._epoch_orders
        ]
        overall = hashlib.sha256()
        for epoch in self._epoch_orders:
            if epoch:
                overall.update(np.concatenate(epoch).astype(np.int64).tobytes())
        kl = post_probe["kl_old_new"]
        return {
            "schema": SCHEMA_VERSION,
            "update_index": update_index,
            "config_post": self._model_config(model),
            "ordering": {
                "minibatches_per_epoch": self._n_minibatches_per_epoch,
                "epoch_sha256": orders,
                "overall_sha256": overall.hexdigest(),
                "rng_snapshot": "rng.pkl",
            },
            "pre_probe": {
                "samples": int(len(pre_probe["observations"])),
                "action_dim": int(pre_probe["actions"].shape[-1]),
                "obs_dim": int(pre_probe["observations"].shape[-1]),
                "old_log_prob_max_abs_diff_vs_buffer": float(
                    np.max(
                        np.abs(
                            pre_probe["old_unperturbed_log_probs"]
                            - pre_probe["old_log_probs"]
                        )
                    )
                ),
                "old_value_max_abs_diff_vs_buffer": float(
                    np.max(np.abs(pre_probe["recomputed_values"] - pre_probe["old_values"]))
                ),
                "log_std_per_action": _clean(pre_probe["log_std"]),
            },
            "post_probe": {
                "kl_old_new": {
                    "mean": float(np.mean(kl)),
                    "max": float(np.max(kl)),
                    "p95": float(np.quantile(kl, 0.95)),
                    "nonfinite": int(np.count_nonzero(~np.isfinite(kl))),
                },
                "log_std_per_action": _clean(post_probe["log_std"]),
                "value_mean_shift": float(
                    np.mean(post_probe["new_values"] - pre_probe["recomputed_values"])
                ),
            },
            "summary": self._summary(),
            "errors": self._errors,
        }


def _sb3_version() -> str:
    try:
        import stable_baselines3

        return stable_baselines3.__version__
    except Exception:
        return "unknown"


def _optional_version(module_name: str) -> str | None:
    try:
        module = __import__(module_name)
        return getattr(module, "__version__", None)
    except Exception:
        return None
