"""Tests for the per-update flight recorder.

These use a tiny Pendulum policy so the checks stay short and CPU-only. They
verify the two properties the recorder exists for:

* attaching a recorder cannot change the update (weights, Adam, RNG), and
* a pre-update snapshot replays the update exactly.
"""

from __future__ import annotations

import hashlib
import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.logger import Logger
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from rpo_ppo import RPO
from update_recording import (
    UpdateRecorder,
    load_rng_state,
    load_rollout_buffer,
    restore_rng_state,
    save_rng_state,
)

ENV_ID = "Pendulum-v1"
SEED = 91
ENV_SEED = 101
N_STEPS = 32
BATCH_SIZE = 16
N_EPOCHS = 3
POLICY_KWARGS = {"net_arch": {"pi": [16, 16], "vf": [16, 16]}}


def make_normalized_env() -> VecNormalize:
    vector_env = DummyVecEnv([lambda: Monitor(gym.make(ENV_ID))])
    env = VecNormalize(vector_env, norm_obs=True, norm_reward=True)
    env.seed(ENV_SEED)
    return env


def make_model(env: VecNormalize, rpo_alpha: float = 0.5) -> RPO:
    return RPO(
        "MlpPolicy",
        env,
        seed=SEED,
        rpo_alpha=rpo_alpha,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        device="cpu",
        policy_kwargs=POLICY_KWARGS,
    )


class UpdateRecorderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls) -> None:
        torch.set_num_threads(cls.original_threads)

    # ------------------------------------------------------------- helpers
    def run_training(self, record: bool, total_timesteps: int = 64):
        env = make_normalized_env()
        model = make_model(env)
        temporary = tempfile.TemporaryDirectory() if record else None
        if record:
            UpdateRecorder(Path(temporary.name)).attach(model)
        try:
            model.learn(total_timesteps=total_timesteps, progress_bar=False)
            return {
                "parameters": {
                    name: value.detach().clone()
                    for name, value in model.policy.state_dict().items()
                },
                "optimizer": model.policy.optimizer.state_dict(),
                "torch_rng": torch.get_rng_state().clone(),
                "numpy_rng": np.random.get_state(),
                "python_rng": random.getstate(),
                "obs_mean": env.obs_rms.mean.copy(),
                "obs_var": env.obs_rms.var.copy(),
                "num_timesteps": model.num_timesteps,
                "recorder": temporary,
            }
        finally:
            env.close()

    def assert_same_state(self, first: dict, second: dict) -> None:
        self.assertEqual(first["num_timesteps"], second["num_timesteps"])
        self.assertEqual(first["parameters"].keys(), second["parameters"].keys())
        for name, value in first["parameters"].items():
            self.assertTrue(
                torch.equal(value, second["parameters"][name]), name
            )
        first_optimizer = first["optimizer"]
        second_optimizer = second["optimizer"]
        self.assertEqual(
            first_optimizer["param_groups"], second_optimizer["param_groups"]
        )
        self.assertEqual(
            set(first_optimizer["state"]), set(second_optimizer["state"])
        )
        for parameter, state in first_optimizer["state"].items():
            for name, value in state.items():
                self.assertTrue(
                    torch.equal(value, second_optimizer["state"][parameter][name]),
                    f"optimizer {parameter}/{name}",
                )
        self.assertTrue(torch.equal(first["torch_rng"], second["torch_rng"]))
        first_numpy = first["numpy_rng"]
        second_numpy = second["numpy_rng"]
        self.assertEqual(first_numpy[0], second_numpy[0])
        self.assertTrue(np.array_equal(first_numpy[1], second_numpy[1]))
        self.assertEqual(first_numpy[2:], second_numpy[2:])
        self.assertEqual(first["python_rng"], second["python_rng"])
        self.assertTrue(np.array_equal(first["obs_mean"], second["obs_mean"]))
        self.assertTrue(np.array_equal(first["obs_var"], second["obs_var"]))

    def assert_same_parameters(self, model_a, model_b) -> None:
        params_a = model_a.get_parameters()
        params_b = model_b.get_parameters()
        self.assertEqual(set(params_a), set(params_b))
        for name in params_a["policy"]:
            self.assertTrue(
                torch.equal(params_a["policy"][name], params_b["policy"][name]),
                name,
            )
        state_a = params_a["policy.optimizer"]["state"]
        state_b = params_b["policy.optimizer"]["state"]
        self.assertEqual(set(state_a), set(state_b))
        for parameter, state in state_a.items():
            for name, value in state.items():
                self.assertTrue(
                    torch.equal(value, state_b[parameter][name]),
                    f"optimizer {parameter}/{name}",
                )

    # --------------------------------------------------------------- tests
    def test_recorder_is_bit_exact_over_two_updates(self) -> None:
        without = self.run_training(record=False)
        with_recorder = self.run_training(record=True)
        try:
            self.assert_same_state(without, with_recorder)
        finally:
            with_recorder["recorder"].cleanup()

    def test_pre_snapshot_replays_the_update_exactly(self) -> None:
        empty = tempfile.TemporaryDirectory()
        env = make_normalized_env()
        model = make_model(env)
        recorder = UpdateRecorder(Path(empty.name)).attach(model)
        try:
            model.learn(total_timesteps=N_STEPS, progress_bar=False)
            update_dir = Path(empty.name) / "update_000000"
            for name in (
                "manifest.json",
                "model_pre.zip",
                "model_post.zip",
                "rollout_buffer.pkl",
                "rng.pkl",
                "vecnormalize.pkl",
                "pre_probe.npz",
                "post_probe.npz",
                "minibatches.jsonl",
                "diagnostics.json",
            ):
                self.assertTrue((update_dir / name).is_file(), name)

            manifest = json.loads((update_dir / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(
                [entry[1] for entry in manifest["status_history"]],
                ["running", "pre_saved", "completed"],
            )
            self.assertEqual(
                manifest["runtime"]["torch_num_threads"],
                torch.get_num_threads(),
            )
            saved_rng = load_rng_state(update_dir / "rng.pkl")
            self.assertEqual(
                saved_rng["torch_num_threads"], torch.get_num_threads()
            )
            self.assertIn("policy.optimizer", manifest["config"]["save_objects"])
            self.assertIn(
                "policy.optimizer.pth", manifest["pre_model_members"]
            )
            self.assertIn(
                "policy.optimizer.pth", manifest["post_model_members"]
            )
            for name, entry in manifest["files"].items():
                self.assertIn("sha256", entry, name)
                self.assertGreater(entry["size"], 0, name)

            rows = [
                json.loads(line)
                for line in (update_dir / "minibatches.jsonl")
                .read_text()
                .splitlines()
            ]
            self.assertEqual(len(rows), N_EPOCHS * (N_STEPS // BATCH_SIZE))
            self.assertEqual({row["epoch"] for row in rows}, set(range(N_EPOCHS)))
            self.assertTrue(all(row["optimizer_stepped"] for row in rows))
            self.assertTrue(all(row["step"] for row in rows))
            self.assertTrue(all(row["gradient"] for row in rows))
            probes = [
                row["gradient_probe"]
                for row in rows
                if row["minibatch_in_epoch"] == 0
            ]
            self.assertEqual(len(probes), N_EPOCHS)
            self.assertTrue(all(probe for probe in probes))
            for row in rows:
                metrics = row["metrics"]
                self.assertIn("loss", metrics)
                self.assertIn("kl", metrics)
                self.assertIn("rpo", metrics)
                self.assertIsNotNone(metrics["rpo"]["noise"])
                self.assertIn("nonfinite", row)
                if row["gradient_probe"] is not None:
                    self.assertEqual(
                        row["gradient_probe"]["grad_probe_modified_dot_grad"],
                        False,
                    )

            diagnostics = json.loads(
                (update_dir / "diagnostics.json").read_text()
            )
            self.assertEqual(diagnostics["errors"], [])
            self.assertAlmostEqual(
                diagnostics["pre_probe"][
                    "old_log_prob_max_abs_diff_vs_buffer"
                ],
                0.0,
                places=6,
            )
            self.assertGreater(len(diagnostics["ordering"]["overall_sha256"]), 0)
            summary = diagnostics["summary"]
            for key in (
                "analytic_kl_old_new",
                "ppo_approx_kl_perturbed",
                "ratio_mean",
                "rpo_noise_abs_mean",
                "gradient_pre_clip_norm",
                "gradient_post_clip_norm",
                "parameter_step_norm",
            ):
                for quantile in ("p50", "p90", "p99"):
                    self.assertIsNotNone(summary[key][quantile], key)

            replay = RPO.load(update_dir / "model_pre.zip", device="cpu")
            replay.set_logger(Logger(folder=None, output_formats=[]))
            replay.rollout_buffer = load_rollout_buffer(
                update_dir / "rollout_buffer.pkl"
            )
            restore_rng_state(load_rng_state(update_dir / "rng.pkl"))
            replay.train()
            stored = RPO.load(update_dir / "model_post.zip", device="cpu")
            self.assert_same_parameters(replay, stored)
        finally:
            recorder.detach(model)
            env.close()
            empty.cleanup()

    def test_alpha_zero_fallback_is_recorded(self) -> None:
        empty = tempfile.TemporaryDirectory()
        env = make_normalized_env()
        model = make_model(env, rpo_alpha=0.0)
        recorder = UpdateRecorder(Path(empty.name)).attach(model)
        try:
            model.learn(total_timesteps=N_STEPS, progress_bar=False)
            update_dir = Path(empty.name) / "update_000000"
            diagnostics = json.loads(
                (update_dir / "diagnostics.json").read_text()
            )
            self.assertEqual(diagnostics["errors"], [])
            rows = [
                json.loads(line)
                for line in (update_dir / "minibatches.jsonl")
                .read_text()
                .splitlines()
            ]
            self.assertEqual(len(rows), N_EPOCHS * (N_STEPS // BATCH_SIZE))
            for row in rows:
                self.assertIsNone(row["metrics"]["rpo"]["noise"])
                ratio = row["metrics"]["ratio"]["unperturbed"]["abs_mean"]
                self.assertIsNotNone(ratio)
            # Before any optimizer step the unperturbed ratio is PPO's ratio.
            first_ratio = rows[0]["metrics"]["ratio"]["unperturbed"]["abs_mean"]
            self.assertLess(abs(first_ratio - 1.0), 0.01)
        finally:
            recorder.detach(model)
            env.close()
            empty.cleanup()

    def test_pre_save_failure_fails_closed(self) -> None:
        empty = tempfile.TemporaryDirectory()
        env = make_normalized_env()
        model = make_model(env)
        try:
            # A full buffer from the previous update is enough; the recorder
            # must fail before it ever invokes the training callable.
            model.learn(total_timesteps=N_STEPS, progress_bar=False)
            weights_before = {
                name: value.detach().clone()
                for name, value in model.policy.state_dict().items()
            }
            optimizer_before = model.policy.optimizer.state_dict()
            recorder = UpdateRecorder(Path(empty.name)).attach(model)
            with patch.object(model, "save", side_effect=RuntimeError("disk full")):
                with self.assertRaisesRegex(RuntimeError, "disk full"):
                    recorder.run_update(model, lambda: self.fail("training ran"))
            for name, value in weights_before.items():
                self.assertTrue(
                    torch.equal(value, model.policy.state_dict()[name]), name
                )
            for parameter, state in optimizer_before["state"].items():
                for name, value in state.items():
                    self.assertTrue(
                        torch.equal(
                            value,
                            model.policy.optimizer.state_dict()["state"][
                                parameter
                            ][name],
                        ),
                        f"optimizer {parameter}/{name}",
                    )
            update_dir = Path(empty.name) / "update_000000"
            manifest = json.loads((update_dir / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "pre_save_failed")
            self.assertIn("disk full", manifest["error"])
            self.assertTrue((update_dir / "traceback.txt").is_file())
            recorder.detach(model)
        finally:
            env.close()
            empty.cleanup()

    def test_instrumentation_failure_fails_closed_after_update(self) -> None:
        empty = tempfile.TemporaryDirectory()
        env = make_normalized_env()
        model = make_model(env)
        recorder = UpdateRecorder(Path(empty.name)).attach(model)
        weights_before = {
            name: value.detach().clone()
            for name, value in model.policy.state_dict().items()
        }
        try:
            # The hook swallows the exception so the optimizer update still
            # runs. The recorder must then mark the package invalid and raise
            # instead of reporting a completed update with missing metrics.
            with patch.object(
                recorder,
                "_end_minibatch",
                side_effect=RuntimeError("probe boom"),
            ):
                with self.assertRaisesRegex(RuntimeError, "instrumentation failed"):
                    model.learn(total_timesteps=N_STEPS, progress_bar=False)
            update_dir = Path(empty.name) / "update_000000"
            manifest = json.loads((update_dir / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "instrumentation_failed")
            self.assertEqual(
                [entry[1] for entry in manifest["status_history"]],
                ["running", "pre_saved", "instrumentation_failed"],
            )
            self.assertIn("probe boom", manifest["error"])
            self.assertTrue((update_dir / "traceback.txt").is_file())
            # Post-update evidence is preserved for diagnosis.
            self.assertTrue((update_dir / "model_post.zip").is_file())
            self.assertIn(
                "policy.optimizer.pth", manifest["post_model_members"]
            )
            diagnostics = json.loads(
                (update_dir / "diagnostics.json").read_text()
            )
            self.assertTrue(diagnostics["errors"])
            self.assertTrue(
                any(
                    not torch.equal(value, model.policy.state_dict()[name])
                    for name, value in weights_before.items()
                )
            )
        finally:
            recorder.detach(model)
            env.close()
            empty.cleanup()

    def test_jsonl_write_failure_restores_hooks_and_keeps_original_error(
        self,
    ) -> None:
        empty = tempfile.TemporaryDirectory()
        env = make_normalized_env()
        model = make_model(env)
        try:
            model.learn(total_timesteps=N_STEPS, progress_bar=False)
            recorder = UpdateRecorder(Path(empty.name)).attach(model)
            original_clip = torch.nn.utils.clip_grad_norm_
            buffer = model.rollout_buffer
            policy = model.policy
            optimizer = policy.optimizer

            def failing_train() -> None:
                handle = recorder._jsonl_handle

                class BrokenHandle:
                    def write(self, data: str) -> None:
                        raise OSError("disk full")

                    def flush(self) -> None:
                        raise OSError("disk full")

                    def close(self) -> None:
                        handle.close()

                recorder._jsonl_handle = BrokenHandle()
                policy.evaluate_actions(
                    torch.zeros(4, *model.observation_space.shape),
                    torch.zeros(4, *model.action_space.shape),
                )
                raise RuntimeError("training stopped")

            with self.assertRaisesRegex(RuntimeError, "training stopped"):
                recorder.run_update(model, failing_train)

            self.assertIs(torch.nn.utils.clip_grad_norm_, original_clip)
            for instance, name in (
                (buffer, "get"),
                (buffer, "_get_samples"),
                (policy, "evaluate_actions"),
                (policy, "_sample_rpo_noise"),
                (optimizer, "step"),
            ):
                self.assertNotIn(name, instance.__dict__, name)
            self.assertIsNone(recorder._jsonl_handle)
            update_dir = Path(empty.name) / "update_000000"
            manifest = json.loads((update_dir / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "train_failed")
            self.assertNotEqual(manifest["status"], "completed")
            errors = manifest["metrics"]["instrumentation_errors"]
            self.assertTrue(
                any("disk full" in entry["error"] for entry in errors), errors
            )
            recorder.detach(model)
        finally:
            env.close()
            empty.cleanup()

    def test_jsonl_close_failure_marks_package_not_completed(self) -> None:
        empty = tempfile.TemporaryDirectory()
        env = make_normalized_env()
        model = make_model(env)
        recorder = UpdateRecorder(Path(empty.name)).attach(model)
        original_clip = torch.nn.utils.clip_grad_norm_
        buffer = model.rollout_buffer
        policy = model.policy
        optimizer = policy.optimizer
        try:
            with patch.object(
                recorder, "_close_jsonl", side_effect=OSError("disk full")
            ):
                with self.assertRaisesRegex(OSError, "disk full"):
                    model.learn(total_timesteps=N_STEPS, progress_bar=False)
            self.assertIs(torch.nn.utils.clip_grad_norm_, original_clip)
            for instance, name in (
                (buffer, "get"),
                (buffer, "_get_samples"),
                (policy, "evaluate_actions"),
                (policy, "_sample_rpo_noise"),
                (optimizer, "step"),
            ):
                self.assertNotIn(name, instance.__dict__, name)
            manifest = json.loads(
                (Path(empty.name) / "update_000000" / "manifest.json").read_text()
            )
            self.assertEqual(manifest["status"], "instrumentation_failed")
            self.assertNotEqual(manifest["status"], "completed")
            errors = manifest["metrics"]["instrumentation_errors"]
            self.assertTrue(
                any("disk full" in entry["error"] for entry in errors), errors
            )
        finally:
            if recorder._jsonl_handle is not None:
                recorder._jsonl_handle.close()
                recorder._jsonl_handle = None
            recorder.detach(model)
            env.close()
            empty.cleanup()

    def test_rebuilt_recorder_continues_after_existing_updates(self) -> None:
        empty = tempfile.TemporaryDirectory()
        env = make_normalized_env()
        model = make_model(env)
        root = Path(empty.name)
        try:
            first = UpdateRecorder(root).attach(model)
            model.learn(total_timesteps=N_STEPS, progress_bar=False)
            first.detach(model)
            first_dir = root / "update_000000"
            hashes_before = {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in first_dir.iterdir()
                if path.is_file()
            }

            # --resume constructs a brand-new recorder on the same run_dir.
            resumed = UpdateRecorder(root).attach(model)
            model.learn(total_timesteps=N_STEPS, progress_bar=False)
            resumed.detach(model)

            second_dir = root / "update_000001"
            self.assertTrue(second_dir.is_dir())
            hashes_after = {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in first_dir.iterdir()
                if path.is_file()
            }
            self.assertEqual(hashes_before, hashes_after)
            manifest = json.loads((second_dir / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(manifest["update_index"], 1)

            # Failed/abandoned numeric package directories advance the index.
            (root / "update_000007").mkdir()
            self.assertEqual(UpdateRecorder(root)._next_index, 8)
        finally:
            env.close()
            empty.cleanup()

    def test_pre_probe_log_std_is_a_pre_update_snapshot(self) -> None:
        empty = tempfile.TemporaryDirectory()
        env = make_normalized_env()
        model = make_model(env)
        recorder = UpdateRecorder(Path(empty.name)).attach(model)
        try:
            model.learn(total_timesteps=N_STEPS, progress_bar=False)
            update_dir = Path(empty.name) / "update_000000"
            manifest = json.loads((update_dir / "manifest.json").read_text())
            diagnostics = json.loads(
                (update_dir / "diagnostics.json").read_text()
            )
            pre_model = RPO.load(update_dir / "model_pre.zip", device="cpu")
            post_model = RPO.load(update_dir / "model_post.zip", device="cpu")
            pre_log_std = pre_model.policy.log_std.detach().cpu().numpy()
            post_log_std = post_model.policy.log_std.detach().cpu().numpy()

            # The update changes log_std, so an un-copied ndarray aliasing the
            # parameter would make the pre-diagnostics show post values.
            self.assertFalse(np.array_equal(pre_log_std, post_log_std))
            self.assertTrue(
                np.allclose(
                    diagnostics["pre_probe"]["log_std_per_action"],
                    pre_log_std,
                    rtol=1e-5,
                    atol=1e-9,
                )
            )
            self.assertTrue(
                np.allclose(
                    manifest["config"]["log_std"],
                    pre_log_std,
                    rtol=0.0,
                    atol=1e-7,
                )
            )
            self.assertTrue(
                np.allclose(
                    diagnostics["post_probe"]["log_std_per_action"],
                    post_log_std,
                    rtol=1e-5,
                    atol=1e-9,
                )
            )
            with np.load(update_dir / "pre_probe.npz") as probe:
                self.assertTrue(np.array_equal(probe["log_std"], pre_log_std))
        finally:
            recorder.detach(model)
            env.close()
            empty.cleanup()

    def test_rng_helpers_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rng.pkl"
            random.random()
            np.random.random(3)
            torch.rand(3)
            save_rng_state(path)
            saved_state = load_rng_state(path)
            self.assertEqual(
                saved_state["torch_num_threads"], torch.get_num_threads()
            )
            random_state_after = random.getstate()
            numpy_state_after = np.random.get_state()
            torch_state_after = torch.get_rng_state().clone()
            random.random()
            np.random.random(3)
            torch.rand(3)
            restore_rng_state(saved_state)
            self.assertEqual(random.getstate(), random_state_after)
            numpy_state = np.random.get_state()
            self.assertEqual(numpy_state[0], numpy_state_after[0])
            self.assertTrue(np.array_equal(numpy_state[1], numpy_state_after[1]))
            self.assertEqual(numpy_state[2:], numpy_state_after[2:])
            self.assertTrue(torch.equal(torch.get_rng_state(), torch_state_after))


if __name__ == "__main__":
    unittest.main()
