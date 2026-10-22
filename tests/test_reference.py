from __future__ import annotations

import json
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np

import train
from diagnostics import (
    JsonlMetricsWriter,
    SaveVecNormalizeOnBest,
    classify_episode_end,
    restore_eval_history,
)


def _reference_args() -> types.SimpleNamespace:
    return types.SimpleNamespace(
        variant=train.REFERENCE_VARIANT,
        seed=3,
        timesteps=1_000_000,
        torch_threads=1,
        eval_every=10_000,
        eval_episodes=10,
        eval_seed=50_000,
    )


def _fake_reference_model(**overrides):
    values = {
        "n_steps": train.REFERENCE_PPO_KWARGS["n_steps"],
        "batch_size": train.REFERENCE_PPO_KWARGS["batch_size"],
        "n_epochs": train.REFERENCE_PPO_KWARGS["n_epochs"],
        "gamma": train.REFERENCE_PPO_KWARGS["gamma"],
        "gae_lambda": train.REFERENCE_PPO_KWARGS["gae_lambda"],
        "ent_coef": train.REFERENCE_PPO_KWARGS["ent_coef"],
        "vf_coef": train.REFERENCE_PPO_KWARGS["vf_coef"],
        "max_grad_norm": train.REFERENCE_PPO_KWARGS["max_grad_norm"],
        "learning_rate": train.REFERENCE_LEARNING_RATE,
        "clip_range": train.REFERENCE_PPO_KWARGS["clip_range"],
        "policy_kwargs": train.REFERENCE_PPO_KWARGS["policy_kwargs"],
        "n_envs": 1,
    }
    values.update(overrides)
    return types.SimpleNamespace(**values)


class ReferenceProfileTests(unittest.TestCase):
    def test_reference_hyperparameters_and_learning_rate_are_fixed(self) -> None:
        self.assertEqual(
            train.REFERENCE_PPO_KWARGS,
            {
                "n_steps": 512,
                "batch_size": 32,
                "n_epochs": 20,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_range": 0.1,
                "ent_coef": 0.000585045,
                "vf_coef": 0.871923,
                "max_grad_norm": 1.0,
                "policy_kwargs": {
                    "net_arch": {"pi": [64, 64], "vf": [64, 64]}
                },
                "device": "cpu",
            },
        )
        self.assertEqual(train.REFERENCE_LEARNING_RATE, 5.05041e-5)
        self.assertEqual(
            train.learning_rate_for_variant(train.REFERENCE_VARIANT),
            train.REFERENCE_LEARNING_RATE,
        )
        self.assertTrue(
            train.normalized_for_variant(train.REFERENCE_VARIANT)
        )
        self.assertIs(
            train.model_kwargs_for_variant(train.REFERENCE_VARIANT),
            train.REFERENCE_PPO_KWARGS,
        )
        self.assertEqual(
            train.default_run_dir(train.REFERENCE_VARIANT, 7),
            Path("runs") / "ppo-reference" / "seed_7",
        )

    def test_old_variants_keep_their_profile(self) -> None:
        self.assertIs(
            train.model_kwargs_for_variant("baseline"),
            train.TOTAL_PPO_KWARGS,
        )
        self.assertEqual(
            train.learning_rate_for_variant("baseline"),
            train.INITIAL_LEARNING_RATE,
        )
        self.assertTrue(
            callable(train.learning_rate_for_variant("ppo-normalized"))
        )
        self.assertEqual(
            train.default_run_dir("baseline", 4),
            Path("runs") / "baseline" / "seed_4",
        )

    def test_config_records_reference_metadata_and_running_status(self) -> None:
        args = _reference_args()
        run_dir = Path("runs") / "ppo-reference" / "seed_3"
        config = train.build_run_config(
            args,
            run_dir,
            normalized=True,
            previous_config={},
            resumed_from=None,
            start_timesteps=0,
            resume_history=[],
            segments=[],
            eval_enabled=True,
            versions={"stable_baselines3": "2.9.0"},
            runtime={"utc_started": "2026-01-01T00:00:00+00:00"},
            git={"commit": "abc", "dirty": False},
            hashes={"train.py": "deadbeef"},
        )

        self.assertEqual(config["status"], "running")
        self.assertEqual(config["last_observed_timesteps"], 0)
        self.assertIsNone(config["error_type"])
        self.assertEqual(config["profile"], train.REFERENCE_VARIANT)
        self.assertEqual(config["ppo"]["n_steps"], 512)
        self.assertEqual(
            config["ppo"]["learning_rate_initial"],
            train.REFERENCE_LEARNING_RATE,
        )
        self.assertEqual(
            config["ppo"]["learning_rate_schedule"], "constant"
        )
        self.assertEqual(config["ppo"]["n_envs"], 1)
        self.assertEqual(
            config["normalization"],
            {
                "observations": True,
                "rewards": True,
                "clip_obs": 10.0,
                "clip_reward": 10.0,
                "gamma": 0.99,
            },
        )
        self.assertEqual(config["eval"]["eval_seed"], 50_000)
        self.assertTrue(config["eval"]["enabled"])
        self.assertEqual(
            config["reference"]["source_environment"], "Walker2d-v4"
        )
        self.assertIn("Walker2d-v5", config["reference"]["environment_label"])
        self.assertEqual(config["monitor_csv"], str(run_dir / "monitor.csv"))
        self.assertEqual(
            config["metrics_jsonl"], str(run_dir / "metrics.jsonl")
        )
        self.assertEqual(config["git"], {"commit": "abc", "dirty": False})

    def test_reference_overwrite_guard(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            (run_dir / "final_model.zip").touch()
            self.assertTrue(train.reference_existing_artifacts(run_dir))
            with self.assertRaises(SystemExit):
                train.ensure_reference_run_dir_available(run_dir, None)
            train.ensure_reference_run_dir_available(run_dir, "auto")


class ResumeValidationTests(unittest.TestCase):
    def _write_config(self, run_dir: Path) -> None:
        ppo = {
            key: value
            for key, value in train.REFERENCE_PPO_KWARGS.items()
            if key not in ("policy_kwargs", "device")
        }
        ppo["learning_rate_initial"] = train.REFERENCE_LEARNING_RATE
        (run_dir / "config.json").write_text(
            json.dumps({
                "variant": train.REFERENCE_VARIANT,
                "seed": 3,
                "profile": train.REFERENCE_VARIANT,
                "ppo": ppo,
                "normalization": {
                    "observations": True,
                    "rewards": True,
                },
            }),
            encoding="utf-8",
        )

    def test_reference_resume_accepts_matching_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            self._write_config(run_dir)
            train.validate_resume(
                run_dir,
                _reference_args(),
                _fake_reference_model(),
                normalized=True,
            )

    def test_reference_resume_rejects_mismatched_hyperparameters(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            self._write_config(run_dir)
            with self.assertRaises(SystemExit):
                train.validate_resume(
                    run_dir,
                    _reference_args(),
                    _fake_reference_model(n_steps=2048),
                    normalized=True,
                )
            with self.assertRaises(SystemExit):
                train.validate_resume(
                    run_dir,
                    _reference_args(),
                    _fake_reference_model(learning_rate=3e-4),
                    normalized=True,
                )

    def test_reference_resume_requires_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(SystemExit):
                train.validate_resume(
                    Path(directory),
                    _reference_args(),
                    _fake_reference_model(),
                    normalized=True,
                )


class DiagnosticsTests(unittest.TestCase):
    def test_jsonl_writer_converts_numpy_and_honors_excludes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.jsonl"
            writer = JsonlMetricsWriter(path)
            writer.write(
                {
                    "train/loss": np.float32(1.5),
                    "train/vector": np.asarray([1, 2]),
                    "skip/me": np.int64(3),
                },
                {"skip/me": ("jsonl",)},
                step=np.int64(7),
            )
            writer.write_event(8, {"eval/rate": np.float64(0.5)})
            writer.close()

            rows = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(rows[0]["step"], 7)
            self.assertEqual(rows[0]["train/loss"], 1.5)
            self.assertEqual(rows[0]["train/vector"], [1, 2])
            self.assertNotIn("skip/me", rows[0])
            self.assertEqual(rows[1]["eval/rate"], 0.5)

    def test_episode_end_classification_uses_time_limit_flag(self) -> None:
        self.assertEqual(
            classify_episode_end({"TimeLimit.truncated": True}),
            "healthy_timeout",
        )
        self.assertEqual(
            classify_episode_end({"TimeLimit.truncated": False}),
            "unhealthy_termination",
        )
        self.assertEqual(
            classify_episode_end({}), "unhealthy_termination"
        )

    def test_save_vecnormalize_on_best_does_not_train(self) -> None:
        class FakeVecNormalize:
            def __init__(self) -> None:
                self.saved: list[str] = []

            def save(self, path: str) -> None:
                self.saved.append(path)

        class FakeModel:
            def __init__(self) -> None:
                self.vec_normalize = FakeVecNormalize()

            def get_vec_normalize_env(self):
                return self.vec_normalize

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "best" / "vecnormalize.pkl"
            callback = SaveVecNormalizeOnBest(path)
            callback.model = FakeModel()

            self.assertTrue(callback._on_step())
            self.assertEqual(
                callback.model.vec_normalize.saved, [str(path)]
            )
            self.assertTrue(path.parent.is_dir())

    def test_restore_eval_history_sets_best_from_npz(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            np.savez(
                run_dir / "evaluations.npz",
                timesteps=np.asarray([512, 1024]),
                results=np.asarray([[1.0, 2.0], [3.0, 4.0]]),
                ep_lengths=np.asarray([[10, 20], [30, 40]]),
            )
            callback = types.SimpleNamespace()

            self.assertTrue(restore_eval_history(callback, run_dir))
            self.assertEqual(callback.evaluations_timesteps, [512, 1024])
            self.assertEqual(callback.evaluations_results, [[1.0, 2.0], [3.0, 4.0]])
            self.assertEqual(callback.best_mean_reward, 3.5)


if __name__ == "__main__":
    unittest.main()
