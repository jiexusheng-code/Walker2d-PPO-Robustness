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


def _args(
    variant: str = train.REFERENCE_VARIANT, seed: int = 3
) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        variant=variant,
        seed=seed,
        timesteps=1_000_000,
        torch_threads=1,
        eval_every=10_000,
        eval_episodes=10,
        eval_seed=50_000,
        rpo_alpha=None,
    )


def _fake_model(variant: str = train.REFERENCE_VARIANT, **overrides):
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
        "truly_alpha": train.TrulyPPO.truly_alpha,
        "truly_delta": train.TrulyPPO.truly_delta,
        "rpo_alpha": train.RPO.DEFAULT_ALPHA,
    }
    values.update(overrides)
    return types.SimpleNamespace(**values)


def _write_config(
    run_dir: Path,
    *,
    variant: str = train.REFERENCE_VARIANT,
    profile: str = train.REFERENCE_PROFILE,
    seed: int = 3,
    ppo_overrides: dict | None = None,
    actor: dict | None | str = "default",
    normalization: dict | None = None,
) -> None:
    ppo = {
        key: train.REFERENCE_PPO_KWARGS[key]
        for key in (
            "n_steps",
            "batch_size",
            "n_epochs",
            "gamma",
            "gae_lambda",
            "clip_range",
            "ent_coef",
            "vf_coef",
            "max_grad_norm",
        )
    }
    ppo["learning_rate_initial"] = train.REFERENCE_LEARNING_RATE
    ppo["learning_rate_schedule"] = "constant"
    ppo["policy_kwargs"] = train.REFERENCE_PPO_KWARGS["policy_kwargs"]
    ppo.update(ppo_overrides or {})
    config = {
        "variant": variant,
        "seed": seed,
        "profile": profile,
        "environment": "Walker2d-v5",
        "algorithm": train.algorithm_name_for_variant(variant),
        "ppo": ppo,
        "normalization": (
            {"observations": True, "rewards": True}
            if normalization is None else normalization
        ),
    }
    if actor == "default":
        config["actor"] = train.actor_config_for_variant(variant)
    elif actor is not None:
        config["actor"] = actor
    (run_dir / "config.json").write_text(
        json.dumps(config), encoding="utf-8"
    )


class SharedProfileTests(unittest.TestCase):
    def test_project_exposes_shared_profile_variants(self) -> None:
        self.assertEqual(
            train.VARIANTS,
            (train.REFERENCE_VARIANT, train.TRULY_VARIANT, train.RPO_VARIANT),
        )
        self.assertEqual(train.DEFAULT_VARIANT, train.REFERENCE_VARIANT)
        self.assertEqual(train.REFERENCE_PROFILE, train.REFERENCE_VARIANT)
        for variant in train.VARIANTS:
            self.assertIs(
                train.model_kwargs_for_variant(variant),
                train.REFERENCE_PPO_KWARGS,
            )
            self.assertEqual(
                train.learning_rate_for_variant(variant),
                train.REFERENCE_LEARNING_RATE,
            )
            self.assertEqual(
                train.default_run_dir(variant, 7),
                Path("runs") / variant / "seed_7",
            )
        self.assertIs(
            train.model_class_for_variant(train.REFERENCE_VARIANT), train.PPO
        )
        self.assertIs(
            train.model_class_for_variant(train.TRULY_VARIANT), train.TrulyPPO
        )
        self.assertIs(
            train.model_class_for_variant(train.RPO_VARIANT), train.RPO
        )
        self.assertEqual(
            train.algorithm_name_for_variant(train.RPO_VARIANT), "RPO"
        )
        with self.assertRaises(SystemExit):
            train.model_kwargs_for_variant("baseline")
        with self.assertRaises(SystemExit):
            train.model_class_for_variant("ppo-normalized")

    def test_config_records_shared_settings_and_actor_difference(self) -> None:
        configs = {}
        for variant in train.VARIANTS:
            run_dir = Path("runs") / variant / "seed_3"
            configs[variant] = train.build_run_config(
                _args(variant),
                run_dir,
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

        reference = configs[train.REFERENCE_VARIANT]
        truly = configs[train.TRULY_VARIANT]
        for key in (
            "environment",
            "profile",
            "seed",
            "requested_timesteps",
            "ppo",
            "normalization",
            "eval",
            "recipe",
            "versions",
            "git",
            "source_sha256",
        ):
            self.assertEqual(reference[key], truly[key], key)

        self.assertEqual(reference["profile"], train.REFERENCE_PROFILE)
        self.assertEqual(truly["profile"], train.REFERENCE_PROFILE)
        self.assertEqual(reference["algorithm"], "PPO")
        self.assertEqual(truly["algorithm"], "TrulyPPO")
        self.assertEqual(reference["ppo"]["n_steps"], 512)
        self.assertEqual(
            reference["ppo"]["learning_rate_initial"],
            train.REFERENCE_LEARNING_RATE,
        )
        self.assertEqual(
            reference["ppo"]["learning_rate_schedule"], "constant"
        )
        self.assertEqual(
            reference["normalization"],
            {
                "observations": True,
                "rewards": True,
                "clip_obs": 10.0,
                "clip_reward": 10.0,
                "gamma": 0.99,
            },
        )
        self.assertEqual(reference["monitor_csv"], str(
            Path("runs") / train.REFERENCE_VARIANT / "seed_3" / "monitor.csv"
        ))

        self.assertEqual(reference["actor"]["objective"], "ppo_clip")
        self.assertTrue(reference["actor"]["clip_range_used"])
        self.assertIsNone(reference["actor"]["kl_rollback"])
        self.assertEqual(
            truly["actor"]["objective"],
            "truly_ppo_kl_triggered_rollback",
        )
        self.assertFalse(truly["actor"]["clip_range_used"])
        self.assertEqual(truly["actor"]["kl_rollback"]["alpha"], 5.0)
        self.assertEqual(truly["actor"]["kl_rollback"]["delta"], 0.03)
        self.assertIn(
            "no parameter", truly["actor"]["kl_rollback"]["note"]
        )

        rpo = configs[train.RPO_VARIANT]
        self.assertEqual(rpo["profile"], train.REFERENCE_PROFILE)
        self.assertEqual(rpo["algorithm"], "RPO")
        self.assertEqual(
            rpo["actor"]["objective"], "rpo_uniform_mean_perturbation"
        )
        self.assertTrue(rpo["actor"]["clip_range_used"])
        self.assertIsNone(rpo["actor"]["kl_rollback"])
        self.assertEqual(
            rpo["actor"]["rpo"]["alpha"], train.RPO.DEFAULT_ALPHA
        )
        self.assertIn("training-update", rpo["actor"]["rpo"]["scope"])
        self.assertIn("alpha=0", rpo["actor"]["rpo"]["alpha_zero"])
        self.assertIn("cleanrl", rpo["recipe"]["rpo_source"])
        self.assertEqual(rpo["ppo"], reference["ppo"])
        self.assertEqual(rpo["normalization"], reference["normalization"])

        self.assertIn("Walker2d-v4", reference["recipe"]["environment_label"])
        self.assertIn("1903.07940", reference["recipe"]["truly_ppo_source"])
        self.assertIn("C. Wen", reference["recipe"]["truly_ppo_source"])

    def test_overwrite_guard_refuses_both_arms(self) -> None:
        for variant in train.VARIANTS:
            with tempfile.TemporaryDirectory() as directory:
                run_dir = Path(directory)
                (run_dir / "final_model.zip").touch()
                with self.assertRaises(SystemExit):
                    train.ensure_run_dir_available(run_dir)
                with self.assertRaises(SystemExit):
                    train.ensure_run_dir_available(
                        run_dir,
                        allow_existing_from=run_dir.parent / "elsewhere.zip",
                    )
                train.ensure_run_dir_available(
                    run_dir,
                    allow_existing_from=run_dir / "final_model.zip",
                )
                self.assertTrue(variant)


class ResumeValidationTests(unittest.TestCase):
    def test_resume_accepts_matching_config_for_both_arms(self) -> None:
        for variant in train.VARIANTS:
            with tempfile.TemporaryDirectory() as directory:
                run_dir = Path(directory)
                _write_config(run_dir, variant=variant)
                train.validate_resume(
                    run_dir, _args(variant), _fake_model(variant)
                )

    def test_resume_rejects_shared_hyperparameter_mismatch_for_both(self) -> None:
        for variant in train.VARIANTS:
            with tempfile.TemporaryDirectory() as directory:
                run_dir = Path(directory)
                _write_config(run_dir, variant=variant)
                with self.assertRaises(SystemExit):
                    train.validate_resume(
                        run_dir,
                        _args(variant),
                        _fake_model(variant, n_steps=2048),
                    )
                with self.assertRaises(SystemExit):
                    train.validate_resume(
                        run_dir,
                        _args(variant),
                        _fake_model(variant, learning_rate=3e-4),
                    )

    def test_resume_requires_config_for_both_arms(self) -> None:
        for variant in train.VARIANTS:
            with tempfile.TemporaryDirectory() as directory:
                with self.assertRaises(SystemExit):
                    train.validate_resume(
                        Path(directory), _args(variant), _fake_model(variant)
                    )

    def test_resume_rejects_invalid_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            (run_dir / "config.json").write_text(
                "{not json", encoding="utf-8"
            )
            with self.assertRaises(SystemExit):
                train.validate_resume(
                    run_dir,
                    _args(train.REFERENCE_VARIANT),
                    _fake_model(),
                )

    def test_resume_rejects_old_truly_base(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            _write_config(
                run_dir,
                variant=train.TRULY_VARIANT,
                profile="truly-ppo",
                ppo_overrides={"n_steps": 2048, "batch_size": 64},
                actor=None,
            )
            with self.assertRaises(SystemExit):
                train.validate_resume(
                    run_dir,
                    _args(train.TRULY_VARIANT),
                    _fake_model(train.TRULY_VARIANT),
                )

    def test_resume_rejects_wrong_variant(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            _write_config(run_dir, variant=train.REFERENCE_VARIANT)
            with self.assertRaises(SystemExit):
                train.validate_resume(
                    run_dir,
                    _args(train.TRULY_VARIANT),
                    _fake_model(train.TRULY_VARIANT),
                )

    def test_resume_rejects_actor_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            _write_config(
                run_dir,
                variant=train.TRULY_VARIANT,
                actor={
                    "objective": "ppo_clip",
                    "clip_range_used": True,
                    "kl_rollback": None,
                },
            )
            with self.assertRaises(SystemExit):
                train.validate_resume(
                    run_dir,
                    _args(train.TRULY_VARIANT),
                    _fake_model(train.TRULY_VARIANT),
                )

            _write_config(
                run_dir,
                variant=train.TRULY_VARIANT,
                actor={
                    "objective": train.ACTOR_OBJECTIVES[
                        train.TRULY_VARIANT
                    ],
                    "clip_range_used": False,
                    "kl_rollback": {
                        "alpha": 5.0,
                        "delta": 0.05,
                    },
                },
            )
            with self.assertRaises(SystemExit):
                train.validate_resume(
                    run_dir,
                    _args(train.TRULY_VARIANT),
                    _fake_model(train.TRULY_VARIANT),
                )

    def test_resume_rejects_missing_normalization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            _write_config(
                run_dir,
                variant=train.REFERENCE_VARIANT,
                normalization={},
            )
            with self.assertRaises(SystemExit):
                train.validate_resume(
                    run_dir, _args(train.REFERENCE_VARIANT), _fake_model()
                )

    def test_legacy_reference_config_without_actor_is_resumable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            _write_config(
                run_dir,
                variant=train.REFERENCE_VARIANT,
                actor=None,
            )
            train.validate_resume(
                run_dir, _args(train.REFERENCE_VARIANT), _fake_model()
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
