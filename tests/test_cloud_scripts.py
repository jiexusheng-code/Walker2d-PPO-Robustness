from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_matrix_lib import (  # noqa: E402
    completed_steps,
    plan_run,
    status_is_incomplete,
)

from evaluate import (  # noqa: E402
    default_vecnormalize_path,
    file_sha256,
    summarize_condition,
    training_metadata,
)

CONDITIONS = {
    "nominal": {"mean_return": 1.0},
    "heldout_low": {"mean_return": 1.0},
    "heldout_high": {"mean_return": 1.0},
}
ACTOR_OBJECTIVES = {
    "ppo-reference": "ppo_clip",
    "truly-ppo": "truly_ppo_kl_triggered_rollback",
}


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def make_run(
    root: Path,
    *,
    variant: str = "ppo-reference",
    seed: int = 0,
    steps: int | None = 1_000_000,
    status: object = None,
    final_model: bool = True,
    stats: bool = True,
    evaluation: bool = True,
    evaluation_model: str | None = None,
    conditions: dict | None = None,
    episodes: int = 100,
    base_seed: int = 40_000,
    deterministic: bool = True,
    model_sha_override: str | None = None,
    stats_sha_override: str | None = None,
    profile: str = "ppo-reference",
    actor: str | None = "default",
) -> Path:
    run_dir = Path(root) / "run"
    config: dict[str, object] = {
        "variant": variant,
        "seed": seed,
        "profile": profile,
        "environment": "Walker2d-v5",
        "algorithm": "TrulyPPO" if variant == "truly-ppo" else "PPO",
        "requested_timesteps": steps,
        "ppo": {
            "n_steps": 512,
            "batch_size": 32,
            "n_epochs": 20,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.1,
            "ent_coef": 0.000585045,
            "vf_coef": 0.871923,
            "max_grad_norm": 1.0,
            "learning_rate_initial": 5.05041e-5,
            "learning_rate_schedule": "constant",
            "policy_kwargs": {"net_arch": {"pi": [64, 64], "vf": [64, 64]}},
        },
        "normalization": {"observations": True, "rewards": True},
    }
    if actor == "default":
        config["actor"] = {
            "objective": ACTOR_OBJECTIVES[variant],
            "clip_range_used": variant == "ppo-reference",
            "kl_rollback": (
                {"alpha": 5.0, "delta": 0.03}
                if variant == "truly-ppo" else None
            ),
        }
    elif actor is not None:
        config["actor"] = actor
    if steps is not None:
        config["actual_timesteps"] = steps
    if status is not None:
        config["status"] = status
    write_json(run_dir / "config.json", config)

    model = run_dir / "final_model.zip"
    if final_model:
        model.write_bytes(b"model")
    stats_path = run_dir / "vecnormalize.pkl"
    if stats:
        stats_path.write_bytes(b"stats")
    if evaluation:
        write_json(
            run_dir / "evaluation" / "summary.json",
            {
                "variant": variant,
                "model": evaluation_model or str(model.resolve()),
                "model_sha256": (
                    model_sha_override
                    if model_sha_override is not None
                    else (sha256_bytes(b"model") if final_model else None)
                ),
                "vecnormalize_sha256": (
                    stats_sha_override
                    if stats_sha_override is not None
                    else (sha256_bytes(b"stats") if stats else None)
                ),
                "episodes_per_condition": episodes,
                "base_seed": base_seed,
                "deterministic": deterministic,
                "conditions": conditions or CONDITIONS,
            },
        )
    return run_dir


def plan(run_dir: Path, *, variant: str = "ppo-reference", seed: int = 0,
         budget: int = 1_000_000, episodes: int = 100,
         eval_seed: int = 40_000, resume: bool = False):
    return plan_run(
        run_dir,
        variant=variant,
        seed=seed,
        budget=budget,
        episodes=episodes,
        eval_seed=eval_seed,
        resume=resume,
    )


class PlanRunTests(unittest.TestCase):
    def test_skip_completed_run_with_matching_evaluation(self) -> None:
        for variant in ("ppo-reference", "truly-ppo"):
            with tempfile.TemporaryDirectory() as directory:
                run_dir = make_run(Path(directory), variant=variant)
                result = plan(run_dir, variant=variant)
                self.assertEqual(result.action, "skip", result.reason)
                self.assertEqual(result.completed_steps, 1_000_000)

    def test_unsupported_legacy_variant_is_an_error(self) -> None:
        for variant in ("baseline", "ppo-normalized"):
            with tempfile.TemporaryDirectory() as directory:
                run_dir = make_run(Path(directory))
                result = plan(run_dir, variant=variant)
                self.assertEqual(result.action, "error")
                self.assertIn("unsupported training variant", result.reason)

    def test_old_truly_profile_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = make_run(
                Path(directory), variant="truly-ppo", profile="truly-ppo"
            )
            result = plan(run_dir, variant="truly-ppo")
            self.assertEqual(result.action, "error")
            self.assertIn("does not match", result.reason)

    def test_truly_without_actor_record_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = make_run(Path(directory), variant="truly-ppo", actor=None)
            result = plan(run_dir, variant="truly-ppo")
            self.assertEqual(result.action, "error")
            self.assertIn("actor record is missing", result.reason)

    def test_legacy_reference_without_actor_record_can_still_skip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = make_run(
                Path(directory), variant="ppo-reference", actor=None
            )
            self.assertEqual(plan(run_dir).action, "skip")

    def test_legacy_config_without_status_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = make_run(Path(directory), status=None)
            self.assertEqual(plan(run_dir).action, "skip")

    def test_higher_budget_without_resume_is_a_friendly_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = make_run(Path(directory), steps=300_000)
            result = plan(run_dir)
            self.assertEqual(result.action, "error")
            self.assertIn("RESUME=1", result.reason)
            self.assertIn("300000", result.reason)

    def test_higher_budget_with_resume_continues(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = make_run(Path(directory), steps=300_000)
            result = plan(run_dir, resume=True)
            self.assertEqual(result.action, "resume")
            self.assertEqual(result.completed_steps, 300_000)

    def test_expanded_budget_with_resume_is_not_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = make_run(Path(directory), steps=1_000_000)
            result = plan(run_dir, budget=1_500_000, resume=True)
            self.assertEqual(result.action, "resume")

    def test_resume_without_any_model_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = make_run(
                Path(directory), final_model=False, evaluation=False
            )
            result = plan(run_dir, resume=True)
            self.assertEqual(result.action, "error")
            self.assertIn("nothing to resume", result.reason)

    def test_resume_mode_starts_untouched_later_arm(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = plan(Path(directory) / "untouched", resume=True)
            self.assertEqual(result.action, "train", result.reason)
            self.assertIsNone(result.completed_steps)

    def test_resume_matrix_continues_interrupted_then_starts_untouched(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = make_run(
                Path(directory),
                steps=300_000,
                final_model=False,
                evaluation=False,
            )
            checkpoint_dir = first / "checkpoints"
            checkpoint_dir.mkdir()
            (checkpoint_dir / "ppo_walker2d_300000_steps.zip").touch()
            (
                checkpoint_dir / "ppo_walker2d_vecnormalize_300000_steps.pkl"
            ).touch()

            first_plan = plan(first, resume=True)
            self.assertEqual(first_plan.action, "resume", first_plan.reason)
            second_plan = plan(Path(directory) / "untouched", resume=True)
            self.assertEqual(second_plan.action, "train", second_plan.reason)

    def test_completed_but_missing_evaluation_is_reevaluated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = make_run(Path(directory), evaluation=False)
            self.assertEqual(plan(run_dir).action, "evaluate")

    def test_single_episode_summary_is_not_mistaken_for_100(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = make_run(Path(directory), episodes=1)
            result = plan(run_dir)
            self.assertEqual(result.action, "evaluate")
            self.assertIn("episodes per condition", result.reason)

    def test_wrong_base_seed_is_reevaluated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = make_run(Path(directory), base_seed=12_345)
            result = plan(run_dir)
            self.assertEqual(result.action, "evaluate")
            self.assertIn("base seed", result.reason)

    def test_changed_model_is_reevaluated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = make_run(Path(directory))
            (run_dir / "final_model.zip").write_bytes(b"model-v2")
            result = plan(run_dir)
            self.assertEqual(result.action, "evaluate")
            self.assertIn("model SHA-256", result.reason)

    def test_changed_statistics_are_reevaluated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = make_run(Path(directory))
            (run_dir / "vecnormalize.pkl").write_bytes(b"stats-v2")
            result = plan(run_dir)
            self.assertEqual(result.action, "evaluate")
            self.assertIn("VecNormalize SHA-256", result.reason)

    def test_missing_hashes_are_not_treated_as_current(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = make_run(Path(directory), model_sha_override="")
            result = plan(run_dir)
            self.assertEqual(result.action, "evaluate")
            self.assertIn("no model SHA-256", result.reason)
        with tempfile.TemporaryDirectory() as directory:
            run_dir = make_run(Path(directory), stats_sha_override="")
            result = plan(run_dir)
            self.assertEqual(result.action, "evaluate")
            self.assertIn("no VecNormalize SHA-256", result.reason)

    def test_evaluation_for_another_model_is_reevaluated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = make_run(
                Path(directory),
                evaluation_model="/somewhere/else/final_model.zip",
            )
            self.assertEqual(plan(run_dir).action, "evaluate")

    def test_evaluation_missing_a_condition_is_reevaluated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = make_run(
                Path(directory),
                conditions={"nominal": {"mean_return": 1.0}},
            )
            self.assertEqual(plan(run_dir).action, "evaluate")

    def test_non_deterministic_evaluation_is_reevaluated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = make_run(Path(directory), deterministic=False)
            self.assertEqual(plan(run_dir).action, "evaluate")

    def test_completed_without_statistics_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = make_run(Path(directory), stats=False, evaluation=False)
            result = plan(run_dir)
            self.assertEqual(result.action, "error")
            self.assertIn("no vecnormalize.pkl", result.reason)

    def test_variant_mismatch_is_an_error_even_with_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = make_run(Path(directory), variant="ppo-reference")
            result = plan(run_dir, variant="truly-ppo", resume=True)
            self.assertEqual(result.action, "error")
            self.assertIn("does not match", result.reason)

    def test_seed_mismatch_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = make_run(Path(directory), seed=1)
            result = plan(run_dir, seed=0)
            self.assertEqual(result.action, "error")

    def test_running_status_is_not_treated_as_complete(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = make_run(Path(directory), status="running")
            blocked = plan(run_dir)
            resumed = plan(run_dir, resume=True)
            self.assertEqual(blocked.action, "error")
            self.assertEqual(resumed.action, "resume")

    def test_missing_actual_step_count_is_not_treated_as_complete(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = make_run(Path(directory), steps=None)
            blocked = plan(run_dir)
            resumed = plan(run_dir, resume=True)
            self.assertEqual(blocked.action, "error")
            self.assertEqual(resumed.action, "resume")

    def test_invalid_config_with_artifacts_refuses_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            run_dir.mkdir()
            (run_dir / "final_model.zip").touch()
            (run_dir / "config.json").write_text("{not json", encoding="utf-8")
            result = plan(run_dir, resume=True)
            self.assertEqual(result.action, "error")
            self.assertIn("invalid", result.reason)

    def test_missing_config_with_artifacts_refuses_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            run_dir.mkdir()
            (run_dir / "final_model.zip").touch()
            (run_dir / "metrics.jsonl").touch()
            result = plan(run_dir, resume=True)
            self.assertEqual(result.action, "error")
            self.assertIn("cannot resume", result.reason)

    def test_empty_directory_starts_fresh(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = plan(Path(directory) / "new_run")
            self.assertEqual(result.action, "train")
            self.assertIsNone(result.completed_steps)

    def test_completed_steps_reads_only_actual_timesteps(self) -> None:
        self.assertEqual(completed_steps({"actual_timesteps": 123}), 123)
        self.assertIsNone(completed_steps({"status": {"timesteps": 456}}))
        self.assertIsNone(completed_steps({"completed_timesteps": 789}))
        self.assertIsNone(completed_steps({"actual_timesteps": True}))

    def test_status_is_incomplete_for_existing_status_strings(self) -> None:
        self.assertTrue(status_is_incomplete({"status": "interrupted"}))
        self.assertTrue(status_is_incomplete({"status": "failed"}))
        self.assertTrue(status_is_incomplete({"status": "running"}))
        self.assertFalse(status_is_incomplete({"status": "completed"}))
        self.assertFalse(status_is_incomplete({"status": {"phase": "running"}}))


class PlanCliTests(unittest.TestCase):
    def _run_cli(self, run_dir: Path, budget: int, resume: bool = False):
        command = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_matrix_lib.py"),
            "plan",
            "--run-dir", str(run_dir),
            "--variant", "ppo-reference",
            "--seed", "0",
            "--budget", str(budget),
            "--episodes", "100",
            "--eval-seed", "40000",
        ]
        if resume:
            command.append("--resume")
        return subprocess.run(
            command, cwd=PROJECT_ROOT, capture_output=True, text=True
        )

    def test_cli_reports_skip_and_budget_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = make_run(Path(directory))
            skipped = self._run_cli(run_dir, 1_000_000)
            self.assertEqual(skipped.returncode, 0, skipped.stderr)
            self.assertTrue(skipped.stdout.startswith("skip|"), skipped.stdout)

            blocked = self._run_cli(run_dir, 1_500_000)
            self.assertEqual(blocked.returncode, 3, blocked.stderr)
            self.assertTrue(blocked.stdout.startswith("error|"), blocked.stdout)


class ShellSyntaxTests(unittest.TestCase):
    def test_shell_scripts_parse(self) -> None:
        bash = shutil.which("bash")
        if bash is None:
            self.skipTest("bash is not available")
        for name in (
            "run_matrix.sh",
            "pilot_reference.sh",
            "cloud_smoke.sh",
            "benchmark.sh",
        ):
            result = subprocess.run(
                [bash, "-n", str(PROJECT_ROOT / "scripts" / name)],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, f"{name}: {result.stderr}")

    def test_run_matrix_defaults_to_the_two_arms(self) -> None:
        text = (PROJECT_ROOT / "scripts" / "run_matrix.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("set -euo pipefail", text)
        self.assertIn("uv sync --locked", text)
        self.assertIn(
            'VARIANTS="${VARIANTS:-ppo-reference truly-ppo}"', text
        )
        self.assertIn('TIMESTEPS="${TIMESTEPS:-1000000}"', text)
        self.assertIn('SEEDS="${SEEDS:-0}"', text)
        self.assertIn('RESUME="${RESUME:-0}"', text)
        self.assertIn("scripts/run_matrix_lib.py plan", text)
        self.assertIn('--episodes "$EPISODES"', text)
        self.assertIn('--eval-seed "$FINAL_EVAL_SEED"', text)
        self.assertIn('RUNS_ROOT="${RUNS_ROOT:-runs/two_arm}"', text)
        self.assertIn(
            'run_dir="${RUNS_ROOT%/}/${variant}/seed_${seed}"', text
        )
        self.assertIn("shared_profile=ppo-reference", text)
        self.assertNotIn('run_dir="runs/main/', text)

    def test_pilot_is_single_group_only(self) -> None:
        text = (PROJECT_ROOT / "scripts" / "pilot_reference.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("set -euo pipefail", text)
        self.assertIn("uv sync --locked", text)
        self.assertIn("--variant ppo-reference", text)
        self.assertNotIn("truly-ppo", text)
        self.assertIn("single-group reference pilot", text)
        self.assertIn('EVAL_SEED="${EVAL_SEED:-50000}"', text)
        self.assertIn('FINAL_EVAL_SEED="${FINAL_EVAL_SEED:-40000}"', text)
        self.assertIn('TIMESTEPS="${TIMESTEPS:-1000000}"', text)
        self.assertIn('SEED="${SEED:-0}"', text)
        self.assertIn(
            'if [ "$RESUME" = "1" ] && [ -z "${RUN_DIR:-}" ]', text
        )
        self.assertIn("runs/main", text)
        self.assertIn("runs/two_arm", text)

    def test_smoke_and_benchmark_default_to_the_two_arms(self) -> None:
        for name in ("cloud_smoke.sh", "benchmark.sh"):
            text = (PROJECT_ROOT / "scripts" / name).read_text(
                encoding="utf-8"
            )
            self.assertIn(
                'VARIANTS="${VARIANTS:-ppo-reference truly-ppo}"', text
            )
            self.assertIn("uv sync --locked", text)
            self.assertIn("runs/", text)


class EvaluateHelpersTests(unittest.TestCase):
    def test_file_sha256(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "blob.bin"
            path.write_bytes(b"abc")
            self.assertEqual(
                file_sha256(path), hashlib.sha256(b"abc").hexdigest()
            )

    def test_default_vecnormalize_path_matches_the_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(
                default_vecnormalize_path(root / "final_model.zip"),
                root / "vecnormalize.pkl",
            )
            best = root / "best"
            self.assertEqual(
                default_vecnormalize_path(best / "best_model.zip"),
                best / "vecnormalize.pkl",
            )
            checkpoints = root / "checkpoints"
            self.assertEqual(
                default_vecnormalize_path(
                    checkpoints / "ppo_walker2d_500000_steps.zip"
                ),
                checkpoints / "ppo_walker2d_vecnormalize_500000_steps.pkl",
            )

    def test_evaluate_cli_rejects_missing_statistics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "final_model.zip"
            model.touch()
            result = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "evaluate.py"),
                    "--model", str(model),
                    "--variant", "truly-ppo",
                    "--output-dir", str(root / "evaluation"),
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("VecNormalize statistics not found", result.stderr)

    def test_evaluate_cli_rejects_statistics_for_legacy_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "final_model.zip"
            model.touch()
            result = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "evaluate.py"),
                    "--model", str(model),
                    "--variant", "baseline",
                    "--vecnormalize", str(root / "vecnormalize.pkl"),
                    "--output-dir", str(root / "evaluation"),
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("not valid for the legacy", result.stderr)

    def test_training_metadata_reads_matching_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            model = run_dir / "final_model.zip"
            model.touch()
            write_json(
                run_dir / "config.json",
                {
                    "variant": "ppo-reference",
                    "seed": 7,
                    "requested_timesteps": 1_000_000,
                    "actual_timesteps": 1_000_000,
                },
            )
            metadata = training_metadata(model, "ppo-reference")
            self.assertEqual(metadata["seed"], 7)
            self.assertEqual(metadata["actual_timesteps"], 1_000_000)
            self.assertTrue(str(metadata["config"]).endswith("config.json"))

    def test_training_metadata_ignores_variant_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            model = run_dir / "final_model.zip"
            model.touch()
            write_json(run_dir / "config.json", {"variant": "baseline", "seed": 7})
            metadata = training_metadata(model, "ppo-reference")
            self.assertIsNone(metadata["seed"])
            self.assertIsNone(metadata["config"])

    def test_training_metadata_reads_best_parent_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            model = run_dir / "best" / "best_model.zip"
            model.parent.mkdir()
            model.touch()
            write_json(run_dir / "config.json", {"variant": "ppo-reference", "seed": 3})
            metadata = training_metadata(model, "ppo-reference")
            self.assertEqual(metadata["seed"], 3)

    def test_summary_keeps_original_fields_and_adds_healthy_metrics(self) -> None:
        rows = [
            self._row(length=10, fell=True, truncated=True, healthy=False),
            self._row(length=1000, fell=False, truncated=True, healthy=True),
            self._row(length=20, fell=True, truncated=False, healthy=False),
            self._row(length=30, fell=False, truncated=False, healthy=False),
        ]
        summary = summarize_condition(rows, max_episode_steps=1000)
        for field in (
            "episodes",
            "mean_return",
            "std_return",
            "mean_episode_length",
            "mean_forward_velocity",
            "mean_displacement",
            "fall_rate",
            "time_limit_rate",
        ):
            self.assertIn(field, summary)
        self.assertAlmostEqual(summary["fall_rate"], 0.5)
        self.assertAlmostEqual(summary["time_limit_rate"], 0.5)
        self.assertAlmostEqual(summary["healthy_time_limit_rate"], 0.25)
        self.assertAlmostEqual(summary["length_full_rate"], 0.25)
        self.assertEqual(summary["max_episode_steps"], 1000)

    def test_length_full_rate_is_null_without_a_limit(self) -> None:
        rows = [self._row(length=10, fell=False, truncated=False, healthy=False)]
        summary = summarize_condition(rows, max_episode_steps=None)
        self.assertIsNone(summary["length_full_rate"])

    @staticmethod
    def _row(
        *, length: int, fell: bool, truncated: bool, healthy: bool
    ) -> dict[str, object]:
        return {
            "return": float(length),
            "length": length,
            "mean_forward_velocity": 1.0,
            "displacement": 2.0,
            "fell": fell,
            "time_limit_reached": truncated,
            "healthy_time_limit": healthy,
        }


if __name__ == "__main__":
    unittest.main()
