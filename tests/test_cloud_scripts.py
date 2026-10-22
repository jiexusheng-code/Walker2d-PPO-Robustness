from __future__ import annotations

import hashlib
import json
import os
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
    file_sha256,
    summarize_condition,
    training_metadata,
)

CONDITIONS = {
    "nominal": {"mean_return": 1.0},
    "heldout_low": {"mean_return": 1.0},
    "heldout_high": {"mean_return": 1.0},
}


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def make_run(
    root: Path,
    *,
    variant: str = "baseline",
    seed: int = 0,
    steps: int | None = 1_000_000,
    status: object = None,
    final_model: bool = True,
    evaluation: bool = True,
    evaluation_model: str | None = None,
    conditions: dict | None = None,
) -> Path:
    run_dir = Path(root) / "run"
    config: dict[str, object] = {
        "variant": variant,
        "seed": seed,
        "requested_timesteps": steps,
    }
    if steps is not None:
        config["actual_timesteps"] = steps
    if status is not None:
        config["status"] = status
    write_json(run_dir / "config.json", config)
    if final_model:
        (run_dir / "final_model.zip").touch()
    if evaluation:
        write_json(
            run_dir / "evaluation" / "summary.json",
            {
                "variant": variant,
                "model": evaluation_model
                or str((run_dir / "final_model.zip").resolve()),
                "conditions": conditions or CONDITIONS,
            },
        )
    return run_dir


class PlanRunTests(unittest.TestCase):
    def test_skip_completed_run_with_matching_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = make_run(Path(directory))
            plan = plan_run(
                run_dir, variant="baseline", seed=0, budget=1_000_000, resume=False
            )
            self.assertEqual(plan.action, "skip")
            self.assertEqual(plan.completed_steps, 1_000_000)

    def test_legacy_config_without_status_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = make_run(Path(directory), status=None)
            plan = plan_run(
                run_dir, variant="baseline", seed=0, budget=1_000_000, resume=False
            )
            self.assertEqual(plan.action, "skip")

    def test_higher_budget_without_resume_is_a_friendly_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = make_run(Path(directory), steps=300_000)
            plan = plan_run(
                run_dir, variant="baseline", seed=0, budget=1_000_000, resume=False
            )
            self.assertEqual(plan.action, "error")
            self.assertIn("RESUME=1", plan.reason)
            self.assertIn("300000", plan.reason)

    def test_higher_budget_with_resume_continues(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = make_run(Path(directory), steps=300_000)
            plan = plan_run(
                run_dir, variant="baseline", seed=0, budget=1_000_000, resume=True
            )
            self.assertEqual(plan.action, "resume")
            self.assertEqual(plan.completed_steps, 300_000)

    def test_expanded_budget_with_resume_is_not_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = make_run(Path(directory), steps=1_000_000)
            plan = plan_run(
                run_dir, variant="baseline", seed=0, budget=1_500_000, resume=True
            )
            self.assertEqual(plan.action, "resume")

    def test_completed_but_missing_evaluation_is_reevaluated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = make_run(Path(directory), evaluation=False)
            plan = plan_run(
                run_dir, variant="baseline", seed=0, budget=1_000_000, resume=False
            )
            self.assertEqual(plan.action, "evaluate")

    def test_evaluation_for_another_model_is_reevaluated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = make_run(
                Path(directory),
                evaluation_model="/somewhere/else/final_model.zip",
            )
            plan = plan_run(
                run_dir, variant="baseline", seed=0, budget=1_000_000, resume=False
            )
            self.assertEqual(plan.action, "evaluate")

    def test_evaluation_missing_a_condition_is_reevaluated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = make_run(
                Path(directory),
                conditions={"nominal": {"mean_return": 1.0}},
            )
            plan = plan_run(
                run_dir, variant="baseline", seed=0, budget=1_000_000, resume=False
            )
            self.assertEqual(plan.action, "evaluate")

    def test_variant_mismatch_is_an_error_even_with_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = make_run(Path(directory), variant="baseline")
            plan = plan_run(
                run_dir, variant="truly-ppo", seed=0, budget=1_000_000, resume=True
            )
            self.assertEqual(plan.action, "error")
            self.assertIn("does not match", plan.reason)

    def test_seed_mismatch_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = make_run(Path(directory), seed=1)
            plan = plan_run(
                run_dir, variant="baseline", seed=0, budget=1_000_000, resume=False
            )
            self.assertEqual(plan.action, "error")

    def test_running_status_is_not_treated_as_complete(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = make_run(Path(directory), status="running")
            blocked = plan_run(
                run_dir, variant="baseline", seed=0, budget=1_000_000, resume=False
            )
            resumed = plan_run(
                run_dir, variant="baseline", seed=0, budget=1_000_000, resume=True
            )
            self.assertEqual(blocked.action, "error")
            self.assertEqual(resumed.action, "resume")

    def test_missing_actual_step_count_is_not_treated_as_complete(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = make_run(Path(directory), steps=None)
            blocked = plan_run(
                run_dir, variant="baseline", seed=0, budget=1_000_000, resume=False
            )
            resumed = plan_run(
                run_dir, variant="baseline", seed=0, budget=1_000_000, resume=True
            )
            self.assertEqual(blocked.action, "error")
            self.assertEqual(resumed.action, "resume")

    def test_changed_model_at_same_path_is_not_current(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = make_run(Path(directory), evaluation=False)
            model = run_dir / "final_model.zip"
            model.write_bytes(b"model-v1")
            write_json(
                run_dir / "evaluation" / "summary.json",
                {
                    "variant": "baseline",
                    "model": str(model.resolve()),
                    "model_sha256": hashlib.sha256(b"model-v1").hexdigest(),
                    "conditions": CONDITIONS,
                },
            )
            self.assertEqual(
                plan_run(
                    run_dir, variant="baseline", seed=0,
                    budget=1_000_000, resume=False,
                ).action,
                "skip",
            )

            model.write_bytes(b"model-v2")
            self.assertEqual(
                plan_run(
                    run_dir, variant="baseline", seed=0,
                    budget=1_000_000, resume=False,
                ).action,
                "evaluate",
            )

    def test_legacy_summary_mtime_must_not_predate_the_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = make_run(Path(directory), evaluation=False)
            model = run_dir / "final_model.zip"
            summary = run_dir / "evaluation" / "summary.json"
            write_json(
                summary,
                {
                    "variant": "baseline",
                    "model": str(model.resolve()),
                    "conditions": CONDITIONS,
                },
            )
            base = 1_700_000_000_000_000_000
            os.utime(model, ns=(base, base))
            os.utime(summary, ns=(base + 1_000_000_000,) * 2)
            self.assertEqual(
                plan_run(
                    run_dir, variant="baseline", seed=0,
                    budget=1_000_000, resume=False,
                ).action,
                "skip",
            )

            os.utime(model, ns=(base + 2_000_000_000,) * 2)
            self.assertEqual(
                plan_run(
                    run_dir, variant="baseline", seed=0,
                    budget=1_000_000, resume=False,
                ).action,
                "evaluate",
            )

    def test_invalid_config_with_artifacts_refuses_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            run_dir.mkdir()
            (run_dir / "final_model.zip").touch()
            (run_dir / "config.json").write_text("{not json", encoding="utf-8")
            plan = plan_run(
                run_dir, variant="ppo-reference", seed=0,
                budget=1_000_000, resume=True,
            )
            self.assertEqual(plan.action, "error")
            self.assertIn("invalid", plan.reason)

    def test_reference_without_config_refuses_resume_over_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            run_dir.mkdir()
            (run_dir / "final_model.zip").touch()
            (run_dir / "metrics.jsonl").touch()
            plan = plan_run(
                run_dir, variant="ppo-reference", seed=0,
                budget=1_000_000, resume=True,
            )
            self.assertEqual(plan.action, "error")
            self.assertIn("cannot resume ppo-reference", plan.reason)

    def test_legacy_variant_without_config_can_still_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            run_dir.mkdir()
            (run_dir / "final_model.zip").touch()
            plan = plan_run(
                run_dir, variant="baseline", seed=0,
                budget=1_000_000, resume=True,
            )
            self.assertEqual(plan.action, "resume")

    def test_empty_directory_starts_fresh(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            plan = plan_run(
                Path(directory) / "new_run",
                variant="baseline",
                seed=0,
                budget=1_000_000,
                resume=False,
            )
            self.assertEqual(plan.action, "train")
            self.assertIsNone(plan.completed_steps)

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
            "--variant", "baseline",
            "--seed", "0",
            "--budget", str(budget),
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
            self.assertTrue(skipped.stdout.startswith("skip|"))

            blocked = self._run_cli(run_dir, 1_500_000)
            self.assertEqual(blocked.returncode, 3, blocked.stderr)
            self.assertTrue(blocked.stdout.startswith("error|"))


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

    def test_pilot_defaults_match_evaluation_protocol(self) -> None:
        text = (PROJECT_ROOT / "scripts" / "pilot_reference.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("set -euo pipefail", text)
        self.assertIn("uv sync --locked", text)
        self.assertIn("--variant ppo-reference", text)
        self.assertIn('EVAL_SEED="${EVAL_SEED:-50000}"', text)
        self.assertIn('FINAL_EVAL_SEED="${FINAL_EVAL_SEED:-40000}"', text)
        self.assertIn('TIMESTEPS="${TIMESTEPS:-1000000}"', text)
        self.assertIn('SEED="${SEED:-0}"', text)
        self.assertIn(
            'if [ "$RESUME" = "1" ] && [ -z "${RUN_DIR:-}" ]', text
        )
        self.assertIn("runs/main", text)

    def test_run_matrix_uses_the_decision_helper(self) -> None:
        text = (PROJECT_ROOT / "scripts" / "run_matrix.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("scripts/run_matrix_lib.py plan", text)
        self.assertIn('RESUME="${RESUME:-0}"', text)


class EvaluateHelpersTests(unittest.TestCase):
    def test_file_sha256(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "blob.bin"
            path.write_bytes(b"abc")
            self.assertEqual(
                file_sha256(path), hashlib.sha256(b"abc").hexdigest()
            )

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
