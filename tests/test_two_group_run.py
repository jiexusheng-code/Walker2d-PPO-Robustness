from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ACTOR_OBJECTIVES = {
    "ppo-reference": "ppo_clip",
    "truly-ppo": "truly_ppo_kl_triggered_rollback",
}


class TwoGroupShortRunTests(unittest.TestCase):
    """Exercise real (short) training, resume, and evaluation for both arms."""

    def test_short_train_resume_and_evaluate_both_arms(self) -> None:
        for variant in ("ppo-reference", "truly-ppo"):
            with self.subTest(variant=variant), tempfile.TemporaryDirectory(
                prefix="walker two group "
            ) as directory:
                run_dir = Path(directory) / "run with spaces"
                self._run([
                    "train.py",
                    "--variant", variant,
                    "--seed", "0",
                    "--timesteps", "512",
                    "--run-dir", str(run_dir),
                    "--checkpoint-every", "512",
                    "--eval-every", "512",
                    "--eval-episodes", "1",
                    "--eval-seed", "50000",
                    "--torch-threads", "1",
                ])
                first = self._config(run_dir)
                self.assertEqual(first["variant"], variant)
                self.assertEqual(first["profile"], "ppo-reference")
                self.assertEqual(
                    first["actor"]["objective"], ACTOR_OBJECTIVES[variant]
                )
                if variant == "truly-ppo":
                    self.assertEqual(
                        first["actor"]["kl_rollback"]["alpha"], 5.0
                    )
                    self.assertEqual(
                        first["actor"]["kl_rollback"]["delta"], 0.03
                    )
                else:
                    self.assertIsNone(first["actor"]["kl_rollback"])
                first_steps = first["actual_timesteps"]
                self.assertGreaterEqual(first_steps, 512)
                for artifact in (
                    "final_model.zip",
                    "vecnormalize.pkl",
                    "best/best_model.zip",
                    "best/vecnormalize.pkl",
                ):
                    self.assertTrue(
                        (run_dir / artifact).is_file(), artifact
                    )

                self._run([
                    "train.py",
                    "--variant", variant,
                    "--seed", "0",
                    "--timesteps", "1024",
                    "--run-dir", str(run_dir),
                    "--resume",
                    "--torch-threads", "1",
                ])
                second = self._config(run_dir)
                self.assertEqual(len(second["resume_history"]), 1)
                self.assertGreater(second["actual_timesteps"], first_steps)

                evaluation_dir = run_dir / "evaluation"
                self._run([
                    "evaluate.py",
                    "--model", str(run_dir / "final_model.zip"),
                    "--variant", variant,
                    "--episodes", "2",
                    "--seed", "40000",
                    "--output-dir", str(evaluation_dir),
                ])
                self._check_summary(
                    evaluation_dir, variant, run_dir / "final_model.zip",
                    run_dir / "vecnormalize.pkl",
                )

                best_dir = run_dir / "best" / "evaluation"
                self._run([
                    "evaluate.py",
                    "--model", str(run_dir / "best" / "best_model.zip"),
                    "--variant", variant,
                    "--vecnormalize",
                    str(run_dir / "best" / "vecnormalize.pkl"),
                    "--episodes", "2",
                    "--seed", "40000",
                    "--output-dir", str(best_dir),
                ])
                self._check_summary(
                    best_dir, variant,
                    run_dir / "best" / "best_model.zip",
                    run_dir / "best" / "vecnormalize.pkl",
                )

    def _run(self, arguments: list[str]) -> None:
        result = subprocess.run(
            [sys.executable, *arguments],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=240,
        )
        self.assertEqual(
            result.returncode, 0, result.stdout + result.stderr
        )

    @staticmethod
    def _config(run_dir: Path) -> dict:
        return json.loads(
            (run_dir / "config.json").read_text(encoding="utf-8")
        )

    def _check_summary(
        self,
        output_dir: Path,
        variant: str,
        model_path: Path,
        stats_path: Path,
    ) -> None:
        from evaluate import file_sha256

        summary = json.loads(
            (output_dir / "summary.json").read_text(encoding="utf-8")
        )
        self.assertEqual(summary["variant"], variant)
        self.assertEqual(summary["episodes_per_condition"], 2)
        self.assertEqual(summary["base_seed"], 40_000)
        self.assertTrue(summary["deterministic"])
        self.assertEqual(summary["model_sha256"], file_sha256(model_path))
        self.assertEqual(
            summary["vecnormalize_sha256"], file_sha256(stats_path)
        )
        self.assertTrue(summary["normalization"]["enabled"])
        self.assertTrue(summary["normalization"]["frozen"])
        self.assertFalse(summary["normalization"]["norm_reward"])
        self.assertEqual(
            set(summary["conditions"]),
            {"nominal", "heldout_low", "heldout_high"},
        )
        self.assertEqual(
            summary["training"]["seed"], 0
        )


if __name__ == "__main__":
    unittest.main()
