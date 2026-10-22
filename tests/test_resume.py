from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from train import (
    clear_stale_artifacts,
    find_resume_point,
    latest_checkpoint,
    resolve_resume,
    vecnormalize_path_for,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ResumePathTests(unittest.TestCase):
    def test_latest_checkpoint_ignores_vecnormalize_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_dir = Path(directory)
            (checkpoint_dir / "ppo_walker2d_250000_steps.zip").touch()
            (checkpoint_dir / "ppo_walker2d_500000_steps.zip").touch()
            (checkpoint_dir / "ppo_walker2d_vecnormalize_500000_steps.pkl").touch()

            self.assertEqual(
                latest_checkpoint(checkpoint_dir).name,
                "ppo_walker2d_500000_steps.zip",
            )

    def test_latest_checkpoint_is_none_for_missing_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self.assertIsNone(latest_checkpoint(Path(directory) / "checkpoints"))

    def test_find_resume_point_returns_none_for_empty_run_dir(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self.assertIsNone(
                find_resume_point(Path(directory), normalized=False)
            )

    def test_find_resume_point_prefers_final_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            (run_dir / "final_model.zip").touch()
            checkpoint_dir = run_dir / "checkpoints"
            checkpoint_dir.mkdir()
            (checkpoint_dir / "ppo_walker2d_500000_steps.zip").touch()

            model_path, stats_path = find_resume_point(run_dir, normalized=False)
            self.assertEqual(model_path.name, "final_model.zip")
            self.assertIsNone(stats_path)

    def test_find_resume_point_uses_matching_checkpoint_statistics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            checkpoint_dir = run_dir / "checkpoints"
            checkpoint_dir.mkdir()
            (checkpoint_dir / "ppo_walker2d_750000_steps.zip").touch()
            stats = checkpoint_dir / "ppo_walker2d_vecnormalize_750000_steps.pkl"
            stats.touch()

            model_path, stats_path = find_resume_point(run_dir, normalized=True)
            self.assertEqual(model_path.name, "ppo_walker2d_750000_steps.zip")
            self.assertEqual(stats_path, stats)

    def test_normalized_resume_requires_statistics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            (run_dir / "final_model.zip").touch()
            with self.assertRaises(FileNotFoundError):
                find_resume_point(run_dir, normalized=True)

    def test_vecnormalize_path_falls_back_to_run_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            model_path = run_dir / "final_model.zip"
            self.assertEqual(
                vecnormalize_path_for(model_path),
                run_dir / "vecnormalize.pkl",
            )

    def test_clear_stale_artifacts_keeps_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            checkpoint_dir = run_dir / "checkpoints"
            checkpoint_dir.mkdir()
            (run_dir / "final_model.zip").touch()
            (run_dir / "vecnormalize.pkl").touch()
            (checkpoint_dir / "ppo_walker2d_250000_steps.zip").touch()
            (checkpoint_dir / "ppo_walker2d_vecnormalize_250000_steps.pkl").touch()
            evaluation_dir = run_dir / "evaluation"
            evaluation_dir.mkdir()
            (evaluation_dir / "summary.json").touch()

            removed = clear_stale_artifacts(run_dir, checkpoint_dir)

            self.assertEqual(len(removed), 4)
            self.assertFalse((run_dir / "final_model.zip").exists())
            self.assertIsNone(latest_checkpoint(checkpoint_dir))
            self.assertTrue((evaluation_dir / "summary.json").exists())

    def test_explicit_zip_and_missing_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            model_path = run_dir / "final_model.zip"
            model_path.touch()
            self.assertEqual(
                resolve_resume(str(model_path), run_dir, normalized=False)[0],
                model_path,
            )
            with self.assertRaises(FileNotFoundError):
                resolve_resume(str(run_dir / "missing.zip"), run_dir, normalized=False)
            with self.assertRaises(ValueError):
                (run_dir / "notes.txt").touch()
                resolve_resume(str(run_dir / "notes.txt"), run_dir, normalized=False)


class ResumeIntegrationTests(unittest.TestCase):
    def test_cli_resume_continues_from_final_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "baseline" / "seed_0"
            first = self._train(run_dir, 2048)
            self.assertFalse(first["resumed_from"])
            first_steps = first["actual_timesteps"]

            second = self._train(run_dir, 2560, extra=["--resume"])
            self.assertTrue(second["resumed_from"].endswith("final_model.zip"))
            self.assertEqual(len(second["resume_history"]), 1)
            self.assertEqual(
                second["resume_history"][0]["start_timesteps"], first_steps
            )
            self.assertGreater(second["actual_timesteps"], first_steps)
            self.assertTrue((run_dir / "final_model.zip").exists())

            # Without --resume the same directory is trained from scratch again,
            # and the discarded run's history is not carried over.
            third = self._train(run_dir, 2048)
            self.assertIsNone(third["resumed_from"])
            self.assertEqual(third["resume_history"], [])
            self.assertEqual(third["actual_timesteps"], 2048)

    def _train(self, run_dir: Path, timesteps: int, extra: list[str] | None = None):
        command = [
            sys.executable,
            "train.py",
            "--variant", "baseline",
            "--seed", "0",
            "--timesteps", str(timesteps),
            "--run-dir", str(run_dir),
            "--checkpoint-every", "1000000",
            *(extra or []),
        ]
        subprocess.run(command, cwd=PROJECT_ROOT, check=True, capture_output=True)
        return json.loads((run_dir / "config.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
