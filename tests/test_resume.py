from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from train import (
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
            self.assertIsNone(find_resume_point(Path(directory)))

    def test_find_resume_point_prefers_final_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            (run_dir / "final_model.zip").touch()
            (run_dir / "vecnormalize.pkl").touch()
            checkpoint_dir = run_dir / "checkpoints"
            checkpoint_dir.mkdir()
            (checkpoint_dir / "ppo_walker2d_500000_steps.zip").touch()

            model_path, stats_path = find_resume_point(run_dir)
            self.assertEqual(model_path.name, "final_model.zip")
            self.assertEqual(stats_path, run_dir / "vecnormalize.pkl")

    def test_find_resume_point_uses_matching_checkpoint_statistics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            checkpoint_dir = run_dir / "checkpoints"
            checkpoint_dir.mkdir()
            (checkpoint_dir / "ppo_walker2d_750000_steps.zip").touch()
            stats = checkpoint_dir / "ppo_walker2d_vecnormalize_750000_steps.pkl"
            stats.touch()

            model_path, stats_path = find_resume_point(run_dir)
            self.assertEqual(model_path.name, "ppo_walker2d_750000_steps.zip")
            self.assertEqual(stats_path, stats)

    def test_find_resume_point_requires_statistics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            (run_dir / "final_model.zip").touch()
            with self.assertRaises(FileNotFoundError):
                find_resume_point(run_dir)

    def test_checkpoint_missing_stats_is_not_replaced_by_directory_fallback(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            checkpoint_dir = run_dir / "checkpoints"
            checkpoint_dir.mkdir()
            model_path = checkpoint_dir / "ppo_walker2d_500000_steps.zip"
            model_path.touch()
            (checkpoint_dir / "vecnormalize.pkl").touch()

            expected = (
                checkpoint_dir / "ppo_walker2d_vecnormalize_500000_steps.pkl"
            )
            self.assertEqual(vecnormalize_path_for(model_path), expected)
            with self.assertRaises(FileNotFoundError):
                find_resume_point(run_dir)

    def test_vecnormalize_path_falls_back_to_run_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            model_path = run_dir / "final_model.zip"
            self.assertEqual(
                vecnormalize_path_for(model_path),
                run_dir / "vecnormalize.pkl",
            )

    def test_explicit_zip_and_missing_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            model_path = run_dir / "final_model.zip"
            stats_path = run_dir / "vecnormalize.pkl"
            model_path.touch()
            stats_path.touch()
            self.assertEqual(
                resolve_resume(str(model_path), run_dir)[0],
                model_path,
            )
            self.assertEqual(
                resolve_resume(str(model_path), run_dir)[1],
                stats_path,
            )
            with self.assertRaises(FileNotFoundError):
                resolve_resume(str(run_dir / "missing.zip"), run_dir)
            with self.assertRaises(ValueError):
                (run_dir / "notes.txt").touch()
                resolve_resume(str(run_dir / "notes.txt"), run_dir)

    def test_explicit_zip_without_statistics_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            model_path = run_dir / "final_model.zip"
            model_path.touch()
            with self.assertRaises(FileNotFoundError):
                resolve_resume(str(model_path), run_dir)


class ResumeIntegrationTests(unittest.TestCase):
    def test_cli_resume_continues_and_fresh_overwrite_is_refused(self) -> None:
        with tempfile.TemporaryDirectory(prefix="walker resume ") as directory:
            run_dir = Path(directory) / "ppo-reference" / "seed_0"
            first = self._train(run_dir, 512)
            self.assertFalse(first["resumed_from"])
            first_steps = first["actual_timesteps"]
            self.assertGreaterEqual(first_steps, 512)

            second = self._train(run_dir, 1024, extra=["--resume"])
            self.assertTrue(second["resumed_from"].endswith("final_model.zip"))
            self.assertEqual(len(second["resume_history"]), 1)
            self.assertEqual(
                second["resume_history"][0]["start_timesteps"], first_steps
            )
            self.assertGreater(second["actual_timesteps"], first_steps)
            self.assertTrue((run_dir / "final_model.zip").exists())

            # Without --resume the same directory must not be restarted from
            # scratch over existing artifacts.
            result = self._run_train(run_dir, 1536)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("refusing to overwrite", result.stderr)
            unchanged = json.loads(
                (run_dir / "config.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                unchanged["actual_timesteps"], second["actual_timesteps"]
            )

    def test_resume_with_nothing_to_load_is_refused(self) -> None:
        with tempfile.TemporaryDirectory(prefix="walker empty ") as directory:
            run_dir = Path(directory) / "ppo-reference" / "seed_0"
            run_dir.mkdir(parents=True)
            (run_dir / "config.json").write_text(
                json.dumps({"variant": "ppo-reference"}), encoding="utf-8"
            )
            result = self._run_train(run_dir, 512, extra=["--resume"])
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("found no final_model.zip or checkpoint", result.stderr)

    def _run_train(
        self, run_dir: Path, timesteps: int, extra: list[str] | None = None
    ) -> subprocess.CompletedProcess:
        command = [
            sys.executable,
            "train.py",
            "--variant", "ppo-reference",
            "--seed", "0",
            "--timesteps", str(timesteps),
            "--run-dir", str(run_dir),
            "--checkpoint-every", "1000000",
            "--torch-threads", "1",
            *(extra or []),
        ]
        return subprocess.run(
            command, cwd=PROJECT_ROOT, capture_output=True, text=True
        )

    def _train(self, run_dir: Path, timesteps: int, extra: list[str] | None = None):
        result = self._run_train(run_dir, timesteps, extra)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return json.loads((run_dir / "config.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
