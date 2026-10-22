"""Exercise the cloud entry point with a fake uv; never launch Python training."""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FAKE_UV = r'''#!/usr/bin/env bash
set -eu
printf '%s\n' "$*" >> "$CALLS_FILE"
if [ "$1" = sync ]; then exit 0; fi
stage="$3"
if [ "$stage" = train.py ]; then
  if [ "${FAIL_TRAIN:-0}" = 1 ]; then exit 23; fi
  shift 3
  while [ "$#" -gt 0 ]; do
    if [ "$1" = --run-dir ]; then run_dir="$2"; break; fi
    shift
  done
  mkdir -p "$run_dir/best"
  touch "$run_dir/final_model.zip" "$run_dir/vecnormalize.pkl"
  touch "$run_dir/best/best_model.zip" "$run_dir/best/vecnormalize.pkl"
  printf 'fake training completed\n'
else
  printf 'fake evaluation completed\n'
fi
'''


class PilotShellTests(unittest.TestCase):
    def setUp(self) -> None:
        self.bash = shutil.which("bash")
        if self.bash is None:
            self.skipTest("Git Bash or Linux bash is required")
        self.temporary = tempfile.TemporaryDirectory(prefix="walker pilot ")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        scripts = self.root / "scripts"
        scripts.mkdir()
        self.script = scripts / "pilot_reference.sh"
        shutil.copyfile(ROOT / "scripts" / self.script.name, self.script)
        bin_dir = self.root / "bin"
        bin_dir.mkdir()
        fake = bin_dir / "uv"
        fake.write_text(FAKE_UV, encoding="utf-8", newline="\n")
        fake.chmod(0o755)
        self.calls = self.root / "calls.log"
        self.run_dir = self.root / "run with spaces"
        self.env = dict(os.environ)
        self.env.update({
            "PATH": str(bin_dir) + os.pathsep + self.env["PATH"],
            "CALLS_FILE": self.calls.as_posix(),
            "RUN_DIR": self.run_dir.as_posix(),
            "RESUME": "0",
        })

    def run_pilot(self) -> subprocess.CompletedProcess:
        return subprocess.run(
            [self.bash, str(self.script)], env=self.env,
            capture_output=True, text=True, timeout=30,
        )

    def test_success_uses_reference_and_distinct_eval_seeds(self) -> None:
        result = self.run_pilot()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        calls = self.calls.read_text(encoding="utf-8").splitlines()
        self.assertEqual(calls[0], "sync --locked")
        self.assertIn("--variant ppo-reference", calls[1])
        self.assertIn("--timesteps 1000000", calls[1])
        self.assertIn("--eval-seed 50000", calls[1])
        evaluations = [line for line in calls if "evaluate.py" in line]
        self.assertEqual(len(evaluations), 2)
        self.assertTrue(all("--seed 40000" in line for line in evaluations))
        log = (self.run_dir / "console.log").read_text(encoding="utf-8")
        self.assertIn("fake training completed", log)
        self.assertIn("exited with code 0", log)

    def test_training_failure_survives_tee_and_stops_evaluation(self) -> None:
        self.env["FAIL_TRAIN"] = "1"
        result = self.run_pilot()
        self.assertEqual(result.returncode, 23, result.stdout + result.stderr)
        self.assertNotIn("evaluate.py", self.calls.read_text(encoding="utf-8"))
        log = (self.run_dir / "console.log").read_text(encoding="utf-8")
        self.assertIn("exited with code 23", log)

    def test_existing_directory_is_not_overwritten(self) -> None:
        self.run_dir.mkdir()
        sentinel = self.run_dir / "keep.txt"
        sentinel.write_text("keep", encoding="utf-8")
        result = self.run_pilot()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")
        self.assertFalse(self.calls.exists())

    def test_main_results_are_protected(self) -> None:
        self.env["RUN_DIR"] = "runs/main/seed_0"
        result = self.run_pilot()
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.calls.exists())

    def test_resume_requires_explicit_directory(self) -> None:
        self.env.pop("RUN_DIR")
        self.env["RESUME"] = "1"
        result = self.run_pilot()
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.calls.exists())


if __name__ == "__main__":
    unittest.main()
