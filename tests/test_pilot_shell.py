"""Exercise the cloud entry points with a fake uv; never launch training."""
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
if [ "$stage" = "scripts/run_matrix_lib.py" ]; then
  action="${PLAN_ACTION:-train}"
  case "$*" in
    *truly-ppo*)
      if [ -n "${TRULY_ACTION:-}" ]; then action="$TRULY_ACTION"; fi
      ;;
  esac
  printf '%s|fake plan\n' "$action"
  if [ "$action" = error ]; then exit 3; fi
  exit 0
fi
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
  if [ "${FAIL_EVAL:-0}" = 1 ]; then exit 24; fi
  printf 'fake evaluation completed\n'
fi
'''


class ShellTestBase(unittest.TestCase):
    script_name = ""

    def setUp(self) -> None:
        self.bash = shutil.which("bash")
        if self.bash is None:
            self.skipTest("Git Bash or Linux bash is required")
        self.temporary = tempfile.TemporaryDirectory(prefix="walker pilot ")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        scripts = self.root / "scripts"
        scripts.mkdir()
        self.script = scripts / self.script_name
        shutil.copyfile(ROOT / "scripts" / self.script_name, self.script)
        bin_dir = self.root / "bin"
        bin_dir.mkdir()
        fake = bin_dir / "uv"
        fake.write_text(FAKE_UV, encoding="utf-8", newline="\n")
        fake.chmod(0o755)
        self.calls = self.root / "calls.log"
        self.env = dict(os.environ)
        self.env.update({
            "PATH": str(bin_dir) + os.pathsep + self.env["PATH"],
            "CALLS_FILE": self.calls.as_posix(),
        })

    def run_script(self, timeout: int = 30) -> subprocess.CompletedProcess:
        return subprocess.run(
            [self.bash, str(self.script)], env=self.env,
            capture_output=True, text=True, timeout=timeout,
        )

    def call_lines(self) -> list[str]:
        return self.calls.read_text(encoding="utf-8").splitlines()


class PilotShellTests(ShellTestBase):
    script_name = "pilot_reference.sh"

    def setUp(self) -> None:
        super().setUp()
        self.run_dir = self.root / "run with spaces"
        self.env.update({
            "RUN_DIR": self.run_dir.as_posix(),
            "RESUME": "0",
        })

    def run_pilot(self) -> subprocess.CompletedProcess:
        return self.run_script()

    def test_success_uses_reference_and_distinct_eval_seeds(self) -> None:
        result = self.run_pilot()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        calls = self.call_lines()
        self.assertEqual(calls[0], "sync --locked")
        self.assertIn("--variant ppo-reference", calls[1])
        self.assertIn("--timesteps 1000000", calls[1])
        self.assertIn("--eval-seed 50000", calls[1])
        evaluations = [line for line in calls if "evaluate.py" in line]
        self.assertEqual(len(evaluations), 2)
        self.assertTrue(all("--seed 40000" in line for line in evaluations))
        log = (self.run_dir / "console.log").read_text(encoding="utf-8")
        self.assertIn("fake training completed", log)
        self.assertIn("single-group reference pilot", log)
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

    def test_two_arm_results_are_protected(self) -> None:
        self.env["RUN_DIR"] = "runs/two_arm/ppo-reference/seed_0"
        result = self.run_pilot()
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.calls.exists())

    def test_resume_requires_explicit_directory(self) -> None:
        self.env.pop("RUN_DIR")
        self.env["RESUME"] = "1"
        result = self.run_pilot()
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.calls.exists())


class MatrixShellTests(ShellTestBase):
    script_name = "run_matrix.sh"

    def test_default_runs_both_arms_with_identical_conditions(self) -> None:
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("shared_profile=ppo-reference", result.stdout)
        self.assertIn("runs_root=runs/two_arm", result.stdout)
        calls = self.call_lines()
        self.assertEqual(calls[0], "sync --locked")
        trains = [line for line in calls if "train.py" in line]
        self.assertEqual(len(trains), 2)
        for variant, line in zip(("ppo-reference", "truly-ppo"), trains):
            self.assertIn(f"--variant {variant}", line)
            self.assertIn("--seed 0", line)
            self.assertIn("--timesteps 1000000", line)
            self.assertIn("--eval-every 50000", line)
            self.assertIn("--eval-episodes 10", line)
            self.assertIn("--eval-seed 50000", line)
            self.assertIn(
                f"--run-dir runs/two_arm/{variant}/seed_0", line
            )
        evaluations = [line for line in calls if "evaluate.py" in line]
        self.assertEqual(len(evaluations), 4)
        primary = [
            line for line in evaluations
            if "--output-dir" in line and "best/" not in line
        ]
        self.assertEqual(len(primary), 2)
        for line in primary:
            self.assertIn("--episodes 100", line)
            self.assertIn("--seed 40000", line)
        best = [line for line in evaluations if "best/" in line]
        self.assertEqual(len(best), 2)
        for line in best:
            self.assertIn("--vecnormalize", line)
        self.assertFalse(any("--rpo-alpha" in line for line in calls))

    def test_old_variant_is_rejected_before_uv(self) -> None:
        self.env["VARIANTS"] = "baseline"
        result = self.run_script()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unsupported training variant", result.stderr)
        self.assertFalse(self.calls.exists())

    def test_explicit_rpo_variant_is_accepted(self) -> None:
        self.env["VARIANTS"] = "rpo"
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        trains = [line for line in self.call_lines() if "train.py" in line]
        self.assertEqual(len(trains), 1)
        self.assertIn("--variant rpo", trains[0])

    def test_rpo_alpha_is_forwarded_to_planning_and_training(self) -> None:
        self.env["VARIANTS"] = "rpo"
        self.env["RPO_ALPHA"] = "0.1"
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        calls = self.call_lines()
        planned = [
            line for line in calls if "scripts/run_matrix_lib.py" in line
        ]
        self.assertEqual(len(planned), 1)
        self.assertIn("--rpo-alpha 0.1", planned[0])
        trains = [line for line in calls if "train.py" in line]
        self.assertEqual(len(trains), 1)
        self.assertIn("--rpo-alpha 0.1", trains[0])

    def test_skip_does_not_train_or_evaluate(self) -> None:
        self.env["PLAN_ACTION"] = "skip"
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        text = self.calls.read_text(encoding="utf-8")
        self.assertNotIn("train.py", text)
        self.assertNotIn("evaluate.py", text)

    def test_evaluate_action_skips_training(self) -> None:
        self.env["PLAN_ACTION"] = "evaluate"
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        text = self.calls.read_text(encoding="utf-8")
        self.assertNotIn("train.py", text)
        self.assertIn("evaluate.py", text)

    def test_resume_passes_the_flag_to_training(self) -> None:
        self.env["RESUME"] = "1"
        self.env["PLAN_ACTION"] = "resume"
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        trains = [line for line in self.call_lines() if "train.py" in line]
        self.assertEqual(len(trains), 2)
        self.assertTrue(all("--resume" in line for line in trains))

    def test_resume_starts_an_untouched_later_arm_without_the_flag(self) -> None:
        self.env["RESUME"] = "1"
        self.env["PLAN_ACTION"] = "resume"
        self.env["TRULY_ACTION"] = "train"
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        trains = [line for line in self.call_lines() if "train.py" in line]
        self.assertEqual(len(trains), 2)
        self.assertIn("--variant ppo-reference", trains[0])
        self.assertIn("--resume", trains[0])
        self.assertIn("--variant truly-ppo", trains[1])
        self.assertNotIn("--resume", trains[1])

    def test_training_failure_stops_the_matrix(self) -> None:
        self.env["FAIL_TRAIN"] = "1"
        result = self.run_script()
        self.assertEqual(result.returncode, 23, result.stdout + result.stderr)
        text = self.calls.read_text(encoding="utf-8")
        self.assertNotIn("evaluate.py", text)
        self.assertEqual(text.count("train.py"), 1)

    def test_plan_error_stops_the_matrix(self) -> None:
        self.env["PLAN_ACTION"] = "error"
        result = self.run_script()
        self.assertNotEqual(result.returncode, 0)
        text = self.calls.read_text(encoding="utf-8")
        self.assertNotIn("train.py", text)
        self.assertNotIn("evaluate.py", text)

    def test_custom_seeds_are_used_for_both_arms(self) -> None:
        self.env["SEEDS"] = "0 1"
        result = self.run_script(timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        trains = [line for line in self.call_lines() if "train.py" in line]
        self.assertEqual(len(trains), 4)
        self.assertIn("--seed 0", trains[0])
        self.assertIn("--seed 1", trains[1])
        self.assertIn("--seed 0", trains[2])
        self.assertIn("--seed 1", trains[3])


if __name__ == "__main__":
    unittest.main()
