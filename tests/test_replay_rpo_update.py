from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from scripts.replay_rpo_update import compare_state, replay


class UpdateReplayTests(unittest.TestCase):
    def test_exact_comparison_checks_optimizer_tensors(self) -> None:
        first = {"state": {1: {"exp_avg": torch.tensor([0.1, 0.2])}}}
        second = {"state": {1: {"exp_avg": torch.tensor([0.1, 0.3])}}}
        self.assertEqual(compare_state(first, first), [])
        self.assertEqual(compare_state(first, second), [".state.1.exp_avg"])

    def test_replay_requires_explicit_trust(self) -> None:
        with self.assertRaisesRegex(ValueError, "trust-local"):
            replay(Path("unused"), Path("unused.json"), trust_local=False)

    def test_missing_required_hash_rejected_before_loading_model(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "manifest.json").write_text(json.dumps({
                "status": "completed", "files": {},
            }), encoding="utf-8")
            with patch("scripts.replay_rpo_update.RPO.load") as load:
                with self.assertRaisesRegex(ValueError, "required replay artifacts"):
                    replay(directory, directory / "result.json", trust_local=True)
                load.assert_not_called()

    def test_hash_mismatch_rejected_before_loading_model(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "model_pre.zip").write_bytes(b"changed")
            (directory / "manifest.json").write_text(json.dumps({
                "status": "completed",
                "files": {name: {
                    "sha256": hashlib.sha256(b"original").hexdigest(),
                    "size": 8,
                } for name in ("model_pre.zip", "model_post.zip",
                               "rollout_buffer.pkl", "rng.pkl")},
            }), encoding="utf-8")
            with patch("scripts.replay_rpo_update.RPO.load") as load:
                with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                    replay(directory, directory / "result.json", trust_local=True)
                load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
