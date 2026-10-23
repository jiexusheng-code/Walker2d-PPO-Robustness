from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from stable_baselines3.common.logger import configure

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rpo_ppo import RPO
from update_recording import load_rng_state, load_rollout_buffer, restore_rng_state


def compare_state(first, second, prefix="") -> list[str]:
    """Compare nested model/Adam state exactly, not with a loose tolerance."""
    if isinstance(first, torch.Tensor) and isinstance(second, torch.Tensor):
        return [] if torch.equal(first.cpu(), second.cpu()) else [prefix]
    if isinstance(first, dict) and isinstance(second, dict):
        if first.keys() != second.keys():
            return [prefix + ".keys"]
        return [issue for key in first for issue in compare_state(
            first[key], second[key], f"{prefix}.{key}"
        )]
    if isinstance(first, (list, tuple)) and isinstance(second, (list, tuple)):
        if len(first) != len(second):
            return [prefix + ".length"]
        return [issue for i, (a, b) in enumerate(zip(first, second))
                for issue in compare_state(a, b, f"{prefix}[{i}]")]
    return [] if first == second else [prefix]


def file_hash(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def replay(directory: Path, output: Path, *, trust_local: bool) -> dict:
    if not trust_local:
        raise ValueError(
            "Snapshots contain pickle/model data capable of executing code. "
            "Use --trust-local only for packages produced by your own training."
        )
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "completed":
        raise ValueError("Only a completed package can be verified against model_post.zip")
    required = {"model_pre.zip", "model_post.zip", "rollout_buffer.pkl", "rng.pkl"}
    if not required.issubset(manifest["files"]):
        raise ValueError("Manifest does not contain hashes for all required replay artifacts")
    # Check all recorded file hashes before loading any pickle/model artifact.
    for name, entry in manifest["files"].items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory.resolve()):
            raise ValueError(f"Artifact path escapes update directory: {name}")
        if file_hash(path) != entry["sha256"]:
            raise ValueError(f"SHA256 mismatch: {name}")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    runtime = manifest.get("runtime", {})
    torch.set_num_threads(int(runtime.get("torch_num_threads", 1)))
    interop = runtime.get("torch_num_interop_threads")
    if interop is not None and int(interop) != torch.get_num_interop_threads():
        torch.set_num_interop_threads(int(interop))
    model = RPO.load(directory / "model_pre.zip", device="cpu")
    expected = RPO.load(directory / "model_post.zip", device="cpu")
    model.rollout_buffer = load_rollout_buffer(directory / "rollout_buffer.pkl")
    model.set_logger(configure(folder=None, format_strings=[]))
    rng = load_rng_state(directory / "rng.pkl")
    restore_rng_state(rng)
    model.train()
    mismatches = compare_state(model.policy.state_dict(), expected.policy.state_dict(), "policy")
    mismatches += compare_state(model.policy.optimizer.state_dict(),
                                expected.policy.optimizer.state_dict(), "optimizer")
    if model._n_updates != expected._n_updates:
        mismatches.append("n_updates")
    result = {
        "source_update": str(directory.resolve()),
        "num_timesteps": int(model.num_timesteps),
        "alpha": float(model.rpo_alpha),
        "exact_policy_and_optimizer_match": not mismatches,
        "mismatches": mismatches,
        "scope": "fixed-buffer optimizer-update replay, not simulator continuation",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as file:
        json.dump(result, file, indent=2)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify a trusted RPO recorded update by exact replay.")
    parser.add_argument("update_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trust-local", action="store_true")
    args = parser.parse_args()
    result = replay(args.update_dir, args.output, trust_local=args.trust_local)
    print(json.dumps(result, indent=2))
    if not result["exact_policy_and_optimizer_match"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
