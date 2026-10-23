from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from diagnostics import HealthyEvalCallback
from envs import make_env
from rpo_ppo import RPO
from training_trace import TrainingTraceCallback
from scripts.replay_rpo_update import compare_state, replay
from train import REFERENCE_LEARNING_RATE, REFERENCE_PPO_KWARGS, make_training_env, make_eval_env
from update_recording import UpdateRecorder, capture_rng_state


class TrainingTraceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    def tearDown(self) -> None:
        torch.set_num_threads(self.threads)

    def run_training(self, directory: Path | None):
        env = VecNormalize(DummyVecEnv([lambda: Monitor(make_env())]))
        try:
            model = RPO("MlpPolicy", env, n_steps=32, batch_size=16,
                        n_epochs=2, seed=912, device="cpu",
                        policy_kwargs={"net_arch": {"pi": [16], "vf": [16]}})
            callback = TrainingTraceCallback(directory) if directory else None
            model.learn(64, callback=callback)
            return ({k: v.clone() for k, v in model.policy.state_dict().items()},
                    torch.get_rng_state().clone(), np.random.get_state())
        finally:
            env.close()

    def test_trace_preserves_training_and_records_transitions(self) -> None:
        plain, torch_plain, np_plain = self.run_training(None)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            traced, torch_traced, np_traced = self.run_training(directory)
            for key, value in plain.items():
                self.assertTrue(torch.equal(value, traced[key]), key)
            self.assertTrue(torch.equal(torch_plain, torch_traced))
            self.assertTrue(np.array_equal(np_plain[1], np_traced[1]))
            traces = sorted(directory.glob("rollout_*.npz"))
            self.assertEqual(len(traces), 2)
            with np.load(traces[0], allow_pickle=False) as data:
                self.assertEqual(data["actions_raw_gaussian"].shape, (32, 1, 6))
                self.assertEqual(data["raw_observations"].shape, (32, 1, 17))
                self.assertTrue(np.isfinite(data["raw_observations"]).all())
                self.assertTrue(np.array_equal(data["step"], np.arange(1, 33)))
            self.assertTrue((directory / "environment_000000000000.xml").is_file())
            self.assertTrue((directory / "environment_000000000000.json").is_file())

    def test_full_recipe_recording_preserves_training_and_replays(self) -> None:
        def run(directory):
            env, eval_env = make_training_env(), make_eval_env()
            try:
                model = RPO(
                    "MlpPolicy", env, seed=941, learning_rate=REFERENCE_LEARNING_RATE,
                    **copy.deepcopy(REFERENCE_PPO_KWARGS),
                )
                callbacks = []
                if directory is not None:
                    UpdateRecorder(directory / "updates").attach(model)
                    callbacks.append(TrainingTraceCallback(directory / "transitions"))
                callbacks.append(HealthyEvalCallback(
                    eval_env, eval_freq=512, n_eval_episodes=2, eval_seed=50000,
                    debug_directory=directory / "evaluations" if directory else None,
                    verbose=0,
                ))
                model.learn(512, callback=callbacks)
                state = copy.deepcopy({
                    "parameters": model.get_parameters(),
                    "torch_rng": torch.get_rng_state(),
                    "obs_mean": env.obs_rms.mean,
                    "obs_var": env.obs_rms.var,
                    "reward_mean": env.ret_rms.mean,
                    "reward_var": env.ret_rms.var,
                    "n_updates": model._n_updates,
                })
                return state, capture_rng_state()
            finally:
                env.close()
                eval_env.close()

        plain, plain_rng = run(None)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            recorded, recorded_rng = run(directory)
            for key in ("obs_mean", "obs_var", "reward_mean", "reward_var"):
                self.assertTrue(np.array_equal(plain.pop(key), recorded.pop(key)), key)
            self.assertEqual(compare_state(plain, recorded), [])
            self.assertEqual(plain_rng["python"], recorded_rng["python"])
            self.assertTrue(np.array_equal(plain_rng["numpy"][1], recorded_rng["numpy"][1]))
            self.assertEqual(plain_rng["numpy"][0], recorded_rng["numpy"][0])
            self.assertEqual(plain_rng["numpy"][2:], recorded_rng["numpy"][2:])
            self.assertTrue(torch.equal(plain_rng["torch_cpu"], recorded_rng["torch_cpu"]))
            result = replay(directory / "updates" / "update_000000",
                            directory / "replay.json", trust_local=True)
            self.assertTrue(result["exact_policy_and_optimizer_match"], result)

    def test_periodic_evaluation_keeps_snapshot_and_episode_records(self) -> None:
        env = VecNormalize(DummyVecEnv([lambda: Monitor(make_env())]))
        eval_env = VecNormalize(DummyVecEnv([lambda: Monitor(make_env())]),
                                training=False, norm_reward=False)
        try:
            with tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                model = RPO("MlpPolicy", env, n_steps=32, batch_size=16,
                            n_epochs=1, seed=912, device="cpu")
                callback = HealthyEvalCallback(
                    eval_env, eval_freq=32, n_eval_episodes=2, eval_seed=50000,
                    debug_directory=directory, verbose=0,
                )
                model.learn(32, callback=callback)
                snapshot = directory / "eval_000000000032"
                events = [json.loads(line) for line in
                          (snapshot / "episodes.jsonl").read_text().splitlines()]
                self.assertEqual(events[0]["n_updates"], 0)
                self.assertEqual(len(events), 3)
                self.assertEqual([r["episode_index"] for r in events[1:]], [0, 1])
                self.assertTrue((snapshot / "model.zip").is_file())
                self.assertTrue((snapshot / "vecnormalize.pkl").is_file())
        finally:
            env.close()
            eval_env.close()


if __name__ == "__main__":
    unittest.main()
