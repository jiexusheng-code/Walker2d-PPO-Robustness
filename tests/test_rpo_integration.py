from __future__ import annotations

import unittest

import numpy as np
import torch
from stable_baselines3 import PPO

import train
from rpo_ppo import RPO


class RPOUpdateIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls) -> None:
        torch.set_num_threads(cls.original_threads)

    def run_updates(self, algorithm, alpha: float | None = None):
        env = train.make_training_env()
        try:
            kwargs = dict(train.REFERENCE_PPO_KWARGS)
            if alpha is not None:
                kwargs["rpo_alpha"] = alpha
            model = algorithm(
                "MlpPolicy", env, seed=731,
                learning_rate=train.REFERENCE_LEARNING_RATE,
                **kwargs,
            )
            model.learn(total_timesteps=1024)
            parameters = {
                key: value.detach().clone()
                for key, value in model.policy.state_dict().items()
            }
            optimizer = model.policy.optimizer.state_dict()
            return {
                "parameters": parameters,
                "optimizer": optimizer,
                "torch_rng": torch.get_rng_state().clone(),
                "numpy_rng": np.random.get_state(),
                "obs_mean": env.obs_rms.mean.copy(),
                "reward_var": env.ret_rms.var.copy(),
                "steps": model.num_timesteps,
            }
        finally:
            env.close()

    def assert_same_training_state(self, first: dict, second: dict) -> None:
        self.assertEqual(first["steps"], 1024)
        self.assertEqual(first["steps"], second["steps"])
        self.assertEqual(first["parameters"].keys(), second["parameters"].keys())
        for name, value in first["parameters"].items():
            self.assertTrue(torch.equal(value, second["parameters"][name]), name)
        first_opt, second_opt = first["optimizer"], second["optimizer"]
        self.assertEqual(first_opt["param_groups"], second_opt["param_groups"])
        for parameter, state in first_opt["state"].items():
            for name, value in state.items():
                self.assertTrue(torch.equal(
                    value, second_opt["state"][parameter][name]
                ), f"optimizer {parameter}/{name}")
        self.assertTrue(torch.equal(first["torch_rng"], second["torch_rng"]))
        first_np, second_np = first["numpy_rng"], second["numpy_rng"]
        self.assertEqual(first_np[0], second_np[0])
        self.assertTrue(np.array_equal(first_np[1], second_np[1]))
        self.assertEqual(first_np[2:], second_np[2:])
        self.assertTrue(np.array_equal(first["obs_mean"], second["obs_mean"]))
        self.assertTrue(np.array_equal(first["reward_var"], second["reward_var"]))

    def test_alpha_zero_matches_reference_over_two_full_updates(self) -> None:
        reference = self.run_updates(PPO)
        zero_alpha = self.run_updates(RPO, 0.0)
        self.assert_same_training_state(reference, zero_alpha)

    def test_nonzero_alpha_updates_are_reproducible_and_change_policy(self) -> None:
        first = self.run_updates(RPO, 0.5)
        second = self.run_updates(RPO, 0.5)
        self.assert_same_training_state(first, second)
        unperturbed = self.run_updates(RPO, 0.0)
        self.assertTrue(any(
            not torch.equal(value, unperturbed["parameters"][name])
            for name, value in first["parameters"].items()
            if name.startswith("action_net.")
        ))


if __name__ == "__main__":
    unittest.main()
