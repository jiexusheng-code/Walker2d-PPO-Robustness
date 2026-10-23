from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs import make_env
from train import make_training_env
from truly_ppo import TrulyPPO, truly_ppo_actor_loss


class TrulyPPOTests(unittest.TestCase):
    def test_infinite_delta_reduces_to_unclipped_policy_gradient(self) -> None:
        ratio = torch.tensor([0.5, 1.0, 1.8])
        advantages = torch.tensor([-1.0, 0.5, 2.0])
        kl = torch.tensor([0.04, 0.04, 0.04])

        loss, trigger = truly_ppo_actor_loss(
            ratio, advantages, kl, delta=float("inf")
        )

        self.assertFalse(bool(trigger.any()))
        self.assertTrue(torch.isclose(loss, -(ratio * advantages).mean()))

    def test_rollback_is_triggered_per_sample(self) -> None:
        ratio = torch.tensor([1.2, 0.8, 1.1, 0.9])
        advantages = torch.tensor([1.0, 1.0, -1.0, 1.0])
        kl = torch.full((4,), 0.04)

        _, trigger = truly_ppo_actor_loss(ratio, advantages, kl)

        self.assertTrue(torch.equal(trigger, torch.tensor([True, False, False, False])))

    def test_rollback_kl_penalty_has_gradient(self) -> None:
        ratio = torch.tensor([1.2])
        advantages = torch.tensor([1.0])
        kl = torch.tensor([0.04], requires_grad=True)

        loss, trigger = truly_ppo_actor_loss(ratio, advantages, kl)
        loss.backward()

        self.assertTrue(bool(trigger.item()))
        self.assertIsNotNone(kl.grad)
        self.assertNotEqual(float(kl.grad.item()), 0.0)

    def test_rollback_gradient_matches_alpha_and_masks_untriggered_samples(
        self,
    ) -> None:
        triggered_kl = torch.tensor([0.04], requires_grad=True)
        triggered_loss, triggered = truly_ppo_actor_loss(
            torch.tensor([1.2]), torch.tensor([1.0]), triggered_kl
        )
        triggered_loss.backward()

        self.assertTrue(bool(triggered.item()))
        self.assertAlmostEqual(float(triggered_loss.item()), -1.0, places=6)
        self.assertAlmostEqual(float(triggered_kl.grad.item()), 5.0, places=6)

        untriggered_kl = torch.tensor([0.04], requires_grad=True)
        untriggered_loss, untriggered = truly_ppo_actor_loss(
            torch.tensor([0.8]), torch.tensor([1.0]), untriggered_kl
        )
        untriggered_loss.backward()

        self.assertFalse(bool(untriggered.item()))
        self.assertAlmostEqual(float(untriggered_loss.item()), -0.8, places=6)
        self.assertAlmostEqual(float(untriggered_kl.grad.item()), 0.0, places=6)

    def test_trigger_boundary_is_inclusive_at_delta(self) -> None:
        ratio = torch.tensor([1.1, 1.1, 1.1])
        advantages = torch.tensor([1.0, 1.0, 1.0])
        kl = torch.tensor([0.03, 0.0299, 0.0301])

        _, trigger = truly_ppo_actor_loss(ratio, advantages, kl)

        self.assertTrue(torch.equal(trigger, torch.tensor([True, False, True])))

    def test_zero_advantage_never_triggers(self) -> None:
        ratio = torch.tensor([1.5, 0.5])
        advantages = torch.tensor([0.0, 0.0])
        kl = torch.tensor([0.04, 0.04])

        _, trigger = truly_ppo_actor_loss(ratio, advantages, kl)

        self.assertFalse(bool(trigger.any()))

    def test_ratio_one_with_high_kl_never_triggers(self) -> None:
        ratio = torch.tensor([1.0, 1.0])
        advantages = torch.tensor([1.0, -1.0])
        kl = torch.tensor([0.04, 0.04])

        _, trigger = truly_ppo_actor_loss(ratio, advantages, kl)

        self.assertFalse(bool(trigger.any()))

    def test_use_sde_is_rejected(self) -> None:
        env = DummyVecEnv([make_env])
        try:
            model = TrulyPPO(
                "MlpPolicy",
                env,
                use_sde=True,
                n_steps=32,
                batch_size=16,
                device="cpu",
                seed=0,
                verbose=0,
            )
            with self.assertRaises(ValueError):
                model.train()
        finally:
            env.close()

    def test_squash_output_is_rejected(self) -> None:
        env = DummyVecEnv([make_env])
        try:
            model = TrulyPPO(
                "MlpPolicy",
                env,
                n_steps=32,
                batch_size=16,
                device="cpu",
                seed=0,
                verbose=0,
            )
            model.policy._squash_output = True
            with self.assertRaises(ValueError):
                model.train()
        finally:
            env.close()

    def test_vecnormalize_statistics_reload_for_evaluation(self) -> None:
        training_env = make_training_env()
        raw_env = make_env()
        loaded_env = None
        try:
            training_env.reset()
            raw_obs, _ = raw_env.reset(seed=9)
            with tempfile.TemporaryDirectory() as directory:
                stats_path = Path(directory) / "vecnormalize.pkl"
                training_env.save(stats_path)
                loaded_env = VecNormalize.load(
                    stats_path, DummyVecEnv([make_env])
                )
                loaded_env.training = False
                loaded_env.norm_reward = False
                normalized_obs = loaded_env.normalize_obs(raw_obs)
            self.assertEqual(normalized_obs.shape, raw_obs.shape)
            self.assertTrue(np.all(np.isfinite(normalized_obs)))
        finally:
            training_env.close()
            raw_env.close()
            if loaded_env is not None:
                loaded_env.close()

    def test_truly_ppo_save_load_preserves_prediction(self) -> None:
        env = DummyVecEnv([make_env])
        try:
            model = TrulyPPO(
                "MlpPolicy",
                env,
                n_steps=32,
                batch_size=16,
                n_epochs=1,
                policy_kwargs={"net_arch": {"pi": [16], "vf": [16]}},
                device="cpu",
                seed=3,
                verbose=0,
            )
            obs = env.reset()
            expected, _ = model.predict(obs, deterministic=True)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "truly_ppo"
                model.save(path)
                restored = TrulyPPO.load(path, env=env, device="cpu")
                actual, _ = restored.predict(obs, deterministic=True)
            self.assertTrue(np.array_equal(expected, actual))
        finally:
            env.close()

    def test_train_runs_and_trigger_metric_is_active(self) -> None:
        env = DummyVecEnv([make_env])
        try:
            model = TrulyPPO(
                "MlpPolicy",
                env,
                n_steps=32,
                batch_size=16,
                n_epochs=2,
                policy_kwargs={"net_arch": {"pi": [16], "vf": [16]}},
                device="cpu",
                seed=3,
                verbose=0,
            )
            model.truly_delta = 0.0
            model.learn(total_timesteps=32)

            self.assertEqual(model.num_timesteps, 32)
            trigger_fraction = model.logger.name_to_value[
                "train/truly_trigger_fraction"
            ]
            self.assertGreater(float(trigger_fraction), 0.0)
            for parameter in model.policy.parameters():
                self.assertTrue(bool(torch.isfinite(parameter).all()))
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
