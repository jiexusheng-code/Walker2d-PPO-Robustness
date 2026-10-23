from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from torch.distributions import Normal

import train
from envs import make_env
from rpo_ppo import RPO, RPOActorCriticPolicy, validate_rpo_alpha


def _make_model(alpha: float = 0.5, seed: int = 0) -> tuple[DummyVecEnv, RPO]:
    env = DummyVecEnv([make_env])
    model = RPO(
        "MlpPolicy",
        env,
        rpo_alpha=alpha,
        n_steps=32,
        batch_size=16,
        n_epochs=1,
        policy_kwargs={"net_arch": {"pi": [16], "vf": [16]}},
        device="cpu",
        seed=seed,
        verbose=0,
    )
    return env, model


def _obs_tensor(model: RPO, obs: np.ndarray) -> torch.Tensor:
    tensor, _ = model.policy.obs_to_tensor(obs)
    return tensor


def _fake_rpo_model(alpha: float = 0.25) -> types.SimpleNamespace:
    values = {
        key: train.REFERENCE_PPO_KWARGS[key]
        for key in (
            "n_steps",
            "batch_size",
            "n_epochs",
            "gamma",
            "gae_lambda",
            "ent_coef",
            "vf_coef",
            "max_grad_norm",
        )
    }
    values.update(
        {
            "learning_rate": train.REFERENCE_LEARNING_RATE,
            "clip_range": train.REFERENCE_PPO_KWARGS["clip_range"],
            "policy_kwargs": train.REFERENCE_PPO_KWARGS["policy_kwargs"],
            "n_envs": 1,
            "rpo_alpha": alpha,
        }
    )
    return types.SimpleNamespace(**values)


class RPOActorCriticPolicyTests(unittest.TestCase):
    def test_perturbed_log_prob_matches_fixed_noise_formula_and_gradients(
        self,
    ) -> None:
        env, model = _make_model(alpha=0.5)
        try:
            policy = model.policy
            obs = env.reset()
            obs_tensor = _obs_tensor(model, obs)
            actions = torch.zeros(
                (obs_tensor.shape[0], env.action_space.shape[0]),
                dtype=torch.float32,
            )

            features = policy.extract_features(obs_tensor)
            latent_pi, latent_vf = policy.mlp_extractor(features)
            mean_actions = policy.action_net(latent_pi)
            fixed_noise = torch.linspace(
                -0.25, 0.25, mean_actions.numel()
            ).reshape_as(mean_actions)
            policy._sample_rpo_noise = lambda reference: fixed_noise

            values, log_prob, entropy = policy.evaluate_actions(
                obs_tensor, actions
            )

            std = torch.exp(policy.log_std)
            expected_log_prob = Normal(
                mean_actions + fixed_noise, std
            ).log_prob(actions).sum(dim=-1)
            expected_entropy = Normal(mean_actions, std).entropy().sum(dim=-1)
            expected_values = policy.value_net(latent_vf).flatten()

            self.assertTrue(
                torch.allclose(log_prob, expected_log_prob, atol=1e-6)
            )
            self.assertTrue(
                torch.allclose(entropy, expected_entropy, atol=1e-6)
            )
            self.assertTrue(
                torch.allclose(
                    values.flatten(), expected_values, atol=1e-6
                )
            )

            policy.zero_grad(set_to_none=True)
            log_prob.sum().backward()
            implementation_grad = policy.action_net.weight.grad.detach().clone()

            policy.zero_grad(set_to_none=True)
            expected_log_prob.sum().backward()
            expected_grad = policy.action_net.weight.grad.detach().clone()

            self.assertTrue(
                torch.allclose(implementation_grad, expected_grad, atol=1e-6)
            )
            self.assertGreater(float(implementation_grad.abs().sum()), 0.0)
        finally:
            env.close()

    def test_alpha_zero_is_exact_ppo_and_uses_no_rng(self) -> None:
        env, model = _make_model(alpha=0.0)
        try:
            policy = model.policy
            obs = env.reset()
            obs_tensor = _obs_tensor(model, obs)
            actions = torch.zeros(
                (obs_tensor.shape[0], env.action_space.shape[0]),
                dtype=torch.float32,
            )

            rng_state = torch.get_rng_state().clone()
            values, log_prob, entropy = policy.evaluate_actions(
                obs_tensor, actions
            )
            self.assertTrue(
                torch.equal(rng_state, torch.get_rng_state()),
                "rpo_alpha=0 must not draw RNG values",
            )

            expected_values, expected_log_prob, expected_entropy = (
                super(RPOActorCriticPolicy, policy).evaluate_actions(
                    obs_tensor, actions
                )
            )
            self.assertTrue(torch.equal(values, expected_values))
            self.assertTrue(torch.equal(log_prob, expected_log_prob))
            self.assertTrue(torch.equal(entropy, expected_entropy))

            with mock.patch.object(
                torch,
                "rand_like",
                side_effect=AssertionError("unexpected RPO noise draw"),
            ):
                policy.evaluate_actions(obs_tensor, actions)
        finally:
            env.close()

    def test_rollout_and_predict_use_the_unperturbed_gaussian(self) -> None:
        env, model = _make_model(alpha=0.5)
        try:
            policy = model.policy
            obs = env.reset()
            obs_tensor = _obs_tensor(model, obs)

            features = policy.extract_features(obs_tensor)
            latent_pi, _ = policy.mlp_extractor(features)
            expected_mean = policy.action_net(latent_pi).detach().numpy()
            predicted, _ = model.predict(obs, deterministic=True)
            self.assertTrue(np.allclose(predicted, expected_mean, atol=1e-6))

            with mock.patch.object(
                policy,
                "evaluate_actions",
                side_effect=AssertionError("evaluate_actions is update-only"),
            ):
                torch.manual_seed(7)
                sampled, _, _ = policy(obs_tensor, deterministic=False)
                torch.manual_seed(7)
                distribution = policy.get_distribution(obs_tensor)
                expected_sample = distribution.get_actions(
                    deterministic=False
                )
                self.assertTrue(
                    torch.allclose(sampled, expected_sample, atol=1e-6)
                )
                model.predict(obs, deterministic=False)
        finally:
            env.close()

    def test_alpha_validation(self) -> None:
        env = DummyVecEnv([make_env])
        try:
            for bad in (-0.1, float("nan"), float("inf"), "abc"):
                with self.subTest(alpha=bad):
                    with self.assertRaises(ValueError):
                        RPO(
                            "MlpPolicy",
                            env,
                            rpo_alpha=bad,
                            n_steps=32,
                            batch_size=16,
                            device="cpu",
                            verbose=0,
                        )
            for good in (0.0, 0.25, 1.0, 2.0):
                self.assertEqual(validate_rpo_alpha(good), good)
            self.assertAlmostEqual(validate_rpo_alpha(0.5), 0.5)
        finally:
            env.close()

    def test_unsupported_configurations_are_rejected(self) -> None:
        env = DummyVecEnv([make_env])
        try:
            with self.assertRaises(ValueError):
                RPO(
                    "MlpPolicy",
                    env,
                    use_sde=True,
                    n_steps=32,
                    batch_size=16,
                    device="cpu",
                    verbose=0,
                )

            model = RPO(
                "MlpPolicy",
                env,
                n_steps=32,
                batch_size=16,
                policy_kwargs={"net_arch": {"pi": [16], "vf": [16]}},
                device="cpu",
                verbose=0,
            )
            model.policy._squash_output = True
            with self.assertRaises(ValueError):
                model.train()

            model.policy._squash_output = False
            model.policy.action_dist = object()
            with self.assertRaises(ValueError):
                model._validate_supported()
        finally:
            env.close()

    def test_save_load_preserves_alpha_policy_and_prediction(self) -> None:
        env, model = _make_model(alpha=0.3)
        try:
            obs = env.reset()
            expected, _ = model.predict(obs, deterministic=True)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "rpo_model"
                model.save(path)
                restored = RPO.load(path, env=env, device="cpu")

            self.assertAlmostEqual(float(restored.rpo_alpha), 0.3)
            self.assertIsInstance(restored.policy, RPOActorCriticPolicy)
            self.assertAlmostEqual(float(restored.policy.rpo_alpha), 0.3)
            actual, _ = restored.predict(obs, deterministic=True)
            self.assertTrue(np.array_equal(expected, actual))
        finally:
            env.close()

    def test_standalone_policy_save_load_preserves_alpha(self) -> None:
        env, model = _make_model(alpha=0.3)
        try:
            obs = env.reset()
            obs_tensor = _obs_tensor(model, obs)
            with torch.no_grad():
                expected, _, _ = model.policy(
                    obs_tensor, deterministic=True
                )
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "rpo_policy"
                model.policy.save(path)
                restored = RPOActorCriticPolicy.load(path, device="cpu")

            self.assertAlmostEqual(float(restored.rpo_alpha), 0.3)
            with torch.no_grad():
                actual, _, _ = restored(obs_tensor, deterministic=True)
            self.assertTrue(torch.equal(expected, actual))
        finally:
            env.close()

    def test_loaded_alpha_is_validated_on_setup(self) -> None:
        env, model = _make_model(alpha=0.3)
        try:
            for bad in (-0.1, float("nan"), float("inf")):
                with self.subTest(alpha=bad):
                    model.rpo_alpha = bad
                    with self.assertRaises(ValueError):
                        model._setup_model()
            model.rpo_alpha = 0.3
            model._setup_model()
        finally:
            env.close()

    def test_load_rejects_a_non_rpo_checkpoint(self) -> None:
        env = DummyVecEnv([make_env])
        try:
            ppo = PPO(
                "MlpPolicy",
                env,
                n_steps=32,
                batch_size=16,
                n_epochs=1,
                policy_kwargs={"net_arch": {"pi": [16], "vf": [16]}},
                device="cpu",
                seed=1,
                verbose=0,
            )
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "plain_ppo"
                ppo.save(path)
                with self.assertRaises(ValueError):
                    RPO.load(path, env=env, device="cpu")
        finally:
            env.close()


class RPOVariantConfigTests(unittest.TestCase):
    def _args(self, alpha: float | None, seed: int = 3):
        return types.SimpleNamespace(
            variant=train.RPO_VARIANT,
            seed=seed,
            timesteps=1_000_000,
            torch_threads=1,
            eval_every=0,
            eval_episodes=1,
            eval_seed=50_000,
            rpo_alpha=alpha,
        )

    def _config(self, alpha: float | None) -> dict:
        return train.build_run_config(
            self._args(alpha),
            Path("runs") / train.RPO_VARIANT / "seed_3",
            previous_config={},
            resumed_from=None,
            start_timesteps=0,
            resume_history=[],
            segments=[],
            eval_enabled=False,
            versions={},
            runtime={},
            git={},
            hashes={},
        )

    def _write_config(self, run_dir: Path, config: dict) -> None:
        (run_dir / "config.json").write_text(
            json.dumps(config), encoding="utf-8"
        )

    def test_config_records_rpo_alpha_and_shared_recipe(self) -> None:
        config = self._config(0.25)
        self.assertEqual(config["variant"], train.RPO_VARIANT)
        self.assertEqual(config["profile"], train.REFERENCE_PROFILE)
        self.assertEqual(config["algorithm"], "RPO")
        self.assertEqual(
            config["actor"]["objective"],
            train.ACTOR_OBJECTIVES[train.RPO_VARIANT],
        )
        self.assertEqual(config["actor"]["rpo"]["alpha"], 0.25)
        self.assertTrue(config["actor"]["clip_range_used"])
        self.assertIsNone(config["actor"]["kl_rollback"])
        self.assertEqual(config["ppo"]["n_steps"], 512)
        self.assertNotIn("rpo_alpha", config["ppo"])
        self.assertIn("cleanrl", config["recipe"]["rpo_source"])
        self.assertEqual(self._config(None)["actor"]["rpo"]["alpha"], 0.5)

    def test_constructor_kwargs_only_add_alpha_for_rpo(self) -> None:
        reference = train.model_constructor_kwargs_for_variant(
            train.REFERENCE_VARIANT
        )
        self.assertNotIn("rpo_alpha", reference)
        self.assertEqual(reference["n_steps"], 512)

        rpo = train.model_constructor_kwargs_for_variant(
            train.RPO_VARIANT, 0.25
        )
        self.assertEqual(rpo["rpo_alpha"], 0.25)
        self.assertEqual(rpo["n_steps"], 512)

    def test_resume_accepts_matching_alpha(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            self._write_config(run_dir, self._config(0.25))
            train.validate_resume(
                run_dir, self._args(0.25), _fake_rpo_model(0.25)
            )
            # Omitting --rpo-alpha accepts the checkpoint's own value.
            train.validate_resume(
                run_dir, self._args(None), _fake_rpo_model(0.25)
            )

    def test_resume_rejects_saved_or_requested_alpha_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            self._write_config(run_dir, self._config(0.25))
            with self.assertRaises(SystemExit):
                train.validate_resume(
                    run_dir, self._args(None), _fake_rpo_model(0.5)
                )
            with self.assertRaises(SystemExit):
                train.validate_resume(
                    run_dir, self._args(0.75), _fake_rpo_model(0.25)
                )

    def test_resume_rejects_a_different_variant(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            config = self._config(0.25)
            config["variant"] = train.REFERENCE_VARIANT
            config["algorithm"] = "PPO"
            config["actor"] = train.actor_config_for_variant(
                train.REFERENCE_VARIANT
            )
            self._write_config(run_dir, config)
            with self.assertRaises(SystemExit):
                train.validate_resume(
                    run_dir, self._args(0.25), _fake_rpo_model(0.25)
                )

    def test_cli_scopes_rpo_alpha_to_the_rpo_variant(self) -> None:
        with mock.patch.object(
            sys,
            "argv",
            ["train.py", "--variant", "ppo-reference", "--seed", "0",
             "--rpo-alpha", "0.2"],
        ):
            with self.assertRaises(SystemExit):
                train.parse_args()

        with mock.patch.object(
            sys,
            "argv",
            ["train.py", "--variant", "rpo", "--seed", "0",
             "--rpo-alpha", "0.2"],
        ):
            args = train.parse_args()
        self.assertAlmostEqual(args.rpo_alpha, 0.2)

        with mock.patch.object(
            sys,
            "argv",
            ["train.py", "--variant", "rpo", "--seed", "0",
             "--rpo-alpha", "-1"],
        ):
            with self.assertRaises(SystemExit):
                train.parse_args()


if __name__ == "__main__":
    unittest.main()
