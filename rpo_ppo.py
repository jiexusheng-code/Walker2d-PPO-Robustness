from __future__ import annotations

import math

import torch as th
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.distributions import DiagGaussianDistribution
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import PyTorchObs


DEFAULT_RPO_ALPHA = 0.5
RPO_SOURCE_URL = (
    "https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/rpo_continuous_action.py"
)


def validate_rpo_alpha(alpha: float) -> float:
    """Return ``alpha`` as a finite, non-negative float or raise ValueError."""
    try:
        value = float(alpha)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"rpo_alpha must be a finite, non-negative number, got {alpha!r}"
        ) from error
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(
            f"rpo_alpha must be a finite, non-negative number, got {alpha!r}"
        )
    return value


class RPOActorCriticPolicy(ActorCriticPolicy):
    """ActorCriticPolicy with RPO's perturbed log-prob during updates only.

    Rollout collection and prediction keep sampling from the plain Gaussian
    ``Normal(mean, std)``. ``evaluate_actions``, which PPO's update calls,
    adds elementwise ``Uniform(-alpha, alpha)`` noise to the mean and evaluates
    the *original* rollout actions under that perturbed distribution. The
    entropy still comes from the unperturbed Gaussian (equal to the perturbed
    entropy, which does not depend on the mean).

    ``rpo_alpha == 0`` calls the parent implementation unchanged, so it makes
    no RNG draw and is exactly PPO-Clip.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule,
        *args,
        rpo_alpha: float = DEFAULT_RPO_ALPHA,
        **kwargs,
    ) -> None:
        self.rpo_alpha = validate_rpo_alpha(rpo_alpha)
        super().__init__(
            observation_space, action_space, lr_schedule, *args, **kwargs
        )

    def _get_constructor_parameters(self) -> dict:
        data = super()._get_constructor_parameters()
        data["rpo_alpha"] = float(self.rpo_alpha)
        return data

    def _sample_rpo_noise(self, mean_actions: th.Tensor) -> th.Tensor:
        alpha = float(self.rpo_alpha)
        return th.rand_like(mean_actions) * (2.0 * alpha) - alpha

    def evaluate_actions(
        self, obs: PyTorchObs, actions: th.Tensor
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor | None]:
        # Zero alpha must be byte-for-byte the PPO path, including no RNG use.
        if float(self.rpo_alpha) == 0.0:
            return super().evaluate_actions(obs, actions)

        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)

        mean_actions = self.action_net(latent_pi)
        self.action_dist.proba_distribution(mean_actions, self.log_std)
        entropy = self.action_dist.entropy()

        noise = self._sample_rpo_noise(mean_actions)
        self.action_dist.proba_distribution(
            mean_actions + noise, self.log_std
        )
        log_prob = self.action_dist.log_prob(actions)

        values = self.value_net(latent_vf)
        return values, log_prob, entropy


class RPO(PPO):
    """PPO-Clip with RPO's uniform perturbation of the update-time mean."""

    DEFAULT_ALPHA = DEFAULT_RPO_ALPHA
    policy_aliases = dict(PPO.policy_aliases)
    policy_aliases["MlpPolicy"] = RPOActorCriticPolicy

    def __init__(
        self,
        policy,
        env,
        *,
        rpo_alpha: float = DEFAULT_RPO_ALPHA,
        **kwargs,
    ) -> None:
        self.rpo_alpha = validate_rpo_alpha(rpo_alpha)
        super().__init__(policy, env, **kwargs)

    def _setup_model(self) -> None:
        # BaseAlgorithm.load() updates __dict__ directly, bypassing __init__,
        # so a loaded checkpoint's alpha must be validated again here.
        self.rpo_alpha = validate_rpo_alpha(
            getattr(self, "rpo_alpha", DEFAULT_RPO_ALPHA)
        )
        super()._setup_model()
        if not isinstance(self.policy, RPOActorCriticPolicy):
            raise ValueError(
                "RPO requires an RPOActorCriticPolicy, but this model uses "
                f"{type(self.policy).__name__}; refusing to mix RPO with a "
                "non-RPO checkpoint or policy."
            )
        self.policy.rpo_alpha = float(self.rpo_alpha)
        self._validate_supported()

    def _validate_supported(self) -> None:
        if self.use_sde:
            raise ValueError("RPO does not support gSDE (use_sde=True)")
        if not isinstance(self.action_space, spaces.Box):
            raise ValueError("RPO supports only Box action spaces")
        if getattr(self.policy, "squash_output", False):
            raise ValueError("RPO does not support squashed Gaussian outputs")
        if not isinstance(
            self.policy.action_dist, DiagGaussianDistribution
        ):
            raise ValueError(
                "RPO supports only the diagonal Gaussian policy distribution"
            )

    def _excluded_save_params(self) -> list[str]:
        # A runtime flight recorder must never be pickled into model.zip.
        return [*super()._excluded_save_params(), "_update_recorder"]

    def train(self) -> None:
        # PPO's training loop is intentionally inherited; the only change is
        # that its evaluate_actions call uses the perturbed mean.
        self._validate_supported()
        recorder = getattr(self, "_update_recorder", None)
        if recorder is None:
            super().train()
        else:
            recorder.run_update(self, super().train)
        self.logger.record("train/rpo_alpha", float(self.rpo_alpha))
