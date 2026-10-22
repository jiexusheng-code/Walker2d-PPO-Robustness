from __future__ import annotations

import copy

import numpy as np
import gymnasium as gym
import torch as th
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.distributions import kl_divergence
from stable_baselines3.common.utils import explained_variance


def truly_ppo_actor_loss(
    ratio: th.Tensor,
    advantages: th.Tensor,
    kl_old_new: th.Tensor,
    *,
    alpha: float = 5.0,
    delta: float = 0.03,
) -> tuple[th.Tensor, th.Tensor]:
    """Return the Truly PPO actor loss and its per-sample rollback mask."""
    trigger = (kl_old_new >= delta) & ((ratio - 1.0) * advantages > 0.0)
    unclipped_objective = ratio * advantages
    rollback_objective = unclipped_objective - alpha * kl_old_new
    objective = th.where(trigger, rollback_objective, unclipped_objective)
    return -objective.mean(), trigger


class TrulyPPO(PPO):
    """Truly PPO's KL-triggered rollback actor objective on top of SB3 PPO."""

    truly_alpha = 5.0
    truly_delta = 0.03

    def train(self) -> None:
        # Truly PPO's analytic KL port supports neither gSDE nor squashed
        # outputs; these are independent unsupported cases.
        if self.use_sde:
            raise ValueError("TrulyPPO's analytic policy KL does not support gSDE")
        if getattr(self.policy, "squash_output", False):
            raise ValueError(
                "TrulyPPO's analytic policy KL does not support squashed outputs"
            )

        # `self.clip_range` and `self.target_kl` are intentionally unused:
        # Truly PPO replaces actor ratio clipping and target-KL early stopping
        # with the KL-triggered rollback below.
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        # SB3 calls train() immediately after collect_rollouts(), so this
        # snapshot is the behavior policy behind rollout_data.old_log_prob.
        old_policy = copy.deepcopy(self.policy)
        old_policy.set_training_mode(False)

        entropy_losses: list[float] = []
        policy_losses: list[float] = []
        value_losses: list[float] = []
        kl_values: list[float] = []
        trigger_fractions: list[float] = []

        for _ in range(self.n_epochs):
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, gym.spaces.Discrete):
                    actions = actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations, actions
                )
                values = values.flatten()
                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (
                        advantages.std() + 1e-8
                    )

                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                with th.no_grad():
                    old_distribution = old_policy.get_distribution(
                        rollout_data.observations
                    )
                new_distribution = self.policy.get_distribution(
                    rollout_data.observations
                )
                kl = kl_divergence(old_distribution, new_distribution)
                if kl.ndim > 1:
                    kl = kl.sum(dim=-1)

                policy_loss, trigger = truly_ppo_actor_loss(
                    ratio,
                    advantages,
                    kl,
                    alpha=self.truly_alpha,
                    delta=self.truly_delta,
                )

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    clip_range_vf = self.clip_range_vf(
                        self._current_progress_remaining
                    )
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values,
                        -clip_range_vf,
                        clip_range_vf,
                    )
                value_loss = F.mse_loss(rollout_data.returns, values_pred)

                if entropy is None:
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                loss = (
                    policy_loss
                    + self.ent_coef * entropy_loss
                    + self.vf_coef * value_loss
                )
                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm
                )
                self.policy.optimizer.step()

                entropy_losses.append(float(entropy_loss.detach().cpu()))
                policy_losses.append(float(policy_loss.detach().cpu()))
                value_losses.append(float(value_loss.detach().cpu()))
                kl_values.append(float(kl.mean().detach().cpu()))
                trigger_fractions.append(float(trigger.float().mean().detach().cpu()))

            self._n_updates += 1

        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(),
            self.rollout_buffer.returns.flatten(),
        )
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(policy_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/truly_kl_mean", np.mean(kl_values))
        self.logger.record(
            "train/truly_trigger_fraction", np.mean(trigger_fractions)
        )
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
