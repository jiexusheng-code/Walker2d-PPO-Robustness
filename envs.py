from __future__ import annotations

import numpy as np
import gymnasium as gym
import mujoco


DEFAULT_FRICTION_RANGE = (0.8, 1.2)
DEFAULT_ACTUATOR_RANGE = (0.9, 1.1)


class DynamicsScaleWrapper(gym.Wrapper):
    """Scale all MuJoCo geom friction and actuator gears at episode resets."""

    def __init__(
        self,
        env: gym.Env,
        *,
        randomize: bool = False,
        friction_range: tuple[float, float] = DEFAULT_FRICTION_RANGE,
        actuator_range: tuple[float, float] = DEFAULT_ACTUATOR_RANGE,
        friction_scale: float = 1.0,
        actuator_scale: float = 1.0,
    ) -> None:
        super().__init__(env)
        self.model = self.unwrapped.model
        self.base_friction = self.model.geom_friction.copy()
        self.base_gear = self.model.actuator_gear.copy()
        self.floor_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor"
        )
        if self.floor_id < 0:
            raise ValueError("Walker2d model has no geom named 'floor'")
        self.randomize = randomize
        self.friction_range = friction_range
        self.actuator_range = actuator_range
        self.friction_scale = friction_scale
        self.actuator_scale = actuator_scale
        self.rng = np.random.default_rng()

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(
                np.random.SeedSequence([seed, 0x5EED])
            )

        friction_scale = self.friction_scale
        actuator_scale = self.actuator_scale
        if self.randomize:
            friction_scale = self.rng.uniform(*self.friction_range)
            actuator_scale = self.rng.uniform(*self.actuator_range)

        self.model.geom_friction[:] = self.base_friction * friction_scale
        self.model.actuator_gear[:] = self.base_gear * actuator_scale
        observation, info = self.env.reset(seed=seed, options=options)
        info = dict(info or {})
        info["geom_friction_scale"] = float(friction_scale)
        info["actuator_gear_scale"] = float(actuator_scale)
        return observation, info


def make_env(
    *,
    randomize: bool = False,
    friction_scale: float = 1.0,
    actuator_scale: float = 1.0,
) -> gym.Env:
    env = gym.make("Walker2d-v5")
    return DynamicsScaleWrapper(
        env,
        randomize=randomize,
        friction_scale=friction_scale,
        actuator_scale=actuator_scale,
    )
