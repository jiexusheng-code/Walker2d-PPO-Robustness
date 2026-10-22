from __future__ import annotations

import unittest

import mujoco
import numpy as np

from envs import make_env


class DynamicsTests(unittest.TestCase):
    def _floor_contact_sliding_friction(self, scale: float) -> list[float]:
        env = make_env(friction_scale=scale)
        try:
            foot_ids = {
                geom_id
                for geom_id in range(env.model.ngeom)
                if mujoco.mj_id2name(
                    env.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id
                ) in {"foot_geom", "foot_left_geom"}
            }
            env.reset(seed=5)
            action = np.zeros(env.action_space.shape, dtype=np.float32)
            for _ in range(100):
                _, _, terminated, truncated, _ = env.step(action)
                data = env.unwrapped.data
                mujoco.mj_forward(env.model, data)
                values = [
                    float(contact.friction[0])
                    for contact in data.contact[:data.ncon]
                    if env.floor_id in (contact.geom1, contact.geom2)
                    and (contact.geom1 in foot_ids or contact.geom2 in foot_ids)
                ]
                if values:
                    return values
                if terminated or truncated:
                    break
        finally:
            env.close()
        self.fail("Walker2d produced no floor contacts in 100 passive steps")

    def test_randomized_friction_changes_effective_contact_friction(self) -> None:
        low = self._floor_contact_sliding_friction(0.7)
        high = self._floor_contact_sliding_friction(1.3)
        self.assertTrue(all(np.isclose(value, 1.33) for value in low), low)
        self.assertTrue(all(np.isclose(value, 2.47) for value in high), high)

    def test_nominal_scale_preserves_model_parameters(self) -> None:
        env = make_env(friction_scale=1.0, actuator_scale=1.0)
        try:
            env.reset(seed=5)
            self.assertTrue(np.array_equal(env.model.geom_friction, env.base_friction))
            self.assertTrue(np.array_equal(env.model.actuator_gear, env.base_gear))
        finally:
            env.close()

    def test_randomization_is_reproducible_for_a_seed(self) -> None:
        first = make_env(randomize=True)
        second = make_env(randomize=True)
        try:
            first.reset(seed=23)
            second.reset(seed=23)
            self.assertTrue(
                np.array_equal(first.model.geom_friction, second.model.geom_friction)
            )
            self.assertTrue(
                np.array_equal(first.model.actuator_gear, second.model.actuator_gear)
            )
        finally:
            first.close()
            second.close()


if __name__ == "__main__":
    unittest.main()
