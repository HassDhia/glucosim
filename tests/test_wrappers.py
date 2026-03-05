"""Tests for environment wrappers."""

import gymnasium as gym
import numpy as np
import pytest

import glucosim  # noqa: F401
from glucosim.envs.wrappers import NormalizeRewardWrapper


class TestNormalizeRewardWrapper:
    def test_wraps_env(self):
        env = gym.make("glucosim/BasalControl-v0")
        wrapped = NormalizeRewardWrapper(env)
        obs, _ = wrapped.reset(seed=42)
        assert obs.shape == (4,)
        wrapped.close()

    def test_clips_reward(self):
        env = gym.make("glucosim/BasalControl-v0")
        wrapped = NormalizeRewardWrapper(env, low=-2.0, high=2.0)
        wrapped.reset(seed=42)
        for _ in range(100):
            _, reward, _, _, _ = wrapped.step(env.action_space.sample())
            assert -2.0 <= reward <= 2.0
        wrapped.close()

    def test_custom_bounds(self):
        env = gym.make("glucosim/BasalControl-v0")
        wrapped = NormalizeRewardWrapper(env, low=-1.0, high=1.0)
        wrapped.reset(seed=42)
        _, reward, _, _, _ = wrapped.step(np.array([0.0], dtype=np.float32))
        assert -1.0 <= reward <= 1.0
        wrapped.close()
