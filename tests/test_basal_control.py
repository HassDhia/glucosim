"""Tests for the BasalControl-v0 environment."""

import gymnasium as gym
import numpy as np
import pytest

import glucosim  # noqa: F401


class TestBasalControlEnv:
    def test_make(self):
        env = gym.make("glucosim/BasalControl-v0")
        assert env is not None
        env.close()

    def test_observation_space(self):
        env = gym.make("glucosim/BasalControl-v0")
        assert env.observation_space.shape == (4,)
        env.close()

    def test_action_space(self):
        env = gym.make("glucosim/BasalControl-v0")
        assert env.action_space.shape == (1,)
        env.close()

    def test_reset_returns_obs_info(self):
        env = gym.make("glucosim/BasalControl-v0")
        obs, info = env.reset(seed=42)
        assert obs.shape == (4,)
        assert isinstance(info, dict)
        env.close()

    def test_step_returns_five_tuple(self):
        env = gym.make("glucosim/BasalControl-v0")
        env.reset(seed=42)
        action = env.action_space.sample()
        result = env.step(action)
        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        assert obs.shape == (4,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        env.close()

    def test_episode_terminates(self):
        env = gym.make("glucosim/BasalControl-v0")
        env.reset(seed=42)
        done = False
        steps = 0
        while not done:
            obs, reward, terminated, truncated, info = env.step(
                np.array([1.0], dtype=np.float32)
            )
            done = terminated or truncated
            steps += 1
        assert steps == 1440  # 24 hours
        env.close()

    def test_info_contains_required_fields(self):
        env = gym.make("glucosim/BasalControl-v0")
        env.reset(seed=42)
        _, _, _, _, info = env.step(np.array([1.0], dtype=np.float32))
        assert "glucose" in info
        assert "insulin" in info
        assert "time_minutes" in info
        assert "in_range" in info
        env.close()

    def test_obs_within_bounds(self):
        env = gym.make("glucosim/BasalControl-v0")
        env.reset(seed=42)
        for _ in range(100):
            obs, _, _, _, _ = env.step(env.action_space.sample())
            assert env.observation_space.contains(obs)
        env.close()

    def test_difficulty_easy(self):
        env = gym.make("glucosim/BasalControl-v0", difficulty="easy")
        obs, _ = env.reset(seed=42)
        assert obs is not None
        env.close()

    def test_difficulty_medium(self):
        env = gym.make("glucosim/BasalControl-v0", difficulty="medium")
        obs, _ = env.reset(seed=42)
        assert obs is not None
        env.close()

    def test_difficulty_hard(self):
        env = gym.make("glucosim/BasalControl-v0", difficulty="hard")
        obs, _ = env.reset(seed=42)
        assert obs is not None
        env.close()

    def test_reproducible_with_seed(self):
        env = gym.make("glucosim/BasalControl-v0")
        obs1, _ = env.reset(seed=42)
        obs2, _ = env.reset(seed=42)
        np.testing.assert_array_equal(obs1, obs2)
        env.close()

    def test_different_seeds_different_obs(self):
        env = gym.make("glucosim/BasalControl-v0")
        obs1, _ = env.reset(seed=1)
        obs2, _ = env.reset(seed=2)
        assert not np.array_equal(obs1, obs2)
        env.close()
