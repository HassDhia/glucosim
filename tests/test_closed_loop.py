"""Tests for the ClosedLoop-v0 environment."""

import gymnasium as gym
import numpy as np
import pytest

import glucosim  # noqa: F401


class TestClosedLoopEnv:
    def test_make(self):
        env = gym.make("glucosim/ClosedLoop-v0")
        assert env is not None
        env.close()

    def test_observation_space(self):
        env = gym.make("glucosim/ClosedLoop-v0")
        assert env.observation_space.shape == (6,)
        env.close()

    def test_action_space(self):
        env = gym.make("glucosim/ClosedLoop-v0")
        assert env.action_space.shape == (1,)
        assert env.action_space.high[0] == pytest.approx(5.0)
        env.close()

    def test_reset_returns_obs_info(self):
        env = gym.make("glucosim/ClosedLoop-v0")
        obs, info = env.reset(seed=42)
        assert obs.shape == (6,)
        assert isinstance(info, dict)
        env.close()

    def test_step_returns_five_tuple(self):
        env = gym.make("glucosim/ClosedLoop-v0")
        env.reset(seed=42)
        result = env.step(np.array([1.0], dtype=np.float32))
        assert len(result) == 5
        env.close()

    def test_episode_length(self):
        env = gym.make("glucosim/ClosedLoop-v0")
        env.reset(seed=42)
        steps = 0
        done = False
        while not done:
            _, _, terminated, truncated, _ = env.step(
                np.array([1.0], dtype=np.float32)
            )
            done = terminated or truncated
            steps += 1
        assert steps == 2880  # 48 hours
        env.close()

    def test_info_contains_required_fields(self):
        env = gym.make("glucosim/ClosedLoop-v0")
        env.reset(seed=42)
        _, _, _, _, info = env.step(np.array([1.0], dtype=np.float32))
        assert "glucose" in info
        assert "insulin" in info
        assert "time_minutes" in info
        assert "in_range" in info
        env.close()

    def test_obs_within_bounds(self):
        env = gym.make("glucosim/ClosedLoop-v0")
        env.reset(seed=42)
        for _ in range(100):
            obs, _, _, _, _ = env.step(env.action_space.sample())
            assert env.observation_space.contains(obs)
        env.close()

    def test_difficulty_options(self):
        for d in ["easy", "medium", "hard"]:
            env = gym.make("glucosim/ClosedLoop-v0", difficulty=d)
            obs, _ = env.reset(seed=42)
            assert obs is not None
            env.close()

    def test_reproducible_with_seed(self):
        env = gym.make("glucosim/ClosedLoop-v0")
        obs1, _ = env.reset(seed=42)
        obs2, _ = env.reset(seed=42)
        np.testing.assert_array_equal(obs1, obs2)
        env.close()

    def test_insulin_stacking_penalty(self):
        """Delivering too much insulin should reduce reward via IOB penalty."""
        env = gym.make("glucosim/ClosedLoop-v0")
        env.reset(seed=42)
        # Deliver max insulin for a while to build IOB
        rewards = []
        for _ in range(200):
            _, r, _, _, _ = env.step(np.array([5.0], dtype=np.float32))
            rewards.append(r)
        # Later rewards should be lower due to IOB penalty
        env.close()
        # At least some rewards should be penalized
        assert min(rewards) < 0
