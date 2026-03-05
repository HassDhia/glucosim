"""Tests for baseline agents."""

import gymnasium as gym
import numpy as np
import pytest

import glucosim  # noqa: F401
from glucosim.agents.random_agent import RandomAgent
from glucosim.agents.heuristic import HeuristicBasalAgent, HeuristicBolusAgent


class TestRandomAgent:
    def test_predict_returns_action(self):
        env = gym.make("glucosim/BasalControl-v0")
        agent = RandomAgent(env.action_space, seed=42)
        obs, _ = env.reset(seed=42)
        action = agent.predict(obs)
        assert env.action_space.contains(action)
        env.close()

    def test_reset(self):
        env = gym.make("glucosim/BasalControl-v0")
        agent = RandomAgent(env.action_space)
        agent.reset()  # Should not raise
        env.close()


class TestHeuristicBasalAgent:
    def test_default_init(self):
        agent = HeuristicBasalAgent()
        assert agent.target == 120.0
        assert agent.base_rate == 1.0

    def test_predict_returns_action(self):
        agent = HeuristicBasalAgent()
        obs = np.array([130.0, 1.0, 0.5, 0.0], dtype=np.float32)
        action = agent.predict(obs)
        assert action.shape == (1,)
        assert 0.0 <= action[0] <= 3.0

    def test_high_glucose_increases_rate(self):
        agent = HeuristicBasalAgent()
        obs_high = np.array([200.0, 0.0, 0.5, 0.0], dtype=np.float32)
        obs_low = np.array([100.0, 0.0, 0.5, 0.0], dtype=np.float32)
        assert agent.predict(obs_high)[0] > agent.predict(obs_low)[0]

    def test_action_clipped(self):
        agent = HeuristicBasalAgent(gain=1.0)
        obs = np.array([500.0, 0.0, 0.5, 0.0], dtype=np.float32)
        action = agent.predict(obs)
        assert action[0] <= 3.0

    def test_reset(self):
        agent = HeuristicBasalAgent()
        agent.reset()


class TestHeuristicBolusAgent:
    def test_default_init(self):
        agent = HeuristicBolusAgent()
        assert agent.icr == 10.0

    def test_no_meal_no_bolus(self):
        agent = HeuristicBolusAgent()
        obs = np.array([120.0, 0.0, 0.0, 0.0, 60.0], dtype=np.float32)
        action = agent.predict(obs)
        assert action[0] == pytest.approx(0.0)

    def test_meal_produces_bolus(self):
        agent = HeuristicBolusAgent()
        obs = np.array([120.0, 0.0, 1.0, 60.0, 0.0], dtype=np.float32)
        action = agent.predict(obs)
        assert action[0] > 0.0

    def test_larger_meal_larger_bolus(self):
        agent = HeuristicBolusAgent()
        obs_small = np.array([120.0, 0.0, 1.0, 30.0, 0.0], dtype=np.float32)
        obs_large = np.array([120.0, 0.0, 1.0, 90.0, 0.0], dtype=np.float32)
        assert agent.predict(obs_large)[0] > agent.predict(obs_small)[0]

    def test_action_clipped(self):
        agent = HeuristicBolusAgent(icr=1.0)
        obs = np.array([300.0, 0.0, 1.0, 150.0, 0.0], dtype=np.float32)
        action = agent.predict(obs)
        assert action[0] <= 20.0

    def test_reset(self):
        agent = HeuristicBolusAgent()
        agent.reset()
