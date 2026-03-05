"""Integration tests: full episode runs with agents."""

import gymnasium as gym
import numpy as np
import pytest

import glucosim  # noqa: F401
from glucosim.agents.random_agent import RandomAgent
from glucosim.agents.heuristic import HeuristicBasalAgent, HeuristicBolusAgent


class TestFullEpisodeBasal:
    def test_random_agent_completes_episode(self):
        env = gym.make("glucosim/BasalControl-v0")
        agent = RandomAgent(env.action_space, seed=42)
        obs, _ = env.reset(seed=42)
        total_reward = 0.0
        done = False
        while not done:
            action = agent.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
        env.close()
        assert isinstance(total_reward, float)

    def test_heuristic_agent_completes_episode(self):
        env = gym.make("glucosim/BasalControl-v0")
        agent = HeuristicBasalAgent()
        obs, _ = env.reset(seed=42)
        total_reward = 0.0
        done = False
        while not done:
            action = agent.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
        env.close()
        assert isinstance(total_reward, float)

    def test_heuristic_beats_random(self):
        """Heuristic should outperform random on easy difficulty."""
        env_id = "glucosim/BasalControl-v0"
        seeds = [42, 43, 44]

        def run_agent(agent_cls, is_random=False, **kwargs):
            total = 0.0
            for s in seeds:
                env = gym.make(env_id, difficulty="easy")
                agent = agent_cls(env.action_space, seed=s) if is_random else agent_cls(**kwargs)
                obs, _ = env.reset(seed=s)
                done = False
                while not done:
                    action = agent.predict(obs)
                    obs, r, terminated, truncated, _ = env.step(action)
                    total += r
                    done = terminated or truncated
                env.close()
            return total / len(seeds)

        random_reward = run_agent(RandomAgent, is_random=True)
        heuristic_reward = run_agent(HeuristicBasalAgent)
        assert heuristic_reward > random_reward


class TestFullEpisodeBolus:
    def test_random_agent_completes_episode(self):
        env = gym.make("glucosim/BolusAdvisor-v0")
        agent = RandomAgent(env.action_space, seed=42)
        obs, _ = env.reset(seed=42)
        done = False
        while not done:
            action = agent.predict(obs)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        env.close()

    def test_heuristic_agent_completes_episode(self):
        env = gym.make("glucosim/BolusAdvisor-v0")
        agent = HeuristicBolusAgent()
        obs, _ = env.reset(seed=42)
        done = False
        while not done:
            action = agent.predict(obs)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        env.close()


class TestFullEpisodeClosedLoop:
    def test_random_agent_completes_episode(self):
        env = gym.make("glucosim/ClosedLoop-v0")
        agent = RandomAgent(env.action_space, seed=42)
        obs, _ = env.reset(seed=42)
        done = False
        while not done:
            action = agent.predict(obs)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        env.close()

    def test_heuristic_agent_completes_episode(self):
        env = gym.make("glucosim/ClosedLoop-v0")
        agent = HeuristicBasalAgent(base_rate=1.5)
        obs, _ = env.reset(seed=42)
        done = False
        while not done:
            # Pad obs to match expectation (heuristic only uses first element)
            action = agent.predict(obs[:4])
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        env.close()


class TestPatientVariation:
    def test_different_patients_different_outcomes(self):
        """Different virtual patients should produce different trajectories."""
        rewards = []
        for pid in range(3):
            env = gym.make("glucosim/BasalControl-v0", patient_id=pid)
            obs, _ = env.reset(seed=pid + 10)
            total = 0.0
            for _ in range(200):
                obs, r, _, _, _ = env.step(np.array([1.0], dtype=np.float32))
                total += r
            rewards.append(total)
            env.close()
        # At least some variation across patients
        assert max(rewards) - min(rewards) > 1.0
