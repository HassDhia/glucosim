"""PPO agent wrapper for GlucoSim using Stable Baselines3.

Provides a unified interface for training and evaluating PPO agents
across all GlucoSim environments.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


class PPOAgent:
    """Wrapper around SB3 PPO for GlucoSim environments.

    Handles training, evaluation, and model saving/loading with
    a consistent interface matching RandomAgent and Heuristic agents.
    """

    def __init__(self, model=None) -> None:
        self._model = model

    @classmethod
    def train(
        cls,
        env_id: str,
        total_timesteps: int = 100000,
        seed: int = 42,
        env_kwargs: dict[str, Any] | None = None,
        **ppo_kwargs: Any,
    ) -> "PPOAgent":
        """Train a PPO agent on the given environment.

        Args:
            env_id: Gymnasium environment ID (e.g. "glucosim/BasalControl-v0")
            total_timesteps: Total training steps
            seed: Random seed
            env_kwargs: Keyword arguments for the environment
            **ppo_kwargs: Additional PPO hyperparameters

        Returns:
            Trained PPOAgent instance
        """
        try:
            import gymnasium as gym
            from stable_baselines3 import PPO
            from stable_baselines3.common.env_util import make_vec_env
        except ImportError:
            raise ImportError(
                "Training requires stable-baselines3 and torch. "
                "Install with: pip install glucosim[train]"
            )

        env_kwargs = env_kwargs or {}
        env = make_vec_env(env_id, n_envs=1, seed=seed, env_kwargs=env_kwargs)

        defaults = {
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "verbose": 1,
        }
        defaults.update(ppo_kwargs)

        model = PPO("MlpPolicy", env, seed=seed, **defaults)
        model.learn(total_timesteps=total_timesteps)

        return cls(model=model)

    def predict(self, obs: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("No model loaded. Train or load a model first.")
        action, _ = self._model.predict(obs, deterministic=True)
        return action

    def save(self, path: str | Path) -> None:
        if self._model is None:
            raise RuntimeError("No model to save.")
        self._model.save(str(path))

    @classmethod
    def load(cls, path: str | Path) -> "PPOAgent":
        try:
            from stable_baselines3 import PPO
        except ImportError:
            raise ImportError("Loading requires stable-baselines3.")
        model = PPO.load(str(path))
        return cls(model=model)

    def reset(self) -> None:
        pass


def main() -> None:
    """CLI entry point for training PPO agents on all environments."""
    import gymnasium as gym
    import glucosim  # noqa: F401

    envs = [
        ("glucosim/BasalControl-v0", 100000),
        ("glucosim/BolusAdvisor-v0", 100000),
        ("glucosim/ClosedLoop-v0", 150000),
    ]

    results = {}
    for env_id, steps in envs:
        print(f"\nTraining PPO on {env_id} for {steps} steps...")
        agent = PPOAgent.train(env_id, total_timesteps=steps, seed=42)

        # Save model
        name = env_id.split("/")[1].replace("-v0", "").lower()
        save_path = Path("checkpoints") / f"ppo_{name}"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        agent.save(save_path)

        # Evaluate
        env = gym.make(env_id)
        total_reward = 0.0
        episodes = 5
        for ep in range(episodes):
            obs, _ = env.reset(seed=ep)
            ep_reward = 0.0
            done = False
            while not done:
                action = agent.predict(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += reward
                done = terminated or truncated
            total_reward += ep_reward
        env.close()

        avg_reward = total_reward / episodes
        results[env_id] = {
            "mean_reward": avg_reward,
            "training_steps": steps,
            "model_path": str(save_path),
        }
        print(f"  Mean reward over {episodes} episodes: {avg_reward:.2f}")

    # Save results
    out_path = Path("results/training_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
