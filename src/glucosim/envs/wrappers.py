"""SB3-compatible wrappers for GlucoSim environments."""

from __future__ import annotations

import gymnasium as gym
import numpy as np


class NormalizeRewardWrapper(gym.RewardWrapper):
    """Clips rewards to a fixed range for training stability."""

    def __init__(self, env: gym.Env, low: float = -3.0, high: float = 3.0) -> None:
        super().__init__(env)
        self._low = low
        self._high = high

    def reward(self, reward: float) -> float:
        return float(np.clip(reward, self._low, self._high))
