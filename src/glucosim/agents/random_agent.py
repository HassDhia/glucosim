"""Random baseline agent for GlucoSim environments."""

from __future__ import annotations

import numpy as np


class RandomAgent:
    """Agent that samples random actions from the action space."""

    def __init__(self, action_space, seed: int | None = None) -> None:
        self.action_space = action_space
        self._rng = np.random.default_rng(seed)

    def predict(self, obs: np.ndarray) -> np.ndarray:
        return self.action_space.sample()

    def reset(self) -> None:
        pass
