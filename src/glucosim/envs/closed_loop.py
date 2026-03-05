"""ClosedLoop-v0: Full closed-loop glucose control environment.

The agent manages total insulin delivery (basal + bolus) to maintain
blood glucose over a 48-hour stress test with variable meals.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from glucosim.models.bergman import BergmanModel
from glucosim.models.meal import MealModel
from glucosim.models.sensor import CGMSensor
from glucosim.models.patient import VirtualPatient


EPISODE_LENGTH = 2880  # 48 hours in minutes


def _glucose_reward(glucose: float, iob: float) -> float:
    """Zone-based reward with insulin stacking penalty."""
    if 70.0 <= glucose <= 180.0:
        reward = 1.0
    elif 54.0 <= glucose < 70.0:
        reward = -0.5
    elif 180.0 < glucose <= 250.0:
        reward = -0.5
    elif glucose < 54.0:
        reward = -2.0
    else:
        reward = -1.0

    # Penalize insulin stacking
    if iob > 10.0:
        reward -= 0.5

    return reward


class ClosedLoopEnv(gym.Env):
    """Gymnasium environment for full closed-loop glucose control.

    Observation (Box, shape=(6,)):
        [0] cgm_glucose: CGM reading (mg/dL), [30, 500]
        [1] insulin_on_board: estimated active insulin (U), [0, 20]
        [2] time_of_day: normalized time [0, 1]
        [3] glucose_rate: rate of glucose change (mg/dL/min), [-10, 10]
        [4] meal_announced: binary flag, [0, 1]
        [5] meal_carbs: announced carbs in grams, [0, 150]

    Action (Box, shape=(1,)):
        [0] insulin_rate: total insulin delivery (U/hr), [0, 5]

    Reward: Zone-based with IOB stacking penalty.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        difficulty: str = "easy",
        patient_type: str = "adult",
        patient_id: int = 0,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()

        self.difficulty = difficulty
        self.patient_type = patient_type
        self.patient_id = patient_id

        self.observation_space = spaces.Box(
            low=np.array([30.0, 0.0, 0.0, -10.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([500.0, 20.0, 1.0, 10.0, 1.0, 150.0], dtype=np.float32),
        )
        self.action_space = spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([5.0], dtype=np.float32),
        )

        self._model: BergmanModel | None = None
        self._meal_model: MealModel | None = None
        self._sensor: CGMSensor | None = None
        self._time: int = 0
        self._prev_glucose: float = 110.0
        self._insulin_on_board: float = 0.0
        self._meals: list[tuple[int, float]] = []
        self._meal_announced: bool = False
        self._meal_carbs: float = 0.0

    def _build_meal_schedule(self, rng: np.random.Generator) -> list[tuple[int, float]]:
        """Generate 48hr meal schedule with day-to-day variation."""
        base_meals = [
            (420, 45), (720, 70), (900, 20), (1080, 80),   # Day 1
            (1860, 50), (2160, 65), (2340, 15), (2520, 75),  # Day 2
        ]
        meals = []
        for time_min, carbs in base_meals:
            if self.difficulty == "easy":
                meals.append((time_min, carbs))
            elif self.difficulty == "medium":
                jitter = int(rng.integers(-30, 31))
                carb_var = float(rng.uniform(0.8, 1.2))
                meals.append((time_min + jitter, carbs * carb_var))
            else:
                jitter = int(rng.integers(-60, 61))
                carb_var = float(rng.uniform(0.6, 1.4))
                meals.append((time_min + jitter, carbs * carb_var))
                if rng.random() > 0.5:
                    snack_time = time_min + int(rng.integers(60, 180))
                    snack_carbs = float(rng.uniform(10, 40))
                    if snack_time < EPISODE_LENGTH:
                        meals.append((snack_time, snack_carbs))
        meals.sort(key=lambda x: x[0])
        return meals

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        rng = np.random.default_rng(seed)

        patient = VirtualPatient(
            patient_type=self.patient_type,
            patient_id=self.patient_id,
            seed=seed,
        )
        noise_coeff = 0.04 if self.difficulty == "hard" else 0.02

        self._model = BergmanModel(params=patient.params, dt=1.0)
        self._meal_model = MealModel(
            params={"BW": patient.body_weight, "VG": patient.params["VG"]}
        )
        self._sensor = CGMSensor(noise_coefficient=noise_coeff, seed=seed)
        self._sensor.reset(initial_glucose=patient.basal_glucose)

        self._time = 0
        self._prev_glucose = patient.basal_glucose
        self._insulin_on_board = 0.0
        self._meals = self._build_meal_schedule(rng)
        self._meal_announced = False
        self._meal_carbs = 0.0

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        assert self._model is not None

        insulin_rate = float(np.clip(action[0], 0.0, 5.0))

        # Check for meals
        self._meal_announced = False
        self._meal_carbs = 0.0
        for meal_time, meal_carbs in self._meals:
            if self._time == meal_time:
                self._meal_model.announce_meal(meal_carbs)
                self._meal_announced = True
                self._meal_carbs = meal_carbs

        meal_rate = self._meal_model.step(dt=1.0)
        result = self._model.step(insulin_rate=insulin_rate, meal_rate=meal_rate)

        # IOB tracking
        iob_decay = 0.997
        self._insulin_on_board = (
            self._insulin_on_board * iob_decay + insulin_rate / 60.0
        )

        cgm = self._sensor.measure(result["glucose"], dt=1.0)
        glucose_rate = cgm - self._prev_glucose
        self._prev_glucose = cgm

        self._time += 1

        reward = _glucose_reward(result["glucose"], self._insulin_on_board)
        terminated = self._time >= EPISODE_LENGTH
        truncated = False

        obs = self._get_obs()
        info = self._get_info()
        return obs, reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        if self._model is None:
            return np.zeros(6, dtype=np.float32)
        cgm = self._sensor.measure(self._model.glucose, dt=0.0)
        return np.array(
            [
                np.clip(cgm, 30.0, 500.0),
                np.clip(self._insulin_on_board, 0.0, 20.0),
                (self._time % 1440) / 1440.0,
                np.clip(cgm - self._prev_glucose, -10.0, 10.0),
                1.0 if self._meal_announced else 0.0,
                np.clip(self._meal_carbs, 0.0, 150.0),
            ],
            dtype=np.float32,
        )

    def _get_info(self) -> dict[str, Any]:
        if self._model is None:
            return {"glucose": 110.0, "insulin": 15.0, "time_minutes": 0, "in_range": True}
        g = self._model.glucose
        return {
            "glucose": g,
            "insulin": self._model.insulin,
            "time_minutes": self._time,
            "in_range": 70.0 <= g <= 180.0,
        }
