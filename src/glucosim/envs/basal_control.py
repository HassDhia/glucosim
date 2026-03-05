"""BasalControl-v0: Agent optimizes continuous basal insulin delivery rate.

The agent learns to adjust basal insulin delivery to maintain blood glucose
within the target range (70-180 mg/dL) over a 24-hour simulation.
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


# Default meal schedule: (time_in_minutes, carbs_grams)
DEFAULT_MEALS = [
    (420, 45),   # Breakfast 7:00am - 45g
    (720, 70),   # Lunch 12:00pm - 70g
    (900, 20),   # Snack 3:00pm - 20g
    (1080, 80),  # Dinner 6:00pm - 80g
]

EPISODE_LENGTH = 1440  # 24 hours in minutes


def _glucose_reward(glucose: float) -> float:
    """Zone-based reward for blood glucose level."""
    if 70.0 <= glucose <= 180.0:
        return 1.0
    elif 54.0 <= glucose < 70.0:
        return -0.5
    elif 180.0 < glucose <= 250.0:
        return -0.5
    elif glucose < 54.0:
        return -2.0
    else:
        return -1.0


class BasalControlEnv(gym.Env):
    """Gymnasium environment for basal insulin rate optimization.

    Observation (Box, shape=(4,)):
        [0] cgm_glucose: CGM reading (mg/dL), [30, 500]
        [1] insulin_on_board: estimated active insulin (U), [0, 20]
        [2] time_of_day: normalized time [0, 1]
        [3] glucose_rate: rate of glucose change (mg/dL/min), [-10, 10]

    Action (Box, shape=(1,)):
        [0] basal_rate: insulin delivery rate (U/hr), [0, 3]

    Reward: Zone-based per-step reward (see _glucose_reward).
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
            low=np.array([30.0, 0.0, 0.0, -10.0], dtype=np.float32),
            high=np.array([500.0, 20.0, 1.0, 10.0], dtype=np.float32),
        )
        self.action_space = spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([3.0], dtype=np.float32),
        )

        # Will be initialized on reset
        self._model: BergmanModel | None = None
        self._meal_model: MealModel | None = None
        self._sensor: CGMSensor | None = None
        self._time: int = 0
        self._prev_glucose: float = 110.0
        self._insulin_on_board: float = 0.0
        self._meals: list[tuple[int, float]] = []

    def _build_meal_schedule(self, rng: np.random.Generator) -> list[tuple[int, float]]:
        meals = []
        for time_min, carbs in DEFAULT_MEALS:
            if self.difficulty == "easy":
                meals.append((time_min, carbs))
            elif self.difficulty == "medium":
                jitter = int(rng.integers(-30, 31))
                carb_var = float(rng.uniform(0.8, 1.2))
                meals.append((time_min + jitter, carbs * carb_var))
            else:  # hard
                jitter = int(rng.integers(-60, 61))
                carb_var = float(rng.uniform(0.6, 1.4))
                meals.append((time_min + jitter, carbs * carb_var))
                # Extra random snack
                if rng.random() > 0.5:
                    snack_time = int(rng.integers(480, 1320))
                    snack_carbs = float(rng.uniform(10, 40))
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
        self._sensor = CGMSensor(
            noise_coefficient=noise_coeff, seed=seed
        )
        self._sensor.reset(initial_glucose=patient.basal_glucose)

        self._time = 0
        self._prev_glucose = patient.basal_glucose
        self._insulin_on_board = 0.0
        self._meals = self._build_meal_schedule(rng)

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        assert self._model is not None, "Call reset() before step()"

        basal_rate = float(np.clip(action[0], 0.0, 3.0))

        # Check if a meal happens at this time step
        for meal_time, meal_carbs in self._meals:
            if self._time == meal_time:
                self._meal_model.announce_meal(meal_carbs)

        # Get meal glucose appearance rate
        meal_rate = self._meal_model.step(dt=1.0)

        # Advance physiological model
        result = self._model.step(insulin_rate=basal_rate, meal_rate=meal_rate)

        # Update insulin on board (simple exponential decay model)
        # SIMPLIFICATION: IOB modeled as exponential decay with 4hr half-life.
        # Clinical IOB curves are non-exponential and insulin-type-specific.
        iob_decay = 0.997  # ~4hr half-life
        self._insulin_on_board = (
            self._insulin_on_board * iob_decay + basal_rate / 60.0
        )

        # Sensor reading
        cgm = self._sensor.measure(result["glucose"], dt=1.0)

        glucose_rate = cgm - self._prev_glucose
        self._prev_glucose = cgm

        self._time += 1

        reward = _glucose_reward(result["glucose"])
        terminated = self._time >= EPISODE_LENGTH
        truncated = False

        obs = self._get_obs()
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        if self._model is None:
            return np.zeros(4, dtype=np.float32)
        cgm = self._sensor.measure(self._model.glucose, dt=0.0)
        return np.array(
            [
                np.clip(cgm, 30.0, 500.0),
                np.clip(self._insulin_on_board, 0.0, 20.0),
                self._time / EPISODE_LENGTH,
                np.clip(cgm - self._prev_glucose, -10.0, 10.0),
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
