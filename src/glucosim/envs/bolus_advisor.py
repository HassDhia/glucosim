"""BolusAdvisor-v0: Agent decides meal bolus insulin doses.

The agent receives meal announcements and decides the appropriate
bolus insulin dose to manage postprandial glucose excursions.
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


DEFAULT_MEALS = [
    (420, 45),
    (720, 70),
    (900, 20),
    (1080, 80),
]

EPISODE_LENGTH = 1440


def _glucose_reward(glucose: float, postprandial: bool) -> float:
    """Zone-based reward with postprandial multiplier."""
    multiplier = 2.0 if postprandial else 1.0
    if 70.0 <= glucose <= 180.0:
        return 1.0 * multiplier
    elif 54.0 <= glucose < 70.0:
        return -0.5 * multiplier
    elif 180.0 < glucose <= 250.0:
        return -0.5 * multiplier
    elif glucose < 54.0:
        return -2.0 * multiplier
    else:
        return -1.0 * multiplier


class BolusAdvisorEnv(gym.Env):
    """Gymnasium environment for meal bolus insulin dosing.

    Observation (Box, shape=(5,)):
        [0] cgm_glucose: CGM reading (mg/dL), [30, 500]
        [1] insulin_on_board: estimated active insulin (U), [0, 20]
        [2] meal_announced: binary flag, [0, 1]
        [3] meal_carbs: announced carbs in grams, [0, 150]
        [4] time_since_meal: minutes since last meal, [0, 480]

    Action (Box, shape=(1,)):
        [0] bolus_dose: insulin bolus in Units, [0, 20]

    Reward: Zone-based with 2x multiplier in 2hr postprandial window.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        difficulty: str = "easy",
        patient_type: str = "adult",
        patient_id: int = 0,
        fixed_basal_rate: float = 1.0,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()

        self.difficulty = difficulty
        self.patient_type = patient_type
        self.patient_id = patient_id
        self.fixed_basal_rate = fixed_basal_rate

        self.observation_space = spaces.Box(
            low=np.array([30.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([500.0, 20.0, 1.0, 150.0, 480.0], dtype=np.float32),
        )
        self.action_space = spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([20.0], dtype=np.float32),
        )

        self._model: BergmanModel | None = None
        self._meal_model: MealModel | None = None
        self._sensor: CGMSensor | None = None
        self._time: int = 0
        self._insulin_on_board: float = 0.0
        self._meals: list[tuple[int, float]] = []
        self._meal_announced: bool = False
        self._meal_carbs: float = 0.0
        self._last_meal_time: int = -480

    def _build_meal_schedule(self, rng: np.random.Generator) -> list[tuple[int, float]]:
        meals = []
        for time_min, carbs in DEFAULT_MEALS:
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

        self._model = BergmanModel(params=patient.params, dt=1.0)
        self._meal_model = MealModel(
            params={"BW": patient.body_weight, "VG": patient.params["VG"]}
        )
        self._sensor = CGMSensor(seed=seed)
        self._sensor.reset(initial_glucose=patient.basal_glucose)

        self._time = 0
        self._insulin_on_board = 0.0
        self._meals = self._build_meal_schedule(rng)
        self._meal_announced = False
        self._meal_carbs = 0.0
        self._last_meal_time = -480

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        assert self._model is not None

        bolus_dose = float(np.clip(action[0], 0.0, 20.0))

        # Check for meal at this time step
        self._meal_announced = False
        self._meal_carbs = 0.0
        for meal_time, meal_carbs in self._meals:
            if self._time == meal_time:
                self._meal_model.announce_meal(meal_carbs)
                self._meal_announced = True
                self._meal_carbs = meal_carbs
                self._last_meal_time = self._time

        # Convert bolus to rate: apply bolus over 5 minutes
        bolus_rate = bolus_dose * 12.0 if self._meal_announced else 0.0
        total_rate = self.fixed_basal_rate + bolus_rate

        meal_rate = self._meal_model.step(dt=1.0)
        result = self._model.step(insulin_rate=total_rate, meal_rate=meal_rate)

        # IOB tracking
        iob_decay = 0.997
        delivered = total_rate / 60.0
        self._insulin_on_board = self._insulin_on_board * iob_decay + delivered

        self._sensor.measure(result["glucose"], dt=1.0)
        self._time += 1

        # Postprandial window: 2 hours after last meal
        postprandial = (self._time - self._last_meal_time) <= 120
        reward = _glucose_reward(result["glucose"], postprandial)

        terminated = self._time >= EPISODE_LENGTH
        truncated = False

        obs = self._get_obs()
        info = self._get_info()
        return obs, reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        if self._model is None:
            return np.zeros(5, dtype=np.float32)
        cgm = self._sensor.measure(self._model.glucose, dt=0.0)
        time_since = self._time - self._last_meal_time
        return np.array(
            [
                np.clip(cgm, 30.0, 500.0),
                np.clip(self._insulin_on_board, 0.0, 20.0),
                1.0 if self._meal_announced else 0.0,
                np.clip(self._meal_carbs, 0.0, 150.0),
                np.clip(float(time_since), 0.0, 480.0),
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
