"""Heuristic baseline agents implementing clinical rules.

These agents encode simplified versions of standard clinical protocols
for insulin delivery in Type 1 diabetes management.
"""

from __future__ import annotations

import numpy as np


class HeuristicBasalAgent:
    """Proportional basal rate controller.

    Implements a simplified proportional control law that adjusts basal
    insulin delivery based on deviation from a target glucose level.
    This mirrors the basic logic of clinical basal rate adjustments.
    """

    def __init__(
        self,
        target_glucose: float = 120.0,
        base_rate: float = 1.0,
        gain: float = 0.005,
    ) -> None:
        self.target = target_glucose
        self.base_rate = base_rate
        self.gain = gain

    def predict(self, obs: np.ndarray) -> np.ndarray:
        cgm = obs[0]
        error = cgm - self.target
        rate = self.base_rate + self.gain * error
        return np.array([np.clip(rate, 0.0, 3.0)], dtype=np.float32)

    def reset(self) -> None:
        pass


class HeuristicBolusAgent:
    """Carbohydrate-ratio bolus calculator.

    Implements the standard insulin-to-carb ratio (ICR) method used
    clinically for meal bolus calculations, with a correction factor
    for current glucose deviation from target.
    """

    def __init__(
        self,
        icr: float = 10.0,
        correction_factor: float = 50.0,
        target_glucose: float = 120.0,
    ) -> None:
        """Initialize bolus calculator.

        Args:
            icr: Insulin-to-carb ratio (grams per unit)
            correction_factor: mg/dL drop per unit of insulin
            target_glucose: Target glucose for corrections
        """
        self.icr = icr
        self.correction_factor = correction_factor
        self.target = target_glucose

    def predict(self, obs: np.ndarray) -> np.ndarray:
        cgm = obs[0]
        meal_announced = obs[2]
        meal_carbs = obs[3]

        bolus = 0.0
        if meal_announced > 0.5 and meal_carbs > 0.0:
            # Carb coverage
            carb_bolus = meal_carbs / self.icr
            # Correction
            correction = max(0.0, (cgm - self.target) / self.correction_factor)
            bolus = carb_bolus + correction

        return np.array([np.clip(bolus, 0.0, 20.0)], dtype=np.float32)

    def reset(self) -> None:
        pass
