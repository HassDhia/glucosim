"""Bergman Minimal Model of glucose-insulin dynamics.

Implements the classic three-equation system from Bergman et al. (1979)
for glucose disappearance and insulin kinetics, solved via RK4 integration.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# Default parameters for a typical adult subject
DEFAULT_PARAMS: dict[str, float] = {
    "p1": 0.028,       # glucose effectiveness (min^-1)
    "p2": 0.025,       # insulin action decay rate (min^-1)
    "p3": 0.000013,    # insulin action gain (min^-2 per mU/L)
    "n": 0.23,         # insulin clearance rate (min^-1)
    "gamma": 0.004,    # pancreatic responsivity (mU/L/min per mg/dL)
    "h": 110.0,        # glucose threshold for insulin secretion (mg/dL)
    "Gb": 110.0,       # basal glucose (mg/dL)
    "Ib": 15.0,        # basal insulin (mU/L)
    "VG": 1.49,        # glucose distribution volume (dL/kg)
    "VI": 0.04,        # insulin distribution volume (L/kg)
    "body_weight": 70.0,  # patient weight (kg)
}


class BergmanModel:
    """Bergman minimal model for glucose-insulin dynamics.

    State variables:
        G - plasma glucose concentration (mg/dL)
        X - insulin action (remote insulin effect, min^-1)
        I - plasma insulin concentration (mU/L)
    """

    def __init__(self, params: dict[str, float] | None = None, dt: float = 1.0) -> None:
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.dt = dt  # time step in minutes

        # State variables
        self._glucose: float = self.params["Gb"]
        self._insulin_action: float = 0.0
        self._insulin: float = self.params["Ib"]

    @property
    def glucose(self) -> float:
        return self._glucose

    @property
    def insulin(self) -> float:
        return self._insulin

    @property
    def insulin_action(self) -> float:
        return self._insulin_action

    def reset(
        self, glucose: float | None = None, insulin: float | None = None
    ) -> None:
        self._glucose = glucose if glucose is not None else self.params["Gb"]
        self._insulin = insulin if insulin is not None else self.params["Ib"]
        self._insulin_action = 0.0

    def _derivatives(
        self,
        state: NDArray[np.float64],
        insulin_rate: float,
        meal_rate: float,
    ) -> NDArray[np.float64]:
        """Compute derivatives for the three-equation system.

        Args:
            state: [G, X, I] current state vector
            insulin_rate: exogenous insulin delivery rate (U/hr)
            meal_rate: glucose appearance rate from meals (mg/dL/min)
        """
        G, X, I = state
        p = self.params

        # Convert insulin rate from U/hr to mU/L/min
        # SIMPLIFICATION: Direct conversion assuming instantaneous subcutaneous absorption.
        # Clinical insulin has delayed absorption (peak at 60-90 min for rapid-acting).
        insulin_input = (insulin_rate * 1000.0) / (60.0 * p["VI"] * p["body_weight"])

        # Glucose dynamics
        dG = -(p["p1"] + X) * G + p["p1"] * p["Gb"] + meal_rate

        # Insulin action dynamics (remote compartment)
        dX = -p["p2"] * X + p["p3"] * (I - p["Ib"])

        # Insulin dynamics with endogenous secretion
        # Clearance acts on deviation from basal (I - Ib), so at basal state
        # dI = 0 when G <= h. Basal insulin is maintained by background
        # secretion implicit in the Ib term.
        secretion = p["gamma"] * max(0.0, G - p["h"])
        dI = -p["n"] * (I - p["Ib"]) + secretion + insulin_input

        return np.array([dG, dX, dI])

    def step(self, insulin_rate: float = 0.0, meal_rate: float = 0.0) -> dict[str, float]:
        """Advance the model by one time step using RK4 integration.

        Args:
            insulin_rate: exogenous insulin delivery rate (U/hr)
            meal_rate: glucose appearance rate from meals (mg/dL/min)

        Returns:
            Dictionary with current glucose, insulin, and insulin_action values.
        """
        state = np.array([self._glucose, self._insulin_action, self._insulin])

        # RK4 integration
        k1 = self._derivatives(state, insulin_rate, meal_rate)
        k2 = self._derivatives(state + 0.5 * self.dt * k1, insulin_rate, meal_rate)
        k3 = self._derivatives(state + 0.5 * self.dt * k2, insulin_rate, meal_rate)
        k4 = self._derivatives(state + self.dt * k3, insulin_rate, meal_rate)

        state = state + (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        # Enforce physiological bounds
        self._glucose = float(np.clip(state[0], 10.0, 600.0))
        self._insulin_action = float(max(0.0, state[1]))
        self._insulin = float(max(0.0, state[2]))

        return {
            "glucose": self._glucose,
            "insulin": self._insulin,
            "insulin_action": self._insulin_action,
        }
