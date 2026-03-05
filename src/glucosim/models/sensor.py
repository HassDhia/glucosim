"""Continuous Glucose Monitor (CGM) sensor model.

Simulates measurement noise and time lag typical of commercial CGM devices.
"""

from __future__ import annotations

import numpy as np


class CGMSensor:
    """CGM sensor model with Gaussian noise and first-order lag.

    Commercial CGM sensors have two primary imperfections:
    1. Measurement noise (typically 2-5% coefficient of variation)
    2. Physiological lag from interstitial fluid sampling (5-15 minutes)

    This model captures both effects with configurable parameters.
    """

    def __init__(
        self,
        noise_coefficient: float = 0.02,
        lag_minutes: float = 10.0,
        sample_interval: float = 5.0,
        seed: int | None = None,
    ) -> None:
        """Initialize CGM sensor.

        Args:
            noise_coefficient: Coefficient of variation for Gaussian noise
            lag_minutes: First-order lag time constant in minutes
            sample_interval: Time between CGM samples in minutes
            seed: Random seed for reproducibility
        """
        self.noise_coefficient = noise_coefficient
        self.lag_minutes = lag_minutes
        self.sample_interval = sample_interval
        self._rng = np.random.default_rng(seed)
        self._lagged_glucose: float = 110.0
        self._last_reading: float = 110.0
        self._time_since_sample: float = 0.0

    def reset(self, initial_glucose: float = 110.0) -> None:
        self._lagged_glucose = initial_glucose
        self._last_reading = initial_glucose
        self._time_since_sample = 0.0

    def measure(self, true_glucose: float, dt: float = 1.0) -> float:
        """Return a noisy, lagged CGM reading.

        Args:
            true_glucose: Actual plasma glucose concentration (mg/dL)
            dt: Time step in minutes since last call

        Returns:
            CGM glucose reading (mg/dL) with noise and lag applied
        """
        # Apply first-order lag filter
        # SIMPLIFICATION: Using exponential smoothing as proxy for
        # interstitial fluid diffusion dynamics. Clinical models use
        # two-compartment diffusion equations.
        alpha = 1.0 - np.exp(-dt / self.lag_minutes)
        self._lagged_glucose += alpha * (true_glucose - self._lagged_glucose)

        # Update sample timer
        self._time_since_sample += dt

        # Only produce a new reading at the sample interval
        if self._time_since_sample >= self.sample_interval:
            self._time_since_sample = 0.0
            noise = self._rng.normal(
                0.0, self.noise_coefficient * self._lagged_glucose
            )
            self._last_reading = float(
                np.clip(self._lagged_glucose + noise, 30.0, 500.0)
            )

        return self._last_reading
