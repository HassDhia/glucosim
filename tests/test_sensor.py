"""Tests for the CGM sensor model."""

import numpy as np
import pytest

from glucosim.models.sensor import CGMSensor


class TestSensorInit:
    def test_default_init(self):
        s = CGMSensor(seed=42)
        reading = s.measure(110.0)
        assert 30.0 <= reading <= 500.0

    def test_custom_noise(self):
        s = CGMSensor(noise_coefficient=0.10, seed=42)
        assert s.noise_coefficient == 0.10


class TestSensorMeasure:
    def test_returns_float(self):
        s = CGMSensor(seed=42)
        reading = s.measure(120.0)
        assert isinstance(reading, float)

    def test_bounded_output(self):
        s = CGMSensor(seed=42)
        for g in [30.0, 100.0, 300.0, 500.0]:
            reading = s.measure(g)
            assert 30.0 <= reading <= 500.0

    def test_low_glucose_clipped(self):
        s = CGMSensor(seed=42)
        reading = s.measure(10.0)
        assert reading >= 30.0

    def test_noise_present(self):
        s = CGMSensor(noise_coefficient=0.05, seed=42)
        readings = []
        for _ in range(20):
            readings.append(s.measure(120.0, dt=5.0))
        assert len(set(readings)) > 1  # Not all identical

    def test_zero_noise(self):
        s = CGMSensor(noise_coefficient=0.0, seed=42)
        # After lag settles
        for _ in range(100):
            s.measure(150.0, dt=5.0)
        reading = s.measure(150.0, dt=5.0)
        assert abs(reading - 150.0) < 1.0

    def test_lag_effect(self):
        """Sensor should lag behind a sudden glucose change."""
        s = CGMSensor(noise_coefficient=0.0, lag_minutes=10.0, seed=42)
        s.reset(initial_glucose=100.0)
        # Jump glucose to 200
        readings = [s.measure(200.0, dt=5.0) for _ in range(20)]
        # First reading should be closer to 100 than 200
        assert readings[0] < 170.0
        # Later readings should approach 200
        assert readings[-1] > 190.0


class TestSensorReset:
    def test_reset(self):
        s = CGMSensor(seed=42)
        s.measure(300.0, dt=5.0)
        s.reset(initial_glucose=100.0)
        # After reset, should be near 100
        reading = s.measure(100.0, dt=5.0)
        assert abs(reading - 100.0) < 20.0

    def test_reproducible_with_same_seed(self):
        s1 = CGMSensor(seed=123)
        s2 = CGMSensor(seed=123)
        for _ in range(20):
            r1 = s1.measure(150.0, dt=5.0)
            r2 = s2.measure(150.0, dt=5.0)
            assert r1 == pytest.approx(r2)
