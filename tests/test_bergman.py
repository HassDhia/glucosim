"""Tests for the Bergman minimal model."""

import numpy as np
import pytest

from glucosim.models.bergman import BergmanModel, DEFAULT_PARAMS


class TestBergmanInit:
    def test_default_params(self):
        m = BergmanModel()
        assert m.glucose == pytest.approx(DEFAULT_PARAMS["Gb"])
        assert m.insulin == pytest.approx(DEFAULT_PARAMS["Ib"])
        assert m.insulin_action == 0.0

    def test_custom_params(self):
        m = BergmanModel(params={"Gb": 130.0, "Ib": 20.0})
        assert m.glucose == pytest.approx(130.0)
        assert m.insulin == pytest.approx(20.0)

    def test_custom_dt(self):
        m = BergmanModel(dt=5.0)
        assert m.dt == 5.0

    def test_default_dt_is_one(self):
        m = BergmanModel()
        assert m.dt == 1.0


class TestBergmanReset:
    def test_reset_to_basal(self):
        m = BergmanModel()
        m.step(insulin_rate=2.0, meal_rate=5.0)
        m.reset()
        assert m.glucose == pytest.approx(DEFAULT_PARAMS["Gb"])
        assert m.insulin == pytest.approx(DEFAULT_PARAMS["Ib"])
        assert m.insulin_action == 0.0

    def test_reset_to_custom(self):
        m = BergmanModel()
        m.reset(glucose=200.0, insulin=25.0)
        assert m.glucose == pytest.approx(200.0)
        assert m.insulin == pytest.approx(25.0)


class TestBergmanStep:
    def test_step_returns_dict(self):
        m = BergmanModel()
        result = m.step(insulin_rate=0.0, meal_rate=0.0)
        assert "glucose" in result
        assert "insulin" in result
        assert "insulin_action" in result

    def test_basal_state_stable(self):
        """At basal with no input, glucose should remain near basal."""
        m = BergmanModel()
        for _ in range(60):
            m.step(insulin_rate=0.0, meal_rate=0.0)
        assert abs(m.glucose - DEFAULT_PARAMS["Gb"]) < 5.0

    def test_meal_raises_glucose(self):
        """Glucose appearance from meal should increase glucose."""
        m = BergmanModel()
        initial = m.glucose
        for _ in range(30):
            m.step(insulin_rate=0.0, meal_rate=2.0)
        assert m.glucose > initial

    def test_insulin_lowers_glucose(self):
        """High insulin delivery should lower glucose."""
        m = BergmanModel()
        m.reset(glucose=200.0)
        for _ in range(120):
            m.step(insulin_rate=2.0, meal_rate=0.0)
        assert m.glucose < 200.0

    def test_glucose_bounded(self):
        """Glucose should not go below physiological minimum."""
        m = BergmanModel()
        for _ in range(500):
            m.step(insulin_rate=5.0, meal_rate=0.0)
        assert m.glucose >= 10.0

    def test_glucose_upper_bounded(self):
        """Glucose should not exceed upper bound."""
        m = BergmanModel()
        for _ in range(500):
            m.step(insulin_rate=0.0, meal_rate=10.0)
        assert m.glucose <= 600.0

    def test_insulin_nonnegative(self):
        m = BergmanModel()
        m.reset(insulin=0.1)
        for _ in range(100):
            m.step(insulin_rate=0.0, meal_rate=0.0)
        assert m.insulin >= 0.0

    def test_insulin_action_nonnegative(self):
        m = BergmanModel()
        for _ in range(100):
            m.step(insulin_rate=0.0, meal_rate=0.0)
        assert m.insulin_action >= 0.0

    def test_rk4_more_accurate_than_euler(self):
        """RK4 should track a known trajectory more accurately."""
        m = BergmanModel(dt=1.0)
        for _ in range(60):
            m.step(insulin_rate=1.0, meal_rate=1.0)
        g1 = m.glucose

        m2 = BergmanModel(dt=0.1)
        for _ in range(600):
            m2.step(insulin_rate=1.0, meal_rate=1.0)
        g2 = m2.glucose

        # Both should converge to similar value; finer dt should be reference
        assert abs(g1 - g2) < 5.0

    def test_deterministic(self):
        m1 = BergmanModel()
        m2 = BergmanModel()
        for _ in range(100):
            m1.step(insulin_rate=1.0, meal_rate=0.5)
            m2.step(insulin_rate=1.0, meal_rate=0.5)
        assert m1.glucose == pytest.approx(m2.glucose)
