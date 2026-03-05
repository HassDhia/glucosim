"""Tests for the meal absorption model."""

import pytest
from glucosim.models.meal import MealModel


class TestMealInit:
    def test_default_init(self):
        m = MealModel()
        assert m.total_remaining == 0.0
        assert not m.is_absorbing

    def test_custom_params(self):
        m = MealModel(params={"BW": 80.0})
        assert m.params["BW"] == 80.0


class TestMealAnnounce:
    def test_single_meal(self):
        m = MealModel()
        m.announce_meal(50.0)
        assert m.total_remaining > 0.0
        assert m.is_absorbing

    def test_multiple_meals(self):
        m = MealModel()
        m.announce_meal(30.0)
        m.announce_meal(20.0)
        assert m.total_remaining == pytest.approx(50000.0)


class TestMealStep:
    def test_produces_glucose_appearance(self):
        m = MealModel()
        m.announce_meal(50.0)
        rate = m.step(dt=1.0)
        assert rate >= 0.0

    def test_glucose_appearance_peaks_then_decays(self):
        m = MealModel()
        m.announce_meal(70.0)
        rates = [m.step(dt=1.0) for _ in range(300)]
        peak_idx = rates.index(max(rates))
        # Peak should occur within first 100 minutes
        assert peak_idx < 100
        # Rate should decay after peak
        assert rates[-1] < rates[peak_idx]

    def test_total_absorption(self):
        """After long enough, nearly all carbs should be absorbed."""
        m = MealModel()
        m.announce_meal(50.0)
        for _ in range(600):
            m.step(dt=1.0)
        assert m.total_remaining < 100.0  # <0.2% remaining

    def test_no_meal_no_glucose(self):
        m = MealModel()
        rate = m.step(dt=1.0)
        assert rate == 0.0

    def test_reset_clears_state(self):
        m = MealModel()
        m.announce_meal(50.0)
        for _ in range(10):
            m.step()
        m.reset()
        assert m.total_remaining == 0.0
        assert not m.is_absorbing

    def test_nonnegative_compartments(self):
        m = MealModel()
        m.announce_meal(100.0)
        for _ in range(1000):
            rate = m.step(dt=1.0)
            assert rate >= 0.0

    def test_deterministic(self):
        m1, m2 = MealModel(), MealModel()
        m1.announce_meal(60.0)
        m2.announce_meal(60.0)
        for _ in range(200):
            r1 = m1.step()
            r2 = m2.step()
            assert r1 == pytest.approx(r2)
