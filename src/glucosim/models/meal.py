"""Meal absorption model based on Dalla Man et al. (2007).

Implements a simplified two-compartment gut absorption model that converts
ingested carbohydrates into a glucose appearance rate (Ra) over time.
"""

from __future__ import annotations

# Default absorption parameters
DEFAULT_MEAL_PARAMS: dict[str, float] = {
    "kgri": 0.0558,   # grinding rate (min^-1)
    "kempt": 0.0680,  # gastric emptying rate (min^-1)
    "kabs": 0.0570,   # intestinal absorption rate (min^-1)
    "f": 0.90,        # bioavailability fraction
    "BW": 70.0,       # body weight (kg)
    "VG": 1.49,       # glucose distribution volume (dL/kg)
}


class MealModel:
    """Two-compartment gut absorption model.

    Tracks carbohydrate flow through stomach (grinding, emptying) and
    gut (absorption) to produce a glucose appearance rate.

    State variables:
        Qsto1 - solid food in stomach (mg)
        Qsto2 - liquid/triturated food in stomach (mg)
        Qgut  - glucose in intestine (mg)
    """

    def __init__(self, params: dict[str, float] | None = None) -> None:
        self.params = {**DEFAULT_MEAL_PARAMS, **(params or {})}
        self._qsto1: float = 0.0
        self._qsto2: float = 0.0
        self._qgut: float = 0.0

    def reset(self) -> None:
        self._qsto1 = 0.0
        self._qsto2 = 0.0
        self._qgut = 0.0

    def announce_meal(self, carbs_grams: float) -> None:
        """Add a meal bolus to the stomach compartment.

        Args:
            carbs_grams: carbohydrate content in grams
        """
        # Convert grams of carbs to mg of glucose equivalent
        # SIMPLIFICATION: Assuming 1g carbs = 1000mg glucose equivalent.
        # In practice, different carb types have different glycemic indices.
        self._qsto1 += carbs_grams * 1000.0

    def step(self, dt: float = 1.0) -> float:
        """Advance the meal model by dt minutes.

        Args:
            dt: time step in minutes

        Returns:
            Glucose appearance rate in mg/dL/min, normalized by
            distribution volume and body weight.
        """
        p = self.params

        # Two-compartment stomach -> gut -> plasma
        dQsto1 = -p["kgri"] * self._qsto1
        dQsto2 = -p["kempt"] * self._qsto2 + p["kgri"] * self._qsto1
        dQgut = -p["kabs"] * self._qgut + p["kempt"] * self._qsto2

        self._qsto1 += dQsto1 * dt
        self._qsto2 += dQsto2 * dt
        self._qgut += dQgut * dt

        # Ensure non-negative
        self._qsto1 = max(0.0, self._qsto1)
        self._qsto2 = max(0.0, self._qsto2)
        self._qgut = max(0.0, self._qgut)

        # Rate of glucose appearance (mg/dL/min)
        Ra = p["f"] * p["kabs"] * self._qgut / (p["BW"] * p["VG"])
        return Ra

    @property
    def is_absorbing(self) -> bool:
        """True if there is still unabsorbed food in the system."""
        return (self._qsto1 + self._qsto2 + self._qgut) > 1.0

    @property
    def total_remaining(self) -> float:
        """Total unabsorbed glucose in mg."""
        return self._qsto1 + self._qsto2 + self._qgut
