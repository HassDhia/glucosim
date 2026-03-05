"""Virtual patient generator for in-silico glucose management experiments.

Generates patient-specific Bergman model parameters with realistic
inter-patient variability across three age groups.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np

# Population means by age group
POPULATION_MEANS: dict[str, dict[str, float]] = {
    "child": {
        "Gb": 120.0, "Ib": 10.0, "body_weight": 35.0,
        "p1": 0.030, "p2": 0.027, "p3": 0.000015,
        "n": 0.25, "gamma": 0.005, "h": 100.0,
        "VG": 1.69, "VI": 0.05,
    },
    "adolescent": {
        "Gb": 115.0, "Ib": 12.0, "body_weight": 55.0,
        "p1": 0.029, "p2": 0.026, "p3": 0.000014,
        "n": 0.24, "gamma": 0.0045, "h": 102.0,
        "VG": 1.59, "VI": 0.045,
    },
    "adult": {
        "Gb": 110.0, "Ib": 15.0, "body_weight": 70.0,
        "p1": 0.028, "p2": 0.025, "p3": 0.000013,
        "n": 0.23, "gamma": 0.004, "h": 105.0,
        "VG": 1.49, "VI": 0.04,
    },
}

VALID_TYPES = ("child", "adolescent", "adult")


class VirtualPatient:
    """A virtual patient with individualized glucose-insulin parameters.

    Parameters are sampled with +/-20% uniform variability around
    population means for the specified age group.
    """

    def __init__(
        self,
        patient_type: str = "adult",
        patient_id: int = 0,
        seed: int | None = None,
    ) -> None:
        if patient_type not in VALID_TYPES:
            raise ValueError(
                f"patient_type must be one of {VALID_TYPES}, got '{patient_type}'"
            )

        self.patient_type = patient_type
        self.patient_id = patient_id
        self._rng = np.random.default_rng(seed)

        # Generate parameters with variability
        means = POPULATION_MEANS[patient_type]
        self._params: dict[str, float] = {}
        for key, mean_val in means.items():
            # +/-20% uniform variability
            low = mean_val * 0.8
            high = mean_val * 1.2
            self._params[key] = float(self._rng.uniform(low, high))

    @property
    def params(self) -> dict[str, float]:
        return dict(self._params)

    @property
    def body_weight(self) -> float:
        return self._params["body_weight"]

    @property
    def basal_glucose(self) -> float:
        return self._params["Gb"]

    @property
    def basal_insulin(self) -> float:
        return self._params["Ib"]

    def __repr__(self) -> str:
        return (
            f"VirtualPatient(type={self.patient_type!r}, id={self.patient_id}, "
            f"Gb={self.basal_glucose:.1f}, Ib={self.basal_insulin:.1f}, "
            f"BW={self.body_weight:.1f})"
        )


class PatientPopulation:
    """A collection of virtual patients for cohort simulations."""

    def __init__(
        self,
        n_patients: int = 10,
        patient_type: str = "adult",
        seed: int = 42,
    ) -> None:
        self._patients = [
            VirtualPatient(
                patient_type=patient_type,
                patient_id=i,
                seed=seed + i,
            )
            for i in range(n_patients)
        ]

    def __len__(self) -> int:
        return len(self._patients)

    def __getitem__(self, idx: int) -> VirtualPatient:
        return self._patients[idx]

    def __iter__(self) -> Iterator[VirtualPatient]:
        return iter(self._patients)

    def __repr__(self) -> str:
        return (
            f"PatientPopulation(n={len(self)}, "
            f"type={self._patients[0].patient_type!r})"
        )
