"""Tests for virtual patient generation."""

import pytest

from glucosim.models.patient import (
    VirtualPatient,
    PatientPopulation,
    POPULATION_MEANS,
    VALID_TYPES,
)


class TestVirtualPatient:
    def test_default_adult(self):
        p = VirtualPatient(seed=42)
        assert p.patient_type == "adult"
        assert p.patient_id == 0

    def test_all_types_valid(self):
        for t in VALID_TYPES:
            p = VirtualPatient(patient_type=t, seed=42)
            assert p.patient_type == t

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError, match="patient_type"):
            VirtualPatient(patient_type="infant")

    def test_params_within_variability(self):
        """Parameters should be within +/-20% of population means."""
        for ptype in VALID_TYPES:
            means = POPULATION_MEANS[ptype]
            for i in range(10):
                p = VirtualPatient(patient_type=ptype, patient_id=i, seed=i)
                for key, mean_val in means.items():
                    val = p.params[key]
                    assert val >= mean_val * 0.8 - 1e-10
                    assert val <= mean_val * 1.2 + 1e-10

    def test_different_seeds_different_params(self):
        p1 = VirtualPatient(seed=1)
        p2 = VirtualPatient(seed=2)
        assert p1.params != p2.params

    def test_same_seed_same_params(self):
        p1 = VirtualPatient(seed=42)
        p2 = VirtualPatient(seed=42)
        assert p1.params == p2.params

    def test_body_weight_property(self):
        p = VirtualPatient(seed=42)
        assert p.body_weight == p.params["body_weight"]

    def test_basal_glucose_property(self):
        p = VirtualPatient(seed=42)
        assert p.basal_glucose == p.params["Gb"]

    def test_basal_insulin_property(self):
        p = VirtualPatient(seed=42)
        assert p.basal_insulin == p.params["Ib"]

    def test_repr(self):
        p = VirtualPatient(seed=42)
        r = repr(p)
        assert "adult" in r
        assert "Gb=" in r

    def test_child_lighter_than_adult(self):
        children = [VirtualPatient("child", i, seed=i).body_weight for i in range(10)]
        adults = [VirtualPatient("adult", i, seed=i).body_weight for i in range(10)]
        assert sum(children) / 10 < sum(adults) / 10


class TestPatientPopulation:
    def test_population_size(self):
        pop = PatientPopulation(n_patients=5)
        assert len(pop) == 5

    def test_iteration(self):
        pop = PatientPopulation(n_patients=3)
        patients = list(pop)
        assert len(patients) == 3
        for p in patients:
            assert isinstance(p, VirtualPatient)

    def test_indexing(self):
        pop = PatientPopulation(n_patients=5)
        assert pop[0].patient_id == 0
        assert pop[4].patient_id == 4

    def test_repr(self):
        pop = PatientPopulation(n_patients=10)
        assert "n=10" in repr(pop)

    def test_different_patients(self):
        pop = PatientPopulation(n_patients=5)
        params = [p.params["Gb"] for p in pop]
        assert len(set(params)) > 1  # Not all identical
