"""Tests for the benchmark suite."""

import pytest
from glucosim.benchmarks.environments import BENCHMARK_CONFIGS, TIER_NAMES


class TestBenchmarkConfigs:
    def test_three_envs_configured(self):
        assert len(BENCHMARK_CONFIGS) == 3

    def test_five_tiers_per_env(self):
        for env_id, configs in BENCHMARK_CONFIGS.items():
            assert len(configs) == 5, f"{env_id} should have 5 tiers"

    def test_tier_names_count(self):
        assert len(TIER_NAMES) == 5

    def test_all_configs_have_difficulty(self):
        for env_id, configs in BENCHMARK_CONFIGS.items():
            for c in configs:
                assert "difficulty" in c
                assert c["difficulty"] in ("easy", "medium", "hard")

    def test_all_configs_have_patient_type(self):
        for env_id, configs in BENCHMARK_CONFIGS.items():
            for c in configs:
                assert "patient_type" in c
                assert c["patient_type"] in ("child", "adolescent", "adult")

    def test_env_ids_valid(self):
        expected = {
            "glucosim/BasalControl-v0",
            "glucosim/BolusAdvisor-v0",
            "glucosim/ClosedLoop-v0",
        }
        assert set(BENCHMARK_CONFIGS.keys()) == expected
