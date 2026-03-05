"""Benchmark environment configurations with difficulty tiers."""

from __future__ import annotations

from typing import Any

# Benchmark suite: 5 difficulty tiers per environment
BENCHMARK_CONFIGS: dict[str, list[dict[str, Any]]] = {
    "glucosim/BasalControl-v0": [
        {"difficulty": "easy", "patient_type": "adult", "patient_id": 0},
        {"difficulty": "easy", "patient_type": "adult", "patient_id": 5},
        {"difficulty": "medium", "patient_type": "adult", "patient_id": 0},
        {"difficulty": "medium", "patient_type": "adolescent", "patient_id": 0},
        {"difficulty": "hard", "patient_type": "child", "patient_id": 0},
    ],
    "glucosim/BolusAdvisor-v0": [
        {"difficulty": "easy", "patient_type": "adult", "patient_id": 0},
        {"difficulty": "easy", "patient_type": "adult", "patient_id": 3},
        {"difficulty": "medium", "patient_type": "adult", "patient_id": 0},
        {"difficulty": "medium", "patient_type": "adolescent", "patient_id": 0},
        {"difficulty": "hard", "patient_type": "child", "patient_id": 0},
    ],
    "glucosim/ClosedLoop-v0": [
        {"difficulty": "easy", "patient_type": "adult", "patient_id": 0},
        {"difficulty": "easy", "patient_type": "adult", "patient_id": 7},
        {"difficulty": "medium", "patient_type": "adult", "patient_id": 0},
        {"difficulty": "medium", "patient_type": "adolescent", "patient_id": 0},
        {"difficulty": "hard", "patient_type": "child", "patient_id": 0},
    ],
}

TIER_NAMES = ["Tier 1 (Easy-Adult)", "Tier 2 (Easy-Variant)", "Tier 3 (Medium-Adult)",
              "Tier 4 (Medium-Adolescent)", "Tier 5 (Hard-Child)"]
