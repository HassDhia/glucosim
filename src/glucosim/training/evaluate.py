"""Evaluation utilities for trained agents."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def load_results(path: str = "results/training_results.json") -> dict[str, Any]:
    """Load training results from JSON."""
    with open(path) as f:
        return json.load(f)


def summarize_results(results: dict[str, Any]) -> str:
    """Generate a human-readable summary of training results."""
    lines = ["GlucoSim Training Results Summary", "=" * 40, ""]

    for key, val in results.items():
        if key == "timestamp":
            lines.append(f"Trained: {val}")
            continue
        if not isinstance(val, dict):
            continue

        env_name = key
        lines.append(f"\n{env_name}:")
        lines.append("-" * 30)

        for agent_type in ["random", "heuristic", "ppo"]:
            if agent_type in val:
                r = val[agent_type]
                lines.append(
                    f"  {agent_type:12s}: reward={r['mean_reward']:.2f} "
                    f"(+/-{r.get('std_reward', 0):.2f}), "
                    f"TIR={r.get('time_in_range', 0)*100:.1f}%"
                )

        ratio = val.get("ppo_vs_random_ratio", "N/A")
        if isinstance(ratio, float):
            lines.append(f"  PPO/Random ratio: {ratio:.2f}x")

    return "\n".join(lines)
