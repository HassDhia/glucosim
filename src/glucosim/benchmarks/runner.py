"""Benchmark runner for GlucoSim environments."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from glucosim.benchmarks.environments import BENCHMARK_CONFIGS, TIER_NAMES


def run_benchmark(
    agent: Any,
    env_id: str,
    configs: list[dict[str, Any]],
    n_episodes: int = 3,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Run benchmark suite for a single environment.

    Returns list of per-tier results.
    """
    import gymnasium as gym

    tier_results = []
    for i, config in enumerate(configs):
        env = gym.make(env_id, **config)
        rewards = []
        tir_values = []

        for ep in range(n_episodes):
            obs, _ = env.reset(seed=seed + ep)
            ep_reward = 0.0
            in_range_steps = 0
            total_steps = 0
            done = False

            while not done:
                action = agent.predict(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += reward
                if info.get("in_range", False):
                    in_range_steps += 1
                total_steps += 1
                done = terminated or truncated

            rewards.append(ep_reward)
            tir_values.append(in_range_steps / max(total_steps, 1))

        env.close()

        tier_results.append({
            "tier": TIER_NAMES[i],
            "config": config,
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "time_in_range": float(np.mean(tir_values)),
            "n_episodes": n_episodes,
        })

    return tier_results


def run_full_benchmark(
    output_dir: str = "results",
) -> dict[str, Any]:
    """Run the full benchmark suite across all environments and agents."""
    import gymnasium as gym
    import glucosim  # noqa: F401
    from glucosim.agents.random_agent import RandomAgent
    from glucosim.agents.heuristic import HeuristicBasalAgent, HeuristicBolusAgent

    results: dict[str, Any] = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}

    for env_id, configs in BENCHMARK_CONFIGS.items():
        env_name = env_id.split("/")[1]
        print(f"\nBenchmark: {env_name}")

        env = gym.make(env_id)
        env_results: dict[str, Any] = {}

        # Random agent
        random_agent = RandomAgent(env.action_space, seed=42)
        env_results["random"] = run_benchmark(random_agent, env_id, configs)

        # Heuristic
        if "Basal" in env_name or "Closed" in env_name:
            heuristic = HeuristicBasalAgent()
        else:
            heuristic = HeuristicBolusAgent()
        env_results["heuristic"] = run_benchmark(heuristic, env_id, configs)

        env.close()
        results[env_name] = env_results

    out_path = Path(output_dir) / "benchmark_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    return results


def main() -> None:
    """CLI entry point for running benchmarks."""
    run_full_benchmark()


if __name__ == "__main__":
    main()
