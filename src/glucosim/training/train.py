"""Training pipeline for GlucoSim agents.

Trains PPO, random, and heuristic agents across all environments and
compiles results into a unified training_results.json.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np


def evaluate_agent(
    agent: Any,
    env_id: str,
    n_episodes: int = 10,
    seed: int = 0,
) -> dict[str, float]:
    """Evaluate an agent on an environment.

    Returns dict with mean_reward, std_reward, time_in_range, hypo_rate, hyper_rate.
    """
    import gymnasium as gym
    import glucosim  # noqa: F401

    env = gym.make(env_id)
    rewards = []
    tir_steps = []
    total_steps = 0

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        ep_reward = 0.0
        ep_in_range = 0
        ep_steps = 0
        done = False

        while not done:
            action = agent.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            if info.get("in_range", False):
                ep_in_range += 1
            ep_steps += 1
            done = terminated or truncated

        rewards.append(ep_reward)
        tir_steps.append(ep_in_range / max(ep_steps, 1))
        total_steps += ep_steps

    env.close()

    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "time_in_range": float(np.mean(tir_steps)),
        "n_episodes": n_episodes,
    }


def train_all(
    output_dir: str = "results",
    training_steps: int = 100000,
    n_eval_episodes: int = 10,
) -> dict[str, Any]:
    """Train and evaluate all agent types across all environments.

    Returns the full results dictionary.
    """
    import gymnasium as gym
    import glucosim  # noqa: F401
    from glucosim.agents.random_agent import RandomAgent
    from glucosim.agents.heuristic import HeuristicBasalAgent, HeuristicBolusAgent
    from glucosim.agents.ppo import PPOAgent

    envs_config = [
        {
            "env_id": "glucosim/BasalControl-v0",
            "steps": training_steps,
            "heuristic_cls": HeuristicBasalAgent,
            "heuristic_kwargs": {},
        },
        {
            "env_id": "glucosim/BolusAdvisor-v0",
            "steps": training_steps,
            "heuristic_cls": HeuristicBolusAgent,
            "heuristic_kwargs": {},
        },
        {
            "env_id": "glucosim/ClosedLoop-v0",
            "steps": int(training_steps * 1.5),
            "heuristic_cls": HeuristicBasalAgent,
            "heuristic_kwargs": {"base_rate": 1.5},
        },
    ]

    results: dict[str, Any] = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}

    for cfg in envs_config:
        env_id = cfg["env_id"]
        env_name = env_id.split("/")[1]
        print(f"\n{'='*60}")
        print(f"Environment: {env_name}")
        print(f"{'='*60}")

        env = gym.make(env_id)
        env_results: dict[str, Any] = {"env_id": env_id}

        # Random baseline
        print("  Evaluating random agent...")
        random_agent = RandomAgent(env.action_space, seed=42)
        env_results["random"] = evaluate_agent(random_agent, env_id, n_eval_episodes)

        # Heuristic baseline
        print("  Evaluating heuristic agent...")
        heuristic = cfg["heuristic_cls"](**cfg["heuristic_kwargs"])
        env_results["heuristic"] = evaluate_agent(heuristic, env_id, n_eval_episodes)

        # PPO
        print(f"  Training PPO for {cfg['steps']} steps...")
        ppo = PPOAgent.train(env_id, total_timesteps=cfg["steps"], seed=42)
        name = env_name.replace("-v0", "").lower()
        save_path = Path("checkpoints") / f"ppo_{name}"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        ppo.save(save_path)
        print("  Evaluating PPO agent...")
        env_results["ppo"] = evaluate_agent(ppo, env_id, n_eval_episodes)
        env_results["ppo"]["training_steps"] = cfg["steps"]
        env_results["ppo"]["model_path"] = str(save_path)

        # Compute ratios
        random_reward = env_results["random"]["mean_reward"]
        ppo_reward = env_results["ppo"]["mean_reward"]
        if random_reward != 0:
            env_results["ppo_vs_random_ratio"] = ppo_reward / abs(random_reward)
        else:
            env_results["ppo_vs_random_ratio"] = float("inf") if ppo_reward > 0 else 0.0

        env.close()
        results[env_name] = env_results

    # Save results
    out_path = Path(output_dir) / "training_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return results
