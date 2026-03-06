# glucosim

**Gymnasium Environments for Reinforcement Learning in Glucose Management**

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Tests](https://img.shields.io/badge/tests-117%20passing-brightgreen.svg)
[![PyPI version](https://img.shields.io/pypi/v/glucosim.svg)](https://pypi.org/project/glucosim/)

---

Three Gymnasium-compatible RL environments for Type 1 diabetes glucose management: basal rate optimization, meal bolus dosing, and full closed-loop insulin delivery control. Includes the Bergman minimal glucose-insulin model, Dalla Man gut absorption dynamics, a CGM sensor noise model, 30 virtual patients across three age groups (child, adolescent, adult), configurable difficulty tiers, heuristic clinical baselines, PPO RL agents, and a five-tier benchmark suite. 117 tests, MIT licensed.

## Installation

```bash
pip install glucosim              # Core (numpy, scipy, gymnasium)
pip install glucosim[train]       # + SB3, PyTorch for RL training
pip install glucosim[all]         # Everything
```

Development install:

```bash
git clone https://github.com/HassDhia/glucosim.git
cd glucosim
pip install -e ".[all]"
```

## Quick Start

```python
import gymnasium as gym
import glucosim

env = gym.make("glucosim/BasalControl-v0")
obs, info = env.reset(seed=42)
for _ in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
env.close()
```

## Environments

| Environment | Paradigm | Observation | Action | Key Challenge |
|---|---|---|---|---|
| `BasalControl-v0` | Continuous basal rate | CGM glucose, IOB, time, glucose rate | Basal rate (U/hr) | Maintain euglycemia over 24h |
| `BolusAdvisor-v0` | Meal bolus dosing | CGM glucose, IOB, meal flag, carbs, time since meal | Bolus dose (U) | Optimal postprandial control |
| `ClosedLoop-v0` | Full closed-loop | CGM glucose, IOB, time, glucose rate, meal flag, carbs | Total insulin rate (U/hr) | 48h stress test with IOB penalty |

## Architecture

GlucoSim implements a modular simulation stack:

- **Bergman Minimal Model** - Three-equation glucose-insulin dynamics with RK4 integration
- **Dalla Man Gut Model** - Two-compartment carbohydrate absorption producing glucose appearance rates
- **CGM Sensor Model** - First-order lag filter with configurable Gaussian noise
- **Virtual Patients** - 30 patients across child/adolescent/adult groups with +/-20% parameter variability
- **Baseline Agents** - Random, proportional basal controller, and ICR bolus calculator
- **PPO Agent** - Stable Baselines3 PPO with tuned hyperparameters
- **Benchmark Suite** - Five difficulty tiers per environment (easy-adult to hard-child)

## Paper

The accompanying paper is available at:
- [PDF (GitHub)](https://github.com/HassDhia/glucosim/blob/main/paper/glucosim.pdf)

## Citation

If you use glucosim in your research, please cite:

```bibtex
@software{dhia2026glucosim,
  author = {Dhia, Hass},
  title = {GlucoSim: Gymnasium Environments for Reinforcement Learning in Glucose Management},
  year = {2026},
  publisher = {Smart Technology Investments Research Institute},
  url = {https://github.com/HassDhia/glucosim}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.

## Contact

Hass Dhia - Smart Technology Investments Research Institute
- Email: partners@smarttechinvest.com
- Web: [smarttechinvest.com/research](https://smarttechinvest.com/research)
