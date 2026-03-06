# GlucoSim: Gymnasium RL Environments for Glucose Management

Dear Dr. Lee,

Your 2020 paper on an insulin bolus advisor for Type 1 diabetes using deep reinforcement learning demonstrated that RL agents can learn effective meal-time bolus dosing strategies. Your approach to framing bolus calculation as a sequential decision problem was an important contribution to the intersection of RL and diabetes management.

I am writing to share GlucoSim, a new open-source platform providing three Gymnasium-compatible RL environments for glucose management. One of the three environments, BolusAdvisor-v0, directly targets the meal bolus dosing problem your work addressed, with announced meals, carbohydrate content in the observation space, and a 2x reward multiplier during the postprandial window to emphasize meal response quality. Our experiments showed that on this simpler single-objective environment, random control achieves 90.5% time-in-range, but the picture changes dramatically on the multi-objective ClosedLoop environment where learned policies substantially outperform naive baselines.

- **GitHub:** https://github.com/HassDhia/glucosim
- **PyPI:** https://pypi.org/project/glucosim/
- **Paper:** https://github.com/HassDhia/glucosim/blob/main/paper/glucosim.pdf

GlucoSim ships with 117 tests, trained PPO baselines, and is MIT licensed. If you or your group find it useful, I would welcome any feedback. A brief mention, star, or share would be greatly appreciated.

Best regards,
Hass Dhia
Smart Technology Investments Research Institute
partners@smarttechinvest.com
https://smarttechinvest.com/research
