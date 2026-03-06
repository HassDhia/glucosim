# GlucoSim: Gymnasium RL Environments for Glucose Management

Dear Dr. Ngo,

Your 2025 paper on a safe-enhanced fully closed-loop artificial pancreas controller based on deep reinforcement learning is directly relevant to our work. Your dual safety mechanism approach, achieving 87.45% median time-in-range, demonstrated that safety constraints are not just practically important but can actively improve RL controller performance. This finding aligns closely with what we observed in our own experiments.

I am writing to share GlucoSim, a new open-source platform providing three Gymnasium-compatible RL environments for glucose management. Our key finding complements your work: we show that safety constraints are not just practically necessary but methodologically essential for RL benchmarking in glucose management. On single-objective environments, random insulin delivery achieves 76.7% TIR due to homeostatic model feedback, but on our ClosedLoop environment with IOB stacking penalties (conceptually similar to your safety mechanisms), PPO achieves 1868.6 mean reward versus -825.8 for random control. We cite your paper as establishing that safety constraints improve both performance and evaluation.

- **GitHub:** https://github.com/HassDhia/glucosim
- **PyPI:** https://pypi.org/project/glucosim/
- **Paper:** https://github.com/HassDhia/glucosim/blob/main/paper/glucosim.pdf

GlucoSim ships with 117 tests, trained PPO baselines, and is MIT licensed. If you find it useful as a complementary benchmarking platform, I would welcome any feedback. A brief mention, star, or share would be greatly appreciated.

Best regards,
Hass Dhia
Smart Technology Investments Research Institute
partners@smarttechinvest.com
https://smarttechinvest.com/research
