# GlucoSim: Gymnasium RL Environments for Glucose Management

Dear Dr. Fox,

Your 2020 paper on deep reinforcement learning for closed-loop blood glucose control was one of the first rigorous demonstrations that deep RL can learn clinically meaningful insulin delivery policies. Your work with Dr. Wiens and colleagues showed that RL approaches can handle the sequential decision-making nature of glucose control in ways that traditional control methods cannot easily replicate.

I am writing to share GlucoSim, a new open-source platform providing three Gymnasium-compatible RL environments for glucose management. The environments span basal rate optimization, meal bolus dosing, and a 48-hour full closed-loop stress test with IOB stacking penalties. Our experiments revealed an interesting finding relevant to your work: on single-objective environments, random insulin delivery achieves 76.7% time-in-range due to the Bergman model's homeostatic feedback, but this advantage collapses on the multi-objective ClosedLoop task where PPO achieves 1868.6 mean reward versus -825.8 for random control. This suggests that multi-objective reward design, similar to what your work explored, is essential for meaningful glucose RL benchmarking.

- **GitHub:** https://github.com/HassDhia/glucosim
- **PyPI:** https://pypi.org/project/glucosim/
- **Paper:** https://github.com/HassDhia/glucosim/blob/main/paper/glucosim.pdf

GlucoSim ships with 117 tests, trained PPO baselines, and is MIT licensed. If you or your group find it useful as a benchmarking tool, I would welcome any feedback. A brief mention, star, or share would be greatly appreciated.

Best regards,
Hass Dhia
Smart Technology Investments Research Institute
partners@smarttechinvest.com
https://smarttechinvest.com/research
