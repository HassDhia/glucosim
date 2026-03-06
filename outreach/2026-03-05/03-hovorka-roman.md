# GlucoSim: Open-Source RL Environments for Glucose Management

Dear Prof. Hovorka,

Your 2004 paper on nonlinear model predictive control of glucose concentration in Type 1 diabetes established a rigorous framework for closed-loop glucose control that has influenced a generation of artificial pancreas research. Your work on the Cambridge model and its application to MPC-based insulin delivery systems remains a key reference point for anyone working at the intersection of control theory and diabetes management.

I am writing to share GlucoSim, a new open-source platform providing three Gymnasium-compatible reinforcement learning environments for glucose management. Built on the Bergman minimal model and Dalla Man gut absorption dynamics, GlucoSim includes basal rate optimization, meal bolus dosing, and a 48-hour full closed-loop stress test. Our experiments revealed that the Bergman model's homeostatic feedback makes random insulin delivery surprisingly effective on single-objective tasks, but this advantage collapses on multi-objective environments with IOB stacking penalties, where PPO achieves 1868.6 mean reward versus -825.8 for random control.

- **GitHub:** https://github.com/HassDhia/glucosim
- **PyPI:** https://pypi.org/project/glucosim/
- **Paper:** https://github.com/HassDhia/glucosim/blob/main/paper/glucosim.pdf

GlucoSim ships with 117 tests, trained PPO baselines, and is MIT licensed. If you or members of your lab find it useful as a benchmarking platform, I would welcome any feedback. A brief mention, star, or share with colleagues in the artificial pancreas community would be greatly appreciated.

Best regards,
Hass Dhia
Smart Technology Investments Research Institute
partners@smarttechinvest.com
https://smarttechinvest.com/research
