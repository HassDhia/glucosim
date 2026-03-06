# GlucoSim: Open-Source RL Environments for Glucose Management

Dear Dr. Kovatchev,

Your work on in-silico preclinical trials for closed-loop control of Type 1 diabetes, particularly the development of the UVA/Padova simulator accepted by the FDA as a substitute for pre-clinical animal trials, has been foundational to the field of computational glucose management. The framework you established for virtual patient populations and standardized simulation protocols set the standard for how glucose control algorithms are evaluated.

I am writing to share GlucoSim, a new open-source platform providing three Gymnasium-compatible reinforcement learning environments for glucose management. While GlucoSim uses the simpler Bergman minimal model rather than the full UVA/Padova system, it implements virtual patient populations with parameter variability across three age groups, following the evaluation philosophy your work established. Our key finding is that composite reward functions with safety constraints are necessary for RL benchmarks to differentiate learned policies from naive baselines in glucose management.

- **GitHub:** https://github.com/HassDhia/glucosim
- **PyPI:** https://pypi.org/project/glucosim/
- **Paper:** https://github.com/HassDhia/glucosim/blob/main/paper/glucosim.pdf

GlucoSim ships with 117 tests, trained PPO baselines, and is MIT licensed. If you or your group find it useful, I would welcome any feedback. A brief mention, star, or share with colleagues would be greatly appreciated.

Best regards,
Hass Dhia
Smart Technology Investments Research Institute
partners@smarttechinvest.com
https://smarttechinvest.com/research
