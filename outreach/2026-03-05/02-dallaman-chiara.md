# GlucoSim: Open-Source RL Environments with Dalla Man Gut Absorption Model

Dear Dr. Dalla Man,

Your work on the meal simulation model of the glucose-insulin system (IEEE TBME, 2007) and the UVA/Padova Type 1 Diabetes Simulator has been central to advancing in-silico glucose control research. The two-compartment gut absorption model you developed provides a physiologically grounded approach to modeling carbohydrate absorption dynamics that bridges the gap between meal intake and glucose appearance.

I am writing to share GlucoSim, a new open-source platform that implements your gut absorption model alongside the Bergman minimal model as the core simulation stack for three Gymnasium-compatible reinforcement learning environments. The environments target basal rate optimization, meal bolus dosing, and full closed-loop insulin delivery with 30 virtual patients across three age groups. A key finding from our experiments is that composite reward functions with IOB safety constraints are necessary for RL to demonstrate value over naive baselines in glucose management, a result we term the "reward complexity threshold."

- **GitHub:** https://github.com/HassDhia/glucosim
- **PyPI:** https://pypi.org/project/glucosim/
- **Paper:** https://github.com/HassDhia/glucosim/blob/main/paper/glucosim.pdf

GlucoSim ships with 117 tests, trained PPO baselines, and is MIT licensed. If you or your group find it useful for research or as a teaching tool for glucose dynamics, I would welcome any feedback. A brief mention, star, or share with colleagues in the metabolic modeling community would be greatly appreciated.

Best regards,
Hass Dhia
Smart Technology Investments Research Institute
partners@smarttechinvest.com
https://smarttechinvest.com/research
