# GlucoSim: Open-Source RL Environments Built on the Bergman Minimal Model

Dear Dr. Bergman,

Your 1979 paper "Quantitative estimation of insulin sensitivity" introduced the minimal glucose-insulin model that has become foundational to computational glucose dynamics. The three-equation system you developed with Dr. Cobelli and colleagues remains remarkably effective for capturing the essential feedback structure of glucose homeostasis, and its tractability has made it the standard starting point for control-oriented glucose research.

I am writing to share GlucoSim, a new open-source platform that implements your minimal model as the core physiological engine for three Gymnasium-compatible reinforcement learning environments targeting Type 1 diabetes glucose management. The environments span basal rate optimization, meal bolus dosing, and full closed-loop insulin delivery, each with configurable difficulty tiers and virtual patient populations. During our experiments, we discovered that the endogenous insulin secretion term in your model creates a homeostatic feedback floor that makes random insulin delivery surprisingly effective on single-objective tasks (76.7% time-in-range), establishing that composite reward functions with safety constraints are necessary for meaningful RL benchmarking in this domain.

- **GitHub:** https://github.com/HassDhia/glucosim
- **PyPI:** https://pypi.org/project/glucosim/
- **Paper:** https://github.com/HassDhia/glucosim/blob/main/paper/glucosim.pdf

GlucoSim ships with 117 tests, trained PPO baselines, and is MIT licensed. If you or your group find it useful for teaching or research, I would welcome any feedback. A brief mention, star, or share with colleagues working in glucose modeling or RL for metabolic control would be greatly appreciated.

Best regards,
Hass Dhia
Smart Technology Investments Research Institute
partners@smarttechinvest.com
https://smarttechinvest.com/research
