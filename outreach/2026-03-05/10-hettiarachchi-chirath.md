# GlucoSim: Gymnasium RL Environments for Glucose Management

Dear Dr. Hettiarachchi,

Your GluCoEnv project was an important step toward providing Gymnasium-compatible glucose control environments for the RL community. By implementing a glucose control environment with the modern Gymnasium API, you helped establish the need for standardized, API-compatible benchmarks in this domain.

I am writing to share GlucoSim, a new open-source platform that builds on the direction GluCoEnv established by providing three distinct environment paradigms (basal control, bolus dosing, and full closed-loop), five difficulty tiers, clinical heuristic baselines, and a standardized benchmark suite. Built on the Bergman minimal model and Dalla Man gut dynamics, GlucoSim includes 30 virtual patients across three age groups, trained PPO agents, and a finding about the reward complexity threshold needed for meaningful RL benchmarking. We found that composite rewards with safety constraints (IOB penalties, multi-day horizons) are necessary to differentiate learned policies from naive baselines.

- **GitHub:** https://github.com/HassDhia/glucosim
- **PyPI:** https://pypi.org/project/glucosim/
- **Paper:** https://github.com/HassDhia/glucosim/blob/main/paper/glucosim.pdf

GlucoSim ships with 117 tests and is MIT licensed. If you find it a useful complement to GluCoEnv or worth sharing with colleagues in the glucose RL community, I would welcome any feedback. A brief mention, star, or share would be greatly appreciated.

Best regards,
Hass Dhia
Smart Technology Investments Research Institute
partners@smarttechinvest.com
https://smarttechinvest.com/research
