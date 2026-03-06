# GlucoSim: A Gymnasium Update to the simglucose Approach

Dear Dr. Xie,

Your simglucose project was one of the first open-source implementations of the UVA/Padova glucose-insulin model with an OpenAI Gym interface, and it has been widely used by the RL-for-diabetes research community since its release in 2018. By making glucose simulation accessible to RL researchers, you helped establish this as a viable research direction.

I am writing to share GlucoSim, a new open-source platform that carries forward the spirit of simglucose with the modern Gymnasium API (the successor to OpenAI Gym). GlucoSim provides three distinct environment paradigms (basal control, bolus dosing, and full closed-loop), configurable difficulty tiers, clinical heuristic baselines, and a benchmark suite. Built on the Bergman minimal model and Dalla Man gut dynamics, it ships with trained PPO agents and an interesting finding about the reward complexity threshold needed for meaningful RL benchmarking in glucose management.

- **GitHub:** https://github.com/HassDhia/glucosim
- **PyPI:** https://pypi.org/project/glucosim/
- **Paper:** https://github.com/HassDhia/glucosim/blob/main/paper/glucosim.pdf

GlucoSim ships with 117 tests and is MIT licensed. If you find it a useful complement to simglucose or worth sharing with your network, I would welcome any feedback. A brief mention, star, or share would be greatly appreciated.

Best regards,
Hass Dhia
Smart Technology Investments Research Institute
partners@smarttechinvest.com
https://smarttechinvest.com/research
