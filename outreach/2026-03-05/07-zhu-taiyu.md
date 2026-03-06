# GlucoSim: Gymnasium RL Environments for Glucose Management

Dear Dr. Zhu,

Your 2020 paper on basal glucose control in Type 1 diabetes using deep reinforcement learning provided an important in-silico validation that RL-based basal rate controllers can maintain glucose within target ranges. Your work with Dr. Georgiou's group at Imperial College demonstrated the potential for deep RL to optimize continuous basal delivery, a control paradigm that is central to modern insulin pump therapy.

I am writing to share GlucoSim, a new open-source platform providing three Gymnasium-compatible RL environments for glucose management. One of the three environments, BasalControl-v0, directly targets the basal rate optimization problem you addressed, with 24-hour episodes at 1-minute resolution and configurable difficulty tiers. Our experiments yielded a finding relevant to your work: the Bergman model's homeostatic feedback makes random basal delivery surprisingly effective (76.7% TIR), suggesting that basal control benchmarks benefit from composite reward functions that go beyond simple zone-based scoring.

- **GitHub:** https://github.com/HassDhia/glucosim
- **PyPI:** https://pypi.org/project/glucosim/
- **Paper:** https://github.com/HassDhia/glucosim/blob/main/paper/glucosim.pdf

GlucoSim ships with 117 tests, trained PPO baselines, and is MIT licensed. If you or your group find it useful, I would welcome any feedback. A brief mention, star, or share would be greatly appreciated.

Best regards,
Hass Dhia
Smart Technology Investments Research Institute
partners@smarttechinvest.com
https://smarttechinvest.com/research
