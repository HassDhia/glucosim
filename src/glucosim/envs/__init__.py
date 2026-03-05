"""Gymnasium environment registration for GlucoSim."""

from gymnasium.envs.registration import register

register(
    id="glucosim/BasalControl-v0",
    entry_point="glucosim.envs.basal_control:BasalControlEnv",
)
register(
    id="glucosim/BolusAdvisor-v0",
    entry_point="glucosim.envs.bolus_advisor:BolusAdvisorEnv",
)
register(
    id="glucosim/ClosedLoop-v0",
    entry_point="glucosim.envs.closed_loop:ClosedLoopEnv",
)
