from gym.envs.registration import register

register(
    id="Drx-v0",
    entry_point="drx_env.envs:DRXEnv",
)
