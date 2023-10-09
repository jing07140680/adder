from gym.envs.registration import register

register(
    id="Drx-v1",
    entry_point="drx_env_plus.envs:DRXEnv",
)
