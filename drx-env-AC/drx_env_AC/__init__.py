from gym.envs.registration import register

register(
    id="Drx-v2",
    entry_point="drx_env_AC.envs:DRXEnv",
)
