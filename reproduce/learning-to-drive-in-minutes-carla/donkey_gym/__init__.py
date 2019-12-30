from gym.envs.registration import register

register(
    id='DonkeyVae-v0',
    entry_point='donkey_gym.envs.vae_env:DonkeyVAEEnv',
)
