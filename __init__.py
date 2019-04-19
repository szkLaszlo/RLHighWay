from gym.envs.registration import register

register(
    id='EPHighWay-v1',
    entry_point='envs.SUMO_Starter:EPHighWayEnv',
)

