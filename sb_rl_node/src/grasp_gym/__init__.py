from gym.envs.registration import register

register(
    id='lane_change-v0',
    entry_point='grasp_gym.envs:LaneChange',
)