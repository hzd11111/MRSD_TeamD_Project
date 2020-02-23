from gym.envs.registration import register

# define the name to call the env
# gym.make("racer-v0")
# print("Debugging here ")
register(id="racer-v1", entry_point="gym_racer.envs:RacerEnv")
