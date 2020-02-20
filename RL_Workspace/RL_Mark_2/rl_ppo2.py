import gym
import gym_racer
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2,DQN
import tensorflow.python.util.deprecation as deprecation
import pygame


deprecation._PRINT_DEPRECATION_WARNINGS = False
mode = "human"
#  mode = "console"
#  sat = "diamond"
sat = "lidar"
FPS = 2
sensor_array_params = {}
sensor_array_params["ray_num"] = 7
sensor_array_params["ray_step"] = 10
sensor_array_params["ray_sensors_per_ray"] = 13
sensor_array_params["ray_max_angle"] = 130
sensor_array_params["viewfield_size"] = 30
sensor_array_params["viewfield_step"] = 8
env = gym.make(
        "racer-v1",
        sensor_array_type=sat,
        render_mode=mode,
        sensor_array_params=sensor_array_params,
    )
# env = gym.make("CartPole-v1")

# Optional: PPO2 requires a vectorized environment to run
# the env is now wrapped automatically when passing it to the constructor
# env = DummyVecEnv([lambda: env])

# print(env.reset().shape)

'''
model = PPO2(MlpPolicy, env, verbose=1,tensorboard_log="./Logs/")
model.learn(total_timesteps=20000)
model.save("car_racer_v1")
'''

model = PPO2.load("car_racer_v1")
obs = env.reset()
clock = pygame.time.Clock()
for i in range(1500):
    action, _states = model.predict(obs)
    print(action)
    obs, reward, done, info = env.step(action)
    env.render(mode=mode, reward=reward)
    pygame.event.get()
    clock.tick(FPS)
