import gym
import gym_racer
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines import DQN,PPO2
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
sensor_array_params["ray_sensors_per_ray"] = 25
sensor_array_params["ray_max_angle"] = 100
sensor_array_params["viewfield_size"] = 30
sensor_array_params["viewfield_step"] = 8

# env = gym.make('CartPole-v1')
env = gym.make(
        "racer-v1",
        sensor_array_type=sat,
        render_mode=mode,
        sensor_array_params=sensor_array_params,
    )



# lr = 1e-4
model = DQN(MlpPolicy, env, verbose=1, tensorboard_log='./Logs/')
model.learn(total_timesteps=35000)
del model # remove to demonstrate saving and loading


model = DQN.load("deepq_cartpole")

obs = env.reset()
clock = pygame.time.Clock()
done = False
while not done:
    action, _states = model.predict(obs)
    print(action)
    obs, reward, done, info = env.step(action)
    env.render(reward=reward, mode=mode)
    pygame.event.get()
    clock.tick(2)

