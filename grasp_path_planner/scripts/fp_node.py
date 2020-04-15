#!/usr/bin/env python
# -----------------------------------Packages------------------------------------------------------#
import sys
import os
import time
from enum import Enum
homedir=os.getenv("HOME")
distro=os.getenv("ROS_DISTRO")
sys.path.remove("/opt/ros/"+distro+"/lib/python2.7/dist-packages")
sys.path.append("/opt/ros/"+distro+"/lib/python2.7/dist-packages")
import rospy
# RL packages
import gym
from stable_baselines import DQN,PPO2
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.env_checker import check_env
from stable_baselines.common.cmd_util import make_vec_env
# Other Packages
from path_planner import PathPlannerManager
from rl_manager import RLManager, CustomEnv, CustomLaneChangePolicy, CustomPedestrianPolicy
from settings import *

# -----------------------------------Global------------------------------------------------------#
dir_path = os.path.dirname(os.path.realpath(__file__))
# -----------------------------------Code------------------------------------------------------#

class FullPlannerManager:
    def __init__(self,event):
        self.path_planner = PathPlannerManager()
        self.behavior_planner = RLManager(event)
        self.event = event

    def initialize(self):
        self.path_planner.initialize()

    def run_train(self):
        env = CustomEnv(self.path_planner, self.behavior_planner, event)
        env = make_vec_env(lambda: env, n_envs=1)
        model = None
        if self.event == Scenario.LANE_CHANGE:
            model = DQN(CustomLaneChangePolicy, env, verbose=1, learning_starts=256, batch_size=256, exploration_fraction=0.9, target_network_update_freq=100, tensorboard_log=dir_path+'/Logs/')
            # model = DQN(MlpPolicy, env, verbose=1, learning_starts=64,  target_network_update_freq=50, tensorboard_log='./Logs/')
            # model = DQN.load("DQN_Model_SimpleSim_30k",env=env,exploration_fraction=0.1,tensorboard_log='./Logs/')
        if self.event == Scenario.PEDESTRIAN:
            model = DQN(CustomPedestrianPolicy, env, verbose=1, learning_starts=256, batch_size=256, exploration_fraction=0.5, target_network_update_freq=100, tensorboard_log=dir_path+'/Logs/Ped')
        model.learn(total_timesteps=10000)
        # model = PPO2(MlpPolicy, env, verbose=1,tensorboard_log="./Logs/")
        # model.learn(total_timesteps=20000)
        model.save(dir_path+"/Models/DQN_Model_SimpleSim_Ped")

    def run_test(self):
        env = CustomEnv(self.path_planner, self.behavior_planner, event)
        env = make_vec_env(lambda: env, n_envs=1)
        model = DQN.load(dir_path+"/DQN_20min.zip")
        obs = env.reset()
        count = 0
        success = 0
        while count < 500:
            done = False
            print("Count ", count, "Success ", success)
            while not done:
                action, _ = model.predict(obs)

                print(action)
                obs, reward, done, info = env.step(action)
                print("Reward",reward)
            count += 1
            if info[0]["success"]:
                success += 1
        print("Success Rate ", success / count, success, count)

if __name__ == '__main__':
    try:
        event = Scenario.LANE_CHANGE
        if event==Scenario.PEDESTRIAN:
            full_planner = FullPlannerManager(Scenario.PEDESTRIAN)
        elif event == Scenario.LANE_CHANGE:
            full_planner = FullPlannerManager(Scenario.LANE_CHANGE)
        full_planner.initialize()
        full_planner.run_test()

    except rospy.ROSInterruptException:
        pass