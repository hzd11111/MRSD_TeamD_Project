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
import numpy as np
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
# For Visualisation
import matplotlib.pyplot as plt
import matplotlib.animation as animation
# Threading
from threading import Thread, Lock
# -----------------------------------Global------------------------------------------------------#
dir_path = os.path.dirname(os.path.realpath(__file__))
animation_lock = Lock()
action_probs = None
if CURRENT_SCENARIO == Scenario.PEDESTRIAN:
    action_probs = [0,0,0]
else:
    action_probs = [0,0,0,0]
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ani_obj = None
# -----------------------------------Code------------------------------------------------------#

def animate_callback(x):
    global action_probs
    print("ENTERED CALLBACK")
    print(action_probs)
    ax.clear()
    animation_lock.acquire()
    if CURRENT_SCENARIO == Scenario.LANE_CHANGE:
        ax.bar(x=[1,2,3,4], height = action_probs, tick_label=["constant","accelerate","decelerate","lane_change"])
    else:
        ax.bar(x=[1,2,3], height = action_probs, tick_label=["constant","accelerate","decelerate"])
    animation_lock.release()

def animate():
    animation.FuncAnimation(fig, animate_callback, interval=1000)
    plt.show()

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

        if self.event == Scenario.PEDESTRIAN:
            model = DQN(CustomPedestrianPolicy, env, verbose=1, learning_starts=256, batch_size=256, exploration_fraction=0.9, target_network_update_freq=100, tensorboard_log=dir_path+'/Logs/Ped', gamma=0.93, learning_rate=0.0001)
        model.learn(total_timesteps=20000)
        model.save(dir_path+"/Models/DQN_Model_CARLA_Ped")

    def run_test(self):
        global action_probs
        env = CustomEnv(self.path_planner, self.behavior_planner, event)
        env = make_vec_env(lambda: env, n_envs=1)
        if(self.event == Scenario.LANE_CHANGE):
            model = DQN.load(dir_path+"/DQN_20min")
        if(self.event == Scenario.PEDESTRIAN):
            model = DQN.load(dir_path+"/Models/DQN_Model_CARLA_Ped")
        obs = env.reset()
        count = 0
        success = 0
        while count < 500:
            done = False
            print("Count ", count, "Success ", success)
            while not done:
                action, _ = model.predict(obs)
                vals = model.action_probability(obs)
                animation_lock.acquire()
                action_probs = list(vals[0])
                animation_lock.release()
                print(action_probs)
                print(action)
                obs, reward, done, info = env.step(action)
                print("Reward",reward)
            count += 1
            if info[0]["success"]:
                success += 1
        print("Success Rate ", success / count, success, count)

if __name__ == '__main__':
    try:
        event = CURRENT_SCENARIO
        if event==Scenario.PEDESTRIAN:
            full_planner = FullPlannerManager(Scenario.PEDESTRIAN)
        elif event == Scenario.LANE_CHANGE:
            full_planner = FullPlannerManager(Scenario.LANE_CHANGE)
        full_planner.initialize()
        thread = Thread(target=full_planner.run_test)
        thread.start()
        ani_obj = animation.FuncAnimation(fig, animate_callback, interval=100)
        plt.show()
        # full_planner.run_test()

    except rospy.ROSInterruptException:
        pass