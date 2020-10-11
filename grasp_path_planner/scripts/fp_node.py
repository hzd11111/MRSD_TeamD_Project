#!/usr/bin/env python
# -----------------------------------Packages------------------------------------------------------#
import sys
print(sys.path)
import os
homedir = os.getenv("HOME")
distro = os.getenv("ROS_DISTRO")
sys.path.remove("/opt/ros/" + distro + "/lib/python2.7/dist-packages")
sys.path.append("/opt/ros/" + distro + "/lib/python2.7/dist-packages")
import rospy
# RL packages
from stable_baselines import DQN
from stable_baselines.common.cmd_util import make_vec_env
# Other Packages
from path_planner import PathPlannerManager
from rl_manager import RLManager
from lane_change_policy import CustomLaneChangePolicy
from lane_following_policy import CustomLaneFollowingPolicy
from custom_env import CustomEnv
from options import Scenario, RLDecision
from settings import Mode
from settings import MODEL_LOAD_PATH, MODEL_SAVE_PATH, CURRENT_MODE, CURRENT_SCENARIO
# -----------------------------------Global------------------------------------------------------#
dir_path = os.path.dirname(os.path.realpath(__file__))
# -----------------------------------Code------------------------------------------------------#


class FullPlannerManager:
    def __init__(self, event):
        self.path_planner = PathPlannerManager()
        self.behavior_planner = RLManager(event)
        self.event = event

    def initialize(self):
        self.path_planner.initialize()

    def run_train(self):
        env = CustomEnv(self.path_planner, self.behavior_planner, event)
        env = make_vec_env(lambda: env, n_envs=1)
        model = None
        if self.event == Scenario.SWITCH_LANE_LEFT:
            model = DQN(CustomLaneChangePolicy, env, verbose=1,
                        learning_starts=256, batch_size=16,
                        exploration_fraction=0.9, target_network_update_freq=100,
                        tensorboard_log=dir_path + '/Logs/')

        if self.event == Scenario.PEDESTRIAN:
            model = DQN(CustomLaneFollowingPolicy, env, verbose=1,
                        learning_starts=256, batch_size=256, exploration_fraction=0.9,
                        target_network_update_freq=100,
                        tensorboard_log=dir_path + '/Logs/Ped', gamma=0.93, learning_rate=0.0001)
        model.learn(total_timesteps=20000)
        model.save(MODEL_SAVE_PATH)

    def run_test(self):
        env = CustomEnv(self.path_planner, self.behavior_planner, event)
        env = make_vec_env(lambda: env, n_envs=1)
        if(self.event == Scenario.SWITCH_LANE_LEFT):
            model = DQN.load(MODEL_LOAD_PATH)
        if(self.event == Scenario.PEDESTRIAN):
            model = DQN.load(MODEL_LOAD_PATH)
        obs = env.reset()
        count = 0
        success = 0
        while count < 500:
            done = False
            while not done:
                action, _ = model.predict(obs)

                print("Action taken:", RLDecision(action))
                obs, reward, done, info = env.step(action)
                # print("Reward",reward)
            count += 1
            if info[0]["success"]:
                success += 1
            print("Count ", count, "Success ", success, "Success Rate:", success * 100 / float(count), "%")
        print("Success Rate ", success / count, success, count)


if __name__ == '__main__':
    try:
        event = CURRENT_SCENARIO
        if event == Scenario.PEDESTRIAN:
            full_planner = FullPlannerManager(Scenario.PEDESTRIAN)
        elif event == Scenario.SWITCH_LANE_LEFT:
            full_planner = FullPlannerManager(Scenario.SWITCH_LANE_LEFT)
        full_planner.initialize()

        if(CURRENT_MODE == Mode.TEST):
            full_planner.run_test()
        else:
            full_planner.run_train()

    except rospy.ROSInterruptException:
        pass
