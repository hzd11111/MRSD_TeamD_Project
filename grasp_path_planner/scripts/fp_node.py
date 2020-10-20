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
import numpy as np
# RL packages
from stable_baselines import DQN
from stable_baselines.common.cmd_util import make_vec_env
# Other Packages
from path_planner import PathPlannerManager
from rl_manager import RLManager, GeneralRLManager
from lane_change_policy import CustomLaneChangePolicy
from intersection_policy import CustomIntersectionStraight, CustomIntersectionLeftTurn, CustomIntersectionRightTurn
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
        if self.event == Scenario.SWITCH_LANE_LEFT or\
                self.event == Scenario.SWITCH_LANE_RIGHT:
            model = DQN(CustomLaneChangePolicy, env, verbose=1,
                        learning_starts=256, batch_size=16,
                        exploration_fraction=0.9, target_network_update_freq=100,
                        tensorboard_log=dir_path + '/Logs/LaneChange')

        if self.event == Scenario.LANE_FOLLOWING:
            model = DQN(CustomLaneFollowingPolicy, env, verbose=1,
                        learning_starts=256, batch_size=256, exploration_fraction=0.9,
                        target_network_update_freq=100,
                        tensorboard_log=dir_path + '/Logs/Follow', gamma=0.93, learning_rate=0.0001)

        if self.event == Scenario.GO_STRAIGHT:
            model = DQN(CustomIntersectionStraight, env, verbose=1,
                        learning_starts=256, batch_size=256, exploration_fraction=0.9,
                        target_network_update_freq=100,
                        tensorboard_log=dir_path + '/Logs/Straight', gamma=0.93, learning_rate=0.0001)

        if self.event == Scenario.LEFT_TURN:
            model = DQN(CustomIntersectionLeftTurn, env, verbose=1,
                        learning_starts=256, batch_size=256, exploration_fraction=0.9,
                        target_network_update_freq=100,
                        tensorboard_log=dir_path + '/Logs/LeftTurn', gamma=0.93, learning_rate=0.0001)

        if self.event == Scenario.RIGHT_TURN:
            model = DQN(CustomIntersectionRightTurn, env, verbose=1,
                        learning_starts=256, batch_size=256, exploration_fraction=0.9,
                        target_network_update_freq=100,
                        tensorboard_log=dir_path + '/Logs/RightTurn', gamma=0.93, learning_rate=0.0001)

        model.learn(total_timesteps=20000)
        model.save(MODEL_SAVE_PATH)

    def run_test(self):
        env = CustomEnv(self.path_planner, self.behavior_planner, self.event)
        env = make_vec_env(lambda: env, n_envs=1)
        decision_maker = GeneralRLManager()
        model = DQN.load(MODEL_LOAD_PATH)
        obs = env.reset()
        count = 0
        success = 0
        while count < 500:
            done = False
            while not done:
                # action, _ = model.predict(obs)
                # import ipdb; ipdb.set_trace()
                action = np.array([env.action_space.sample()])
                action_enum = decision_maker.convertDecision(action[0], self.event)
                print("Action taken:", action_enum)
                obs, reward, done, info = env.step(action)
                # print("Reward",reward)
            count += 1
            if info[0]["success"]:
                success += 1
            print("Count ", count, "Success ", success,
                  "Success Rate:", success * 100 / float(count), "%")
        print("Success Rate ", success / count, success, count)


if __name__ == '__main__':
    try:
        event = CURRENT_SCENARIO
        full_planner = FullPlannerManager(event)
        full_planner.initialize()

        if(CURRENT_MODE == Mode.TEST):
            full_planner.run_test()
        else:
            full_planner.run_train()

    except rospy.ROSInterruptException:
        pass
