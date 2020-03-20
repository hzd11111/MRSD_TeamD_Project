#!/usr/bin/env python3
# Software License Agreement (BSD License)
#
# Copyright (c) 2008, Willow Garage, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of Willow Garage, Inc. nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Revision $Id$

## Simple talker demo that published std_msgs/Strings messages
## to the 'chatter' topic

import sys
sys.path.append("/home/arcot/GRASP/src")
# to remove tensorflow warnings
import warnings
warnings.filterwarnings("ignore")
import os,logging
logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import rospy
import copy
import threading
import numpy as np
from std_msgs.msg import String
from grasp_path_planner.msg import LanePoint
from grasp_path_planner.msg import Lane
from grasp_path_planner.msg import VehicleState
from grasp_path_planner.msg import RewardInfo
from grasp_path_planner.msg import EnvironmentState
from grasp_path_planner.msg import RLCommand
import gym
import time
from gym import spaces
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

from stable_baselines import DQN,PPO2
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.deepq.policies import MlpPolicy
# import tensorflow.python.util.deprecation as deprecation
from stable_baselines.common.env_checker import check_env
from stable_baselines.common.cmd_util import make_vec_env


# deprecation._PRINT_DEPRECATION_WARNINGS = False


NODE_NAME = 'sb_rl_node'
RL_TOPIC_NAME = 'rl_decision'
ENVIRONMENT_TOPIC_NAME = 'environment_state'
N_DISCRETE_ACTIONS = 4
LANE_CHANGE = 1
CONSTANT_SPEED = 0
ACCELERATE = 2
DECELERATE = 3

# make custom environment class to only read values. It should not directly change values in the other thread.
class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface"""

    metadata = {'render.modes': ['human']}

    def __init__(self, rl_manager):
        super(CustomEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        self.observation_space = spaces.Box(low = -1000, high=1000, shape = (1,25))
        self.rl_manager = rl_manager

    def step(self, action):
        # reset sb_event flag if previously set in previous action
        print("SENDING ACTION",action)
        self.rl_manager.sb_event.clear()
        rl_manager.make_rl_message(action, False)
        self.rl_manager.sb_event.wait()
        # access values from the other thread
        # rl_manager.lock.acquire()
        reward=rl_manager.reward
        cur_state=rl_manager.previous_state
        done=cur_state.reward.collision
        # if lane change was the action then reset the sim as it was a success
        if action==LANE_CHANGE:
            done=True
        # rl_manager.lock.release()
        # return observation, reward, done, info
        print(reward,done)
        return rl_manager.make_state_vector(cur_state), reward, done, {}

    def reset(self):
        # reset sb_event flag if previously set in previous action
        print("SENDING RESET")
        self.rl_manager.sb_event.clear()
        rl_manager.make_rl_message(None, True)
        self.rl_manager.sb_event.wait()
        print("Returned to RESET")
        self.rl_manager.previous_reward = None
        return rl_manager.make_state_vector(rl_manager.cur_state)
        # return observation  # reward, done, info can't be included

    def render(self, mode='human'):
        pass

    def close (self):
        pass


class RLManager:
    def __init__(self):
        self.sb_event=threading.Event()
        self.action_event=threading.Event()
        self.lock=threading.Lock()
        self.cur_id=0
        self.cur_state=None
        self.previous_state=None
        self.terminal_option_data=None
        self.reward=0
        self.k1=1
        self.cur_action=None
        self.lane_term_th=1e-2
        self.previous_reward=None

    def is_terminal_option(self, data):
        if(self.previous_reward == None):
            print("Option Time elapsed is", data.reward.time_elapsed)
        else:
            print("Option Time elapsed is", data.reward.time_elapsed - self.previous_reward.time_elapsed)
        if self.cur_action.constant_speed or self.cur_action.accelerate or self.cur_action.decelerate:
            if(self.previous_reward == None):
                return data.reward.time_elapsed>1
            return (data.reward.time_elapsed - self.previous_reward.time_elapsed)>1
        else:
            return data.reward.new_run

    def calculate_lane_change_reward(self, data):
        # TODO: Generalise for curved roads
        diff_1=data.next_lane.lane[0].pose.y-data.cur_vehicle_state.vehicle_location.y
        diff_2=data.next_lane.lane[0].pose.y-data.current_lane.lane[0].pose.y
        return abs(diff_1/diff_2)

    def calculate_ego_reward(self, data):
        return 0

    def update_reward(self, data):
        if data.reward.collision:
            self.reward=-1
        elif self.cur_action.change_lane:
            if data.reward.new_run and self.previous_state is not None:
                self.reward=1
        else:
            self.reward=0
            
        self.previous_reward = data.reward

    def is_new_episode(self,data):
        return data.reward.new_run or data.reward.time_elapsed>20
    
    def is_terminate_episode(self, data):
        print("Episode Time elapsed is", data.reward.time_elapsed)
        return data.reward.collision or data.reward.time_elapsed>10

    def simCallback(self, data):
        # Pause until an action is given to execute action to execute
        if not self.action_event.is_set():
            print("Paused")
            return
        # entering critical section 
        # self.lock.acquire()
        # print("Entered SimCallback")
        # Check if new message
        if data.id==self.cur_id:
            # print("Got same message ", self.cur_action)
            if self.cur_action is not None:
                self.pub_rl.publish(self.cur_action)
            # self.lock.release()
            return
        # update reward
        # self.update_reward(data)
        # set current state
        self.previous_state=self.cur_state
        self.cur_state=data
        self.cur_id=data.id
        # reset conditions check
        if self.is_new_episode(data) and \
                self.cur_action is not None and \
                    self.cur_action.reset_run is True:
            # print("New Episode ",self.is_new_episode(data))
            if not self.sb_event.is_set():
                self.sb_event.set()
                self.action_event.clear()     
        # check if current option is terminated or collision
        elif self.is_terminal_option(data) or self.is_terminate_episode(data): 
            # print("Option ",self.is_terminal_option(data), data.reward.new_run)
            # print("Collision ",data.reward.collision)
            self.update_reward(data)
            if not self.sb_event.is_set():
                self.sb_event.set()
                self.action_event.clear()
        else:
            self.cur_action.id=self.cur_id
            self.pub_rl.publish(self.cur_action)
            # print("Republishing",self.cur_action)
        # self.lock.release()
        # exiting critcal section

    def initialize(self):
        #initialize node
        self.sb_event.clear()
        self.action_event.clear()
        rospy.init_node(NODE_NAME, anonymous=True)
        self.env_sub = rospy.Subscriber(ENVIRONMENT_TOPIC_NAME, 
                            EnvironmentState, self.simCallback)
        self.pub_rl = rospy.Publisher(RL_TOPIC_NAME, RLCommand, queue_size = 10)

    def spin(self):
        rospy.spin()

    def append_vehicle_state(self, env_state, vehicle_state):
        env_state.append(vehicle_state.vehicle_location.x)
        env_state.append(vehicle_state.vehicle_location.y)
        env_state.append(vehicle_state.vehicle_location.theta)
        env_state.append(vehicle_state.vehicle_speed)
        return

    def make_state_vector(self, data):
        '''
        create a state vector from the message recieved
        '''
        i=0
        env_state = []
        env_state.append(data.cur_vehicle_state.vehicle_speed)
        self.append_vehicle_state(env_state, data.back_vehicle_state)
        for _, veh_state in enumerate(data.adjacent_lane_vehicles):
            if i < 5:
                self.append_vehicle_state(env_state, veh_state)
            else:
                break
            i+=1
        dummy = VehicleState()
        dummy.vehicle_location.x = 100
        dummy.vehicle_location.y = 100
        dummy.vehicle_location.theta = 0
        dummy.vehicle_speed = 0
        while i<5:
            self.append_vehicle_state(env_state, dummy)
            i+=1
        return env_state

    def make_rl_message(self, action, is_reset=False):
        self.reward=0
        # Uncomment for debugging
        # action=1
        rl_command=RLCommand()
        if action == LANE_CHANGE:
            rl_command.change_lane=1
        elif action == CONSTANT_SPEED:
            rl_command.constant_speed=1
        elif action == ACCELERATE:
            rl_command.accelerate=1
        elif action == DECELERATE:
            rl_command.decelerate=1
            
        if is_reset:
            rl_command.reset_run=True
        rl_command.id = self.cur_id
        self.cur_action=rl_command
        self.pub_rl.publish(self.cur_action)
        self.action_event.set()
        return rl_command

    def convert_to_local(self, veh_state, x,y):
        H = np.eye(3)
        H[-1,-1] = 1
        H[0,-1] = -veh_state.vehicle_location.x
        H[1,-1] = -veh_state.vehicle_location.y
        H[0,0] = np.cos(-veh_state.vehicle_location.theta)
        H[0,1] = np.sin(-veh_state.vehicle_location.theta)
        H[1,0] = -np.sin(-veh_state.vehicle_location.theta)
        H[1,1] = np.cos(-veh_state.vehicle_location.theta)
        res = np.matmul(H, np.array([x,y,1]).reshape(3,1))
        return res[0,0], res[1,0]


def sb_model_train(rl_manager):
    env=CustomEnv(rl_manager)
    env=make_vec_env(lambda:env, n_envs=1)
    model = DQN(MlpPolicy, env, verbose=1, tensorboard_log='./Logs/')
    model.learn(total_timesteps=500)
    model.save("DQN_Model_SimpleSim")
    return

if __name__ == '__main__':
    try:
        rl_manager = RLManager()
        rl_manager.initialize()
        sb_model_thread = threading.Thread(target=sb_model_train, args=(rl_manager,))
        sb_model_thread.start()
        rl_manager.spin()

    except rospy.ROSInterruptException:
        pass
