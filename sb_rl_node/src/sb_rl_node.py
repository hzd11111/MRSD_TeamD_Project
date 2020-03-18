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
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from std_msgs.msg import String
from grasp_path_planner.msg import LanePoint
from grasp_path_planner.msg import Lane
from grasp_path_planner.msg import VehicleState
from grasp_path_planner.msg import RewardInfo
from grasp_path_planner.msg import EnvironmentState
from grasp_path_planner.msg import RLCommand
import gym
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
N_DISCRETE_ACTIONS = 2
LANE_CHANGE = 1
CONSTANT_SPEED = 0

class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface"""

    metadata = {'render.modes': ['console']}

    def __init__(self, rl_manager):
        super(CustomEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        self.observation_space = spaces.Box(low = -1000, high=1000, shape = (1,25))
        self.rl_manager = rl_manager

    def step(self, action):
        # reset reward to zero
        self.rl_manager.reward=0
        while rl_manager.cur_state is None:
            print("Waiting in step")
            pass
        rl_command = rl_manager.make_rl_message(action, rl_manager.cur_state.id, False)
        # acquire lock
        rl_manager.sb_lock.acquire()
        rl_manager.pub_rl.publish(rl_command)
        # wait till option finishes
        rl_manager.sb_lock.acquire()
        print("Action executed",action)
        rl_manager.sb_lock.release()
        reward=rl_manager.reward
        cur_state=rl_manager.cur_state
        cur_state_vec=np.array(rl_manager.make_state_vector(cur_state)).astype(np.float32)
        rl_manager.lock.acquire()
        if action==LANE_CHANGE:
            done=True
        else:    
            done=cur_state.reward.collision
        rl_manager.lock.release()
        # return observation, reward, done, info
        return rl_manager.make_state_vector(cur_state), reward, done, None

    def reset(self):
        id=None
        while rl_manager.cur_state is None:
            # print("Waiting in reset")
            pass
        id=rl_manager.cur_state.id
        rl_manager.sb_lock.acquire()
        rl_manager.make_rl_message(id,CONSTANT_SPEED,True)
        rl_manager.sb_lock.acquire()
        print("Reset executed")
        rl_manager.sb_lock.release()
        return np.array(rl_manager.make_state_vector(rl_manager.cur_state)).astype(np.float32)
        # return observation  # reward, done, info can't be included

    def render(self, mode='console'):
        pass

    def close (self):
        pass


class RLManager:
    def __init__(self):
        self.sb_lock=threading.Lock()
        self.lock=threading.Lock()
        self.action_lock=threading.Lock()
        self.cur_id=None
        self.cur_state=None
        self.terminal_option_data=None
        self.reward=0
        self.k1=1
        self.cur_action=None
        self.lane_term_th=1e-2
        self.pub_rl=None
        self.env_sub=None

    def is_terminal_option(self, data,action):
        if action==CONSTANT_SPEED:
            # print("Constant speed termination")
            return True
        else:
            lateral_dist=np.abs(data.cur_vehicle_state.vehicle_location.y- \
                                    data.next_lane.lane[0].pose.y)
            print("Lateral Dist",lateral_dist)
            if lateral_dist<self.lane_term_th:
                return True
            return False

    def calculate_lane_change_reward(self, data):
        # TODO: Generalise for curved roads
        diff_1=data.next_lane.lane[0].pose.y-data.cur_vehicle_state.vehicle_location.y
        diff_2=data.next_lane.lane[0].pose.y-data.current_lane.lane[0].pose.y
        return abs(diff_1/diff_2)

    def calculate_ego_reward(self, data):
        return 0

    def update_reward(self, data):
        r_ego = self.calculate_ego_reward(data)
        r_lane_change = self.calculate_lane_change_reward(data)
        self.reward+=r_ego
        self.reward+=r_lane_change

    def is_new_episode(self,data):
        x=data.cur_vehicle_state.vehicle_location.x
        y=data.cur_vehicle_state.vehicle_location.y
        cur=np.array([x,y])
        start=np.array([10.08,4])
        if np.allclose(cur,start):
            return True

    def simCallback(self, data):
        self.lock.acquire()
        # Check if new message
        if data.id==self.cur_id:
            if(self.sb_lock.locked()):
                self.sb_lock.release()
            self.lock.release()
            return
        # update reward
        self.update_reward(data)
        # check if collision
        if data.reward.collision:              
            # print("Collision ",data.reward.collision)
            # print(self.cur_action)
            if self.sb_lock.locked():
                self.sb_lock.release()
        else:
            # copy current command
            self.action_lock.acquire()
            ongoing_action=self.cur_action
            self.action_lock.release()
            # decisions to make
            if ongoing_action is not None:
                # identify current action
                if ongoing_action.constant_speed:
                    action=CONSTANT_SPEED
                else:
                    action=LANE_CHANGE
                # take decisions
                if ongoing_action.reset_run:
                    if self.is_new_episode(data):
                        print("New Episode ",self.is_new_episode(data))
                        if self.sb_lock.locked():
                            self.sb_lock.release()
                elif self.is_terminal_option(data,action):
                    print("Option ",self.is_terminal_option(data,action))
                    if self.sb_lock.locked():
                        self.sb_lock.release()
                else:
                    self.action_lock.acquire()
                    self.cur_action.id = data.id
                    self.pub_rl.publish(self.cur_action)
                    print("Republishing same decision ", data.id)
                    print("Published:")
                    print(self.cur_action)
                    self.action_lock.release()
        # set current state
        self.cur_state=data
        self.cur_id=data.id
        self.lock.release()

    def initialize(self):
        #initialize node
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
        dummy.vehicle_location.x = 10000
        dummy.vehicle_location.y = 10000
        dummy.vehicle_location.theta = 10000
        dummy.vehicle_speed = 0
        while i<5:
            self.append_vehicle_state(env_state, dummy)
            i+=1
        return env_state

    def make_rl_message(self, action, id, is_reset=False):
        rl_command=RLCommand()
        if action is LANE_CHANGE:
            rl_command.change_lane=1
        else:
            rl_command.constant_speed=1
        if is_reset:
            rl_command.reset_run=True
        rl_command.id = id
        self.action_lock.acquire()
        self.cur_action=rl_command
        self.action_lock.release()
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
    
    def publishFunc(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            self.lock.acquire()
            if self.cur_action:
                self.pub_rl.publish(self.cur_action)
            self.lock.release()
            rate.sleep()


def sb_model_train(rl_manager):
    env=CustomEnv(rl_manager)
    print("Checking environment")
    check_env(env)
    # while(True):
    #     print("Here")
    #     state,reward,done,_=env.step(LANE_CHANGE)
    #     print("state", state)
    #     print("reward", reward)
    #     print("done",done)
    #     if done:
    #         env.reset()
    # env=make_vec_env(lambda:env, n_envs=1)
    # model = DQN(MlpPolicy, env, verbose=1, tensorboard_log='./Logs/')
    # model.learn(total_timesteps=35000)
    return

if __name__ == '__main__':
    try:
        print("Start of main")
        rl_manager = RLManager()
        env=CustomEnv(rl_manager)
        rl_manager.initialize()
        # pub_thread = threading.Thread(target=rl_manager.publishFunc)
        # pub_thread.start()
        sb_model_thread = threading.Thread(target=sb_model_train, args=(rl_manager,))
        sb_model_thread.start()
        print("Started thread")
        rl_manager.spin()

    except rospy.ROSInterruptException:
        pass
