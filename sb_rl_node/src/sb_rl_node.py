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
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import tensorflow as tf
import tensorflow.contrib as tf_contrib
import tensorflow.contrib.layers as tf_layers
from stable_baselines import DQN,PPO2
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines.deepq.policies import DQNPolicy
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

# make a custom policy
class CustomPolicy(DQNPolicy):
    """
    Policy object that implements DQN policy, using a MLP (2 layers of 64)

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param obs_phs: (TensorFlow Tensor, TensorFlow Tensor) a tuple containing an override for observation placeholder
        and the processed observation placeholder respectively
    :param dueling: (bool) if true double the output MLP to compute a baseline for action scores
    :param _kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """
    def embedding_net(self,input_vec):
        out = input_vec
        with tf.variable_scope("embedding_network", reuse=tf.AUTO_REUSE):
            out=tf_layers.fully_connected(out, num_outputs=16, activation_fn=tf.nn.relu)
            out=tf_layers.fully_connected(out, num_outputs=32, activation_fn=tf.nn.relu)
            out=tf_layers.fully_connected(out, num_outputs=64, activation_fn=tf.nn.relu)
        return out
    
    def q_net(self,input_vec):
        out=input_vec
        with tf.variable_scope("action_value"):
            out = tf_layers.fully_connected(out, num_outputs=64, activation_fn=tf.nn.relu)
            out = tf_layers.fully_connected(out, num_outputs=128, activation_fn=tf.nn.relu)
            out = tf_layers.fully_connected(out, num_outputs=4, activation_fn=tf.nn.tanh)
        return out
    
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False,
                 obs_phs=None, dueling=False, **kwargs):
        super(CustomPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps,
                                                n_batch, dueling=dueling, reuse=reuse,
                                                scale=False, obs_phs=obs_phs)
        with tf.variable_scope("model", reuse=reuse):
            out_ph = tf.layers.flatten(self.processed_obs)
            embed_list=[]
            for i in range(5):
                embed_list.append(self.embedding_net(tf.concat([out_ph[:,:4],out_ph[:,(i+1)*4:(i+2)*4]],axis=1)))
            stacked_out = tf.stack(embed_list,axis=1)
            max_out = tf.reduce_max(stacked_out, axis=1)
            q_out=self.q_net(max_out)
        self.q_values=q_out
        self._setup_init()

    def step(self, obs, state=None, mask=None, deterministic=True):
        q_values, actions_proba = self.sess.run([self.q_values, self.policy_proba], {self.obs_ph: obs})
        if deterministic:
            actions = np.argmax(q_values, axis=1)
        else:
            # Unefficient sampling
            # TODO: replace the loop
            # maybe with Gumbel-max trick ? (http://amid.fish/humble-gumbel)
            actions = np.zeros((len(obs),), dtype=np.int64)
            for action_idx in range(len(obs)):
                actions[action_idx] = np.random.choice(self.n_actions, p=actions_proba[action_idx])

        return actions, q_values, None

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})

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
        self.observation_space = spaces.Box(low = -1000, high=1000, shape = (1,24))
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
        done=self.rl_manager.done
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
        self.rl_manager.done=False
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
        self.episode_duration=30
        self.option_duration=0.05
        self.done=False
        self.finish=False

    def is_terminal_option(self, data):
        if(self.previous_reward == None):
            print("Option Time elapsed is", data.reward.time_elapsed)
        else:
            print("Option Time elapsed is", data.reward.time_elapsed - self.previous_reward.time_elapsed)
        if self.cur_action.constant_speed or self.cur_action.accelerate or self.cur_action.decelerate:
            if(self.previous_reward == None):
                return data.reward.time_elapsed>self.option_duration
            return (data.reward.time_elapsed - self.previous_reward.time_elapsed)>self.option_duration
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
                self.reward=5
        else:
            self.reward=0
            
        self.previous_reward = data.reward

    def is_new_episode(self,data):
        print("Episode time elapsed is ",data.reward.time_elapsed)
        return data.reward.new_run or data.reward.time_elapsed>self.episode_duration
    
    def is_terminate_episode(self, data):
        print("Episode Time elapsed is", data.reward.time_elapsed)
        self.done=data.reward.collision or data.reward.time_elapsed>self.episode_duration
        return self.done

    def simCallback(self, data):
        # Pause until an action is given to execute action to execute
        if self.finish:
            sys.exit()
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
        self.append_vehicle_state(env_state, data.cur_vehicle_state)
        # self.append_vehicle_state(env_state, data.back_vehicle_state)
        # self.append_vehicle_state(env_state, data.front_vehicle_state)
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

def sb_model_test(rl_manager):
    env=CustomEnv(rl_manager)
    env=make_vec_env(lambda:env, n_envs=1)
    model = DQN.load("DQN_Model_SimpleSim_40k")
    obs = env.reset()
    count=0
    success=0
    while count<100:
        done = False
        print("Count ",count, "Success ", success)
        while not done:
            action, _ = model.predict(obs)
            
            print(action)
            obs, reward, done, info = env.step(action)
        count+=1
        if reward==5:
            success+=1
    print("Success Rate ",success/count, success, count)
    rl_manager.finish=True

def sb_model_train(rl_manager):
    env=CustomEnv(rl_manager)
    env=make_vec_env(lambda:env, n_envs=1)
    # model = DQN(CustomPolicy, env, verbose=1, learning_starts=256, batch_size=256, exploration_fraction=0.5,  target_network_update_freq=10, tensorboard_log='./Logs/')
    # model = DQN(MlpPolicy, env, verbose=1, learning_starts=64,  target_network_update_freq=50, tensorboard_log='./Logs/')
    model = DQN.load("DQN_Model_SimpleSim_30k",env=env,exploration_fraction=0.1,tensorboard_log='./Logs/')
    model.learn(total_timesteps=10000)
    # model = PPO2(MlpPolicy, env, verbose=1,tensorboard_log="./Logs/")
    # model.learn(total_timesteps=20000)
    model.save("DQN_Model_SimpleSim_40k_1")
    # sb_model_test(rl_manager)
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
