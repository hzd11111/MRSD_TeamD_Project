#!/usr/bin/env python
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
import rospy
import copy
import threading
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
print(sys.path)
print(sys.version_info)
from std_msgs.msg import String
from grasp_path_planner.msg import LanePoint
from grasp_path_planner.msg import Lane
from grasp_path_planner.msg import VehicleState
from grasp_path_planner.msg import RewardInfo
from grasp_path_planner.msg import EnvironmentState
from grasp_path_planner.msg import RLCommand

import dqn_manager

QUEUE_SIZE = 1
NODE_NAME = 'rl_node'
RL_TOPIC_NAME = 'rl_decision'
ENVIRONMENT_TOPIC_NAME = 'environment_state'

# DQN Settings
##############################################
settings = {
	"BATCH_SIZE" : 64,
	"GAMMA" : 0.99,
	"EPS_START" : 0.9,
	"EPS_END" : 0.05,
	"EPS_DECAY" : 200,
	"TARGET_UPDATE" : 10,
	"INPUT_HEIGHT" : 0,
	"INPUT_WIDTH" : 25,
	"CAPACITY" : 10000,
	"N_ACTIONS": 4,
	"DEVICE" : "cpu"
}
##############################################
class RLManager:
	def __init__(self):
		self.pub_rl = None
		self.env_sub = None
		self.rl_agent = None
		self.lock = threading.Lock()
		self.rl_decision = False
		self.previous_id = -1
		self.manager = dqn_manager.DQNManager(*settings.values())
		self.manager.initialize()
		self.previous_state = None
		self.previous_action = None

	def simCallback(self, data):
		self.lock.acquire()
		if data.id == self.previous_id:
			self.lock.release()
			return
		iter_start_time = rospy.Time.now()
		print("RL Sim Msg Delay", (rospy.Time.now() - data.sent_time).nsecs * 1e-6)
		env_state = self.make_state_vector(data)
		env_state = np.array(env_state)
		# computation with rl_agent
		rl_decision = self.make_decision(env_state, data.id)
		rl_decision.reset_run = self.manager.exitCondition(data)
		print("Decision Duration", (rospy.Time.now() - iter_start_time).nsecs * 1e-6)

		# create a record
		if self.previous_action is not None and self.previous_state is not None:
			iter_start_time = rospy.Time.now()
			reward = self.manager.rewardCalculation(data)
			reward = torch.tensor([reward]).to(settings["DEVICE"])
			env_tensor = torch.tensor(env_state).float().view(1,-1).to(settings["DEVICE"])
			self.manager.memory.push(self.previous_state,self.previous_action,env_tensor,reward)
			self.manager.optimize_model()
			print("Model Optimization Duration", (rospy.Time.now() - iter_start_time).nsecs * 1e-6)
		self.previous_id = data.id
		self.previous_state = torch.tensor(env_state).float().view(1,-1).to(settings["DEVICE"])
		# publish message
		rl_decision.sent_time = rospy.Time.now()
		self.pub_rl.publish(rl_decision)
		self.rl_decision = rl_decision
		#print("Published:",rl_decision.id)
		self.lock.release()

	def initialize(self):
		#initialize node
		rospy.init_node(NODE_NAME, anonymous=True)
		# initialize subscriber
		self.env_sub = rospy.Subscriber(ENVIRONMENT_TOPIC_NAME, 
							EnvironmentState, self.simCallback)
		# initialize pulbisher
		self.pub_rl = rospy.Publisher(RL_TOPIC_NAME, RLCommand, queue_size = QUEUE_SIZE)
		# initlialize rl class
		
	def publishFunc(self):
		rate = rospy.Rate(10)
		while not rospy.is_shutdown():
			self.lock.acquire()
			if self.rl_decision:
				self.rl_decision.sent_time = rospy.Time.now()
				self.pub_rl.publish(self.rl_decision)
			self.lock.release()
			rate.sleep()

	def make_state_vector(self, data):
		'''
		create a state vector from the message recieved
		'''
		env_state = []
		def append_vehicle_state(env_state, vehicle_state):
			env_state.append(vehicle_state.vehicle_location.x)
			env_state.append(vehicle_state.vehicle_location.y)
			env_state.append(vehicle_state.vehicle_location.theta)
			env_state.append(vehicle_state.vehicle_speed)
			return
		# items needed
		# current vehicle velocity
		env_state.append(data.cur_vehicle_state.vehicle_speed)
		# rear vehicle state
			# position
			# velocity
		append_vehicle_state(env_state, data.back_vehicle_state)
		# adjacent vehicle state
			# position
			# velocity
		i = 0
		for _, veh_state in enumerate(data.adjacent_lane_vehicles):
			if i < 5:
				append_vehicle_state(env_state, veh_state)
			else:
				break
			i+=1
		dummy = VehicleState()
		dummy.vehicle_location.x = 10000
		dummy.vehicle_location.y = 10000
		dummy.vehicle_location.theta = 10000
		dummy.vehicle_speed = 0
		while i<5:
			append_vehicle_state(env_state, dummy)
			i+=1
		return env_state
		
	def spin(self):
		# spin
		rospy.spin()

	def make_decision(self, env_state, id):
		'''
		env_state is an array of a fixed length
		'''
		# action_vals = self.offline_network(env_state)
		# max_val, arg_val = torch.max(action_vals,1)
		state_tensor = torch.Tensor(env_state).to(settings["DEVICE"])
		arg_val = self.manager.selectAction(state_tensor).item()
		#arg_val = self.manager.target_net(state_tensor).view(-1,4).max(1)[1].view(1,1).item()
		self.previous_action = torch.tensor([[arg_val]]).long().to(settings["DEVICE"])
		# arg_val = 1
		rl_command = RLCommand()
		# arg_val = 0
		if arg_val == 0:
			rl_command.accelerate = 1
		elif arg_val == 1:
			rl_command.decelerate = 1
		elif arg_val == 2:
			rl_command.change_lane = 1
		elif arg_val == 3:
			rl_command.constant_speed = 1
		rl_command.id = id
		return rl_command

	def is_terminate(self, env_state):
		if env_state.reward.collision:
			return True
		return False

	def train(self, env_state):
		self.manager.optimize_model()
		return
if __name__ == '__main__':
	try:
		rl_manager = RLManager()
		rl_manager.initialize()
		pub_thread = threading.Thread(target=rl_manager.publishFunc)
		pub_thread.start()
		rl_manager.spin()
	except rospy.ROSInterruptException:
		pass
