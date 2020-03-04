#!/usr/bin/env python2.7
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

import rospy
import copy
import threading

from std_msgs.msg import String
from grasp_path_planner.msg import LanePoint
from grasp_path_planner.msg import Lane
from grasp_path_planner.msg import VehicleState
from grasp_path_planner.msg import RewardInfo
from grasp_path_planner.msg import EnvironmentState
from grasp_path_planner.msg import RLCommand

NODE_NAME = 'rl_node'
RL_TOPIC_NAME = 'rl_decision'
ENVIRONMENT_TOPIC_NAME = 'environment_state'

class RLManager:
	def __init__(self):
		self.pub_rl = None
		self.env_sub = None
		self.rl_agent = None
		self.lock = threading.Lock()
		self.rl_decision = False
		self.previous_id = -1
	def simCallback(self, data):
		self.lock.acquire()
		if data.id == self.previous_id:
			self.lock.release()
			return
		# sample code for data loading ToDo: Delete
		current_vehicle = data.cur_vehicle_state
		adjacent_lane_vehicles = data.adjacent_lane_vehicles
		current_lane = data.current_lane
		
		cur_vehicle_location = current_vehicle.vehicle_location
		cur_vehicle_speed = current_vehicle.vehicle_speed
		
		print('New Environment State')
		print('Current Vehicle Speed', cur_vehicle_speed)
		
		for vehicle in adjacent_lane_vehicles:
			# load them the same as current_vehicle
			speed = vehicle.vehicle_speed

		# computation with rl_agent
	
		
		# dummy msg ToDo: Delete these when rl is done
		rl_decision = RLCommand()
		if True:#data.id < 150:
			rl_decision.change_lane = 0
			rl_decision.constant_speed = 1
			rl_decision.accelerate = 0
			rl_decision.decelerate = 0
			rl_decision.reset_run = 0
			rl_decision.id = data.id		
		else:
			rl_decision.change_lane = 1
			rl_decision.constant_speed = 0
			rl_decision.accelerate = 0
			rl_decision.decelerate = 0
			rl_decision.reset_run = 0
			rl_decision.id = data.id		
			
		self.previous_id = data.id
		# publish message
		self.pub_rl.publish(rl_decision)
		self.rl_decision = rl_decision
		print "Published:",rl_decision.id
		
		self.lock.release()

	def initialize(self):
		
		#initialize node
		rospy.init_node(NODE_NAME, anonymous=True)

		# initialize subscriber
		self.env_sub = rospy.Subscriber(ENVIRONMENT_TOPIC_NAME, EnvironmentState, self.simCallback)

		# initialize pulbisher
		self.pub_rl = rospy.Publisher(RL_TOPIC_NAME, RLCommand, queue_size = 10)
		# initlialize rl class
		
	def publishFunc(self):
		rate = rospy.Rate(10)
		while not rospy.is_shutdown():
			self.lock.acquire()
			if self.rl_decision:
				self.pub_rl.publish(self.rl_decision)
			self.lock.release()
			rate.sleep()

	def spin(self):
		# spin
		rospy.spin()

if __name__ == '__main__':
    try:
        rl_manager = RLManager()
	rl_manager.initialize()
	pub_thread = threading.Thread(target=rl_manager.publishFunc)
	rl_manager.spin()
    except rospy.ROSInterruptException:
        pass
