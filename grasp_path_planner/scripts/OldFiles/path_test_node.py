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
from std_msgs.msg import String
from grasp_path_planner.msg import LanePoint
from grasp_path_planner.msg import Lane
from grasp_path_planner.msg import VehicleState
from grasp_path_planner.msg import RewardInfo
from grasp_path_planner.msg import EnvironmentState
from grasp_path_planner.msg import RLCommand
from grasp_path_planner.msg import PathPlan
from ros_message_test.msg import PathPlan

NODE_NAME = 'talker'
SIM_TOPIC_NAME = "environment_state"
RL_TOPIC_NAME = "rl_decision"

def talker():
    # initialize publishers
    pub = rospy.Publisher(SIM_TOPIC_NAME, EnvironmentState, queue_size=10)
    pub_rl = rospy.Publisher(RL_TOPIC_NAME, RLCommand, queue_size = 10)

    rospy.init_node(NODE_NAME, anonymous=True)
    rate = rospy.Rate(10) # 10hz
    while not rospy.is_shutdown():
	# current Lane
	lane_cur = Lane()
	lane_points = []
	for i in range(100):
		lane_point = LanePoint()
		lane_point.pose.y = 0
		lane_point.pose.x = i/10.
		lane_point.pose.theta = 0
		lane_point.width = 3
		lane_points.append(lane_point)
	lane_cur.lane = lane_points
	# next Lane
	lane_next = Lane()
	lane_points_next = []
	for i in range(100):
		lane_point = LanePoint()
		lane_point.pose.y = 3
		lane_point.pose.x = i/10.
		lane_point.pose.theta = 0
		lane_point.width = 3
		lane_points_next.append(lane_point)
	lane_next.lane = lane_points_next
	
	# current vehicle	
	vehicle = VehicleState()
	vehicle.vehicle_location.x = 2
	vehicle.vehicle_location.y = 0
	vehicle.vehicle_location.theta = 0
	vehicle.vehicle_speed = 10
	vehicle.length = 5
	vehicle.width = 2

	# sample enviroment state
	env_state = EnvironmentState()
	env_state.cur_vehicle_state = copy.copy(vehicle)
	env_state.front_vehicle_state = copy.copy(vehicle)
	env_state.back_vehicle_state = copy.copy(vehicle)
	env_state.adjacent_lane_vehicles = [copy.copy(vehicle), copy.copy(vehicle)]
	env_state.max_num_vehicles = 2
	env_state.speed_limit = 40
	env_state.current_lane = copy.copy(lane_cur)
	env_state.next_lane = copy.copy(lane_next)

	# sample RL command
	rl_command = RLCommand()
	rl_command.change_lane = 1

        rospy.loginfo("Publishing")
        pub.publish(env_state)
	pub_rl.publish(rl_command)
        rate.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
