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
from ros_message_test.msg import Example
from ros_message_test.msg import LanePoint
from ros_message_test.msg import Lane
from ros_message_test.msg import VehicleState
from ros_message_test.msg import RewardInfo
from ros_message_test.msg import EnvironmentState

NODE_NAME = 'talker'
PUBLISHING_TOPIC_NAME = 'test_chat'

def talker():
    pub = rospy.Publisher(PUBLISHING_TOPIC_NAME, EnvironmentState, queue_size=10)
    rospy.init_node(NODE_NAME, anonymous=True)
    rate = rospy.Rate(10) # 10hz
    while not rospy.is_shutdown():
        # sample Lane Point
        lane_point_sample = LanePoint()
        lane_point_sample.pose.x = 1
	lane_point_sample.pose.y = 1
	lane_point_sample.pose.theta = 0
	
	# sample Lane
	lane = Lane()
	lane.lane = [copy.copy(lane_point_sample), copy.copy(lane_point_sample)]
	
	# sample vehicle
	vehicle = VehicleState()
	vehicle.vehicle_location.x = 2
	vehicle.vehicle_location.y = 2
	vehicle.vehicle_location.theta = 1
	vehicle.vehicle_speed = 10
	vehicle.length = 5
	vehicle.width = 2

	# sample reward info
	reward = RewardInfo()
	reward.collision = 1
	reward.time_elapsed = 10.0
	reward.new_run = 0

	# sample enviroment state
	env_state = EnvironmentState()
	env_state.cur_vehicle_state = copy.copy(vehicle)
	env_state.front_vehicle_state = copy.copy(vehicle)
	env_state.back_vehicle_state = copy.copy(vehicle)
	env_state.adjacent_lane_vehicles = [copy.copy(vehicle), copy.copy(vehicle)]
	env_state.max_num_vehicles = 2
	env_state.speed_limit = 40
	env_state.current_lane = copy.copy(lane)
	env_state.next_lane = copy.copy(lane)
	env_state.reward = copy.copy(reward)

        rospy.loginfo("Publishing")
        pub.publish(env_state)
        rate.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
