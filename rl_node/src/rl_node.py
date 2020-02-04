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
from ros_message_test.msg import RLCommand
from ros_message_test.msg import PathPlan

NODE_NAME = 'rl_node'
RL_TOPIC_NAME = 'reinforcement_learning'
ENVIRONMENT_TOPIC_NAME = 'environment_state'
VEHICLE_TOPIC_NAME = 'vehicle_state'
LANE_TOPIC_NAME = 'lane'

def callback_env(message):
	pass

def callback_vehicle(message):
	pass

def callback_lane(message):
	pass

def main():
    # initialize publishers
	pub_rl = rospy.Publisher(RL_TOPIC_NAME, RLCommand, queue_size = 10)
	rospy.Subscriber(ENVIRONMENT_TOPIC_NAME, EnvironmentState, callback_env)
	rospy.Subscriber(VEHICLE_TOPIC_NAME, VehicleState, callback_vehicle)
	rospy.Subscriber(LANE_TOPIC_NAME, Lane, callback_lane)
	rospy.init_node(NODE_NAME, anonymous=True)
	rate = rospy.Rate(10) # 10hz
	while not rospy.is_shutdown():
		rospy.loginfo("Publishing")
		rl_command = RLCommand()
		rl_command.constant_speed = 1
		pub_rl.publish(rl_command)
		rate.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
