#!/usr/bin/env python
import rospy
from carla_utils.msg import PathPlanMsg
from carla_utils.msg import GlobalPathPointMsg
from carla_utils.msg import GlobalPathMsg
from carla_utils.msg import LaneStatusMsg
from carla_utils.msg import CurrentLaneMsg
from carla_utils.msg import ParallelLaneMsg
from carla_utils.msg import PerpendicularLaneMsg

from utility import * 

TEST_NODE_NAME = 'custom_talker'
TEST_TOPIC_NAME = 'custom_chatter'

def talker():
    pub = rospy.Publisher(TEST_TOPIC_NAME, EnvDescMsg)
    rospy.init_node(TEST_NODE_NAME, anonymous=True)
    r = rospy.Rate(10) #10hz
    msg = EnvDesc().toRosMsg()
    while not rospy.is_shutdown():
        rospy.loginfo(msg)
        pub.publish(msg)
        r.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException: pass