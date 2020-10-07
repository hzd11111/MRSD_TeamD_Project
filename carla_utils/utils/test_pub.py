#!/usr/bin/env python
import rospy
from carla_utils.msg import PathPlanMsg
from carla_utils.msg import GlobalPathPointMsg
from carla_utils.msg import GlobalPathMsg
from carla_utils.msg import LaneStatusMsg
from carla_utils.msg import CurrentLaneMsg
from carla_utils.msg import ParallelLaneMsg
from carla_utils.msg import PerpendicularLaneMsg
from carla_utils.msg import ActorMsg, VehicleMsg, PedestrainMsg, FrenetMsg

from utility import * 
from actors import * 
from options import * 

TEST_NODE_NAME = 'custom_talker'
TEST_TOPIC_NAME = 'actor_test'

def talker():
    pub = rospy.Publisher(TEST_TOPIC_NAME, PedestrainMsg)
    rospy.init_node(TEST_NODE_NAME, anonymous=True)
    r = rospy.Rate(10) #10hz
    msg = Pedestrian(actor_id=-1, priority_status=PedestrainPriority(1)).toRosMsg()
    while not rospy.is_shutdown():
        rospy.loginfo(msg)
        pub.publish(msg)
        r.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException: pass