#!/usr/bin/env python
import rospy
from carla_utils.msg import GlobalPathPointMsg, GlobalPathMsg, PathPlanMsg
from carla_utils.msg import LaneStatusMsg, CurrentLaneMsg, ParallelLaneMsg, PerpendicularLaneMsg
from carla_utils.msg import ActorMsg, VehicleMsg, PedestrainMsg, FrenetMsg, LanePointMsg, EnvDescMsg

from utility import * 
from actors import * 
from options import * 

TEST_NODE_NAME = 'custom_talker'
TEST_TOPIC_NAME = 'custom_chatter'

def talker():
    pub = rospy.Publisher(TEST_TOPIC_NAME, EnvDescMsg)
    rospy.init_node(TEST_NODE_NAME, anonymous=True)
    r = rospy.Rate(10) #10hz
    p = Pedestrian(location_global=Pose2D(1,2,3), priority_status=PedestrainPriority(2))
    lane = PerpendicularLane(crossing_pedestrain=[p], lane_id=99, origin_global_pose=Pose2D(6,7,8), intersecting_distance = 12)
    env = EnvDesc(next_intersection=[lane], reward_info=RewardInfo(current_action=RLDecision(1)))

    # lane = ParallelLane(crossing_pedestrain=[p], lane_id=99, origin_global_pose=Pose2D(6,7,8), lane_distance = 12)

    msg = env.toRosMsg()
    while not rospy.is_shutdown():
        rospy.loginfo(msg)
        pub.publish(msg)
        r.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException: pass