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
    pub = rospy.Publisher(TEST_TOPIC_NAME, PathPlanMsg)
    rospy.init_node(TEST_NODE_NAME, anonymous=True)
    r = rospy.Rate(10) #10hz
    # p = Pedestrian(priority_status=PedestrainPriority(2))
    # v = Vehicle(world=None, 
    #             actor_id=99, 
    #             speed=99, 
    #             acceleration=99, 
    #             location_global=Pose2D(99,99,99), 
    #             location_frenet=Frenet(99,99,99), 
    #             length=99, 
    #             width=99,
    #             traffic_light_status=None)
    
    # lane = PerpendicularLane(crossing_pedestrain=[p], right_turning_lane=True)
    # msg = EnvDesc(next_intersection=[lane], reward_info=RewardInfo(current_action=RLDecision(1))).toRosMsg()
    # msg = p.toRosMsg()
    # msg = v.toRosMsg()
    msg = PathPlan(scenario_chosen = Scenario(2)).toRosMsg()
    

    while not rospy.is_shutdown():
        rospy.loginfo(msg)
        pub.publish(msg)
        r.sleep()
        # input()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException: pass