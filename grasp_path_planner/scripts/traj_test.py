#!/usr/bin/env python
# -----------------------------------Packages------------------------------------------------------#

import rospy
import sys
from trajectory_generator import *
from options import RLDecision

sys.path.append("../../carla_utils/utils")


TRAJ_PARAM = {'look_up_distance': 0, \
              'lane_change_length': 30, \
              'lane_change_time_constant': 1.05, \
              'lane_change_time_disc': 0.4, \
              'action_time_disc': 0.2, \
              'action_duration': 0.5, \
              'accelerate_amt': 5, \
              'decelerate_amt': 5, \
              'min_speed': 20
              }

if __name__ == '__main__':
    try:
        rospy.init_node("TrajTest", anonymous=True)
        print("Hello World")
        generator = TrajGenerator(TRAJ_PARAM)
        generator.trajPlan(RLDecision.CONSTANT_SPEED)
    except rospy.ROSInterruptException:
        pass
