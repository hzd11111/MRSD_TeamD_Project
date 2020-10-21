#!/usr/bin/env python

import sys
print(sys.path)
import os
homedir = os.getenv("HOME")
distro = os.getenv("ROS_DISTRO")
sys.path.remove("/opt/ros/" + distro + "/lib/python2.7/dist-packages")
sys.path.append("/opt/ros/" + distro + "/lib/python2.7/dist-packages")
import rospy
from settings import *
from grasp_path_planner.srv import SimService, SimServiceRequest
from utility import PathPlan, EnvDesc

TRAJ_PARAM = {'look_up_distance': 0, \
              'lane_change_length': 30, \
              'lane_change_time_constant': 1.05, \
              'lane_change_time_disc': 0.4, \
              'action_time_disc': 0.2, \
              'action_duration': 0.5, \
              'accelerate_amt': 5, \
              'decelerate_amt': 5, \
              'min_speed': 0
              }

class Point2PointPlanner:
    def __init__(self):
        self.prev_env_desc = None
        self.neural_network_selector = NeuralNetworkSelector()
        self.nn_manager = NNManager()
        self.traj_generator = TrajGenerator(TRAJ_PARAM)

    def initialize(self):
        rospy.init_node("Point2PointPlanner", anonymous=True)
        rospy.wait_for_service(SIM_SERVICE_NAME)
        self.sim_service_interface = rospy.ServiceProxy(SIM_SERVICE_NAME, SimService)

    def resetSim(self):
        reset_msg = PathPlan()
        reset_msg.reset_sim = True
        reset_msg.end_of_action = True
        req = SimServiceRequest()
        req.path_plan = reset_msg
        self.prev_env_desc = EnvDesc.fromRosMsg(self.sim_service_interface(req).env)
        return self.prev_env_desc

    def run(self):
        selected_scenario = self.neural_network_selector.selectNeuralNetwork(self.prev_env_desc)
        while not selected_scenario is Scenarios.DONE:
            if selected_scenario is Scenarios.STOP:
                
            else:

        print("Location Reached!")



if __name__ == '__main__':
    try:
        # initialize p2p planner
        p2p_planner = Point2PointPlanner()
        p2p_planner.initialize()
        p2p_planner.reset()

    except rospy.ROSInterruptException:
        pass
