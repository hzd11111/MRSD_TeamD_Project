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
        self.nn_manager.initialize("path to nn")
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

    # perform action for a single timestep
    def performAction(self, action):
        path_plan = self.traj_generator.trajPlan(action, self.prev_env_desc)

        req = SimServiceRequest()
        req.path_plan = path_plan
        # import ipdb; ipdb.set_trace()
        self.prev_env_desc = EnvDesc.fromRosMsg(self.sim_service_interface(req).env)
        return self.prev_env_desc, path_plan.end_of_action, path_plan.path_planner_terminate

    # perform action for a single RL decision
    def performRLDecision(self, decision):
        end_of_action = False
        path_planner_terminate = False
        while not end_of_action:
            env_desc, end_of_action, path_planner_terminate = self.path_planner.performAction(decision)
            end_of_action = end_of_action
        return path_planner_terminate

    def run(self):
        selected_scenario = self.neural_network_selector.selectNeuralNetwork(self.prev_env_desc, True)
        while not selected_scenario is Scenarios.DONE:
            path_planner_terminate = False
            if selected_scenario is Scenarios.STOP:
                self.performRLDecision(RLDecision.STOP)
                path_planner_terminate = True
            else:
                # call neural network manager
                rl_decision = self.nn_manager.makeDecision(self.prev_env_desc, selected_scenario)
                path_planner_terminate = self.performRLDecision(rl_decision)
            selected_scenario = self.neural_network_selector.selectNeuralNetwork(self.prev_env_desc,
                                                                                 path_planner_terminate,
                                                                                 selected_scenario)
        print("Location Reached!")



if __name__ == '__main__':
    try:
        # initialize p2p planner
        p2p_planner = Point2PointPlanner()
        p2p_planner.initialize()
        p2p_planner.reset()
        p2p_planner.run()

    except rospy.ROSInterruptException:
        pass
