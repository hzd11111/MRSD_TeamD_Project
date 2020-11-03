#!/usr/bin/env python

import sys
print(sys.path)
import os


sys.path.append("../../carla_utils/utils")
homedir = os.getenv("HOME")
distro = os.getenv("ROS_DISTRO")
sys.path.remove("/opt/ros/" + distro + "/lib/python2.7/dist-packages")
sys.path.append("/opt/ros/" + distro + "/lib/python2.7/dist-packages")

from stable_baselines import DQN
import rospy
from settings import *

from grasp_path_planner.srv import SimService, SimServiceRequest
from utility import PathPlan, EnvDesc
from neural_network_manager import NNManager, NeuralNetworkSelector
from trajectory_generator import TrajGenerator
from options import RLDecision, Scenario


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
        self.nn_manager.initialize({}) #ToDo: Fillin the paths
        rospy.init_node("Point2PointPlanner", anonymous=True)
        rospy.wait_for_service(SIM_SERVICE_NAME)
        self.sim_service_interface = rospy.ServiceProxy(SIM_SERVICE_NAME, SimService)

    def resetSim(self):
        reset_msg = PathPlan()
        reset_msg.reset_sim = True
        reset_msg.end_of_action = True
        reset_msg.scenario_chosen = Scenario.LANE_FOLLOWING.value # Default to start LaneFollowing
        req = SimServiceRequest()
        req.path_plan = reset_msg
        if (self.sim_service_interface(req) is None):
            print("Getting None from Msg and feed Empty Env")
            self.prev_env_desc = EnvDesc()            
        else :
            self.prev_env_desc = EnvDesc.fromRosMsg(self.sim_service_interface(req).env)
        return self.prev_env_desc

    # perform action for a single timestep
    def performAction(self, action, selected_scenario, auto_pilot=False):
        if auto_pilot:
            path_plan = PathPlan()
            path_plan.auto_pilot = True
            path_plan.scenario_chosen = selected_scenario.value
            req = SimServiceRequest()
            req.path_plan = path_plan
            self.prev_env_desc = EnvDesc.fromRosMsg(self.sim_service_interface(req).env)
            return self.prev_env_desc, True, False

        path_plan = self.traj_generator.trajPlan(action, self.prev_env_desc, selected_scenario)
        dummy_scenario_selection = selected_scenario
        if selected_scenario is Scenario.STOP:
            dummy_scenario_selection = Scenario.LANE_FOLLOWING
        path_plan.scenario_chosen = dummy_scenario_selection.value
        req = SimServiceRequest()
        req.path_plan = path_plan
        # import ipdb; ipdb.set_trace()
        self.prev_env_desc = EnvDesc.fromRosMsg(self.sim_service_interface(req).env)
        return self.prev_env_desc, path_plan.end_of_action, path_plan.path_planner_terminate

    # perform action for a single RL decision
    def performRLDecision(self, decision, selected_scenario, auto_pilot=False):
        if auto_pilot:
            self.performAction(None, selected_scenario, True)
            return True
        end_of_action = False
        path_planner_terminate = False
        while not end_of_action:
            env_desc, end_of_action, path_planner_terminate = self.performAction(decision, selected_scenario)
            end_of_action = end_of_action
        return path_planner_terminate

    def run(self):
        selected_scenario, new_scenario = self.neural_network_selector.selectNeuralNetwork(self.prev_env_desc, True)
        while selected_scenario is not Scenario.DONE:
            path_planner_terminate = False
            print(selected_scenario)
            if selected_scenario is Scenario.STOP:
                self.performRLDecision(RLDecision.STOP, selected_scenario)
                path_planner_terminate = True
            else:
                path_planner_terminate = self.performRLDecision(None, selected_scenario, True)
                # call neural network manager
                #rl_decision = self.nn_manager.makeDecision(self.prev_env_desc, selected_scenario)
                #path_planner_terminate = self.performRLDecision(rl_decision)
            selected_scenario, new_scenario = self.neural_network_selector.selectNeuralNetwork(self.prev_env_desc,
                                                                                 path_planner_terminate,
                                                                                 selected_scenario)
        print("Location Reached!")



if __name__ == '__main__':
    try:
        # initialize p2p planner
        p2p_planner = Point2PointPlanner()
        p2p_planner.initialize()
        p2p_planner.resetSim()
        p2p_planner.run()

    except rospy.ROSInterruptException:
        pass