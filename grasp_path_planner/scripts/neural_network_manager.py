#!/usr/bin/env python
import numpy as np

import sys
print(sys.path)
import os
homedir = os.getenv("HOME")
distro = os.getenv("ROS_DISTRO")
# os.environ["WANDB_MODE"] = "dryrun"
sys.path.remove("/opt/ros/" + distro + "/lib/python2.7/dist-packages")
sys.path.append("/opt/ros/" + distro + "/lib/python2.7/dist-packages")

# RL Packages
from stable_baselines import DQN

# other packages
from state_manager import StateManager
from rl_manager import GeneralRLManager
from options import Scenario, RLDecision, GlobalPathAction


class NNManager:
    """
    A neural network manager class to perform inference
    """
    def __init__(self):
        """
        Class constructor.
        Handles the neural_network inference based on the scenario
        """
        self.state_manager = StateManager()
        self.decision_maker = GeneralRLManager()
        self.neural_nets = {}

    def initialize(self, model_paths):
        """
        loads the model
        Args:
        :param model_paths: dict (Scenario Enu,path) the path to the model
        """
        for key in model_paths:
            print(key, "model loaded")
            self.neural_nets[key] = DQN.load(model_paths[key])

    def makeDecision(self, env_desc, scenario) -> RLDecision:
        """
        Makes a decision using the neural network
        Args:
        :param env_embedding: Embedding of environment information
        """
        state = self.state_manager.embedState(env_desc, scenario).reshape(1, -1)
        action, _ = self.neural_nets[scenario].predict(state)
        action_enum = self.decision_maker.convertDecision(action, scenario)
        print(action_enum)
        return action_enum


class NeuralNetworkSelector:
    def __init__(self):
        self.global_path_pointer = 0
        self.prev_lane_change_pointer = 0

    def updateGlobalPathProgress(self, global_path, curr_vehicle_global_pose):
        print("Vehicle Pose:", curr_vehicle_global_pose.x, curr_vehicle_global_pose.y, curr_vehicle_global_pose.theta)
        while self.global_path_pointer < len(global_path.path_points):
            single_pose = global_path.path_points[self.global_path_pointer].global_pose
            print(single_pose.isInfrontOf(curr_vehicle_global_pose))
            print(single_pose.distance(curr_vehicle_global_pose))
            print("Global Path Pose:", single_pose.x, single_pose.y, single_pose.theta)
            if single_pose.isInfrontOf(curr_vehicle_global_pose) and \
                   single_pose.distance(curr_vehicle_global_pose) > 0.2 and \
                    single_pose.distance(curr_vehicle_global_pose) < 15:
                break
            print("Next Global Point:",self.global_path_pointer, "/", len(global_path.path_points))
            self.global_path_pointer += 1

    def priorityDeterminatoin(self, scenario):
        if scenario is Scenario.DONE:
            return 0
        elif scenario is Scenario.STOP:
            return 1
        elif scenario is Scenario.LEFT_TURN:
            return 2.1
        elif scenario is Scenario.RIGHT_TURN:
            return 2.2
        elif scenario is Scenario.GO_STRAIGHT:
            return 2.3
        elif scenario is Scenario.SWITCH_LANE_LEFT:
            return 3
        elif scenario is Scenario.SWITCH_LANE_RIGHT:
            return 3
        elif scenario is Scenario.LANE_FOLLOWING:
            return 4
        else:
            print("Invalid Scenario")

    def doneStateCondition(self, env_desc):
        # get the last point of the global path
        global_path = env_desc.global_path.path_points
        last_global_path_point = global_path[-1]
        last_global_path_pose = last_global_path_point.global_pose

        # determine the distance to the last pose of global path
        curr_pose = env_desc.cur_vehicle_state.location_global
        distance_to_goal = curr_pose.distance(last_global_path_pose)
        return distance_to_goal < 2

    def stopStateCondition(self, env_desc):
        # get the last point of the global path
        global_path = env_desc.global_path.path_points
        last_global_path_point = global_path[-1]
        last_global_path_pose = last_global_path_point.global_pose

        # determine the distance to the last pose of global path
        curr_pose = env_desc.cur_vehicle_state.location_global
        distance_to_goal = curr_pose.distance(last_global_path_pose)
        return distance_to_goal < 7

    def leftTurnStateCondition(self, env_desc):
        # check if there is a intersection point in the next 20 meters
        cul_distance = 0
        temp_global_path_pointer = self.global_path_pointer
        left_turn_action_found = False
        while temp_global_path_pointer < (len(env_desc.global_path.path_points) - 1) and \
            cul_distance < 20:
            if env_desc.global_path.path_points[temp_global_path_pointer].action == GlobalPathAction.LEFT_TURN:
                #print("Left Turn Found")
                left_turn_action_found = True
                break
            cul_distance += env_desc.global_path.path_points[temp_global_path_pointer].global_pose.distance(env_desc.global_path.path_points[temp_global_path_pointer+1].global_pose)
            temp_global_path_pointer += 1

        if not left_turn_action_found:
            return False
        return True
        # check if left turn is allowed
        #return env_desc.current_lane.left_turning_lane

    def rightTurnStateCondition(self, env_desc):
        # check if there is a intersection point in the next 20 meters
        cul_distance = 0
        temp_global_path_pointer = self.global_path_pointer
        right_turn_action_found = False
        while temp_global_path_pointer < (len(env_desc.global_path.path_points) - 1) and \
            cul_distance < 20:
            if env_desc.global_path.path_points[temp_global_path_pointer].action == GlobalPathAction.RIGHT_TURN:
                #print("Right Turn Found")
                right_turn_action_found = True
                break
            cul_distance += env_desc.global_path.path_points[temp_global_path_pointer].global_pose.distance(env_desc.global_path.path_points[temp_global_path_pointer+1].global_pose)
            temp_global_path_pointer += 1

        if not right_turn_action_found:
            return False
        return True
        # check if left turn is allowed
        #return env_desc.current_lane.right_turning_lane

    def goStraightStateCondition(self, env_desc):
        # check if there is a intersection point in the next 20 meters
        cul_distance = 0
        temp_global_path_pointer = self.global_path_pointer
        go_straight_action_found = False
        while temp_global_path_pointer < (len(env_desc.global_path.path_points) - 1) and \
                cul_distance < 20:
            if env_desc.global_path.path_points[temp_global_path_pointer].action == GlobalPathAction.GO_STRAIGHT:
                print("Go Straight Found")
                go_straight_action_found = True
                break
            cul_distance += env_desc.global_path.path_points[temp_global_path_pointer].global_pose.distance(
                env_desc.global_path.path_points[temp_global_path_pointer + 1].global_pose)
            temp_global_path_pointer += 1

        return go_straight_action_found

    def switchLaneLeftStateCondition(self, env_desc):
        # check if there is a left lane switch action in the next 50 meters
        cul_distance = 0
        temp_global_path_pointer = self.global_path_pointer
        lane_change_action_found = False
        while temp_global_path_pointer < (len(env_desc.global_path.path_points) - 1) and \
                cul_distance < 50:
            if env_desc.global_path.path_points[temp_global_path_pointer].action == GlobalPathAction.SWITCH_LANE_LEFT:
                print("Left Lane Change Found")
                if temp_global_path_pointer >= self.prev_lane_change_pointer:
                    self.prev_lane_change_pointer = temp_global_path_pointer
                    lane_change_action_found = True
                    break
            if env_desc.global_path.path_points[temp_global_path_pointer].action == GlobalPathAction.LEFT_TURN or\
                env_desc.global_path.path_points[temp_global_path_pointer].action == GlobalPathAction.RIGHT_TURN:
                return False
            cul_distance += env_desc.global_path.path_points[temp_global_path_pointer].global_pose.distance(
                env_desc.global_path.path_points[temp_global_path_pointer + 1].global_pose)
            temp_global_path_pointer += 1

        return lane_change_action_found

    def switchLaneRightStateCondition(self, env_desc):
        # check if there is a left lane switch action in the next 50 meters
        cul_distance = 0
        temp_global_path_pointer = self.global_path_pointer
        lane_change_action_found = False
        while temp_global_path_pointer < (len(env_desc.global_path.path_points) - 1) and \
                cul_distance < 50:
            if env_desc.global_path.path_points[temp_global_path_pointer].action == GlobalPathAction.SWITCH_LANE_RIGHT:
                print("Right Lane Change Found")
                if temp_global_path_pointer >= self.prev_lane_change_pointer:
                    self.prev_lane_change_pointer = temp_global_path_pointer
                    lane_change_action_found = True
                    break
            if env_desc.global_path.path_points[temp_global_path_pointer].action == GlobalPathAction.LEFT_TURN or\
                env_desc.global_path.path_points[temp_global_path_pointer].action == GlobalPathAction.RIGHT_TURN:
                return False
            cul_distance += env_desc.global_path.path_points[temp_global_path_pointer].global_pose.distance(
                env_desc.global_path.path_points[temp_global_path_pointer + 1].global_pose)
            temp_global_path_pointer += 1

        return lane_change_action_found

    def laneFollowingStateCondition(self, env_desc):
        return True

    def selectNeuralNetwork(self, env_desc, prev_state_exited, current_state=None):
        # priority of the current state
        current_state_priority = 10
        if prev_state_exited and (current_state == Scenario.SWITCH_LANE_LEFT or
            current_state == Scenario.SWITCH_LANE_RIGHT):
            self.prev_lane_change_pointer += 1
        if current_state and (not prev_state_exited):
            current_state_priority = self.priorityDeterminatoin(current_state)

        # check the condition of each state starting from the highest priority
        # exit when the current state priority is smaller than the state

        # Done state
        if current_state_priority <= self.priorityDeterminatoin(Scenario.DONE):
            return current_state, False
        if self.doneStateCondition(env_desc):
            return Scenario.DONE, True

        # update global path progress
        self.updateGlobalPathProgress(env_desc.global_path, env_desc.cur_vehicle_state.location_global)

        # STOP state
        if current_state_priority <= self.priorityDeterminatoin(Scenario.STOP):
            return current_state, False
        if self.stopStateCondition(env_desc):
            return Scenario.STOP, True

        # LEFT_TURN state
        if current_state_priority <= self.priorityDeterminatoin(Scenario.LEFT_TURN):
            return current_state, False
        if self.leftTurnStateCondition(env_desc):
            return Scenario.LEFT_TURN, True

        # RIGHT_TURN state
        if current_state_priority <= self.priorityDeterminatoin(Scenario.RIGHT_TURN):
            return current_state, False
        if self.rightTurnStateCondition(env_desc):
            return Scenario.RIGHT_TURN, True

        # GO_STRAIGHT state
        if current_state_priority <= self.priorityDeterminatoin(Scenario.GO_STRAIGHT):
            return current_state, False
        if self.goStraightStateCondition(env_desc):
            return Scenario.GO_STRAIGHT, True

        # SWITCH_LANE_LEFT state
        if current_state_priority <= self.priorityDeterminatoin(Scenario.SWITCH_LANE_LEFT):
            return current_state, False
        if self.switchLaneLeftStateCondition(env_desc):
            return Scenario.SWITCH_LANE_LEFT, True

        # SWITCH_LANE_RIGHT state
        if current_state_priority <= self.priorityDeterminatoin(Scenario.SWITCH_LANE_RIGHT):
            return current_state, False
        if self.switchLaneRightStateCondition(env_desc):
            return Scenario.SWITCH_LANE_RIGHT, True

        # LANE_FOLLOWING state
        if current_state_priority <= self.priorityDeterminatoin(Scenario.LANE_FOLLOWING):
            return current_state, False
        if self.laneFollowingStateCondition(env_desc):
            return Scenario.LANE_FOLLOWING, True
