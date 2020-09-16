import numpy as np
import logging
# import time

# ROS Packages
from grasp_path_planner.msg import VehicleState
from grasp_path_planner.msg import EnvironmentState

# # to remove tensorflow warnings
# import warnings
# warnings.filterwarnings("ignore")
# import os,logging
# logging.disable(logging.WARNING)
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# # RL packages
# import gym
# from gym import spaces
# import tensorflow as tf
# import tensorflow.contrib as tf_contrib
# import tensorflow.contrib.layers as tf_layers
# from stable_baselines.deepq.policies import MlpPolicy
# from stable_baselines.deepq.policies import DQNPolicy

# other packages
from reward_manager import reward_selector  # , Reward, LaneChangeReward, PedestrianReward
from settings import Scenario, RLDecision
from state_manager import StateManager, convertToLocal


class RLManager:
    def __init__(self, event: Scenario):
        self.eps_time = 40
        self.reward_manager = reward_selector(event)
        self.event = event
        self.state_manager = StateManager()

    def terminate(self, env_desc: EnvironmentState) -> bool:
        """
        A function which returns true if the episode must terminate.
        This is decided by the following conditions.
        * Has the vehicle collinded?
        * Is the episode time elapsed more than the threshold self.eps_time?
        * Is the maneouver completed?

        Args:
        env_desc: (EnvironmentState) ROS Message for the environment
        Returns:
        true if episode must terminate
        """
        if self.event == Scenario.LANE_CHANGE:
            # return true if any of the conditions described in the description is true
            return env_desc.reward.collision or \
                env_desc.reward.path_planner_terminate or \
                env_desc.reward.time_elapsed > self.eps_time
        elif self.event == Scenario.PEDESTRIAN:
            # return true if vehicle has moved past the pedestrian

            # treat a pedestrian as a vehicle object
            ped_vehicle = VehicleState()
            relative_pose = VehicleState()

            # if there is a pedestrian closeby then check if we have passed him
            if env_desc.nearest_pedestrian.exist:
                # populate the "pedestrian vehicle" parameters
                ped_vehicle.vehicle_location = env_desc.nearest_pedestrian.pedestrian_location
                ped_vehicle.vehicle_speed = env_desc.nearest_pedestrian.pedestrian_speed
                relative_pose = convertToLocal(env_desc.cur_vehicle_state, ped_vehicle)
                # if the relative pose of the pedestrian is behind the ego vehicle then return true
                if relative_pose.vehicle_location.x < -1:
                    return True
            # usual conditions
            return env_desc.reward.collision or \
                env_desc.reward.path_planner_terminate or \
                env_desc.reward.time_elapsed > self.eps_time

    def convertDecision(self, action: int) -> RLDecision:
        """
        Converts the action in int into an RLDecision enum

        Args:
        :param action: (int) neural network decision given as an argmax
        Returns:
        RLDecision enum
        """
        if action == RLDecision.CONSTANT_SPEED.value:
            return RLDecision.CONSTANT_SPEED
        elif action == RLDecision.ACCELERATE.value:
            return RLDecision.ACCELERATE
        elif action == RLDecision.DECELERATE.value:
            return RLDecision.DECELERATE
        elif action == RLDecision.SWITCH_LANE.value:
            return RLDecision.SWITCH_LANE
        else:
            logging.error("Bug in decision conversion")
            raise RuntimeError("Invalid action given")

    def rewardCalculation(self) -> np.ndarray:
        raise NotImplementedError()

    def makeStateVector(self, env_desc: EnvironmentState, local: bool = False) -> np.ndarray:
        """
        Creates a state embedding

        Args:
        :param env_desc: (EnvironmentState) A ROS Message describing the state
        :param local: (bool) Flag to select the frame of reference for the embedding
        Returns:
        A state embedding vector (np.ndarray)
        """
        return self.state_manager(env_desc, self.event, local)
