import numpy as np
import logging

# ROS Packages
from carla_utils.msg import EnvDescMsg

# other packages
from reward_manager import reward_selector
from options import Scenario, RLDecision
from state_manager import StateManager


class RLManager:
    def __init__(self, event: Scenario):
        self.eps_time = 40
        self.reward_manager = reward_selector(event)
        self.event = event
        self.state_manager = StateManager(event)

    def terminate(self, env_desc: EnvDescMsg) -> bool:
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
        if self.event == Scenario.SWITCH_LANE_LEFT:
            # return true if any of the conditions described in the description is true
            return env_desc.reward_info.collision or \
                env_desc.reward_info.path_planner_terminate or \
                env_desc.reward_info.time_elapsed > self.eps_time

        elif self.event == Scenario.SWITCH_LANE_RIGHT:
            # return true if any of the conditions described in the description is true
            return env_desc.reward_info.collision or \
                env_desc.reward_info.path_planner_terminate or \
                env_desc.reward_info.time_elapsed > self.eps_time

        elif self.event == Scenario.LANE_FOLLOWING:
            # return true if any of the conditions described in the description is true
            return env_desc.reward_info.collision or \
                env_desc.reward_info.path_planner_terminate or \
                env_desc.reward_info.time_elapsed > self.eps_time

        elif self.event == Scenario.GO_STRAIGHT:
            # return true if any of the conditions described in the description is true
            return env_desc.reward_info.collision or \
                env_desc.reward_info.path_planner_terminate or \
                env_desc.reward_info.time_elapsed > self.eps_time

        elif self.event == Scenario.LEFT_TURN:
            # return true if any of the conditions described in the description is true
            return env_desc.reward_info.collision or \
                env_desc.reward_info.path_planner_terminate or \
                env_desc.reward_info.time_elapsed > self.eps_time

        elif self.event == Scenario.RIGHT_TURN:
            # return true if any of the conditions described in the description is true
            return env_desc.reward_info.collision or \
                env_desc.reward_info.path_planner_terminate or \
                env_desc.reward_info.time_elapsed > self.eps_time

    def convertDecisionSwitchLaneLeft(self, action):
        """
        Converts the action in int into an RLDecision enum

        Args:
        :param action: (int) neural network decision given as an argmax
        Returns:
        RLDecision enum
        """

        if action == 0:
            return RLDecision.CONSTANT_SPEED
        elif action == 1:
            return RLDecision.ACCELERATE
        elif action == 2:
            return RLDecision.DECELERATE
        elif action == 3:
            return RLDecision.SWITCH_LANE_LEFT
        else:
            logging.error("Bug in decision conversion")
            raise RuntimeError("Invalid action given")

    def convertDecisionSwitchLaneRight(self, action):
        """
        Converts the action in int into an RLDecision enum

        Args:
        :param action: (int) neural network decision given as an argmax
        Returns:
        RLDecision enum
        """

        if action == 0:
            return RLDecision.CONSTANT_SPEED
        elif action == 1:
            return RLDecision.ACCELERATE
        elif action == 2:
            return RLDecision.DECELERATE
        elif action == 3:
            return RLDecision.SWITCH_LANE_RIGHT
        else:
            logging.error("Bug in decision conversion")
            raise RuntimeError("Invalid action given")

    def convertDecisionIntersection(self, action):
        """
        Converts the action in int into an RLDecision enum

        Args:
        :param action: (int) neural network decision given as an argmax
        Returns:
        RLDecision enum
        """

        if action == 0:
            return RLDecision.GLOBAL_PATH_CONSTANT_SPEED
        elif action == 1:
            return RLDecision.GLOBAL_PATH_ACCELERATE
        elif action == 2:
            return RLDecision.GLOBAL_PATH_DECELERATE

    def convertDecision(self, action: int) -> RLDecision:
        """
        Converts the action in int into an RLDecision enum

        Args:
        :param action: (int) neural network decision given as an argmax
        Returns:
        RLDecision enum
        """
        if self.event == Scenario.SWITCH_LANE_LEFT:
            return self.convertDecisionSwitchLaneLeft(action)
        if self.event == Scenario.SWITCH_LANE_RIGHT:
            return self.convertDecisionRightTurn(action)
        if self.event == Scenario.LEFT_TURN or \
            self.event == Scenario.RIGHT_TURN or \
                self.event == Scenario.GO_STRAIGHT:
            return self.convertDecisionIntersection(action)
        else:
            logging.error("Bug in decision conversion")
            raise RuntimeError("Invalid scenario given")

    def rewardCalculation(self) -> np.ndarray:
        raise NotImplementedError()

    def makeStateVector(self, env_desc: EnvDescMsg, local: bool = False) -> np.ndarray:
        """
        Creates a state embedding

        Args:
        :param env_desc: (EnvironmentState) A ROS Message describing the state
        :param local: (bool) Flag to select the frame of reference for the embedding
        Returns:
        A state embedding vector (np.ndarray)
        """
        return self.state_manager.embedState(env_desc, self.event)
