# normal packages
import copy
import numpy as np
from typing import Tuple
import sys

sys.path.append("../../carla_utils/utils")

# RL Packages
import gym
from gym import spaces

# ROS Packages
from utility import EnvDesc
# other packages
from options import Scenario
from path_planner import PathPlannerManager
from rl_manager import RLManager


# make custom environment class to only read values. It should not directly change values in the other thread.
class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface"""

    metadata = {"render.modes": ["human"]}

    def __init__(
        self, path_planner: PathPlannerManager, rl_manager: RLManager, event: Scenario
    ):
        super(CustomEnv, self).__init__()
        """
        Define the action and state spaces based on the scenario
        Args:
        :param path_planner: (PathPlannerManager) A planner which executes the high-level action in sim
        :param rl_manager: (RLManager) An RL container which modulates the embeddings and behavior based on the scenario
        :param event: (Scanrion) An enum which tells what scenario it is to train on
        """
        N_ACTIONS = 0
        if event == Scenario.LANE_FOLLOWING:
            N_ACTIONS = 3
            self.action_space = spaces.Discrete(N_ACTIONS)
            self.observation_space = spaces.Box(low=-1000, high=1000, shape=(1, 41))
        elif event == Scenario.SWITCH_LANE_LEFT or event == Scenario.SWITCH_LANE_RIGHT:
            N_ACTIONS = 4
            self.action_space = spaces.Discrete(N_ACTIONS)
            self.observation_space = spaces.Box(low=-1000, high=1000, shape=(1, 77))
        elif event == Scenario.LEFT_TURN:
            N_ACTIONS = 3
            self.action_space = spaces.Discrete(N_ACTIONS)
            self.observation_space = spaces.Box(low=-1000, high=1000, shape=(1, 264))
        elif event == Scenario.RIGHT_TURN:
            N_ACTIONS = 3
            self.action_space = spaces.Discrete(N_ACTIONS)
            self.observation_space = spaces.Box(low=-1000, high=1000, shape=(1, 141))
        elif event == Scenario.GO_STRAIGHT:
            N_ACTIONS = 3
            self.action_space = spaces.Discrete(N_ACTIONS)
            self.observation_space = spaces.Box(low=-1000, high=1000, shape=(1, 214))

        self.path_planner = path_planner
        self.rl_manager = rl_manager

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Steps through the episode for one high-level action
        Args:
        :param action: (int)
        Returns:
        A tuple of the final observation, reward, if episode is done, and a (info) dictionary
        The dictionary can contain additional information. Currently we send if the episode was a
        success or not.
        """
        print("Action in env step:", action)
        # reset sb_event flag if previously set in previous action
        decision = self.rl_manager.convertDecision(action)
        # print(decision)
        env_desc, end_of_action = self.path_planner.performAction(decision)
        env_copy = env_desc
        self.rl_manager.reward_manager.update(env_copy, action)
        done = self.rl_manager.terminate(env_copy)
        end_of_action = end_of_action or done
        while not end_of_action:
            env_desc, end_of_action = self.path_planner.performAction(decision)
            env_copy = env_desc
            self.rl_manager.reward_manager.update(env_copy, action)
            done = self.rl_manager.terminate(env_copy)
            end_of_action = end_of_action or done
        env_state = self.rl_manager.makeStateVector(env_copy)
        reward = None
        reward = self.rl_manager.reward_manager.get_reward(env_copy, action)
        # for sending success signal during testing
        success = not (
            env_desc.reward_info.collision
            or (env_desc.reward_info.time_elapsed > self.rl_manager.eps_time)
        )
        info = {}
        info["success"] = success
        # time.sleep(2)
        return env_state, reward, done, info

    def reset(self):
        """
        Resets the environment
        """
        self.rl_manager.reward_manager.reset()
        env_desc = self.path_planner.resetSim()
        env_copy = env_desc
        env_state = self.rl_manager.makeStateVector(env_copy)
        return env_state
        # return observation  # reward, done, info can't be included

    def render(self, mode="human"):
        pass

    def close(self):
        pass
