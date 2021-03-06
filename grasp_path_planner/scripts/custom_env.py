# normal packages
import copy
import numpy as np
from typing import Tuple

# RL Packages
import gym
from gym import spaces

# ROS Packages
from grasp_path_planner.msg import EnvironmentState

# other packages
from settings import Scenario, CONVERT_TO_LOCAL, INVERT_ANGLES, OLD_REWARD
from path_planner import PathPlannerManager
from rl_manager import RLManager


# make custom environment class to only read values. It should not directly change values in the other thread.
class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface"""

    metadata = {'render.modes': ['human']}

    def __init__(self, path_planner: PathPlannerManager, rl_manager: RLManager, event: Scenario):
        super(CustomEnv, self).__init__()
        """
        Define the action and state spaces based on the scenario
        Args:
        :param path_planner: (PathPlannerManager) A planner which executes the high-level action in sim
        :param rl_manager: (RLManager) An RL container which modulates the embeddings and behavior based on the scenario
        :param event: (Scanrion) An enum which tells what scenario it is to train on
        """
        N_ACTIONS = 0
        if event == Scenario.PEDESTRIAN:
            N_ACTIONS = 3
            self.action_space = spaces.Discrete(N_ACTIONS)
            self.observation_space = spaces.Box(low=-1000, high=1000, shape=(1, 8))
        elif event == Scenario.LANE_CHANGE:
            N_ACTIONS = 4
            self.action_space = spaces.Discrete(N_ACTIONS)
            self.observation_space = spaces.Box(low=-1000, high=1000, shape=(1, 24))
        self.path_planner = path_planner
        self.rl_manager = rl_manager
        self.to_local = CONVERT_TO_LOCAL

    def invert_angles(self, env_desc: EnvironmentState) -> EnvironmentState:
        """
        Inverts the angles to account for the coordinate system. yuck
        Args:
        :param env_desc: (EnvironmentState) A ROS Message containing the environment state
        """
        # TODO: Why do we not invert the angle for pedestrians? We should probably do that
        env_copy = copy.deepcopy(env_desc)
        for vehicle in env_copy.adjacent_lane_vehicles:
            vehicle.vehicle_location.theta *= -1
        env_copy.back_vehicle_state.vehicle_location.theta *= -1
        env_copy.front_vehicle_state.vehicle_location.theta *= -1
        env_copy.cur_vehicle_state.vehicle_location.theta *= -1
        return env_copy

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
        # reset sb_event flag if previously set in previous action
        decision = self.rl_manager.convertDecision(action)
        env_desc, end_of_action = self.path_planner.performAction(decision)
        env_copy = env_desc
        if INVERT_ANGLES:
            env_copy = self.invert_angles(env_desc)
        if not OLD_REWARD:
            self.rl_manager.reward_manager.update(env_copy, action)
        done = self.rl_manager.terminate(env_copy)
        end_of_action = end_of_action or done
        while not end_of_action:
            env_desc, end_of_action = self.path_planner.performAction(decision)
            env_copy = env_desc
            if INVERT_ANGLES:
                env_copy = self.invert_angles(env_desc)
            if not OLD_REWARD:
                self.rl_manager.reward_manager.update(env_copy, action)
            done = self.rl_manager.terminate(env_copy)
            end_of_action = end_of_action or done
        env_state = self.rl_manager.makeStateVector(env_copy, self.to_local)
        reward = None
        if OLD_REWARD and self.rl_manager.event == Scenario.LANE_CHANGE:
            reward = self.rl_manager.rewardCalculation(env_copy)
        else:
            reward = self.rl_manager.reward_manager.get_reward(env_copy, action)
        # for sending success signal during testing
        success = not(env_desc.reward.collision or env_desc.reward.time_elapsed > self.rl_manager.eps_time)
        info = {}
        info["success"] = success
        # time.sleep(2)
        return env_state, reward, done, info

    def reset(self):
        """
        Resets the environment
        """
        if not OLD_REWARD:
            self.rl_manager.reward_manager.reset()
        env_desc = self.path_planner.resetSim()
        env_copy = env_desc
        if INVERT_ANGLES:
            env_copy = self.invert_angles(env_desc)
        env_state = self.rl_manager.makeStateVector(env_copy, self.to_local)
        return env_state
        # return observation  # reward, done, info can't be included

    def render(self, mode='human'):
        pass

    def close(self):
        pass
