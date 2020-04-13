
# ROS Packages
import rospy
import numpy as np
import copy
import time
from geometry_msgs.msg import Pose2D
from std_msgs.msg import String
from grasp_path_planner.msg import LanePoint
from grasp_path_planner.msg import Lane
from grasp_path_planner.msg import VehicleState
from grasp_path_planner.msg import RewardInfo
from grasp_path_planner.msg import Pedestrian
from grasp_path_planner.msg import EnvironmentState
from grasp_path_planner.msg import RLCommand
from grasp_path_planner.msg import PathPlan
from grasp_path_planner.srv import SimService, SimServiceResponse, SimServiceRequest
# to remove tensorflow warnings
import warnings
warnings.filterwarnings("ignore")
import os,logging
logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# RL packages
import gym
import time
from gym import spaces
import tensorflow as tf
import tensorflow.contrib as tf_contrib
import tensorflow.contrib.layers as tf_layers
from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines.deepq.policies import DQNPolicy
# other packages
from reward_manager import reward_selector, Reward, LaneChangeReward, PedestrianReward
from settings import *

# Convert to local
def convert_to_local(cur_vehicle, adj_vehicle):
        result_state = VehicleState()
        x = adj_vehicle.vehicle_location.x
        y = adj_vehicle.vehicle_location.y
        theta = adj_vehicle.vehicle_location.theta
        speed = adj_vehicle.vehicle_speed
        vx = speed*np.cos(theta)
        vy = speed*np.sin(theta)
        # get current_vehicle_speeds
        cvx = cur_vehicle.vehicle_speed*np.cos(cur_vehicle.vehicle_location.theta)
        cvy = cur_vehicle.vehicle_speed*np.sin(cur_vehicle.vehicle_location.theta)
        # make homogeneous transform
        H_Rot = np.eye(3)
        H_Rot[-1,-1] = 1
        H_Rot[0,-1] = 0
        H_Rot[1,-1] = 0
        H_Rot[0,0] = np.cos(cur_vehicle.vehicle_location.theta)
        H_Rot[0,1] = -np.sin(cur_vehicle.vehicle_location.theta)
        H_Rot[1,0] = np.sin(cur_vehicle.vehicle_location.theta)
        H_Rot[1,1] = np.cos(cur_vehicle.vehicle_location.theta)
        H_trans = np.eye(3)
        H_trans[0,-1] = -cur_vehicle.vehicle_location.x
        H_trans[1,-1] = -cur_vehicle.vehicle_location.y
        H = np.matmul(H_Rot,H_trans)
        # calculate and set relative position
        res = np.matmul(H, np.array([x,y,1]).reshape(3,1))
        result_state.vehicle_location.x = res[0,0]
        result_state.vehicle_location.y = res[1,0]
        # calculate and set relative orientation
        result_state.vehicle_location.theta = theta-cur_vehicle.vehicle_location.theta
        # calculate and set relative speed
        res_vel = np.array([vx-cvx,vy-cvy])
        result_state.vehicle_speed = speed # np.linalg.norm(res_vel)
        # print("ADJ-----------------")
        # print(adj_vehicle)
        # print("CUR-----------------")
        # print(cur_vehicle)
        # print("RESULT--------------")
        # print(result_state)
        # time.sleep(5)
        return result_state

# make a custom policy
class CustomLaneChangePolicy(DQNPolicy):
    """
    Policy object that implements DQN policy, using a MLP (2 layers of 64)

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param obs_phs: (TensorFlow Tensor, TensorFlow Tensor) a tuple containing an override for observation placeholder
        and the processed observation placeholder respectively
    :param dueling: (bool) if true double the output MLP to compute a baseline for action scores
    :param _kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def embedding_net(self, input_vec):
        out = input_vec
        with tf.variable_scope("embedding_network", reuse=tf.AUTO_REUSE):
            out = tf_layers.fully_connected(out, num_outputs=16, activation_fn=tf.nn.relu)
            out = tf_layers.fully_connected(out, num_outputs=32, activation_fn=tf.nn.relu)
            out = tf_layers.fully_connected(out, num_outputs=64, activation_fn=tf.nn.relu)
        return out

    def q_net(self, input_vec, out_num):
        out = input_vec
        with tf.variable_scope("action_value"):
            out = tf_layers.fully_connected(out, num_outputs=64, activation_fn=tf.nn.relu)
            out = tf_layers.fully_connected(out, num_outputs=128, activation_fn=tf.nn.relu)
            out = tf_layers.fully_connected(out, num_outputs=out_num, activation_fn=tf.nn.tanh)
        return out

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False,
                 obs_phs=None, dueling=False, **kwargs):
        super(CustomLaneChangePolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps,
                                            n_batch, dueling=dueling, reuse=reuse,
                                            scale=False, obs_phs=obs_phs)
        with tf.variable_scope("model", reuse=reuse):
            out_ph = tf.layers.flatten(self.processed_obs)
            embed_list = []
            for i in range(5):
                embed_list.append(
                    self.embedding_net(tf.concat([out_ph[:, :4], out_ph[:, (i + 1) * 4:(i + 2) * 4]], axis=1)))
            stacked_out = tf.stack(embed_list, axis=1)
            max_out = tf.reduce_max(stacked_out, axis=1)
            q_out = self.q_net(max_out,ac_space.n)
        self.q_values = q_out
        self._setup_init()

    def step(self, obs, state=None, mask=None, deterministic=True):
        q_values, actions_proba = self.sess.run([self.q_values, self.policy_proba], {self.obs_ph: obs})
        if deterministic:
            actions = np.argmax(q_values, axis=1)
        else:
            # Inefficient sampling
            # TODO: replace the loop
            # maybe with Gumbel-max trick ? (http://amid.fish/humble-gumbel)
            actions = np.zeros((len(obs),), dtype=np.int64)
            for action_idx in range(len(obs)):
                actions[action_idx] = np.random.choice(self.n_actions, p=actions_proba[action_idx])

        return actions, q_values, None

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})


# make a custom policy
class CustomPedestrianPolicy(DQNPolicy):
    """
    Policy object that implements DQN policy, using a MLP (2 layers of 64)

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param obs_phs: (TensorFlow Tensor, TensorFlow Tensor) a tuple containing an override for observation placeholder
        and the processed observation placeholder respectively
    :param dueling: (bool) if true double the output MLP to compute a baseline for action scores
    :param _kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def embedding_net(self, input_vec):
        out = input_vec
        with tf.variable_scope("embedding_network", reuse=tf.AUTO_REUSE):
            out = tf_layers.fully_connected(out, num_outputs=8, activation_fn=tf.nn.relu)
            out = tf_layers.fully_connected(out, num_outputs=16, activation_fn=tf.nn.relu)
            out = tf_layers.fully_connected(out, num_outputs=32, activation_fn=tf.nn.relu)
        return out

    def q_net(self, input_vec, out_num):
        out = input_vec
        with tf.variable_scope("action_value"):
            out = tf_layers.fully_connected(out, num_outputs=32, activation_fn=tf.nn.relu)
            out = tf_layers.fully_connected(out, num_outputs=64, activation_fn=tf.nn.relu)
            out = tf_layers.fully_connected(out, num_outputs=out_num, activation_fn=tf.nn.tanh)
        return out

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False,
                 obs_phs=None, dueling=False, **kwargs):
        super(CustomPedestrianPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps,
                                            n_batch, dueling=dueling, reuse=reuse,
                                            scale=False, obs_phs=obs_phs)
        with tf.variable_scope("model", reuse=reuse):
            out_ph = tf.layers.flatten(self.processed_obs)
            embed_list = []
            for i in range(1):
                embed_list.append(
                    self.embedding_net(tf.concat([out_ph[:, :4], out_ph[:, (i + 1) * 4:(i + 2) * 4]], axis=1)))
            q_out = self.q_net(embed_list[0],ac_space.n)
        self.q_values = q_out
        self._setup_init()

    def step(self, obs, state=None, mask=None, deterministic=True):
        q_values, actions_proba = self.sess.run([self.q_values, self.policy_proba], {self.obs_ph: obs})
        if deterministic:
            actions = np.argmax(q_values, axis=1)
        else:
            # Inefficient sampling
            # TODO: replace the loop
            # maybe with Gumbel-max trick ? (http://amid.fish/humble-gumbel)
            actions = np.zeros((len(obs),), dtype=np.int64)
            for action_idx in range(len(obs)):
                actions[action_idx] = np.random.choice(self.n_actions, p=actions_proba[action_idx])

        return actions, q_values, None

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})


# make custom environment class to only read values. It should not directly change values in the other thread.
class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface"""

    metadata = {'render.modes': ['human']}

    def __init__(self, path_planner, rl_manager,event):
        super(CustomEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        N_ACTIONS=0
        if event==Scenario.PEDESTRIAN:
            N_ACTIONS=3
            self.action_space = spaces.Discrete(N_ACTIONS)
            self.observation_space = spaces.Box(low = -1000, high=1000, shape = (1,8))
        elif event==Scenario.LANE_CHANGE:
            N_ACTIONS=4
            self.action_space = spaces.Discrete(N_ACTIONS)
            self.observation_space = spaces.Box(low = -1000, high=1000, shape = (1,24))
        self.path_planner = path_planner
        self.rl_manager = rl_manager
        self.to_local = CONVERT_TO_LOCAL

    def invert_angles(self,env_desc):
        env_copy = copy.deepcopy(env_desc)
        for vehicle in env_copy.adjacent_lane_vehicles:
            vehicle.vehicle_location.theta*=-1
        env_copy.back_vehicle_state.vehicle_location.theta*=-1
        env_copy.front_vehicle_state.vehicle_location.theta*=-1
        env_copy.cur_vehicle_state.vehicle_location.theta*=-1
        return env_copy

    def step(self, action):
        # reset sb_event flag if previously set in previous action
        decision = self.rl_manager.convertDecision(action)
        env_desc, end_of_action = self.path_planner.performAction(decision)
        env_copy = env_desc
        if INVERT_ANGLES:
            env_copy = self.invert_angles(env_desc)
        self.rl_manager.reward_manager.update(env_copy,action)
        done = self.rl_manager.terminate(env_copy)
        end_of_action = end_of_action or done
        while not end_of_action:
            env_desc, end_of_action = self.path_planner.performAction(decision)
            env_copy = env_desc
            if INVERT_ANGLES:
                env_copy = self.invert_angles(env_desc)
            self.rl_manager.reward_manager.update(env_copy,action)
            done = self.rl_manager.terminate(env_copy)
            end_of_action = end_of_action or done
        env_state = self.rl_manager.makeStateVector(env_copy, self.to_local)
        reward = self.rl_manager.reward_manager.get_reward(env_copy,action)
        # for sending success signal during testing
        success = not( env_desc.reward.collision or \
                    env_desc.reward.time_elapsed > self.rl_manager.eps_time)
        info = {}
        info["success"] = success
        print("REWARD",reward)
        # time.sleep(2)
        return env_state, reward, done, info

    def reset(self):
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

    def close (self):
        pass


class RLManager:
    def __init__(self, event):
        self.eps_time = 40
        self.reward_manager = reward_selector(event)
        self.event = event

    def convertDecision(self, action):
        if action == RLDecision.CONSTANT_SPEED.value:
            return RLDecision.CONSTANT_SPEED
        elif action == RLDecision.ACCELERATE.value:
            return RLDecision.ACCELERATE
        elif action == RLDecision.DECELERATE.value:
            return RLDecision.DECELERATE
        elif action == RLDecision.SWITCH_LANE.value:
            return RLDecision.SWITCH_LANE
        else:
            print("Bug in decision conversion")

    def terminate(self, env_desc):
        if self.event == Scenario.LANE_CHANGE:
            return env_desc.reward.collision or \
                env_desc.reward.path_planner_terminate or \
                    env_desc.reward.time_elapsed > self.eps_time
        elif self.event == Scenario.PEDESTRIAN:
            # return true if vehicle has moved past the vehicle
            ped_vehicle = VehicleState()
            relative_pose = ped_vehicle
            if env_desc.nearest_pedestrian.exist:
                ped_vehicle.vehicle_location = env_desc.nearest_pedestrian.pedestrian_location
                ped_vehicle.vehicle_speed = env_desc.nearest_pedestrian.pedestrian_speed
                relative_pose = convert_to_local(env_desc.cur_vehicle_state,ped_vehicle)
                if relative_pose.vehicle_location.x < -1:
                    return True
            # usual conditions
            return env_desc.reward.collision or \
                env_desc.reward.path_planner_terminate or \
                    env_desc.reward.time_elapsed > self.eps_time

    def rewardCalculation(self, env_desc):
        reward = 0
        if env_desc.reward.collision:
            reward = reward - 1
            print("Collision")
        elif env_desc.reward.path_planner_terminate:
            reward += env_desc.reward.action_progress
            print("Progress",env_desc.reward.action_progress)
        return reward

    def append_vehicle_state(self, env_state, vehicle_state):
        env_state.append(vehicle_state.vehicle_location.x)
        env_state.append(vehicle_state.vehicle_location.y)
        env_state.append(vehicle_state.vehicle_location.theta)
        env_state.append(vehicle_state.vehicle_speed)
        return

    def makeStateVector(self, data, local=False):
        '''
        create a state vector from the message recieved
        '''
        if self.event==Scenario.LANE_CHANGE:
            i=0
            env_state = []
            if not local:
                self.append_vehicle_state(env_state, data.cur_vehicle_state)
                # self.append_vehicle_state(env_state, data.back_vehicle_state)
                # self.append_vehicle_state(env_state, data.front_vehicle_state)
                for _, vehicle in enumerate(data.adjacent_lane_vehicles):
                    if i < 5:
                        self.append_vehicle_state(env_state, vehicle)
                    else:
                        break
                    i+=1
            else:
                cur_vehicle_state = VehicleState()
                cur_vehicle_state.vehicle_location.x = 0
                cur_vehicle_state.vehicle_location.y = 0
                cur_vehicle_state.vehicle_location.theta = 0
                cur_vehicle_state.vehicle_speed = data.cur_vehicle_state.vehicle_speed
                self.append_vehicle_state(env_state, cur_vehicle_state)
                for _, vehicle in enumerate(data.adjacent_lane_vehicles):
                    converted_state = convert_to_local(data.cur_vehicle_state, vehicle)
                    if i < 5:
                        self.append_vehicle_state(env_state, converted_state)
                    else:
                        break
                    i+=1
            dummy = VehicleState()
            dummy.vehicle_location.x = 100
            dummy.vehicle_location.y = 100
            dummy.vehicle_location.theta = 0
            dummy.vehicle_speed = 0
            while i<5:
                self.append_vehicle_state(env_state, dummy)
                i+=1
            return env_state
        elif self.event==Scenario.PEDESTRIAN:
            env_state = []
            if not local:
                self.append_vehicle_state(env_state, data.cur_vehicle_state)
                ped_vehicle = VehicleState()
                ped_vehicle.vehicle_location = data.nearest_pedestrian.pedestrian_location
                ped_vehicle.pedestrian_speed = data.nearest_pedestrian.pedestrian_speed
                self.append_vehicle_state(env_state, ped_vehicle)
            else:
                cur_vehicle_state = VehicleState()
                cur_vehicle_state.vehicle_location.x = 0
                cur_vehicle_state.vehicle_location.y = 0
                cur_vehicle_state.vehicle_location.theta = 0
                cur_vehicle_state.vehicle_speed = data.cur_vehicle_state.vehicle_speed
                self.append_vehicle_state(env_state, cur_vehicle_state)
                ped_vehicle = VehicleState()
                ped_vehicle.vehicle_location = data.nearest_pedestrian.pedestrian_location
                ped_vehicle.vehicle_speed = data.nearest_pedestrian.pedestrian_speed
                converted_state = convert_to_local(data.cur_vehicle_state, ped_vehicle)
                self.append_vehicle_state(env_state, converted_state)
            return env_state