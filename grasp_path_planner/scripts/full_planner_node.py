#!/usr/bin/env python
import sys
import os
import time
homedir=os.getenv("HOME")
distro=os.getenv("ROS_DISTRO")
sys.path.remove("/opt/ros/"+distro+"/lib/python2.7/dist-packages")
sys.path.append("/opt/ros/"+distro+"/lib/python2.7/dist-packages")
import copy
# to remove tensorflow warnings
import warnings
warnings.filterwarnings("ignore")
import os,logging
logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# path planner packages
import math
from enum import Enum
import rospy
import numpy as np
from geometry_msgs.msg import Pose2D
from std_msgs.msg import String
from grasp_path_planner.msg import LanePoint
from grasp_path_planner.msg import Lane
from grasp_path_planner.msg import VehicleState
from grasp_path_planner.msg import RewardInfo
from grasp_path_planner.msg import EnvironmentState
from grasp_path_planner.msg import RLCommand
from grasp_path_planner.msg import PathPlan
from grasp_path_planner.srv import SimService, SimServiceResponse, SimServiceRequest
# RL packages
import gym
import time
from gym import spaces
import tensorflow as tf
import tensorflow.contrib as tf_contrib
import tensorflow.contrib.layers as tf_layers
from stable_baselines import DQN,PPO2
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines.deepq.policies import DQNPolicy
from stable_baselines.common.env_checker import check_env
from stable_baselines.common.cmd_util import make_vec_env

SIM_SERVICE_NAME = "simulator"
NODE_NAME = "full_grasp_planner"

TRAJ_PARAM = {'look_up_distance' : 0 ,\
    'lane_change_length' : 30,\
    'lane_change_time_constant' : 1.05,\
    'lane_change_time_disc' : 0.4,\
    'action_time_disc' : 0.2,\
    'action_duration' : 2,\
    'accelerate_amt' : 5,\
    'decelerate_amt' : 5,\
    'min_speed' : 0
}

N_DISCRETE_ACTIONS = 4
CONVERT_TO_LOCAL = True
INVERT_ANGLES = True
dir_path = os.path.dirname(os.path.realpath(__file__))

class RLDecision(Enum):
    CONSTANT_SPEED = 0
    ACCELERATE = 2
    DECELERATE = 3
    SWITCH_LANE = 1
    NO_ACTION = 4


class VecTemp:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    # def __init__(self, x, y):
    #	self.x = x
    #	self.y = y

    # def __init__(self, pose):
    #	self.x = pose.x
    #	self.y = pose.y

    def norm(self):
        return math.sqrt(self.x ** 2 + self.y ** 2)

    def add(self, other):
        return VecTemp(self.x + other.x, self.y + other.y)

    def sub(self, other):
        return VecTemp(self.x - other.x, self.y - other.y)

    def dot(self, other):
        upper = self.x * other.x + self.y + other.y
        lower = self.norm() * other.norm()
        if lower <= 0.00001:
            return 1
        return upper / lower


class PoseTemp:
    def __init__(self, ros_pose=False):
        if ros_pose:
            self.x = ros_pose.x
            self.y = ros_pose.y
            self.theta = self.wrapToPi(ros_pose.theta)
        else:
            self.x = 0
            self.y = 0
            self.theta = 0

    # def __init__(self, ros_pose):
    #	self.x = ros_pose.x
    #	self.y = ros_pose.y
    #	self.theta = self.wrapToPi(ros_pose.theta)

    def wrapToPi(self, theta):
        return (theta + math.pi) % (2. * math.pi) - math.pi

    def distance(self, pose):
        return math.sqrt((self.x - pose.x) ** 2. + (self.y - pose.y) ** 2.)

    def add(self, pose):
        new_pose = PoseTemp()
        new_pose.x = self.x + pose.x * math.cos(self.theta) - pose.y * math.sin(self.theta)
        new_pose.y = self.y + pose.x * math.sin(self.theta) + pose.y * math.cos(self.theta)
        new_pose.theta = self.wrapToPi(self.theta + pose.theta)
        return new_pose

    def vecTo(self, pose):
        new_vec = VecTemp()
        new_vec.x = pose.x - self.x
        new_vec.y = pose.y - self.y
        return new_vec

    def vecFromTheta(self):
        return VecTemp(math.cos(self.theta), math.sin(self.theta))

    def isInfrontOf(self, pose):
        diff_vec = pose.vecTo(self)
        other_vec = pose.vecFromTheta()
        return diff_vec.dot(other_vec) > 0

    def scalarMultiply(self, scalar):
        new_pose = PoseTemp()
        new_pose.x = self.x * scalar
        new_pose.y = self.y * scalar
        new_pose.theta = self.theta * scalar


class PoseSpeedTemp(PoseTemp):
    def __init__(self):
        PoseTemp.__init__(self)
        self.speed = 0

    # def __init__(self, pose, speed):
    #	self.speed = speed
    #	PoseTemp.__init__(self,pose)
    def addToPose(self, pose):
        new_pose = PoseSpeedTemp()
        new_pose.x = pose.x + self.x * math.cos(pose.theta) - self.y * math.sin(pose.theta)
        new_pose.y = pose.y + self.x * math.sin(pose.theta) + self.y * math.cos(pose.theta)
        new_pose.theta = self.wrapToPi(self.theta + pose.theta)
        new_pose.speed = self.speed
        return new_pose

class TrajGenerator:
    SAME_POSE_THRESHOLD = 2
    SAME_POSE_LOWER_THRESHOLD = 0.02

    def __init__(self, traj_parameters):
        self.traj_parameters = traj_parameters
        self.current_action = RLDecision.NO_ACTION
        self.generated_path = []
        self.path_pointer = 0
        self.action_start_time = 0
        self.start_speed = 0
        self.reset()

    def reset(self, cur_act=RLDecision.NO_ACTION, start_speed=0):
        self.current_action = cur_act
        self.generated_path = []
        self.path_pointer = 0
        self.action_start_time = 0
        self.start_speed = start_speed

    def trajPlan(self, rl_data, sim_data):
        if len(sim_data.current_lane.lane) <= 0 or len(sim_data.next_lane.lane) <= 0:
            print("Lane has no component")
        #print("RL_DATA:", rl_data)
        #print("SIWTCH_LANE", RLDecision.SWITCH_LANE)
        #print("ACCELERATE", RLDecision.ACCELERATE)
        #print("DECELERATE", RLDecision.DECELERATE)
        #print("CONSTANT SPEED", RLDecision.CONSTANT_SPEED)
        if self.current_action == RLDecision.SWITCH_LANE:
            #print("SWITCH LANE")
            return self.laneChangeTraj(sim_data)
        elif self.current_action == RLDecision.ACCELERATE:
            #print("ACCELERATE")
            return self.accelerateTraj(sim_data)
        elif self.current_action == RLDecision.DECELERATE:
            #print("DECELERATE")
            return self.decelerateTraj(sim_data)
        elif self.current_action == RLDecision.CONSTANT_SPEED:
            #print("CONSTANT SPEED")
            return self.constSpeedTraj(sim_data)
        elif rl_data == RLDecision.CONSTANT_SPEED:
            #print("CONSTANT SPEED")
            return self.constSpeedTraj(sim_data)
        elif rl_data == RLDecision.ACCELERATE:
            #print("ACCELERATE")
            new_path_plan = self.accelerateTraj(sim_data)
            return new_path_plan
        elif rl_data == RLDecision.DECELERATE:
            #print("DECELERATE")
            new_path_plan = self.decelerateTraj(sim_data)
            return new_path_plan
        elif rl_data == RLDecision.SWITCH_LANE:
            #print("SWITCH LANE")
            return self.laneChangeTraj(sim_data)
        else:
            print("RL Decision Error")

    def findNextLanePose(self, cur_vehicle_pose, lane_pose_array):
        closest_pose = lane_pose_array[0]

        for lane_waypoint in lane_pose_array:
            closest_pose = lane_waypoint
            way_pose = PoseTemp(lane_waypoint.pose)
            if way_pose.distance(cur_vehicle_pose) < TrajGenerator.SAME_POSE_THRESHOLD and \
                    way_pose.isInfrontOf(cur_vehicle_pose) and \
                    way_pose.distance(cur_vehicle_pose) > TrajGenerator.SAME_POSE_LOWER_THRESHOLD:
                return closest_pose
        return closest_pose

    def constSpeedTraj(self, sim_data):
        # duration evaluation
        if not self.current_action == RLDecision.CONSTANT_SPEED:
            self.reset(RLDecision.CONSTANT_SPEED, sim_data.cur_vehicle_state.vehicle_speed)

            # set action start time
            self.action_start_time = sim_data.reward.time_elapsed
        new_sim_time = sim_data.reward.time_elapsed
        # current vehicle state
        cur_vehicle_state = sim_data.cur_vehicle_state
        cur_vehicle_pose = PoseTemp(cur_vehicle_state.vehicle_location)
        cur_vehicle_speed = cur_vehicle_state.vehicle_speed

        # determine the closest next pose in the current lane
        lane_pose_array = sim_data.current_lane.lane
        closest_pose = self.findNextLanePose(cur_vehicle_pose, lane_pose_array)

        end_of_action = False
        #if rl_data.reset_run:
        #    end_of_action = True

        action_progress = (new_sim_time - self.action_start_time) / self.traj_parameters['action_duration']

        if action_progress >= 1.:
            end_of_action = True
            action_progress = 1.

        new_path_plan = PathPlan()
        new_path_plan.tracking_pose = closest_pose.pose
        new_path_plan.reset_sim = False
        new_path_plan.tracking_speed = self.start_speed
        new_path_plan.end_of_action = end_of_action
        new_path_plan.action_progress = action_progress
        new_path_plan.path_planner_terminate = False

        # add future poses
        new_path_plan.future_poses = []
        closest_pose_temp = PoseTemp(closest_pose.pose)
        for ts in np.arange(self.traj_parameters['action_time_disc'], \
                self.traj_parameters['action_duration'] - (new_sim_time - self.action_start_time), \
                            self.traj_parameters['action_time_disc']):
            delta_pose = PoseTemp()
            delta_pose.x = ts * self.start_speed / 3.6
            new_pose = closest_pose_temp.add(delta_pose)
            new_pose_ros = Pose2D()
            new_pose_ros.x = new_pose.x
            new_pose_ros.y = new_pose.y
            new_pose_ros.theta = new_pose.theta
            new_path_plan.future_poses.append(new_pose_ros)

        if end_of_action:
            self.reset()

        return new_path_plan

    def accelerateTraj(self, sim_data):
        # duration evaluation
        if not self.current_action == RLDecision.ACCELERATE:
            self.reset(RLDecision.ACCELERATE, sim_data.cur_vehicle_state.vehicle_speed)

            # set action start time
            self.action_start_time = sim_data.reward.time_elapsed
        new_sim_time = sim_data.reward.time_elapsed
        # current vehicle state
        cur_vehicle_state = sim_data.cur_vehicle_state
        cur_vehicle_pose = PoseTemp(cur_vehicle_state.vehicle_location)
        cur_vehicle_speed = cur_vehicle_state.vehicle_speed

        # determine the closest next pose in the current lane
        lane_pose_array = sim_data.current_lane.lane
        closest_pose = self.findNextLanePose(cur_vehicle_pose, lane_pose_array)

        end_of_action = False
        #if rl_data.reset_run:
        #    end_of_action = True

        action_progress = (new_sim_time - self.action_start_time) / self.traj_parameters['action_duration']

        if action_progress >= 1.:
            end_of_action = True
            action_progress = 1.

        new_path_plan = PathPlan()
        new_path_plan.tracking_pose = closest_pose.pose
        new_path_plan.reset_sim = False
        new_path_plan.tracking_speed = self.start_speed + action_progress * self.traj_parameters['accelerate_amt']
        new_path_plan.end_of_action = end_of_action
        new_path_plan.action_progress = action_progress
        new_path_plan.path_planner_terminate = False

        # add future poses
        new_path_plan.future_poses = []
        closest_pose_temp = PoseTemp(closest_pose.pose)
        acc_per_sec = self.traj_parameters['accelerate_amt'] / self.traj_parameters['action_duration']
        for ts in np.arange(self.traj_parameters['action_time_disc'], \
                self.traj_parameters['action_duration'] - (new_sim_time - self.action_start_time), \
                            self.traj_parameters['action_time_disc']):
            delta_pose = PoseTemp()
            delta_pose.x = (ts * (self.start_speed + action_progress * self.traj_parameters['accelerate_amt']) +\
                            acc_per_sec * (ts**2.) / 2.) / 3.6
            new_pose = closest_pose_temp.add(delta_pose)
            new_pose_ros = Pose2D()
            new_pose_ros.x = new_pose.x
            new_pose_ros.y = new_pose.y
            new_pose_ros.theta = new_pose.theta
            new_path_plan.future_poses.append(new_pose_ros)

        if end_of_action:
            self.reset()
        return new_path_plan

    def decelerateTraj(self, sim_data):
        # duration evaluation
        if not self.current_action == RLDecision.DECELERATE:
            self.reset(RLDecision.DECELERATE, sim_data.cur_vehicle_state.vehicle_speed)

            # set action start time
            self.action_start_time = sim_data.reward.time_elapsed
        new_sim_time = sim_data.reward.time_elapsed
        # current vehicle state
        cur_vehicle_state = sim_data.cur_vehicle_state
        cur_vehicle_pose = PoseTemp(cur_vehicle_state.vehicle_location)
        cur_vehicle_speed = cur_vehicle_state.vehicle_speed

        # determine the closest next pose in the current lane
        lane_pose_array = sim_data.current_lane.lane
        closest_pose = self.findNextLanePose(cur_vehicle_pose, lane_pose_array)

        end_of_action = False
        #if rl_data.reset_run:
        #    end_of_action = True

        action_progress = (new_sim_time - self.action_start_time) / self.traj_parameters['action_duration']

        if action_progress >= 1.:
            end_of_action = True
            action_progress = 1.

        new_path_plan = PathPlan()
        new_path_plan.tracking_pose = closest_pose.pose
        new_path_plan.reset_sim = False
        new_path_plan.tracking_speed = max(0,
                                           self.start_speed - action_progress * self.traj_parameters['decelerate_amt'])
        new_path_plan.end_of_action = end_of_action
        new_path_plan.action_progress = action_progress
        new_path_plan.path_planner_terminate = False

        # add future poses
        new_path_plan.future_poses = []
        closest_pose_temp = PoseTemp(closest_pose.pose)
        dec_per_sec = self.traj_parameters['decelerate_amt'] / self.traj_parameters['action_duration']
        for ts in np.arange(self.traj_parameters['action_time_disc'], \
                self.traj_parameters['action_duration'] - (new_sim_time - self.action_start_time), \
                            self.traj_parameters['action_time_disc']):
            delta_pose = PoseTemp()
            delta_pose.x = (ts * (self.start_speed - action_progress * self.traj_parameters['accelerate_amt']) -\
                            dec_per_sec * (ts**2.) / 2.) / 3.6
            new_pose = closest_pose_temp.add(delta_pose)
            new_pose_ros = Pose2D()
            new_pose_ros.x = new_pose.x
            new_pose_ros.y = new_pose.y
            new_pose_ros.theta = new_pose.theta
            new_path_plan.future_poses.append(new_pose_ros)

        if end_of_action:
            self.reset()

        return new_path_plan

    def cubicSplineGen(self, cur_lane_width, next_lane_width, v_cur):
        v_cur = v_cur / 3.6
        if v_cur < 5:
            v_cur = 5
        # determine external parameters
        w = (cur_lane_width + next_lane_width) / 2.
        l = self.traj_parameters['lane_change_length']
        r = self.traj_parameters['lane_change_time_constant']
        tf = math.sqrt(l ** 2 + w ** 2) * r / v_cur

        # parameters for x
        dx = 0
        cx = v_cur
        ax = (2. * v_cur * tf - 2. * l) / (tf ** 3.)
        bx = -3. / 2 * ax * tf

        # parameters for y
        dy = 0
        cy = 0
        ay = -2. * w / (tf ** 3.)
        by = 3 * w / (tf ** 2.)

        # return result
        neutral_traj = []

        # time loop
        time_disc = self.traj_parameters['lane_change_time_disc']
        total_loop_count = int(tf / time_disc + 1)
        for i in range(total_loop_count):
            t = i * time_disc
            x_value = ax * (t ** 3.) + bx * (t ** 2.) + cx * t + dx
            y_value = -(ay * (t ** 3.) + by * (t ** 2.) + cy * t + dy)
            x_deriv = 3. * ax * (t ** 2.) + 2. * bx * t + cx
            y_deriv = -(3. * ay * (t ** 2.) + 2. * by * t + cy)
            theta = math.atan2(y_deriv, x_deriv)
            speed = math.sqrt(y_deriv ** 2. + x_deriv ** 2.)
            pose = PoseSpeedTemp()
            pose.speed = speed * 3.6
            pose.x = x_value
            pose.y = y_value
            pose.theta = theta
            neutral_traj.append(pose)

        return neutral_traj

    def laneChangeTraj(self, sim_data):
        cur_vehicle_pose = PoseTemp(sim_data.cur_vehicle_state.vehicle_location)

        # generate trajectory
        if not self.current_action == RLDecision.SWITCH_LANE:
            self.reset(RLDecision.SWITCH_LANE)
            # ToDo: Use closest pose for lane width
            # print("reached here")
            neutral_traj = self.cubicSplineGen(sim_data.current_lane.lane[0].width, \
                            sim_data.next_lane.lane[0].width, sim_data.cur_vehicle_state.vehicle_speed)
            # determine the closest next pose in the current lane
            lane_pose_array = sim_data.current_lane.lane
            closest_pose = lane_pose_array[0].pose

            for lane_waypoint in lane_pose_array:
                closest_pose = lane_waypoint.pose
                way_pose = PoseTemp(lane_waypoint.pose)
                if way_pose.distance(cur_vehicle_pose) < TrajGenerator.SAME_POSE_THRESHOLD and \
                        way_pose.isInfrontOf(cur_vehicle_pose) and \
                        way_pose.distance(cur_vehicle_pose) > TrajGenerator.SAME_POSE_LOWER_THRESHOLD:
                    break
            closest_pose = PoseTemp(closest_pose)

            # offset the trajectory with the closest pose
            for pose_speed in neutral_traj:
                self.generated_path.append(pose_speed.addToPose(closest_pose))

            self.path_pointer = 0

        # find the next tracking point
        while (self.path_pointer < len(self.generated_path)):
            # traj pose
            pose_speed = self.generated_path[self.path_pointer]

            if pose_speed.isInfrontOf(cur_vehicle_pose) and \
                    pose_speed.distance(cur_vehicle_pose) > TrajGenerator.SAME_POSE_LOWER_THRESHOLD:
                break

            self.path_pointer += 1

        new_path_plan = PathPlan()
        # print("Total Path Length", len(self.generated_path))
        # determine if lane switch is completed

        action_progress = self.path_pointer / float(len(self.generated_path))
        end_of_action = False
        #if rl_data.reset_run:
        #    end_of_action = True
        if action_progress >= 0.9999:
            end_of_action = True
            action_progress = 1.

        if end_of_action:
            self.reset()
        else:
            new_path_plan.tracking_pose.x = self.generated_path[self.path_pointer].x
            new_path_plan.tracking_pose.y = self.generated_path[self.path_pointer].y
            new_path_plan.tracking_pose.theta = self.generated_path[self.path_pointer].theta
            new_path_plan.tracking_speed = self.generated_path[self.path_pointer].speed
        new_path_plan.reset_sim = False
        new_path_plan.end_of_action = end_of_action
        new_path_plan.action_progress = action_progress
        new_path_plan.path_planner_terminate = end_of_action

        # future poses
        path_pointer = self.path_pointer
        new_path_plan.future_poses = []
        while path_pointer < len(self.generated_path):
            new_pose_ros = Pose2D()
            new_pose_ros.x = self.generated_path[self.path_pointer].x
            new_pose_ros.y = self.generated_path[self.path_pointer].y
            new_pose_ros.theta = self.generated_path[self.path_pointer].theta
            new_path_plan.future_poses.append(new_pose_ros)
            path_pointer += 1

        return new_path_plan


# make a custom policy
class CustomPolicy(DQNPolicy):
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

    def q_net(self, input_vec):
        out = input_vec
        with tf.variable_scope("action_value"):
            out = tf_layers.fully_connected(out, num_outputs=64, activation_fn=tf.nn.relu)
            out = tf_layers.fully_connected(out, num_outputs=128, activation_fn=tf.nn.relu)
            out = tf_layers.fully_connected(out, num_outputs=4, activation_fn=tf.nn.tanh)
        return out

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False,
                 obs_phs=None, dueling=False, **kwargs):
        super(CustomPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps,
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
            q_out = self.q_net(max_out)
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

    def __init__(self, path_planner, rl_manager):
        super(CustomEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        self.observation_space = spaces.Box(low = -1000, high=1000, shape = (1,24))
        self.path_planner = path_planner
        self.rl_manager = rl_manager
        self.to_local = CONVERT_TO_LOCAL

    def invert_angles(self,env_desc):
        for vehicle in env_desc.adjacent_lane_vehicles:
            vehicle.vehicle_location.theta*=-1
        env_desc.back_vehicle_state.vehicle_location.theta*=-1
        env_desc.front_vehicle_state.vehicle_location.theta*=-1
        env_desc.cur_vehicle_state.vehicle_location.theta*=-1
        return

    def step(self, action):
        # reset sb_event flag if previously set in previous action
        decision = self.rl_manager.convertDecision(action)
        env_desc, end_of_action = self.path_planner.performAction(decision)
        self.rl_manager.reward_manager.update(env_desc,action)
        done = self.rl_manager.terminate(env_desc)
        end_of_action = end_of_action or done

        while not end_of_action:
            env_desc, end_of_action = self.path_planner.performAction(decision)
            if INVERT_ANGLES:
                self.invert_angles(env_desc)
            self.rl_manager.reward_manager.update(env_desc,action)
            done = self.rl_manager.terminate(env_desc)
            end_of_action = end_of_action or done

        env_state = self.rl_manager.makeStateVector(env_desc, self.to_local)
        reward = self.rl_manager.reward_manager.get_reward(env_desc,action)
        return env_state, reward, done, {}

    def reset(self):
        self.rl_manager.reward_manager.reset()
        env_desc = self.path_planner.resetSim()
        env_state = self.rl_manager.makeStateVector(env_desc, self.to_local)
        return env_state
        # return observation  # reward, done, info can't be included

    def render(self, mode='human'):
        pass

    def close (self):
        pass


class PathPlannerManager:
    def __init__(self):
        self.traj_generator = TrajGenerator(TRAJ_PARAM)
        self.sim_service_interface = None
        self.prev_env_desc = None

    def initialize(self):
        rospy.init_node(NODE_NAME, anonymous=True)
        rospy.wait_for_service(SIM_SERVICE_NAME)
        self.sim_service_interface = rospy.ServiceProxy(SIM_SERVICE_NAME, SimService)

    def performAction(self, action):
        path_plan = self.traj_generator.trajPlan(action, self.prev_env_desc)
        #print("------------------Path Plan--------------------")
        #print(path_plan)
        req = SimServiceRequest()
        req.path_plan = path_plan
        self.prev_env_desc = self.sim_service_interface(req).env
        return self.prev_env_desc, path_plan.end_of_action

    def resetSim(self):
        #print("RESET SIM CALLED")
        self.traj_generator.reset()
        reset_msg = PathPlan()
        reset_msg.reset_sim = True
        reset_msg.end_of_action = True
        reset_msg.sent_time = rospy.Time.now()
        req = SimServiceRequest()
        req.path_plan = reset_msg
        self.prev_env_desc = self.sim_service_interface(req).env
        return self.prev_env_desc


# Reward calculation functionality
class Reward(object):
    '''
    ^x
    |
    |---->y
    '''
    def __init__(self):
        self.min_vel = 15
        self.max_vel = 20
        self.min_dist = 0.5
        self.k_vel = 1
        self.p_vel = 0.01
        self.k_pos = 1
        self.k_col = -10
        self.k_succ = 20
        self.closest_dist = 1e5
        self.cum_vel_err = 0
        self.max_reward = 20
    
    def project(self, bb, normal):
        projected_points = []
        for point in bb:
            dist = normal[0]*point[0]+normal[1]*point[1]+normal[2]
            alpha = np.arctan(-normal[1]/normal[0])+np.pi/2
            xp,yp=point
            xp = point[0]-abs(dist)*np.sin(alpha)
            yp = point[1]-abs(dist)*np.cos(alpha)
            if round(abs(normal[0]*xp+normal[1]*yp+normal[2]),3)>0.01:   
                xp = point[0]+abs(dist)*np.sin(alpha)
                yp = point[1]+abs(dist)*np.cos(alpha)
            projected_points.append((xp,yp))
        projected_points = sorted(projected_points, key=lambda b:b[1])
        return np.array(projected_points[0]),np.array(projected_points[-1])

    def project_and_calculate(self, bb1, normals1, bb2, normals2):
        # get max seperation between two objects (max seperation is closest meaningful seperation)
        closest = 0
        normals = normals1+normals2
        for normal in normals:
            l1,r1 = self.project(bb1,normal)
            l2,r2 = self.project(bb2,normal)
            if l2[1] > r1[1] and r2[1] > r1[1]:
                # no intersection
                dist = np.linalg.norm(l2-r1)
            elif l1[1] > r2[1] and r1[1] > r2[1]:
                # no intersecton
                dist = np.linalg.norm(l1-r2)
            else:
                dist=0
            if dist > closest:
                closest=dist
        return closest

    def get_closest_distance(self,desc):
        # make a list of bounding boxes of adjacent vehicles
        # NPCs = desc.adjacent_lane_vehicles+front_vehicle_state+back_vehicle_state
        NPCs = desc.adjacent_lane_vehicles
        NPC_bb = []
        NPC_normals = []
        for vehicle in NPCs:
            cx,cy, = vehicle.vehicle_location.x, vehicle.vehicle_location.y
            l,w = vehicle.length, vehicle.width
            NPC_bb.append([[cx+l/2,cy-w/2], [cx+l/2, cy+w/2], [cx-l/2, cy+w/2], [cx-l/2, cy-w/2]])
            normals=[]
            for i in range(0,4):
                x1,y1 = NPC_bb[-1][i%4]
                x2,y2 = NPC_bb[-1][(i+1)%4]
                b = (x2-x1)
                a = -(y2-y1)
                c = (x1*(y2-y1)-(x2-x1)*y1)
                norm = np.linalg.norm([a,b])
                a, b, c = a/norm, b/norm, c/norm
                normals.append([a,b,c])
            NPC_normals.append(copy.deepcopy(normals))

        # get (a,b,c)s of the lines of the ego vehicle
        ego_lines = []
        cx,cy = desc.cur_vehicle_state.vehicle_location.x, desc.cur_vehicle_state.vehicle_location.y
        l,w = desc.cur_vehicle_state.length, desc.cur_vehicle_state.width
        ego_bb = [[cx+l/2,cy-w/2], [cx+l/2, cy+w/2], [cx-l/2, cy+w/2], [cx-l/2, cy-w/2]]
        for i in range(0,4):
            x1,y1 = ego_bb[i%4]
            x2,y2 = ego_bb[(i+1)%4]
            a = (y2-y1)
            b = -(x2-x1)
            c = (y1*(x2-x1)-(y2-y1)*x1)
            norm = np.linalg.norm([a,b])
            a, b, c = a/norm, b/norm, c/norm
            ego_lines.append([a,b,c])
        # find closest distance
        closest = 1e5
        for bb,normals in zip(NPC_bb,NPC_normals):
            distance = self.project_and_calculate(bb,normals,ego_bb,ego_lines)
            if closest > distance:
                closest = distance
        return closest

    def get_velocity_error(self,veh_speed):
        if veh_speed < self.min_vel:
            return self.p_vel*abs(veh_speed-self.min_vel)
        elif veh_speed > self.max_vel:
            return self.p_vel*abs(veh_speed-self.max_vel)
        else:
            return 0

    def update(self,desc,action):
        cur_dist = self.get_closest_distance(desc)
        if self.closest_dist > cur_dist:
            self.closest_dist=cur_dist
        
        self.cum_vel_err+=self.get_velocity_error(desc.cur_vehicle_state.vehicle_speed)

    def vel_cost(self):
        return self.k_vel*self.cum_vel_err

    def position_cost(self):
        r_inv = 1/self.max_reward
        cost = max(0,1/(self.closest_dist+r_inv)-1/(self.closest_dist+r_inv))
        return self.k_pos*cost
    
    def lane_change_reward(self,desc,action):
        if desc.reward.collision:
            return -self.max_reward
        else:
            reward=0
            reward+=self.max_reward
            reward-=self.vel_cost()
            reward-=self.position_cost()
            return reward

    def speed_reward(self,desc,action):
        reward=0
        reward-=self.vel_cost()
        reward-=self.position_cost()
        # reset at end of each action to keep negative reward bounded
        self.cum_vel_err=0
        return reward

    def get_reward(self,desc,action):
        if action==RLDecision.SWITCH_LANE.value:
            # call lane change reward function
            return self.lane_change_reward(desc,action)
        elif action==RLDecision.CONSTANT_SPEED.value:
            # call constant speed reward functon
            return self.speed_reward(desc,action)
        elif action==RLDecision.ACCELERATE.value:
            # call accelerate reward function
            return self.speed_reward(desc,action)
        elif action==RLDecision.DECELERATE.value:
            # call decelerate reward function
            return self.speed_reward(desc,action)
    
    def reset(self):
        self.closest_dist=1e5


class RLManager:
    def __init__(self):
        self.eps_time = 40
        self.reward_manager = Reward()

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
                converted_state = self.convert_to_local(data.cur_vehicle_state, vehicle)
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

    def convert_to_local(self, cur_vehicle, adj_vehicle):
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


class FullPlannerManager:
    def __init__(self):
        self.path_planner = PathPlannerManager()
        self.behavior_planner = RLManager()

    def initialize(self):
        self.path_planner.initialize()

    def run_train(self):
        env = CustomEnv(self.path_planner, self.behavior_planner)
        env = make_vec_env(lambda: env, n_envs=1)
        model = DQN(CustomPolicy, env, verbose=1, learning_starts=256, batch_size=256, exploration_fraction=0.9, target_network_update_freq=100, tensorboard_log=dir_path+'/Logs/')
        # model = DQN(MlpPolicy, env, verbose=1, learning_starts=64,  target_network_update_freq=50, tensorboard_log='./Logs/')
        # model = DQN.load("DQN_Model_SimpleSim_30k",env=env,exploration_fraction=0.1,tensorboard_log='./Logs/')
        model.learn(total_timesteps=10000)
        # model = PPO2(MlpPolicy, env, verbose=1,tensorboard_log="./Logs/")
        # model.learn(total_timesteps=20000)
        model.save(dir_path+"/DQN_Model_SimpleSim_Local")

    def run_test(self):
        env = CustomEnv(self.path_planner, self.behavior_planner)
        env = make_vec_env(lambda: env, n_envs=1)
        model = DQN.load(dir_path+"/DQN_Model_CARLA_10k.zip")
        obs = env.reset()
        count = 0
        success = 0
        while count < 500:
            done = False
            print("Count ", count, "Success ", success)
            while not done:
                action, _ = model.predict(obs)

                print(action)
                obs, reward, done, _ = env.step(action)
                print("Reward",reward)
            count += 1
            if reward == 1:
                success += 1
        print("Success Rate ", success / count, success, count)

if __name__ == '__main__':
    try:
        full_planner = FullPlannerManager()
        full_planner.initialize()
        full_planner.run_train()

    except rospy.ROSInterruptException:
        pass