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
import tensorflow as tf
import tensorflow.contrib as tf_contrib
import tensorflow.contrib.layers as tf_layers
from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines.deepq.policies import DQNPolicy

# other packages
from reward_manager import reward_selector  # , Reward, LaneChangeReward, PedestrianReward
from settings import Scenario, RLDecision
from state_manager import StateManager, convertToLocal


class RLManager:
    def __init__(self, event: Scenario):
        self.eps_time = 40
        self.reward_manager = reward_selector(event)
        self.event = event
        self.state_manager = StateManager(event)

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
                relative_pose = convertToLocal(env_desc.cur_vehicle_state,ped_vehicle)
                if relative_pose.vehicle_location.x < -10:
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
        return self.state_manager.embedState(env_desc, self.event, local)



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
            out = tf_layers.fully_connected(out, num_outputs=16, activation_fn=tf.nn.relu)
            out = tf_layers.fully_connected(out, num_outputs=32, activation_fn=tf.nn.relu)
        return out

    def q_net(self, input_vec, out_num):
        out = input_vec
        with tf.variable_scope("action_value"):
            out = tf_layers.fully_connected(out, num_outputs=32, activation_fn=tf.nn.relu)
            out = tf_layers.fully_connected(out, num_outputs=16, activation_fn=tf.nn.relu)
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