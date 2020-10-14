import numpy as np
# import gym
# from gym import spaces
import tensorflow as tf
import tensorflow.contrib.layers as tf_layers

from stable_baselines.deepq.policies import DQNPolicy


class CustomIntersectionStraight(DQNPolicy):
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

    def embedding_net_pedestrian(self, input_vec):
        out = input_vec
        with tf.variable_scope("embedding_network_pedestrian", reuse=tf.compat.v1.AUTO_REUSE):
            out = tf_layers.fully_connected(
                out, num_outputs=32, activation_fn=tf.nn.relu)
        return out

    def embedding_net_front_back(self, input_vec):
        out = input_vec
        with tf.variable_scope("embedding_network_front_back", reuse=tf.compat.v1.AUTO_REUSE):
            out = tf_layers.fully_connected(
                out, num_outputs=32, activation_fn=tf.nn.relu)
        return out

    def embedding_net_perpendicular(self, input_vec):
        out = input_vec
        with tf.variable_scope("embedding_network_perpendicular",
                               reuse=tf.compat.v1.AUTO_REUSE):
            out = tf_layers.fully_connected(
                out, num_outputs=32, activation_fn=tf.nn.relu)
        return out

    def embedding_net_opposite(self, input_vec):
        out = input_vec
        with tf.variable_scope("embedding_network_opposite", reuse=tf.compat.v1.AUTO_REUSE):
            out = tf_layers.fully_connected(
                out, num_outputs=32, activation_fn=tf.nn.relu)
        return out

    def q_net(self, input_vec, out_num):
        out = input_vec
        with tf.variable_scope("action_value"):
            out = tf_layers.fully_connected(
                out, num_outputs=64, activation_fn=tf.nn.relu)
            out = tf_layers.fully_connected(
                out, num_outputs=128, activation_fn=tf.nn.relu)
            out = tf_layers.fully_connected(
                out, num_outputs=out_num, activation_fn=tf.nn.tanh)
        return out

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False,
                 obs_phs=None, dueling=False, **kwargs):
        super(CustomIntersectionStraight, self).__init__(sess, ob_space, ac_space, n_env, n_steps,
                                                         n_batch, dueling=dueling, reuse=reuse,
                                                         scale=False, obs_phs=obs_phs)
        
        with tf.variable_scope("model", reuse=reuse):
            veh_state_len = 7
            ped_state_len = 6
            current_lane_len = 8
            out_ph = tf.layers.flatten(self.processed_obs)

            # Add front vehicle
            front_veh_start = current_lane_len + veh_state_len
            embed_front_vehicle = []
            embed_front_vehicle.append(
                self.embedding_net_front_back(
                    tf.concat([out_ph[:, current_lane_len:current_lane_len + veh_state_len],
                               out_ph[:, front_veh_start:front_veh_start * veh_state_len]], axis=1)))

            # Add back vehicle
            back_veh_start = current_lane_len + 2 * veh_state_len
            embed_back_vehicle = []
            embed_back_vehicle.append(
                self.embedding_net_front_back(
                    tf.concat([out_ph[:, current_lane_len:current_lane_len + veh_state_len],
                               out_ph[:, back_veh_start:back_veh_start + veh_state_len]], axis=1)))

            # Add pedestrians
            ped_start_index = current_lane_len + 3 * veh_state_len
            embed_pedestrians = []
            for j in range(3):
                embed_pedestrians.append(
                    self.embedding_net_pedestrian(
                        tf.concat(
                            [out_ph[:, current_lane_len:current_lane_len + veh_state_len],
                             out_ph[:, ped_start_index + j * ped_state_len:
                                    ped_start_index + (j + 1) * ped_state_len]],
                            axis=1)))

            # Add perpendicular lane vehicles
            perp_start_index = current_lane_len + 3 * veh_state_len + 3 * ped_state_len
            embed_perp_lane_vehs = []
            for j in range(10):
                embed_perp_lane_vehs.append(
                    self.embedding_net_perpendicular(
                        tf.concat(
                            [out_ph[:, current_lane_len:current_lane_len + veh_state_len],
                             out_ph[:, perp_start_index + j * veh_state_len:
                                    perp_start_index + (j + 1) * veh_state_len]],
                            axis=1)))

            # Add opposite lane vehicles
            opp_lane_start_index = current_lane_len + 3 * veh_state_len + 3 * ped_state_len + \
                10 * veh_state_len
            embed_opp_lane_vehs = []
            for j in range(5):
                embed_opp_lane_vehs.append(
                    self.embedding_net_opposite(
                        tf.concat(
                            [out_ph[:, current_lane_len:current_lane_len + veh_state_len],
                             out_ph[:, opp_lane_start_index + j * veh_state_len:
                                    opp_lane_start_index + (j + 1) * veh_state_len]],
                            axis=1)))

            # stack them and take max
            embed_list = embed_front_vehicle + embed_back_vehicle + embed_pedestrians + \
                embed_perp_lane_vehs + embed_opp_lane_vehs
            stacked_out = tf.stack(embed_list, axis=1)
            max_out = tf.reduce_max(stacked_out, axis=1)

            # concatenate the current_lane_status
            max_out = tf.concat([max_out, out_ph[:, :8][:, None]], axis=1)
            q_out = self.q_net(max_out, ac_space.n)

        self.q_values = q_out
        self._setup_init()

    def step(self, obs, state=None, mask=None, deterministic=True):
        q_values, actions_proba = self.sess.run(
            [self.q_values, self.policy_proba], {self.obs_ph: obs})
        if deterministic:
            actions = np.argmax(q_values, axis=1)
        else:
            # Inefficient sampling
            # TODO: replace the loop
            # maybe with Gumbel-max trick ? (http://amid.fish/humble-gumbel)
            actions = np.zeros((len(obs),), dtype=np.int64)
            for action_idx in range(len(obs)):
                actions[action_idx] = np.random.choice(
                    self.n_actions, p=actions_proba[action_idx])

        return actions, q_values, None

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})


class CustomIntersectionLeftTurn(DQNPolicy):
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

    def embedding_net_pedestrian(self, input_vec):
        out = input_vec
        with tf.variable_scope("embedding_network_pedestrian", reuse=tf.compat.v1.AUTO_REUSE):
            out = tf_layers.fully_connected(
                out, num_outputs=32, activation_fn=tf.nn.relu)
        return out

    def embedding_net_front_back(self, input_vec):
        out = input_vec
        with tf.variable_scope("embedding_network_front_back", reuse=tf.compat.v1.AUTO_REUSE):
            out = tf_layers.fully_connected(
                out, num_outputs=32, activation_fn=tf.nn.relu)
        return out

    def embedding_net_perpendicular(self, input_vec):
        out = input_vec
        with tf.variable_scope("embedding_network_perpendicular",
                               reuse=tf.compat.v1.AUTO_REUSE):
            out = tf_layers.fully_connected(
                out, num_outputs=32, activation_fn=tf.nn.relu)
        return out

    def embedding_net_opposite(self, input_vec):
        out = input_vec
        with tf.variable_scope("embedding_network_opposite", reuse=tf.compat.v1.AUTO_REUSE):
            out = tf_layers.fully_connected(
                out, num_outputs=32, activation_fn=tf.nn.relu)
        return out

    def q_net(self, input_vec, out_num):
        out = input_vec
        with tf.variable_scope("action_value"):
            out = tf_layers.fully_connected(
                out, num_outputs=64, activation_fn=tf.nn.relu)
            out = tf_layers.fully_connected(
                out, num_outputs=128, activation_fn=tf.nn.relu)
            out = tf_layers.fully_connected(
                out, num_outputs=out_num, activation_fn=tf.nn.tanh)
        return out

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False,
                 obs_phs=None, dueling=False, **kwargs):
        super(CustomIntersectionLeftTurn, self).__init__(sess, ob_space, ac_space, n_env, n_steps,
                                                         n_batch, dueling=dueling, reuse=reuse,
                                                         scale=False, obs_phs=obs_phs)
        with tf.variable_scope("model", reuse=reuse):
            veh_state_len = 7
            ped_state_len = 6
            current_lane_len = 8
            out_ph = tf.layers.flatten(self.processed_obs)

            # Add front vehicle
            front_veh_start = current_lane_len + veh_state_len
            embed_front_vehicle = []
            embed_front_vehicle.append(
                self.embedding_net_front_back(
                    tf.concat([out_ph[:, current_lane_len:current_lane_len + veh_state_len],
                               out_ph[:, front_veh_start:front_veh_start + veh_state_len]],
                              axis=1)))

            # Add back vehicle
            back_veh_start = current_lane_len + 2 * veh_state_len
            embed_back_vehicle = []
            embed_back_vehicle.append(
                self.embedding_net_front_back(
                    tf.concat([out_ph[:, current_lane_len:current_lane_len + veh_state_len],
                               out_ph[:, back_veh_start:back_veh_start + veh_state_len]], axis=1)))

            # Add pedestrians
            ped_start_index = current_lane_len + 3 * veh_state_len
            embed_pedestrians = []
            for j in range(3):
                embed_pedestrians.append(
                    self.embedding_net_pedestrian(
                        tf.concat(
                            [out_ph[:, current_lane_len:current_lane_len + veh_state_len],
                             out_ph[:, ped_start_index + j * ped_state_len:
                                    ped_start_index + (j + 1) * ped_state_len]],
                            axis=1)))

            # Add perpendicular lane vehicles
            perp_start_index = current_lane_len + 3 * veh_state_len + 3 * ped_state_len
            embed_perp_lane_vehs = []
            for j in range(10):
                # import ipdb; ipdb.set_trace()
                embed_perp_lane_vehs.append(
                    self.embedding_net_perpendicular(
                        tf.concat(
                            [out_ph[:, current_lane_len:current_lane_len + veh_state_len],
                             out_ph[:, perp_start_index + j * veh_state_len:
                                    perp_start_index + (j + 1) * veh_state_len]],
                            axis=1)))

            # Add opposite lane vehicles
            opp_lane_start_index = current_lane_len + 3 * veh_state_len + \
                3 * ped_state_len + 10 * veh_state_len
            embed_opp_lane_vehs = []
            for j in range(10):
                embed_opp_lane_vehs.append(
                    self.embedding_net_opposite(
                        tf.concat(
                            [out_ph[:, current_lane_len:current_lane_len + veh_state_len],
                             out_ph[:, opp_lane_start_index + j * veh_state_len:
                                    opp_lane_start_index + (j + 1) * veh_state_len]],
                            axis=1)))

            # stack them and take max
            embed_list = embed_front_vehicle + embed_back_vehicle + embed_pedestrians + \
                embed_perp_lane_vehs + embed_opp_lane_vehs
            stacked_out = tf.stack(embed_list, axis=1)
            max_out = tf.reduce_max(stacked_out, axis=1)

            # concatenate the current_lane_status
            max_out = tf.concat([max_out, out_ph[:, :8]], axis=1)
            q_out = self.q_net(max_out, ac_space.n)

        self.q_values = q_out
        self._setup_init()

    def step(self, obs, state=None, mask=None, deterministic=True):
        q_values, actions_proba = self.sess.run(
            [self.q_values, self.policy_proba], {self.obs_ph: obs})
        if deterministic:
            actions = np.argmax(q_values, axis=1)
        else:
            # Inefficient sampling
            # TODO: replace the loop
            # maybe with Gumbel-max trick ? (http://amid.fish/humble-gumbel)
            actions = np.zeros((len(obs),), dtype=np.int64)
            for action_idx in range(len(obs)):
                actions[action_idx] = np.random.choice(
                    self.n_actions, p=actions_proba[action_idx])

        return actions, q_values, None

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})


class CustomIntersectionRightTurn(DQNPolicy):
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

    def embedding_net_ped_perp(self, input_vec):
        out = input_vec
        with tf.variable_scope("embedding_network_pedestrian_perp", reuse=tf.compat.v1.AUTO_REUSE):
            out = tf_layers.fully_connected(
                out, num_outputs=32, activation_fn=tf.nn.relu)
        return out

    def embedding_net_ped_cur(self, input_vec):
        out = input_vec
        with tf.variable_scope("embedding_network_pedestrian_current", reuse=tf.compat.v1.AUTO_REUSE):
            out = tf_layers.fully_connected(
                out, num_outputs=32, activation_fn=tf.nn.relu)
        return out

    def embedding_net_front_back(self, input_vec):
        out = input_vec
        with tf.variable_scope("embedding_network_front_back", reuse=tf.compat.v1.AUTO_REUSE):
            out = tf_layers.fully_connected(
                out, num_outputs=32, activation_fn=tf.nn.relu)
        return out

    def embedding_net_perp_veh(self, input_vec):
        out = input_vec
        with tf.variable_scope("embedding_network_perpendicular_vehicles", reuse=tf.compat.v1.AUTO_REUSE):
            out = tf_layers.fully_connected(
                out, num_outputs=32, activation_fn=tf.nn.relu)
        return out

    def q_net(self, input_vec, out_num):
        out = input_vec
        with tf.variable_scope("action_value"):
            out = tf_layers.fully_connected(
                out, num_outputs=64, activation_fn=tf.nn.relu)
            out = tf_layers.fully_connected(
                out, num_outputs=128, activation_fn=tf.nn.relu)
            out = tf_layers.fully_connected(
                out, num_outputs=out_num, activation_fn=tf.nn.tanh)
        return out

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False,
                 obs_phs=None, dueling=False, **kwargs):
        super(CustomIntersectionLeftTurn, self).__init__(sess, ob_space, ac_space, n_env, n_steps,
                                                         n_batch, dueling=dueling, reuse=reuse,
                                                         scale=False, obs_phs=obs_phs)
        with tf.variable_scope("model", reuse=reuse):
            veh_state_len = 7
            ped_state_len = 6
            current_lane_len = 8
            out_ph = tf.layers.flatten(self.processed_obs)

            # Add front vehicle
            front_veh_start = current_lane_len + veh_state_len
            embed_front_vehicle = []
            embed_front_vehicle.append(
                self.embedding_net_front_back(
                    tf.concat([out_ph[:, current_lane_len:current_lane_len + veh_state_len],
                               out_ph[:, front_veh_start:front_veh_start * veh_state_len]],
                              axis=1)))

            # Add back vehicle
            back_veh_start = current_lane_len + 2 * veh_state_len
            embed_back_vehicle = []
            embed_back_vehicle.append(
                self.embedding_net_front_back(
                    tf.concat([out_ph[:, current_lane_len:current_lane_len + veh_state_len],
                               out_ph[:, back_veh_start:back_veh_start + veh_state_len]], axis=1)))

            # Add pedestrians
            cur_ped_start_index = current_lane_len + 3 * veh_state_len
            embed_pedestrians_cur = []
            for j in range(3):
                embed_pedestrians_cur.append(
                    self.embedding_net_ped_cur(
                        tf.concat(
                            [out_ph[:, current_lane_len:current_lane_len + veh_state_len],
                             out_ph[:, cur_ped_start_index + j * ped_state_len:
                                    cur_ped_start_index + (j + 1) * ped_state_len]],
                            axis=1)))

            # Add perpendicular lane vehicles
            perp_ped_start_index = current_lane_len + 3 * veh_state_len + 3 * ped_state_len
            embedding_pedestrians_perp = []
            for j in range(10):
                embedding_pedestrians_perp.append(
                    self.embedding_net_perpendicular(
                        tf.concat(
                            [out_ph[:, current_lane_len:current_lane_len + veh_state_len],
                             out_ph[:, perp_ped_start_index + j * veh_state_len:
                                    perp_ped_start_index + (j + 1) * veh_state_len]],
                            axis=1)))

            # Add opposite lane vehicles
            perp_start_index = current_lane_len + 3 * veh_state_len + \
                3 * ped_state_len + 3 * ped_state_len
            embed_perp_lane_vehs = []
            for j in range(10):
                embed_perp_lane_vehs.append(
                    self.embedding_net_opposite(
                        tf.concat(
                            [out_ph[:, current_lane_len:current_lane_len + veh_state_len],
                             out_ph[:, perp_start_index + j * veh_state_len:
                                    perp_start_index + (j + 1) * veh_state_len]],
                            axis=1)))

            # stack them and take max
            embed_list = embed_front_vehicle + embed_back_vehicle + embed_pedestrians_cur + \
                embedding_pedestrians_perp + embed_perp_lane_vehs
            stacked_out = tf.stack(embed_list, axis=1)
            max_out = tf.reduce_max(stacked_out, axis=1)

            # concatenate the current_lane_status
            max_out = tf.concat([max_out, out_ph[:, :8][:, None]], axis=1)
            q_out = self.q_net(max_out, ac_space.n)

        self.q_values = q_out
        self._setup_init()

    def step(self, obs, state=None, mask=None, deterministic=True):
        q_values, actions_proba = self.sess.run(
            [self.q_values, self.policy_proba], {self.obs_ph: obs})
        if deterministic:
            actions = np.argmax(q_values, axis=1)
        else:
            # Inefficient sampling
            # TODO: replace the loop
            # maybe with Gumbel-max trick ? (http://amid.fish/humble-gumbel)
            actions = np.zeros((len(obs),), dtype=np.int64)
            for action_idx in range(len(obs)):
                actions[action_idx] = np.random.choice(
                    self.n_actions, p=actions_proba[action_idx])

        return actions, q_values, None

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})
