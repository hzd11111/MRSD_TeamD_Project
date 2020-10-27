import numpy as np
import sys
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
            # out = tf.Print(out, [out], summarize=200, message="PED:")
            out = tf_layers.fully_connected(
                out, num_outputs=16, activation_fn=tf.nn.relu)
            out = tf_layers.fully_connected(
                out, num_outputs=32, activation_fn=tf.nn.relu)
        return out

    def embedding_net_front_back(self, input_vec):
        out = input_vec
        with tf.variable_scope("embedding_network_front_back", reuse=tf.compat.v1.AUTO_REUSE):
            # out = tf.Print(out, [out], summarize=200, message="FRONT_BACK:")
            out = tf_layers.fully_connected(
                out, num_outputs=16, activation_fn=tf.nn.relu)
            out = tf_layers.fully_connected(
                out, num_outputs=32, activation_fn=tf.nn.relu)
        return out

    def embedding_net_perpendicular(self, input_vec):
        out = input_vec
        with tf.variable_scope("embedding_network_perpendicular",
                               reuse=tf.compat.v1.AUTO_REUSE):
            # out = tf.Print(out, [out], summarize=200, message="PERP_VEH:")
            out = tf_layers.fully_connected(
                out, num_outputs=16, activation_fn=tf.nn.relu)
            out = tf_layers.fully_connected(
                out, num_outputs=32, activation_fn=tf.nn.relu)
        return out

    def embedding_net_opposite(self, input_vec):
        out = input_vec
        with tf.variable_scope("embedding_network_opposite", reuse=tf.compat.v1.AUTO_REUSE):
            # out = tf.Print(out, [out], summarize=200, message="OPP_VEH:")
            out = tf_layers.fully_connected(
                out, num_outputs=16, activation_fn=tf.nn.relu)
            out = tf_layers.fully_connected(
                out, num_outputs=32, activation_fn=tf.nn.relu)
        return out

    def embedding_net_intersection(self, input_vec):
        out = input_vec
        with tf.variable_scope("embedding_network_intersection", reuse=tf.compat.v1.AUTO_REUSE):
            out = tf.Print(out, [out], summarize=200, message="INTER_VEH:")
            out = tf_layers.fully_connected(
                out, num_outputs=16, activation_fn=tf.nn.relu)
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
            veh_state_len = 9
            ped_state_len = 8
            current_lane_len = 8
            mask = 1
            out_ph = tf.layers.flatten(self.processed_obs)
            out_ph = tf.Print(out_ph, [out_ph], summarize=300, message="OUT_PH:")
            cur_veh = out_ph[:, current_lane_len:current_lane_len + veh_state_len]
            # Add front vehicle
            front_veh_start = current_lane_len + veh_state_len
            front_veh_mask = out_ph[:, front_veh_start + veh_state_len][:, None]
            embed_front_vehicle = []
            front_veh = out_ph[:, front_veh_start:front_veh_start + veh_state_len]
            front_veh_state = tf.concat([cur_veh, front_veh], axis=1)
            embed_front_vehicle.append(
                self.embedding_net_front_back(front_veh_state) * front_veh_mask)

            # Add back vehicle
            back_veh_start = current_lane_len + veh_state_len + 1 * (veh_state_len + mask)
            back_veh_mask = out_ph[:, back_veh_start + veh_state_len][:, None]
            back_veh = out_ph[:, back_veh_start:back_veh_start + veh_state_len]
            back_veh_state = tf.concat([cur_veh, back_veh], axis=1)
            embed_back_vehicle = []
            embed_back_vehicle.append(
                self.embedding_net_front_back(back_veh_state) * back_veh_mask)

            # Add pedestrians
            # ped_start_index = current_lane_len + veh_state_len + 2 * (veh_state_len + mask)
            embed_pedestrians = []
            # for j in range(3):
            #     ped_mask = out_ph[:, ped_start_index + (j + 1) * (ped_state_len + mask) - 1][:, None]
            #     start = ped_start_index + j * (ped_state_len + mask)
            #     ped = out_ph[:, start:start + ped_state_len]
            #     ped_state = tf.concat([cur_veh, ped], axis=1)
            #     embed_pedestrians.append(
            #         self.embedding_net_pedestrian(ped_state) * ped_mask)

            # Add perpendicular lane vehicles
            perp_start_index = current_lane_len + veh_state_len + \
                2 * (veh_state_len + mask) + 3 * (ped_state_len + mask)

            embed_perp_lane_vehs = []
            for j in range(10):
                perp_mask = out_ph[:, perp_start_index + (j + 1) * (veh_state_len + mask) - 1][:, None]
                start = perp_start_index + j * (veh_state_len + mask)
                perp_veh = out_ph[:, start:start + veh_state_len]
                perp_veh_state = tf.concat([cur_veh, perp_veh], axis=1)
                embed_perp_lane_vehs.append(
                    self.embedding_net_perpendicular(perp_veh_state) * perp_mask)

            # Add opposite lane vehicles
            opp_lane_start_index = current_lane_len + veh_state_len + \
                2 * (veh_state_len + mask) + 3 * (ped_state_len + mask) + \
                10 * (veh_state_len + mask)

            embed_opp_lane_vehs = []
            for j in range(5):
                opp_lane_mask = out_ph[:, opp_lane_start_index + (j + 1) * (veh_state_len + mask) - 1][:, None]
                start = opp_lane_start_index + j * (veh_state_len + mask)
                opp_lane_veh = out_ph[:, start:start + veh_state_len]
                opp_lane_veh_state = tf.concat([cur_veh, opp_lane_veh], axis=1)
                embed_opp_lane_vehs.append(
                    self.embedding_net_opposite(opp_lane_veh_state) * opp_lane_mask)

            # Add vehicles in the intersection
            inter_start_index = current_lane_len + veh_state_len + \
                2 * (veh_state_len + mask) + 3 * (ped_state_len + mask) + \
                10 * (veh_state_len + mask) + 5 * (veh_state_len + mask)

            embed_inter_vehs = []
            for j in range(3):
                inter_lane_mask = out_ph[:, inter_start_index + (j + 1) * (veh_state_len + mask) - 1][:, None]
                start = inter_start_index + j * (veh_state_len + mask)
                inter_veh = out_ph[:, start:start + veh_state_len]
                inter_veh_state = tf.concat([cur_veh, inter_veh], axis=1)
                embed_inter_vehs.append(
                    self.embedding_net_intersection(inter_veh_state) * inter_lane_mask)

            # stack them and take max
            embed_list = embed_front_vehicle + embed_back_vehicle + embed_pedestrians + \
                embed_perp_lane_vehs + embed_opp_lane_vehs + embed_inter_vehs
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
            # out = tf.Print(out, [out], summarize=200, message="FRONT_BACK:")
            out = tf_layers.fully_connected(
                out, num_outputs=32, activation_fn=tf.nn.relu)
        return out

    def embedding_net_perpendicular(self, input_vec):
        out = input_vec
        with tf.variable_scope("embedding_network_perpendicular",
                               reuse=tf.compat.v1.AUTO_REUSE):
            # out = tf.Print(out, [out], summarize=200, message="PERP_VEH:")
            out = tf_layers.fully_connected(
                out, num_outputs=32, activation_fn=tf.nn.relu)
        return out

    def embedding_net_opposite(self, input_vec):
        out = input_vec
        with tf.variable_scope("embedding_network_opposite", reuse=tf.compat.v1.AUTO_REUSE):
            # out = tf.Print(out, [out], summarize=200, message="OPPOSITE:")
            out = tf_layers.fully_connected(
                out, num_outputs=32, activation_fn=tf.nn.relu)
        return out

    def embedding_net_intersection(self, input_vec):
        out = input_vec
        with tf.variable_scope("embedding_network_intersection", reuse=tf.compat.v1.AUTO_REUSE):
            # out = tf.Print(out, [out], summarize=200, message="INTER_VEH:")
            out = tf_layers.fully_connected(
                out, num_outputs=16, activation_fn=tf.nn.relu)
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
            veh_state_len = 9
            ped_state_len = 8
            current_lane_len = 8
            mask = 1
            out_ph = tf.layers.flatten(self.processed_obs)
            # out_ph = tf.Print(out_ph, [out_ph], summarize=300, message="OUT_PH:")
            cur_veh = out_ph[:, current_lane_len:current_lane_len + veh_state_len]

            # Add front vehicle
            front_veh_start = current_lane_len + veh_state_len
            front_veh_mask = out_ph[:, front_veh_start + veh_state_len][:, None]
            embed_front_vehicle = []
            front_veh = out_ph[:, front_veh_start:front_veh_start + veh_state_len]
            front_veh_state = tf.concat([cur_veh, front_veh], axis=1)
            embed_front_vehicle.append(
                self.embedding_net_front_back(front_veh_state) * front_veh_mask)

            # Add back vehicle
            back_veh_start = current_lane_len + veh_state_len + 1 * (veh_state_len + mask)
            back_veh_mask = out_ph[:, back_veh_start + veh_state_len][:, None]
            back_veh = out_ph[:, back_veh_start:back_veh_start + veh_state_len]
            back_veh_state = tf.concat([cur_veh, back_veh], axis=1)
            embed_back_vehicle = []
            embed_back_vehicle.append(
                self.embedding_net_front_back(back_veh_state) * back_veh_mask)

            # Add pedestrians
            ped_start_index = current_lane_len + veh_state_len + 2 * (veh_state_len + mask)
            embed_pedestrians = []
            # for j in range(3):
            #     ped_mask = out_ph[:, ped_start_index + (j + 1) * (ped_state_len + mask) - 1][:, None]
            #     start = ped_start_index + j * (ped_state_len + mask)
            #     ped = out_ph[:, start:start + ped_state_len]
            #     ped_state = tf.concat([cur_veh, ped], axis=1)
            #     embed_pedestrians.append(
            #         self.embedding_net_pedestrian(ped_state) * ped_mask)

            # Add perpendicular lane vehicles
            perp_start_index = current_lane_len + veh_state_len + \
                2 * (veh_state_len + mask) + 3 * (ped_state_len + mask)

            embed_perp_lane_vehs = []
            for j in range(10):
                perp_mask = out_ph[:, perp_start_index + (j + 1) * (veh_state_len + mask) - 1][:, None]
                start = perp_start_index + j * (veh_state_len + mask)
                perp_veh = out_ph[:, start:start + veh_state_len]
                perp_veh_state = tf.concat([cur_veh, perp_veh], axis=1)
                embed_perp_lane_vehs.append(
                    self.embedding_net_front_back(perp_veh_state) * perp_mask)

            # Add opposite lane vehicles
            opp_lane_start_index = current_lane_len + veh_state_len + \
                2 * (veh_state_len + mask) + 3 * (ped_state_len + mask) + \
                10 * (veh_state_len + mask)

            embed_opp_lane_vehs = []
            for j in range(10):
                opp_lane_mask = out_ph[:, opp_lane_start_index + (j + 1) * (veh_state_len + mask) - 1][:, None]
                start = opp_lane_start_index + j * (veh_state_len + mask)
                opp_lane_veh = out_ph[:, start:start + veh_state_len]
                opp_lane_veh_state = tf.concat([cur_veh, opp_lane_veh], axis=1)
                embed_opp_lane_vehs.append(
                    self.embedding_net_front_back(opp_lane_veh_state) * opp_lane_mask)

            # Add vehicles in the intersection
            inter_start_index = current_lane_len + veh_state_len + \
                2 * (veh_state_len + mask) + 3 * (ped_state_len + mask) + \
                10 * (veh_state_len + mask) + 10 * (veh_state_len + mask)

            embed_inter_vehs = []
            for j in range(3):
                inter_lane_mask = out_ph[:, inter_start_index + (j + 1) * (veh_state_len + mask) - 1][:, None]
                start = inter_start_index + j * (veh_state_len + mask)
                inter_veh = out_ph[:, start:start + veh_state_len]
                inter_veh_state = tf.concat([cur_veh, inter_veh], axis=1)
                embed_inter_vehs.append(
                    self.embedding_net_front_back(inter_veh_state) * inter_lane_mask)

            # stack them and take max
            embed_list = embed_front_vehicle + embed_back_vehicle + embed_pedestrians + \
                embed_perp_lane_vehs + embed_opp_lane_vehs + embed_inter_vehs
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
            # out = tf.Print(out, [out], summarize=200, message="FRONT_BACK:")
            out = tf_layers.fully_connected(
                out, num_outputs=16, activation_fn=tf.nn.relu)
            out = tf_layers.fully_connected(
                out, num_outputs=32, activation_fn=tf.nn.relu)
        return out

    def embedding_net_perp_veh(self, input_vec):
        out = input_vec
        with tf.variable_scope("embedding_network_perpendicular_vehicles", reuse=tf.compat.v1.AUTO_REUSE):
            # out = tf.Print(out, [out], summarize=200, message="PERP_VEH:")
            out = tf_layers.fully_connected(
                out, num_outputs=16, activation_fn=tf.nn.relu)
            out = tf_layers.fully_connected(
                out, num_outputs=32, activation_fn=tf.nn.relu)
        return out

    def embedding_net_intersection(self, input_vec):
        out = input_vec
        with tf.variable_scope("embedding_network_intersection", reuse=tf.compat.v1.AUTO_REUSE):
            out = tf.Print(out, [out], summarize=200, message="INTER_VEH:")
            out = tf_layers.fully_connected(
                out, num_outputs=16, activation_fn=tf.nn.relu)
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
        super(CustomIntersectionRightTurn, self).__init__(sess, ob_space, ac_space, n_env, n_steps,
                                                         n_batch, dueling=dueling, reuse=reuse,
                                                         scale=False, obs_phs=obs_phs)
        with tf.variable_scope("model", reuse=reuse):
            veh_state_len = 9
            ped_state_len = 8
            current_lane_len = 8
            mask = 1
            out_ph = tf.layers.flatten(self.processed_obs)
            # out_ph = tf.Print(out_ph, [out_ph], summarize=200, message="OUT_PH:")
            cur_veh = out_ph[:, current_lane_len:current_lane_len + veh_state_len]

            # Add front vehicle
            front_veh_start = current_lane_len + veh_state_len
            front_veh_mask = out_ph[:, front_veh_start + veh_state_len][:, None]
            embed_front_vehicle = []
            front_veh = out_ph[:, front_veh_start:front_veh_start + veh_state_len]
            front_veh_state = tf.concat([cur_veh, front_veh], axis=1)
            embed_front_vehicle.append(
                self.embedding_net_front_back(front_veh_state) * front_veh_mask)

            # Add back vehicle
            back_veh_start = current_lane_len + veh_state_len + 1 * (veh_state_len + mask)
            back_veh_mask = out_ph[:, back_veh_start + veh_state_len][:, None]
            back_veh = out_ph[:, back_veh_start:back_veh_start + veh_state_len]
            back_veh_state = tf.concat([cur_veh, back_veh], axis=1)
            embed_back_vehicle = []
            embed_back_vehicle.append(
                self.embedding_net_front_back(back_veh_state) * back_veh_mask)

            # Add pedestrians from cur lane
            cur_ped_start_index = current_lane_len + veh_state_len + 2 * (veh_state_len + mask)
            embed_pedestrians_cur = []
            # for j in range(3):
            #     cur_ped_mask = out_ph[:, cur_ped_start_index + (j + 1) * (ped_state_len + mask) - 1][:, None]
            #     start = cur_ped_start_index + j * (ped_state_len + mask)
            #     cur_ped = out_ph[:, start:start + ped_state_len]
            #     cur_ped_state = tf.concat([cur_veh, cur_ped], axis=1)
            #     embed_pedestrians_cur.append(
            #         self.embedding_net_ped_cur(cur_ped_state) * cur_ped_mask)

            # Add perpendicular lane pedestrians
            perp_ped_start_index = current_lane_len + veh_state_len + 2 * (veh_state_len + mask) \
                + 3 * (ped_state_len + mask)
            embedding_pedestrians_perp = []
            # for j in range(3):
            #     perp_ped_mask = out_ph[:, perp_ped_start_index + (j + 1) * (ped_state_len + mask) - 1][:, None]
            #     start = perp_ped_start_index + j * (ped_state_len + mask)
            #     perp_ped = out_ph[:, start:start + ped_state_len]
            #     perp_ped_state = tf.concat([cur_veh, perp_ped], axis=1)
            #     embedding_pedestrians_perp.append(
            #         self.embedding_net_ped_perp(perp_ped_state) * perp_ped_mask)

            # Add perpendicular lane vehicles
            perp_veh_start_index = current_lane_len + veh_state_len + 2 * (veh_state_len + mask) \
                + 6 * (ped_state_len + mask)
            embed_perp_lane_vehs = []
            for j in range(5):
                perp_veh_mask = out_ph[:, perp_veh_start_index + (j + 1) * (veh_state_len + mask) - 1][:, None]
                start = perp_veh_start_index + j * (veh_state_len + mask)
                print(start, start + veh_state_len, veh_state_len)
                perp_veh = out_ph[:, start:start + veh_state_len]
                perp_veh_state = tf.concat([cur_veh, perp_veh], axis=1)
                embed_perp_lane_vehs.append(
                    self.embedding_net_perp_veh(perp_veh_state) * perp_veh_mask)

            # Add vehicles in the intersection
            inter_start_index = current_lane_len + veh_state_len + 2 * (veh_state_len + mask) \
                + 6 * (ped_state_len + mask) + 5 * (veh_state_len + mask)

            embed_inter_vehs = []
            for j in range(3):
                inter_lane_mask = out_ph[:, inter_start_index + (j + 1) * (veh_state_len + mask) - 1][:, None]
                start = inter_start_index + j * (veh_state_len + mask)
                inter_veh = out_ph[:, start:start + veh_state_len]
                inter_veh_state = tf.concat([cur_veh, inter_veh], axis=1)
                embed_inter_vehs.append(
                    self.embedding_net_intersection(inter_veh_state) * inter_lane_mask)

            # stack them and take max
            embed_list = embed_front_vehicle + embed_back_vehicle + embed_pedestrians_cur + \
                embedding_pedestrians_perp + embed_perp_lane_vehs + embed_inter_vehs
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
