import numpy as np
# import gym
# from gym import spaces
import tensorflow as tf
import tensorflow.contrib.layers as tf_layers

from stable_baselines.deepq.policies import DQNPolicy


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

    def embedding_net_pedestrian(self, input_vec):
        out = input_vec
        with tf.variable_scope("embedding_network_pedestrian", reuse=tf.compat.v1.AUTO_REUSE):
            out = tf_layers.fully_connected(out, num_outputs=32, activation_fn=tf.nn.relu)
        return out

    def embedding_net_front(self, input_vec):
        out = input_vec
        with tf.variable_scope("embedding_network_front", reuse=tf.compat.v1.AUTO_REUSE):
            # out = tf.Print(out, [out], summarize=200, message="Front:")
            out = tf_layers.fully_connected(out, num_outputs=32, activation_fn=tf.nn.relu)
        return out

    def embedding_net_back(self, input_vec):
        out = input_vec
        with tf.variable_scope("embedding_network_back", reuse=tf.compat.v1.AUTO_REUSE):
            # out = tf.Print(out, [out], summarize=200, message="Back:")
            out = tf_layers.fully_connected(out, num_outputs=32, activation_fn=tf.nn.relu)
        return out

    def embedding_net_adjacent(self, input_vec):
        out = input_vec
        with tf.variable_scope("embedding_network_adjacent", reuse=tf.compat.v1.AUTO_REUSE):
            # out = tf.Print(out, [out], summarize=200, message="Adjacent:")
            out = tf_layers.fully_connected(out, num_outputs=32, activation_fn=tf.nn.relu)
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
            # out_ph = tf.Print(out_ph, [out_ph], summarize=300, message="OUT_PH:")
            embed_adjacent_vehicles = []
            # import ipdb; ipdb.set_trace()
            # Add adjacent vehicles
            veh_state_len = 6
            ped_state_len = 6
            mask = 1
            cur_veh = out_ph[:, :veh_state_len]
            adj_veh_start = veh_state_len
            for j in range(5):
                adj_veh_mask = out_ph[:, adj_veh_start + (j + 1) * (veh_state_len + mask) - 1][:, None]
                start = adj_veh_start + j * (veh_state_len + mask)
                adj_veh = out_ph[:, start:start + veh_state_len]
                adj_veh_state = tf.concat([cur_veh, adj_veh], axis=1)
                # filtered_adj_veh_state = tf.boolean_mask(adj_veh_state, adj_veh_mask, name="adj_veh_masking"+str(j))
                embed_adjacent_vehicles.append(
                    self.embedding_net_adjacent(adj_veh_state) * adj_veh_mask)

            # Add front vehicle
            embed_front_vehicle = []
            front_veh_start = veh_state_len + 5 * (veh_state_len + mask)
            front_veh_mask = out_ph[:, front_veh_start + veh_state_len][:, None]
            front_veh = out_ph[:, front_veh_start:front_veh_start + veh_state_len]
            front_veh_state = tf.concat([cur_veh, front_veh], axis=1)
            # filtered_front_veh_state = tf.boolean_mask(front_veh_state, front_veh_mask, name="front_veh_masking")
            embed_front_vehicle.append(
                self.embedding_net_front(front_veh_state) * front_veh_mask)

            # Add back vehicle
            embed_back_vehicle = []
            back_veh_start = veh_state_len + 6 * (veh_state_len + mask)
            back_veh_mask = out_ph[:, back_veh_start + veh_state_len][:, None]
            back_veh = out_ph[:, back_veh_start:back_veh_start + veh_state_len]
            back_veh_state = tf.concat([cur_veh, back_veh], axis=1)
            # filtered_back_veh_state = tf.boolean_mask(back_veh_state, back_veh_mask, name="back_veh_mask")
            embed_back_vehicle.append(
                self.embedding_net_back(back_veh_state) * back_veh_mask)

            # Add pedestrians
            ped_veh_start = veh_state_len + 7 * (veh_state_len + mask)
            embed_pedestrians = []
            for j in range(3):
                ped_veh_mask = out_ph[:, ped_veh_start + (j + 1) * (ped_state_len + mask) - 1][:, None]
                start = ped_veh_start + j * (ped_state_len + mask)
                ped = out_ph[:, start:start + ped_state_len]
                ped_state = tf.concat([cur_veh, ped], axis=1)
                embed_pedestrians.append(
                    self.embedding_net_pedestrian(ped_state) * ped_veh_mask)

            embed_list = embed_adjacent_vehicles + embed_front_vehicle + embed_back_vehicle  # embed_pedestrians
            stacked_out = tf.stack(embed_list, axis=1)
            max_out = tf.reduce_max(stacked_out, axis=1)
            # concatenate the lane distance
            max_out = tf.concat([max_out, out_ph[:, -2:][:, None]], axis=1)
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
                actions[action_idx] = np.random.choice(self.n_actions, p=actions_proba[action_idx])

        return actions, q_values, None

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})
