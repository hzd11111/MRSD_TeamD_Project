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
            q_out = self.q_net(max_out, ac_space.n)
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


