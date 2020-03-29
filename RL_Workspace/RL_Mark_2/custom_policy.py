import gym
import tensorflow as tf
import tensorflow.contrib as tf_contrib
from stable_baselines.deepq.policies import DQNPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines import A2C,DQN

# Custom MLP policy of three layers of size 128 each for the actor and 2 layers of 32 for the critic,
# with a nature_cnn feature extractor
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
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False,
                 obs_phs=None, dueling=False, **kwargs):
        super(CustomPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps,
                                                n_batch, dueling=dueling, reuse=reuse,
                                                scale=False, obs_phs=obs_phs)

        with tf.variable_scope("model", reuse=reuse):
            action_out = tf.layers.flatten(self.processed_obs)
            embedding_layers=[16,32,64]
            '''embed_model = tf.keras.Sequential()
            embed_model.add(tf.keras.layers.Dense(embedding_layers[0],input_shape=(8,), activation=tf.nn.relu))
            for layer in embedding_layers[1:]:
                embed_model.add(tf.keras.layers.Dense(layer,activation=tf.nn.relu))
            '''    
            action_out = tf_layers.fully_connected(action_out, num_outputs=64, activation_fn=tf.nn.relu)
            action_out = tf_layers.fully_connected(action_out, num_outputs=128, activation_fn=tf.nn.relu)
            action_out = tf_layers.fully_connected(action_out, num_outputs=4, activation_fn=tf.nn.tanh)
            '''
            with tf.variable_scope("embedding_inputs", reuse=False):
                out_ph=extracted_features
                embed_list=[]
                for i in range(5):
                    embed_list.append(tf.concat([out_ph[:,:4],out_ph[:,(i+1)*4:(i+2)*4]],axis=1))
            with tf.variable_scope("embedding_outputs", reuse=False):
                embed_out_list=[]
                for item in embed_list:
                    embed_out_list.append(embed_model(item))
            stacked_out = tf.stack(embed_out_list,axis=1)
            max_out = tf.reduce_max(stacked_out, axis=1)
            q_out=network_model(max_out)
            '''

        self.q_values=action_out
        self._setup_init()

    def step(self, obs, state=None, mask=None, deterministic=True):
        q_values, actions_proba = self.sess.run([self.q_values, self.policy_proba], {self.obs_ph: obs})
        if deterministic:
            actions = np.argmax(q_values, axis=1)
        else:
            # Unefficient sampling
            # TODO: replace the loop
            # maybe with Gumbel-max trick ? (http://amid.fish/humble-gumbel)
            actions = np.zeros((len(obs),), dtype=np.int64)
            for action_idx in range(len(obs)):
                actions[action_idx] = np.random.choice(self.n_actions, p=actions_proba[action_idx])

        return actions, q_values, None

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})


# Create and wrap the environment
env = DummyVecEnv([lambda: gym.make('Breakout-v0')])

model = DQN(CustomPolicy, env, verbose=1)
# Train the agent
model.learn(total_timesteps=100000)