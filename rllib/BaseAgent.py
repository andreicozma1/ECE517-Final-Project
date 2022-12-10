import abc
import logging
import pprint
from typing import List, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import GradientTape

from rllib.BaseEnvironment import BaseEnvironment
from rllib.Networks import TransformerNetwork

keras = tf.keras
eps = np.finfo(np.float32).eps.item()


class BaseAgent:

    def __init__(self, nn: TransformerNetwork, gamma: float):
        self.env = None
        self.nn: TransformerNetwork = nn
        self.gamma: float = gamma
        logging.info(f"Args:\n{pprint.pformat(self.__dict__, width=30)}")

    @property
    def save_path_env(self):
        if self.env is None:
            raise ValueError("Environment not set")
        return self.env.save_path_env

    @property
    def config(self):
        return {
                "nn"   : self.nn.config,
                "gamma": self.gamma,
        }

    def expected_return(self, rewards: tf.Tensor) -> tf.Tensor:
        """Compute expected returns per timestep."""

        n = tf.shape(rewards)[0]
        returns = tf.TensorArray(dtype=tf.float32, size=n)

        # Start from the end of `rewards` and accumulate reward sums
        # into the `returns` array
        rewards = tf.cast(rewards[::-1], dtype=tf.float32)
        discounted_sum = tf.constant(0.0)
        discounted_sum_shape = discounted_sum.shape
        for i in tf.range(n):
            reward = rewards[i]
            discounted_sum = reward + self.gamma * discounted_sum
            discounted_sum.set_shape(discounted_sum_shape)
            returns = returns.write(i, discounted_sum)

        returns = returns.stack()[::-1]
        return returns

    @abc.abstractmethod
    def get_inputs(self, curr_timestep, state_hist, actions_hist, rewards_hist):
        pass

    @abc.abstractmethod
    def get_action(self, model_outputs, deterministic):
        pass

    def rollout(self, env: BaseEnvironment,
                deterministic: bool = False):
        self.env: BaseEnvironment = env

        observation, episode_return, episode_length = env.tf_reset(), 0, 0
        observation_shape = observation.shape

        sum_return = 0
        sum_length = 0
        num_episodes = 0
        buffer = Buffer(observation_shape, self.nn.max_timesteps, gamma=0.99, lam=0.95)
        hist_model_o = []

        for _ in tf.range(self.num_rollout_episodes):
            for t in tf.range(self.nn.max_timesteps):

                # Get the logits, action, and take one step in the environment
                model_inputs = self.get_inputs(t,
                                               observation,
                                               buffer)
                model_outputs = self.nn.model(model_inputs)
                action: tf.Tensor = self.get_action(model_outputs, deterministic)
                observation_new, reward, done = env.tf_step(action)
                episode_return += reward
                episode_length += 1

                # Get the value and log-probability of the action
                logits, value_t = model_outputs
                value_t = tf.squeeze(value_t)
                logits = logits[0]
                log_prob_t = self.get_log_probs(logits, tf.expand_dims(action, axis=0))

                # Store obs, act, rew, v_t, logp_pi_t
                buffer.store(observation, action, reward, value_t, log_prob_t)
                hist_model_o.append(model_outputs)

                # Update the observation
                observation = observation_new

                # Finish trajectory if reached to a terminal state
                if tf.cast(done, tf.bool):
                    # last_value = 0 if tf.cast(done, tf.bool) else value_t
                    buffer.finish_trajectory(0)
                    sum_return += episode_return
                    sum_length += episode_length
                    num_episodes += 1
                    observation, episode_return, episode_length = env.reset(), 0, 0

        hist_model_o = list(zip(*hist_model_o))
        return buffer, hist_model_o

    # @tf.function
    def train_step(self, env: BaseEnvironment,
                   max_steps_per_episode: int) -> Tuple[Tuple[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor, tf.Tensor]:
        self.env: BaseEnvironment = env
        with tf.GradientTape() as tape:
            hist_sar, hist_out = self.run_episode(env, max_steps_per_episode)
            loss = self.compute_loss(hist_sar, hist_out)

        self.apply_grads(loss, tape)

        return hist_sar, hist_out, loss

    def apply_grads(self, loss, tape: GradientTape):
        grads = tape.gradient(loss, self.nn.model.trainable_variables)
        # Apply the gradients to the model's parameters
        self.nn.optimizer.apply_gradients(zip(grads, self.nn.model.trainable_variables))

    @abc.abstractmethod
    def compute_loss(self, rewards, extras):
        pass

    def normalize(self, values: tf.Tensor) -> tf.Tensor:
        c_min = tf.reduce_min(values)
        c_max = tf.reduce_max(values)
        c_range = c_max - c_min
        values = (values - c_min) / c_range
        return values

    def standardize(self, values: tf.Tensor) -> tf.Tensor:
        mean = tf.reduce_mean(values)
        std = tf.math.reduce_std(values)
        values = (values - mean) / std
        return values

    def get_log_probs(self, logits, a):
        # Compute the log-probabilities of taking actions a by using the logits (i.e. the output of the actor)
        logprobabilities_all = tf.nn.log_softmax(logits)
        logprobability = tf.reduce_sum(
                tf.one_hot(a, self.nn.num_actions) * logprobabilities_all, axis=1
        )
        return logprobability

    def get_action_1(self, model_outputs, deterministic):
        action_logits_t, critic_value = model_outputs
        action_logits_t = tf.reshape(action_logits_t, shape=(1, self.nn.num_actions))
        action = tf.random.categorical(action_logits_t, 1)[0, 0]
        action_probs_t = tf.nn.softmax(action_logits_t)
        if deterministic:
            action = tf.argmax(action_probs_t, axis=1)[0]
        action = tf.cast(action, tf.int32)
        return action

    def get_entropy_loss(self, action_probs: tf.Tensor) -> tf.Tensor:
        entropy_loss = tf.math.multiply(action_probs, tf.math.log(action_probs))
        entropy_loss = -1 * tf.reduce_sum(entropy_loss, axis=-1)
        return entropy_loss

    def gen_episode(self, env: BaseEnvironment,
                    max_steps: int,
                    deterministic: bool = False) -> Tuple[Tuple[tf.Tensor, tf.Tensor, tf.Tensor], list]:

        self.env: BaseEnvironment = env

        initial_state = env.tf_reset()
        initial_state_shape = initial_state.shape

        hist_s = tf.TensorArray(dtype=tf.float32, size=0,
                                dynamic_size=True,
                                element_shape=initial_state_shape)
        hist_a = tf.TensorArray(dtype=tf.int32, size=0,
                                dynamic_size=True,
                                element_shape=())
        hist_r = tf.TensorArray(dtype=tf.float32, size=0,
                                dynamic_size=True,
                                element_shape=())

        hist_model_o = []

        current_state = initial_state
        # tq = tqdm(tf.range(max_steps), desc=f"Ep. {self.global_episode:>6}", leave=False)
        for curr_timestep in tf.range(max_steps):
            hist_s = hist_s.write(curr_timestep, current_state)

            model_inputs = self.get_inputs(curr_timestep, hist_s, hist_a, hist_r)
            model_outputs = self.nn.model(model_inputs)
            # print("model_outputs", model_outputs)
            hist_model_o.append(model_outputs)

            action: tf.Tensor = self.get_action(model_outputs, deterministic)
            hist_a = hist_a.write(curr_timestep, action)

            tf_step: List[tf.Tensor] = env.tf_step(action)
            current_state, reward, done = tf_step

            hist_r = hist_r.write(curr_timestep, reward)
            current_state.set_shape(initial_state_shape)
            # hist_s = hist_s.write(curr_timestep + 1, current_state)
            if tf.cast(done, tf.bool):
                break

        hist_s = hist_s.stack()
        hist_a = hist_a.stack()
        hist_r = hist_r.stack()

        hist_model_o = zip(*hist_model_o)

        return (hist_s, hist_a, hist_r), hist_model_o
