import abc
import logging
import pprint
from typing import List, Tuple

import numpy as np
import tensorflow as tf

from rllib.BaseEnvironment import BaseEnvironment
from rllib.Network import Network

keras = tf.keras
eps = np.finfo(np.float32).eps.item()


class BaseAgent:

    def __init__(self, nn: Network, gamma: float):
        self.env = None
        self.nn: Network = nn
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

    def run_episode(self, env: BaseEnvironment,
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

    @abc.abstractmethod
    def get_inputs(self, curr_timestep, state_hist, actions_hist, rewards_hist):
        pass

    @abc.abstractmethod
    def get_action(self, model_outputs, deterministic):
        pass

    # @tf.function
    def train_step(self, env: BaseEnvironment,
                   max_steps_per_episode: int) -> Tuple[Tuple[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor, tf.Tensor]:
        self.env: BaseEnvironment = env
        with tf.GradientTape() as tape:
            hist_sar, hist_out = self.run_episode(env, max_steps_per_episode)
            loss = self.compute_loss(hist_sar, hist_out)

        grads = tape.gradient(loss, self.nn.model.trainable_variables)
        # Apply the gradients to the model's parameters
        self.nn.optimizer.apply_gradients(zip(grads, self.nn.model.trainable_variables))

        return hist_sar, hist_out, loss

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
