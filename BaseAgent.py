import abc
import logging
import pprint
from typing import Tuple

import numpy as np
import pygame
import tensorflow as tf
from matplotlib import pyplot as plt

from PongEnvironment import PongEnvironment
from NeuralNet import NeuralNet

keras = tf.keras


class BaseAgent:

    def __init__(self, gamma: float = 0.97):
        self.gamma: float = gamma
        logging.info(f"Agent Args: {pprint.pformat(self.__dict__)}")

    def get_expected_return(self,
                            rewards: tf.Tensor) -> tf.Tensor:
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
        returns = tf.math.subtract(returns, tf.math.reduce_mean(returns))
        returns = tf.math.divide(returns, tf.math.reduce_std(returns) + np.finfo(np.float32).eps.item())
        # transform such that only positive values are returned
        # returns = tf.math.add(returns, tf.math.abs(tf.math.reduce_min(returns)))
        # returns = tf.math.divide(returns, tf.math.reduce_max(returns))
        return returns

    def run_episode(self, env: PongEnvironment, max_steps: int, deterministic: bool = False) -> Tuple[tf.Tensor, dict]:
        self.on_episode_start()
        reward_hist = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)

        state = env.reset()
        initial_state_shape = state.shape

        for t in tf.range(max_steps):
            # state = tf.convert_to_tensor(state)
            action = self.get_action(t, state)

            state, reward, done = env.tf_step(action)
            state.set_shape(initial_state_shape)
            reward_hist = reward_hist.write(t, reward)
            if done:
                break

        reward_hist = reward_hist.stack()
        self.on_episode_end()

        stats = {
                "steps"       : len(reward_hist.numpy()),
                "total_reward": float(tf.math.reduce_sum(reward_hist)),
        }
        return reward_hist, stats

    @abc.abstractmethod
    def on_episode_start(self):
        pass

    @abc.abstractmethod
    def get_action(self, t, state):
        pass

    @abc.abstractmethod
    def on_episode_end(self):
        pass

    # @tf.function
    def train_step(self, env: PongEnvironment, max_steps_per_episode: int) -> dict:
        """Runs a model training step."""

        with tf.GradientTape() as tape:
            # Run the model for one episode to collect training data
            rewards, stats = self.run_episode(env, max_steps_per_episode, deterministic=False)

            self.on_update(rewards, tape)

        return stats

    @abc.abstractmethod
    def on_update(self, rewards, tape):
        pass