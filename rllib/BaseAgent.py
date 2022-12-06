import abc
import logging
import pprint
from typing import Tuple

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

    def run_episode(self, env: BaseEnvironment, max_steps: int, deterministic: bool = False) -> Tuple[tf.Tensor,
                                                                                                      Tuple[tf.Tensor,
                                                                                                            tf.Tensor]]:
        self.env: BaseEnvironment = env

        action_probs_hist = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        critic_returns_hist = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

        state_hist = tf.TensorArray(dtype=tf.float32, size=self.nn.max_timesteps,
                                    dynamic_size=True,
                                    element_shape=(self.nn.num_features,))
        actions_hist = tf.TensorArray(dtype=tf.float32, size=self.nn.max_timesteps,
                                      dynamic_size=True,
                                      element_shape=(self.nn.num_actions,))

        rewards = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

        initial_state = env.tf_reset()
        initial_state_shape = initial_state.shape
        current_state = initial_state

        # tq = tqdm(tf.range(max_steps), desc=f"Ep. {self.global_episode:>6}", leave=False)
        for timestep in tf.range(max_steps):

            state_hist = state_hist.write(self.nn.max_timesteps + timestep, tf.squeeze(current_state))

            # action = self.get_action(step, state, deterministic)
            model_inputs = self.get_model_inputs(timestep, state_hist, actions_hist)

            # print("=" * 80)
            # print(current_state)
            # print(current_action_hist)
            # print(last_t_states)

            # logging.info(f"state: {state} \t shape: {state.shape}")
            action_logits_t, critic_value = self.nn.model(model_inputs)
            # logging.debug(f"action_logits_t: {action_logits_t} | critic_value: {critic_value}")

            action_logits_t = tf.reshape(action_logits_t, shape=(1, self.nn.num_actions))
            critic_value = tf.squeeze(critic_value)

            # action = tfp.distributions.Categorical(logits=action_logits_t[0]).sample()
            action = tf.random.categorical(action_logits_t, 1)[0, 0]
            action_probs_t = tf.nn.softmax(action_logits_t)
            if deterministic:
                action = tf.argmax(action_probs_t, axis=1)[0]

            # logging.debug(f"action: {action} | action_probs_t: {action_probs_t}")

            critic_returns_hist = critic_returns_hist.write(timestep, critic_value)
            action_probs_hist = action_probs_hist.write(timestep, action_probs_t[0, action])
            actions_hist = actions_hist.write(self.nn.max_timesteps + timestep, tf.one_hot(action, self.nn.num_actions))

            current_state, reward, done = env.tf_step(action)
            current_state.set_shape(initial_state_shape)

            rewards = rewards.write(timestep, reward)
            if tf.cast(done, tf.bool):
                break

        state_hist.mark_used()
        actions_hist.mark_used()
        rewards = rewards.stack()
        action_probs_hist = action_probs_hist.stack()
        critic_returns_hist = critic_returns_hist.stack()
        # histories = self.on_episode_end(histories)

        return rewards, (action_probs_hist, critic_returns_hist)

    def get_model_inputs(self, curr_timestep, state_hist, actions_hist):
        positions = tf.range(start=curr_timestep - self.nn.max_timesteps + 1, limit=curr_timestep + 1, delta=1)
        positions = tf.clip_by_value(positions, 0, curr_timestep)
        positions = tf.reshape(positions, self.nn.input_t_shape)

        states = state_hist.stack()
        states = states[-self.nn.max_timesteps:, :]
        states = tf.reshape(states, self.nn.input_state_shape)

        actions = actions_hist.stack()
        actions = actions[-self.nn.max_timesteps:, :]
        actions = tf.reshape(actions, self.nn.input_actions_shape)

        return [positions, states, actions]

    @abc.abstractmethod
    def on_episode_start(self):
        pass

    @abc.abstractmethod
    def get_action(self, t, state, deterministic):
        pass

    # @tf.function
    def train_step(self, env: BaseEnvironment, max_steps_per_episode: int) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        self.env: BaseEnvironment = env
        with tf.GradientTape() as tape:
            rewards, extras = self.run_episode(env, max_steps_per_episode)

            compute_error_vals = self.compute_error(rewards, extras)

        self.on_update(compute_error_vals, tape)
        return rewards, compute_error_vals

    @abc.abstractmethod
    def compute_error(self, rewards, extras):
        pass

    @abc.abstractmethod
    def on_update(self, compute_error_vals, tape):
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

    # def run_episode(self, env: BaseEnvironment, max_steps: int, deterministic: bool = False) -> Tuple[tf.Tensor,
    #                                                                                                   tf.Tensor,
    #                                                                                                   tf.Tensor,
    #                                                                                                   Tuple[tf.Tensor,
    #                                                                                                         tf.Tensor]]:
    #     self.env: BaseEnvironment = env
    #
    #     action_probs_hist = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    #     critic_returns_hist = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    #     state_hist = tf.TensorArray(dtype=tf.float32, size=self.nn.max_timesteps,
    #                                 dynamic_size=True,
    #                                 element_shape=(self.nn.num_features,))
    #     actions_hist = tf.TensorArray(dtype=tf.float32, size=self.nn.max_timesteps,
    #                                   dynamic_size=True,
    #                                   element_shape=(self.nn.num_actions,))
    #
    #     rewards = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    #
    #     initial_state = env.tf_reset()
    #     initial_state_shape = initial_state.shape
    #     state = initial_state
    #
    #     # tq = tqdm(tf.range(max_steps), desc=f"Ep. {self.global_episode:>6}", leave=False)
    #     steps = 0
    #     for t in tf.range(max_steps):
    #         steps = t
    #
    #         # action = self.get_action(step, state, deterministic)
    #         state_hist = state_hist.write(self.nn.max_timesteps + t, tf.squeeze(state))
    #         current_state = state_hist.stack()
    #         current_state = current_state[-self.nn.max_timesteps:, :]
    #         current_state = tf.reshape(current_state, self.nn.input_state_shape)
    #         current_action_hist = actions_hist.stack()
    #         current_action_hist = current_action_hist[-self.nn.max_timesteps:, :]
    #         current_action_hist = tf.reshape(current_action_hist, self.nn.input_actions_shape)
    #         last_t_states = tf.range(start=t - self.nn.max_timesteps + 1, limit=t + 1, delta=1)
    #         last_t_states = tf.clip_by_value(last_t_states, 0, t)
    #         last_t_states = tf.reshape(last_t_states, self.nn.input_t_shape)
    #
    #         # print("=" * 80)
    #         # print(current_state)
    #         # print(current_action_hist)
    #         # print(last_t_states)
    #
    #         # logging.info(f"state: {state} \t shape: {state.shape}")
    #         action_logits_t, critic_value = self.nn.model([last_t_states, current_state, current_action_hist])
    #         # logging.debug(f"action_logits_t: {action_logits_t} | critic_value: {critic_value}")
    #
    #         action_logits_t = tf.reshape(action_logits_t, shape=(1, self.nn.num_actions))
    #         critic_value = tf.squeeze(critic_value)
    #
    #         # action = tfp.distributions.Categorical(logits=action_logits_t[0]).sample()
    #         action = tf.random.categorical(action_logits_t, 1)[0, 0]
    #         action_probs_t = tf.nn.softmax(action_logits_t)
    #         if deterministic:
    #             action = tf.argmax(action_probs_t, axis=1)[0]
    #
    #         # logging.debug(f"action: {action} | action_probs_t: {action_probs_t}")
    #
    #         critic_returns_hist = critic_returns_hist.write(t, critic_value)
    #         action_probs_hist = action_probs_hist.write(t, action_probs_t[0, action])
    #         actions_hist = actions_hist.write(self.nn.max_timesteps + t, tf.one_hot(action, self.nn.num_actions))
    #
    #         state, reward, done = env.tf_step(action)
    #         state.set_shape(initial_state_shape)
    #
    #         rewards = rewards.write(t, reward)
    #         if tf.cast(done, tf.bool):
    #             break
    #
    #     state_hist.mark_used()
    #     actions_hist.mark_used()
    #     rewards = rewards.stack()
    #     action_probs_hist = action_probs_hist.stack()
    #     critic_returns_hist = critic_returns_hist.stack()
    #     # histories = self.on_episode_end(histories)
    #
    #     total_reward = tf.math.reduce_sum(rewards)
    #     return steps, total_reward, rewards, (action_probs_hist, critic_returns_hist)
