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

    def get_inputs(self, curr_timestep, state_hist, actions_hist, rewards_hist):
        print("=" * 80)
        print("curr_timestep", curr_timestep)
        states, actions, rewards = state_hist.stack(), actions_hist.stack(), rewards_hist.stack()
        # print("s", states)
        # print("a", actions)
        # print("r", rewards)

        pos_s = tf.range(start=curr_timestep - self.nn.max_timesteps + 1, limit=curr_timestep + 1, delta=1)
        pos_s = tf.clip_by_value(pos_s, clip_value_min=-1, clip_value_max=curr_timestep)

        states = tf.gather(states, pos_s, axis=0)

        actions = tf.one_hot(actions, depth=self.nn.num_actions)
        actions = tf.gather(actions, pos_s, axis=0)

        positions = tf.reshape(pos_s, shape=self.nn.inp_p_shape)
        states = tf.reshape(states, shape=self.nn.inp_s_shape)
        actions = tf.reshape(actions, shape=self.nn.inp_a_shape)
        print("positions", positions)
        print("states", states)
        print("actions", actions)
        print("=" * 80)

        return [positions, states, actions]

    def get_action(self, model_outputs, deterministic):
        action_logits_t, critic_value = model_outputs
        action_logits_t = tf.reshape(action_logits_t, shape=(1, self.nn.num_actions))
        action = tf.random.categorical(action_logits_t, 1)[0, 0]
        action_probs_t = tf.nn.softmax(action_logits_t)
        if deterministic:
            action = tf.argmax(action_probs_t, axis=1)[0]
        action = tf.cast(action, tf.int32)
        return action

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
