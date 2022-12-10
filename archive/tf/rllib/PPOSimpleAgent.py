import abc
import logging
import pprint
from typing import List, Tuple

import numpy as np
import tensorflow as tf

from rllib.BaseEnvironment import BaseEnvironment
from rllib.Network import Network
from rllib.BaseAgent import BaseAgent
from rllib.Environments import LunarLander
from rllib.Experiment import Experiment
from rllib.Network import Network
from rllib.PlotHelper import PlotHelper
from rllib.utils import logging_setup
from rllib.Buffer import Buffer

keras = tf.keras
eps = np.finfo(np.float32).eps.item()


class PPOSimpleAgent(BaseAgent):

    def __init__(self, nn: Network,
                 epsilon: float = 0.2,
                 gamma: float = 0.97,
                 steps_per_epoch: int = 4000,
                 train_iterations: int = 80,
                 clip_ratio: float = 0.2,
                 ):
        super().__init__(nn=nn, gamma=gamma)
        self.epsilon = epsilon
        self.clip_ratio = clip_ratio
        self.steps_per_epoch = steps_per_epoch
        self.train_iterations = train_iterations

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

    def get_inputs(self, curr_timestep, curr_observation, state_hist, actions_hist, rewards_hist, stack=True):
        print("=" * 80)
        # print("curr_timestep", curr_timestep)
        if stack:
            states, actions, rewards = state_hist.stack(), actions_hist.stack(), rewards_hist.stack()
        else:
            states, actions, rewards = state_hist, actions_hist, rewards_hist
        # self.nn.max_timesteps

        print(curr_observation)
        print(states)
        # print(states)
        state = states[-states:]
        if states.shape[0] < self.nn.max_timesteps:
            state = tf.pad(state, [[self.nn.max_timesteps - states.shape[0], 0], [0, 0]])
        return [state]

    def get_action(self, model_outputs, deterministic):
        action_logits_t, _ = model_outputs
        action_logits_t = tf.reshape(action_logits_t, shape=(1, self.nn.num_actions))
        action = tf.random.categorical(action_logits_t, 1)[0, 0]
        action_probs_t = tf.nn.softmax(action_logits_t)
        if deterministic:
            action = tf.argmax(action_probs_t, axis=1)[0]
        action = tf.cast(action, tf.int32)
        return action

    def get_log_probs(self, logits, a):
        # Compute the log-probabilities of taking actions a by using the logits (i.e. the output of the actor)
        logprobabilities_all = tf.nn.log_softmax(logits)
        logprobability = tf.reduce_sum(
            tf.one_hot(a, self.nn.num_actions) * logprobabilities_all, axis=1
        )
        return logprobability

    def rollout(self, env: BaseEnvironment,
                deterministic: bool = False):
        self.env: BaseEnvironment = env

        observation, episode_return, episode_length = env.tf_reset(), 0, 0
        observation_shape = observation.shape

        sum_return = 0
        sum_length = 0
        num_episodes = 0
        buffer = Buffer(observation_shape, 1, gamma=0.99, lam=0.95)
        hist_model_o = []

        for t in tf.range(self.steps_per_epoch):
            # Get the logits, action, and take one step in the environment
            # model_inputs = self.get_inputs(t,
            #                                observation,
            #                                buffer.observation_buffer,
            #                                buffer.action_buffer,
            #                                buffer.reward_buffer, stack=True)
            model_outputs = self.nn.model(tf.expand_dims(observation, axis=1))
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
            value_t = tf.squeeze(value_t)
            buffer.store(observation, action, reward, value_t, log_prob_t)
            hist_model_o.append(model_outputs)

            # Update the observation
            observation = observation_new

            # Finish trajectory if reached to a terminal state
            terminal = done
            if terminal or (t == self.steps_per_epoch - 1):
                last_value = 0 if done else value_t
                buffer.finish_trajectory(last_value)
                sum_return += episode_return
                sum_length += episode_length
                num_episodes += 1
                observation, episode_return, episode_length = env.reset(), 0, 0

        hist_model_o = list(zip(*hist_model_o))
        return buffer, hist_model_o

    # @tf.function
    def train_step(self, env: BaseEnvironment,
                   max_steps_per_episode: int):
        self.env: BaseEnvironment = env
        self.episode_history = []
        self.model_outputs_history = []

        # generate training data
        print("=" * 80)
        print("Generating training data")
        buffer, hist_model_o = self.rollout(env, deterministic=False)
        rollout_data = buffer.get()

        (
            _, _, advantage_buffer, return_buffer, _, value_buffer, _
        ) = rollout_data
        self.plot_ret(return_buffer, advantage_buffer, value_buffer)

        print("=" * 80)
        print("Training on generated data")
        for e in range(self.train_iterations):
            loss = self.update(rollout_data, hist_model_o)

        return rollout_data, hist_model_o, loss

    # @tf.function
    def update(self, rollout_data, hist_model_out: list):
        (
            observation_buffer,
            action_buffer,
            advantage_buffer,
            return_buffer,
            log_prob_buffer,
            value_buffer,
            reward_buffer
        ) = rollout_data

        # self.plot_ret(return_buffer, advantage_buffer, value_buffer)
        with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
            action_probs, value_t = self.nn.model(observation_buffer)
            new_log_prob = self.get_log_probs(tf.squeeze(action_probs, axis=1), action_buffer)
            ratio = tf.exp(
                new_log_prob
                - log_prob_buffer
            )
            min_advantage = tf.where(
                advantage_buffer > 0,
                (1 + self.clip_ratio) * advantage_buffer,
                (1 - self.clip_ratio) * advantage_buffer,
            )
            # min_advantage = tf.clip_by_value(advantage_buffer, clip_value_min=1-self.clip_ratio, clip_value_max=1 + self.clip_ratio)
            policy_loss = -tf.reduce_mean(tf.minimum(ratio * advantage_buffer, min_advantage))
            value_loss = tf.reduce_mean((return_buffer - value_t) ** 2)
            total_loss = tf.math.add(policy_loss, value_loss)
            loss = tf.reduce_sum(total_loss)
        grads = tape.gradient(loss, self.nn.model.trainable_variables)
        # Apply the gradients to the model's parameters
        self.nn.optimizer.apply_gradients(zip(grads, self.nn.model.trainable_variables))
        return loss

    def plot_ret(self, actual_returns, advantage, critic_returns):
        plot_returns = {
            "plot": [
                {
                    "args": [tf.squeeze(critic_returns)],
                    "label": "Critic Val",
                    "color": "red"
                },
                {
                    "args": [tf.squeeze(actual_returns)],
                    "label": "Actual Val",
                    "color": "green"
                },
                {
                    "args": [tf.squeeze(advantage)],
                    "label": "Advantage",
                    "color": "purple"
                }
            ],
            "fill_between": [
                {
                    "args": [tf.range(tf.shape(actual_returns)[0]), tf.squeeze(actual_returns),
                             tf.squeeze(critic_returns)],
                    "where": tf.squeeze(actual_returns) > tf.squeeze(critic_returns),
                    "color": "green",
                    "alpha": 0.15
                },
                {
                    "args": [tf.range(tf.shape(actual_returns)[0]), tf.squeeze(actual_returns),
                             tf.squeeze(critic_returns)],
                    "where": tf.squeeze(actual_returns) < tf.squeeze(critic_returns),
                    "color": "red",
                    "alpha": 0.15
                }
            ],
            "axhline": [
                {
                    "y": 0,
                    "color": "black",
                    "linestyle": "--"
                }
            ],
            "suptitle": f"PPO Returns ({self.env.name}): "
                        f"{self.nn.name} - {self.nn.inp_s_shape}" +
                        f" + ({self.env.state_scaler.__class__.__name__})"
            if self.env.state_scaler_enable is True else "",
        }
        PlotHelper.plot_from_dict(plot_returns, savefig="plots/ppo_returns.pdf")

    def plot_loss(self, clip_losses, value_losses, total_losses, entropy_loss=None):
        # pass
        plot_losses = {
            "plot": [
                {
                    "args": [tf.squeeze(clip_losses)],
                    "label": "Clip Loss",
                    "color": "lightskyblue"
                },
                {
                    "args": [tf.squeeze(value_losses)],
                    "label": "Value Loss",
                    "color": "salmon"
                },
                {
                    "args": [total_losses],
                    "label": "Total Loss",
                    "color": "black"
                }

            ],
            "axhline": [
                {
                    "y": 0,
                    "color": "black",
                    "linestyle": "--"
                }
            ],
            "title": f"PPO Losses ({self.env.name}):"
            # f" {self.nn.name} + "
            # f"{self.nn.critic_loss.__class__.__name__} + "
            # f"{self.nn.optimizer.__class__.__name__} - "
            # f"LR: {self.nn.learning_rate}",
        }
        if entropy_loss is not None:
            plot_losses["plot"].append({
                "args": [tf.squeeze(entropy_loss)],
                "label": "Entropy Loss",
                "color": "darkorange"
            })

        PlotHelper.plot_from_dict(plot_losses, savefig="plots/ppo_losses.pdf")

# %%
