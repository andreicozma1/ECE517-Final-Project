import logging
import pprint
from typing import Tuple

import numpy as np
import pygame
import tensorflow as tf
from matplotlib import pyplot as plt

from Environment import Environment
from NeuralNet import NeuralNet

keras = tf.keras


class Agent:

    def __init__(self, nn: NeuralNet, gamma: float = 0.97):
        self.nn: NeuralNet = nn
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
        returns = tf.math.divide(returns, tf.math.reduce_max(returns))

        return returns

    def compute_loss(self,
                     action_probs: tf.Tensor,
                     critic_vals: tf.Tensor,
                     actual_vals: tf.Tensor) -> tf.Tensor:
        """Computes the combined Actor-Critic loss."""
        if self.nn.loss is None:
            raise ValueError("Loss is None")

        advantage = tf.math.subtract(actual_vals, critic_vals)
        print()

        print(f"ARs: {actual_vals.shape} | CRs: {critic_vals.shape} | Adv: {advantage.shape}")

        actor_losses = -tf.math.multiply(tf.math.log(action_probs), advantage)
        actor_loss_sum = tf.math.reduce_sum(actor_losses)

        # map the critic returns and actual returns to the loss function independently and then reduce sum
        critic_losses = self.nn.loss(critic_vals, actual_vals)
        # reshape the critic losses to match the actor losses
        critic_losses = tf.reshape(critic_losses, shape=(tf.shape(actor_losses)))
        # deal with the case where the advantage is negative
        # critic_losses = tf.where(advantage > 0, critic_losses, -critic_losses)
        critic_loss_sum = tf.math.reduce_sum(critic_losses)

        print(f"ALs: {actor_losses.shape} | CLs: {critic_losses.shape}")
        print(f"AL: {actor_loss_sum} | CL: {critic_loss_sum}")

        total_losses = actor_losses + critic_losses

        # incorporate the entropy loss
        entropy_loss = tf.math.reduce_sum(action_probs * tf.math.log(action_probs))
        total_losses += entropy_loss

        total_loss_sum = tf.math.reduce_sum(total_losses)

        self.plot_tr(action_probs, actor_losses, actual_vals, advantage, critic_losses, critic_vals, total_losses)

        return total_loss_sum

    def plot_tr(self, action_probs, actor_losses, actual_vals, advantage, critic_losses, critic_vals, total_losses):
        plt.plot(tf.squeeze(action_probs), label='Actor Probs', color='steelblue')
        # action log probs
        # plt.plot(tf.squeeze(action_log_probs), label='action_log_probs', color='blue')
        plt.plot(tf.squeeze(actor_losses), label='Actor Loss', color="lightskyblue")
        plt.plot(tf.squeeze(critic_losses), label='Critic Loss', color='salmon')
        plt.plot(tf.squeeze(total_losses), label='Total Loss', color='black')
        plt.plot(tf.squeeze(critic_vals), label='Critic Val', color='red')
        plt.plot(tf.squeeze(actual_vals), label='Actual Val', color="green")
        plt.plot(tf.squeeze(advantage), label='Advantage', color='purple')
        # Highlight the difference between the actual returns and the critic returns
        plt.fill_between(tf.range(tf.shape(actual_vals)[0]), tf.squeeze(actual_vals), tf.squeeze(critic_vals),
                         where=tf.squeeze(actual_vals) > tf.squeeze(critic_vals), color='green', alpha=0.15)
        plt.fill_between(tf.range(tf.shape(actual_vals)[0]), tf.squeeze(actual_vals), tf.squeeze(critic_vals),
                         where=tf.squeeze(actual_vals) < tf.squeeze(critic_vals), color='red', alpha=0.15)
        plt.ylim(-2, 5)
        plt.grid(color='lightgray', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.legend(loc='lower left')
        plt.show()

    def run_episode(self, env: Environment, max_steps: int, deterministic: bool = False) -> Tuple[tf.Tensor, tf.Tensor,
                                                                                                  tf.Tensor, dict]:
        if self.nn.model is None:
            raise ValueError("Model is None")
        reward_hist = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
        action_probs_hist = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        critic_returns_hist = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

        state = env.reset()
        initial_state_shape = state.shape

        for t in tf.range(max_steps):
            # state = tf.convert_to_tensor(state)
            state = tf.reshape(state, self.nn.input_shape)
            # Predict action probabilities and estimated future rewards from environment state
            action_logits_t, critic_value = self.nn.model(state)
            action_logits_t = tf.reshape(action_logits_t, shape=(1, self.nn.num_actions))
            critic_value = tf.squeeze(critic_value)

            # Sample next action from the action probability distribution
            action = tf.random.categorical(action_logits_t, 1)[0, 0]
            action_probs_t = tf.nn.softmax(action_logits_t)

            # print(f"Logits: {action_logits_t} | Action: {action} | Probs: {action_probs_t} | Critic Value: {critic_value}")

            critic_returns_hist = critic_returns_hist.write(t, critic_value)
            action_probs_hist = action_probs_hist.write(t, action_probs_t[0, action])

            state, reward, done = env.tf_step(action)
            state.set_shape(initial_state_shape)
            reward_hist = reward_hist.write(t, reward)
            if done:
                break

        reward_hist = reward_hist.stack()
        action_probs_hist = action_probs_hist.stack()
        critic_returns_hist = critic_returns_hist.stack()
        stats = {
                "steps"       : len(reward_hist.numpy()),
                "total_reward": float(tf.math.reduce_sum(reward_hist)),
        }
        return action_probs_hist, critic_returns_hist, reward_hist, stats

    # @tf.function
    def train_step(self, env: Environment, max_steps_per_episode: int) -> dict:
        """Runs a model training step."""
        if self.nn.optimizer is None:
            raise ValueError("Optimizer is None")

        with tf.GradientTape() as tape:
            # Run the model for one episode to collect training data
            action_probs, critic_returns, rewards, stats = self.run_episode(env,
                                                                            max_steps_per_episode,
                                                                            deterministic=False)

            # Calculate the expected returns
            actual_returns = self.get_expected_return(rewards)

            # Convert training data to appropriate TF tensor shapes
            action_probs, critic_returns, actual_returns = [
                    tf.expand_dims(x, 1) for x in [action_probs, critic_returns, actual_returns]]

            # Calculate the loss values to update our network
            loss = self.compute_loss(action_probs, critic_returns, actual_returns)

        # Compute the gradients from the loss
        grads = tape.gradient(loss, self.nn.model.trainable_variables)
        # Pygame stopped responding fix
        pygame.event.pump()
        # Apply the gradients to the model's parameters
        self.nn.optimizer.apply_gradients(zip(grads, self.nn.model.trainable_variables))
        return stats
