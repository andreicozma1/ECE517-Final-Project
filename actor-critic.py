import logging
import os
import random

import numpy as np
import pygame
import tensorflow as tf
from matplotlib import pyplot as plt
import tensorflow_probability as tfp
from Environment import Environment
from Experiment import Experiment
from NeuralNet import NeuralNet
from Agent import Agent
# This is a hacky fix for tensorflow imports to work with intellisense
from rllib.utils import logging_setup

# Logging to stdout and file with logging class

log = logging_setup(file=__file__, name=__name__, level=logging.INFO)
os.environ['WANDB_SILENT'] = "true"

# Set seed for experiment reproducibility
seed = 42

tf.random.set_seed(seed)
np.random.seed(seed)
random.seed(seed)


class A2CAgent(Agent):

    def __init__(self, nn: NeuralNet, gamma: float = 0.97):
        super().__init__(gamma)
        self.nn: NeuralNet = nn
        self.action_probs_hist = None
        self.critic_returns_hist = None

    def on_episode_start(self):
        self.action_probs_hist = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        self.critic_returns_hist = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

    def get_action(self, t, state):
        if self.nn.model is None:
            raise ValueError("Model is None")
        state = tf.reshape(state, self.nn.input_shape)
        # Predict action probabilities and estimated future rewards from environment state
        action_logits_t, critic_value = self.nn.model(state)
        action_logits_t = tf.reshape(action_logits_t, shape=(1, self.nn.num_actions))
        critic_value = tf.squeeze(critic_value)

        # with probability 0.2, use deterministic action
        # Sample next action from the action probability distribution
        # action = tfp.distributions.Categorical(logits=action_logits_t[0]).sample()
        action = tf.random.categorical(action_logits_t, 1)[0, 0]
        action_probs_t = tf.nn.softmax(action_logits_t)
        # epsilon decay based on t value

        # print(f"Logits: {action_logits_t} | Action: {action} | Probs: {action_probs_t} | Critic Value: {critic_value}")
        self.critic_returns_hist = self.critic_returns_hist.write(t, critic_value)
        self.action_probs_hist = self.action_probs_hist.write(t, action_probs_t[0, action])
        return action

    def on_episode_end(self):
        self.nn.model.reset_states()
        self.action_probs_hist = self.action_probs_hist.stack()
        self.critic_returns_hist = self.critic_returns_hist.stack()

    def on_update(self, rewards, tape):
        if self.nn.optimizer is None:
            raise ValueError("Optimizer is None")
        # Calculate the expected returns
        actual_returns = self.get_expected_return(rewards)

        # Convert training data to appropriate TF tensor shapes
        action_probs, critic_returns, actual_returns = [
                tf.expand_dims(x, 1) for x in [self.action_probs_hist, self.critic_returns_hist, actual_returns]]

        # Calculate the loss values to update our network
        loss = self.compute_loss(action_probs, critic_returns, actual_returns)

        # Compute the gradients from the loss
        grads = tape.gradient(loss, self.nn.model.trainable_variables)
        # Pygame stopped responding fix
        # pygame.event.pump()
        # Apply the gradients to the model's parameters
        self.nn.optimizer.apply_gradients(zip(grads, self.nn.model.trainable_variables))

    def compute_loss(self,
                     action_probs: tf.Tensor,
                     critic_vals: tf.Tensor,
                     actual_vals: tf.Tensor) -> tf.Tensor:
        """Computes the combined Actor-Critic loss."""
        if self.nn.loss is None:
            raise ValueError("Loss is None")

        advantage = tf.math.subtract(critic_vals, critic_vals)
        # advantage = tf.square(advantage)
        # print(f"ARs: {actual_vals.shape} | CRs: {critic_vals.shape} | Adv: {advantage.shape}")

        actor_losses = tf.math.multiply(-tf.math.log(action_probs), advantage) * 0.8
        actor_loss_sum = tf.math.reduce_sum(actor_losses)

        critic_losses = self.nn.loss(critic_vals, actual_vals)
        critic_losses = tf.reshape(critic_losses, shape=(tf.shape(actor_losses)))
        critic_loss_sum = tf.math.reduce_sum(critic_losses)

        total_losses = actor_losses + critic_losses

        total_loss_sum = tf.math.reduce_sum(total_losses)

        self.plot_tr(action_probs, actor_losses, actual_vals, advantage, critic_losses, critic_vals, total_losses)

        return total_loss_sum

    def plot_tr(self, action_probs, actor_losses, actual_vals, advantage, critic_losses, critic_vals, total_losses):
        plt.ion()
        plt.plot(tf.squeeze(action_probs), label='Actor Probs', color='steelblue')
        # action log probs
        # plt.plot(tf.squeeze(action_log_probs), label='action_log_probs', color='blue')
        plt.plot(tf.squeeze(actor_losses), label='Actor Loss', color="lightskyblue")
        plt.plot(tf.squeeze(critic_losses), label='Critic Loss', color='salmon')
        plt.plot(tf.squeeze(total_losses), label='Total Loss', color='black')
        plt.plot(tf.squeeze(critic_vals), label='Critic Val', color='red')
        plt.plot(tf.squeeze(actual_vals), label='Actual Val', color="green")
        plt.plot(tf.squeeze(advantage), label='Advantage', color='purple')
        # draw a line through 0
        plt.axhline(y=0, color='black', linestyle='--')
        # Highlight the difference between the actual returns and the critic returns
        plt.fill_between(tf.range(tf.shape(actual_vals)[0]), tf.squeeze(actual_vals), tf.squeeze(critic_vals),
                         where=tf.squeeze(actual_vals) > tf.squeeze(critic_vals), color='green', alpha=0.15)
        plt.fill_between(tf.range(tf.shape(actual_vals)[0]), tf.squeeze(actual_vals), tf.squeeze(critic_vals),
                         where=tf.squeeze(actual_vals) < tf.squeeze(critic_vals), color='red', alpha=0.15)
        plt.ylim(-5, 10)
        plt.grid(color='lightgray', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.legend(loc='lower left')
        plt.show()


def main():
    env = Environment(draw=True)
    nn = NeuralNet("dense", env.num_states, env.num_actions)
    agent = A2CAgent(nn)

    exp = Experiment(env, agent)
    exp.run_experiment()


if __name__ == "__main__":
    main()
