import logging
import os
import random

import numpy as np
import pygame
import tensorflow as tf
from matplotlib import pyplot as plt
import tensorflow_probability as tfp
from PongEnvironment import PongEnvironment
from Experiment import Experiment
from NeuralNet import NeuralNet
from BaseAgent import BaseAgent
# This is a hacky fix for tensorflow imports to work with intellisense
from rllib.utils import logging_setup

# Logging to stdout and file with logging class

log = logging_setup(file=__file__, name=__name__, level=logging.INFO)
os.environ['WANDB_SILENT'] = "true"


# Set seed for experiment reproducibility
# seed = 42
#
# tf.random.set_seed(seed)
# np.random.seed(seed)
# random.seed(seed)


class A2CAgent(BaseAgent):

    def __init__(self, nn: NeuralNet, gamma: float = 0.97):
        super().__init__(gamma=gamma)
        self.nn: NeuralNet = nn
        self.action_probs_hist = None
        self.critic_returns_hist = None
        self.state = None
        self.num_timesteps = self.nn.input_shape[1]

    def on_episode_start(self):
        self.action_probs_hist = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        self.critic_returns_hist = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        self.state = np.zeros(self.nn.input_shape)

    def get_action(self, t, state_t):
        if self.nn.model is None:
            raise ValueError("Model is None")
        # state = tf.reshape(state, self.nn.input_shape)
        state_t = np.expand_dims(state_t, axis=0)
        self.state = np.append(self.state, state_t, axis=1)
        self.state = self.state[:, -self.num_timesteps:, :]

        # print("State: ", self.state)
        # print("State shape: ", self.state.shape)
        action_logits_t, critic_value = self.nn.model(self.state)
        # print(f"Logits: {action_logits_t} | Critic Value: {critic_value}")

        action_logits_t = tf.reshape(action_logits_t, shape=(1, self.nn.num_actions))
        critic_value = tf.squeeze(critic_value)

        # action = tfp.distributions.Categorical(logits=action_logits_t[0]).sample()
        action = tf.random.categorical(action_logits_t, 1)[0, 0]
        action_probs_t = tf.nn.softmax(action_logits_t)

        # print(f"Logits: {action_logits_t} | Action: {action} | Probs: {action_probs_t} | Critic Value: {critic_value}")
        self.critic_returns_hist = self.critic_returns_hist.write(t, critic_value)
        self.action_probs_hist = self.action_probs_hist.write(t, action_probs_t[0, action])
        return action

    def on_episode_end(self):
        # self.nn.model.reset_states()
        self.action_probs_hist = self.action_probs_hist.stack()
        self.critic_returns_hist = self.critic_returns_hist.stack()

    def compute_loss(self,
                     action_probs: tf.Tensor,
                     critic_returns: tf.Tensor,
                     actual_returns: tf.Tensor) -> tf.Tensor:
        """Computes the combined Actor-Critic loss."""
        if self.nn.critic_loss is None:
            raise ValueError("Loss is None")
        # create multipliers for actor and critic losses
        actor_loss_multiplier = 0.5
        critic_loss_multiplier = 1.0

        advantage = tf.math.subtract(actual_returns, critic_returns)
        # print(f"ARs: {actual_vals.shape} | CRs: {critic_vals.shape} | Adv: {advantage.shape}")

        # action_log_probs = -1 * tf.math.log(action_probs)
        # actor_losses = actor_loss_multiplier * tf.math.multiply(action_log_probs, advantage)

        actor_losses = actor_loss_multiplier * self.nn.actor_loss(advantage, action_probs)

        critic_losses = critic_loss_multiplier * self.nn.critic_loss(critic_returns, actual_returns)
        critic_losses = tf.reshape(critic_losses, shape=(tf.shape(actor_losses)))

        total_losses = actor_losses + critic_losses
        total_loss_sum = tf.math.reduce_sum(total_losses)

        self.plot_tr(action_probs, actor_losses, actual_returns, advantage, critic_losses, critic_returns, total_losses)
        return total_loss_sum

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
        if self.env.draw:
            pygame.event.pump()
        # Apply the gradients to the model's parameters
        self.nn.optimizer.apply_gradients(zip(grads, self.nn.model.trainable_variables))

    def plot_tr(self, action_probs, actor_losses, actual_vals, advantage, critic_losses, critic_vals, total_losses):
        # network_state = self.network_state_hist.stack()
        # network_state = tf.squeeze(network_state)
        # network_state = tf.transpose(network_state)
        # plt.imshow(network_state)
        plt.clf()
        plt.plot(tf.squeeze(action_probs), label='Actor Probs', color='steelblue')
        plt.plot(tf.squeeze(actor_losses), label='Actor Loss', color="lightskyblue")
        plt.plot(tf.squeeze(critic_losses), label='Critic Loss', color='salmon')
        plt.plot(tf.squeeze(total_losses), label='Total Loss', color='black')
        plt.plot(tf.squeeze(critic_vals), label='Critic Val', color='red')
        plt.plot(tf.squeeze(actual_vals), label='Actual Val', color="green")
        plt.plot(tf.squeeze(advantage), label='Advantage', color='purple')

        plt.fill_between(tf.range(tf.shape(actual_vals)[0]), tf.squeeze(actual_vals), tf.squeeze(critic_vals),
                         where=tf.squeeze(actual_vals) > tf.squeeze(critic_vals), color='green', alpha=0.15)
        plt.fill_between(tf.range(tf.shape(actual_vals)[0]), tf.squeeze(actual_vals), tf.squeeze(critic_vals),
                         where=tf.squeeze(actual_vals) < tf.squeeze(critic_vals), color='red', alpha=0.15)

        plt.axhline(y=0, color='black', linestyle='--')

        plt.ylim(-5, 2)
        plt.grid(color='lightgray', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.legend(loc='lower left')
        plt.draw()
        plt.savefig('tr.pdf')


def main():
    env = PongEnvironment(draw=False)
    nn = NeuralNet("lstm", env.num_states, env.num_actions)
    agent = A2CAgent(nn)

    exp = Experiment(env, agent)
    exp.run_experiment()


if __name__ == "__main__":
    main()
