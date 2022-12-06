import logging
import os
import pprint
import random
from typing import Tuple

import numpy as np
import pygame
import tensorflow as tf
from PongEnvironment import LunarLander, PongEnvironment
from Experiment import Experiment
from NeuralNet import NeuralNet
from BaseAgent import BaseAgent
# This is a hacky fix for tensorflow imports to work with intellisense
from rllib.PlotHelper import PlotHelper
from rllib.utils import logging_setup

# Logging to stdout and file with logging class

log = logging_setup(file=__file__, name=__name__, level=logging.INFO)
# os.environ['WANDB_SILENT'] = "true"

# Set seed for experiment reproducibility
seed = 42

tf.random.set_seed(seed)
np.random.seed(seed)
random.seed(seed)


class A2CAgent(BaseAgent):

    def __init__(self, nn: NeuralNet,
                 gamma: float = 0.97,
                 actor_loss_multiplier: float = 1.0,
                 critic_loss_multiplier: float = 0.5,
                 entropy_loss_multiplier: float = 0.05,
                 ):
        super().__init__(nn=nn, gamma=gamma)
        self.actor_loss_multiplier = actor_loss_multiplier
        self.critic_loss_multiplier = critic_loss_multiplier
        self.entropy_loss_multiplier = entropy_loss_multiplier
        logging.info(f"Args:\n{pprint.pformat(self.__dict__, width=30)}")

    def normalize(self, values):
        c_min = tf.reduce_min(values)
        c_max = tf.reduce_max(values)
        c_range = c_max - c_min
        values = (values - c_min) / c_range
        return values

    def standardize(self, values):
        mean = tf.reduce_mean(values)
        std = tf.math.reduce_std(values)
        values = (values - mean) / std
        return values

    def compute_error(self, rewards, extras):
        action_probs, critic_returns = extras
        actual_returns = self.get_expected_return(rewards)

        actual_returns = self.standardize(actual_returns)
        critic_returns = self.standardize(critic_returns)

        action_probs = tf.reshape(action_probs, shape=(-1, 1))
        critic_returns = tf.reshape(critic_returns, shape=(-1, 1))
        actual_returns = tf.reshape(actual_returns, shape=(-1, 1))

        loss, advantage = self.compute_loss(action_probs, critic_returns, actual_returns)

        self.plot_ret(actual_returns, advantage, critic_returns)

        return loss

    def get_entropy_loss(self, action_probs: tf.Tensor) -> tf.Tensor:
        entropy_loss = tf.math.multiply(action_probs, tf.math.log(action_probs))
        entropy_loss = -1 * tf.reduce_sum(entropy_loss, axis=-1)
        return entropy_loss

    def compute_loss(self,
                     action_probs: tf.Tensor,
                     critic_returns: tf.Tensor,
                     actual_returns: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Computes the combined Actor-Critic loss."""
        if self.nn.critic_loss is None:
            raise ValueError("Loss is None")
        # If action is better than average, the advantage function is positive,
        # if worse, it is negative.
        advantage = actual_returns - critic_returns

        actor_losses = self.nn.actor_loss(action_probs, advantage)
        # actor_losses = self.standardize(actor_losses)
        actor_losses = tf.math.multiply(actor_losses, self.actor_loss_multiplier)

        entropy_loss = self.get_entropy_loss(action_probs)
        # entropy_loss = self.standardize(entropy_loss)
        entropy_loss = tf.math.multiply(entropy_loss, self.entropy_loss_multiplier)

        critic_losses = self.nn.critic_loss(critic_returns, actual_returns)
        # critic_losses = self.standardize(critic_losses)
        critic_losses = tf.math.multiply(critic_losses, self.critic_loss_multiplier)

        total_losses = tf.math.add(actor_losses, critic_losses, entropy_loss)

        loss = tf.reduce_sum(total_losses)

        self.plot_loss(actor_losses, critic_losses, total_losses, entropy_loss=entropy_loss)

        return loss, advantage

    def on_update(self, loss, tape):
        # Compute the gradients from the loss
        grads = tape.gradient(loss, self.nn.model.trainable_variables)
        # Apply the gradients to the model's parameters
        self.nn.optimizer.apply_gradients(zip(grads, self.nn.model.trainable_variables))

    def plot_ret(self, actual_returns, advantage, critic_returns):
        plot_returns = {
                "plot"        : [
                        {
                                "args" : [tf.squeeze(critic_returns)],
                                "label": "Critic Val",
                                "color": "red"
                        },
                        {
                                "args" : [tf.squeeze(actual_returns)],
                                "label": "Actual Val",
                                "color": "green"
                        },
                        {
                                "args" : [tf.squeeze(advantage)],
                                "label": "Advantage",
                                "color": "purple"
                        }
                ],
                "fill_between": [
                        {
                                "args" : [tf.range(tf.shape(actual_returns)[0]), tf.squeeze(actual_returns),
                                          tf.squeeze(critic_returns)],
                                "where": tf.squeeze(actual_returns) > tf.squeeze(critic_returns),
                                "color": "green",
                                "alpha": 0.15
                        },
                        {
                                "args" : [tf.range(tf.shape(actual_returns)[0]), tf.squeeze(actual_returns),
                                          tf.squeeze(critic_returns)],
                                "where": tf.squeeze(actual_returns) < tf.squeeze(critic_returns),
                                "color": "red",
                                "alpha": 0.15
                        }
                ],
                "axhline"     : [
                        {
                                "y"        : 0,
                                "color"    : "black",
                                "linestyle": "--"
                        }
                ],
                "suptitle"    : f"A2C Returns ({self.env.name}): "
                                f"{self.nn.name} - {self.nn.input_shape}" +
                                f" + ({self.env.state_scaler.__class__.__name__})"
                if self.env.state_scaler_enable is True else "",
        }
        PlotHelper.plot_from_dict(plot_returns, savefig="plots/a2c_returns.pdf")

    def plot_loss(self, actor_losses, critic_losses, total_losses, entropy_loss=None):
        plot_losses = {
                "plot"   : [
                        {
                                "args" : [tf.squeeze(actor_losses)],
                                "label": "Actor Loss",
                                "color": "lightskyblue"
                        },
                        {
                                "args" : [tf.squeeze(critic_losses)],
                                "label": "Critic Loss",
                                "color": "salmon"
                        },
                        {
                                "args" : [total_losses],
                                "label": "Total Loss",
                                "color": "black"
                        }

                ],
                "axhline": [
                        {
                                "y"        : 0,
                                "color"    : "black",
                                "linestyle": "--"
                        }
                ],
                "title"  : f"A2C Losses ({self.env.name}): {self.nn.name} + "
                           f"{self.nn.critic_loss.__class__.__name__} + "
                           f"{self.nn.optimizer.__class__.__name__} - "
                           f"LR: {self.nn.learning_rate}",
        }
        if entropy_loss is not None:
            plot_losses["plot"].append({
                    "args" : [tf.squeeze(entropy_loss)],
                    "label": "Entropy Loss",
                    "color": "darkorange"
            })

        PlotHelper.plot_from_dict(plot_losses, savefig="plots/a2c_losses.pdf")


def main():
    # env = PongEnvironment(draw=True, draw_speed=None, state_scaler_enable=True)
    env = LunarLander(draw=True, draw_speed=None, state_scaler_enable=True)

    nn = NeuralNet("transformer",
                   env.num_states, env.num_actions,
                   max_timesteps=50, learning_rate=0.00001)
    agent = A2CAgent(nn)

    exp = Experiment(env, agent)

    exp.run_experiment()
    exp.run_test()


if __name__ == "__main__":
    main()
