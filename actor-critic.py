import logging
import os

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
os.environ['WANDB_SILENT'] = "true"


# Set seed for experiment reproducibility
# seed = 42
#
# tf.random.set_seed(seed)
# np.random.seed(seed)
# random.seed(seed)


class A2CAgent(BaseAgent):

    def __init__(self, nn: NeuralNet, gamma: float):
        super().__init__(gamma=gamma)
        self.nn: NeuralNet = nn
        self.action_probs_hist = None
        self.critic_returns_hist = None
        self.state = None

    def on_episode_start(self):
        self.action_probs_hist = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        self.critic_returns_hist = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        # tensorflow collection of self.nn.input_shape[1] zeros
        self.state = tf.TensorArray(dtype=tf.float32, size=self.nn.max_timesteps, dynamic_size=True)

    def add_to_history(self, t, state_t):
        self.state = self.state.write(self.nn.max_timesteps + t, tf.reshape(state_t, self.nn.num_features))

    def get_action(self, t, state_t):
        if self.nn.model is None:
            raise ValueError("Model is None")
        # logging.debug("=" * 40 + f" @{t} " + "=" * 40)
        self.add_to_history(t, state_t)

        state = self.state.stack()
        state = state[-self.nn.max_timesteps:, :]
        state = tf.reshape(state, self.nn.input_shape)
        # logging.info(f"state: {state} \t shape: {state.shape}")

        action_logits_t, critic_value = self.nn.model(state)
        # logging.debug(f"action_logits_t: {action_logits_t} | critic_value: {critic_value}")

        action_logits_t = tf.reshape(action_logits_t, shape=(1, self.nn.num_actions))
        critic_value = tf.squeeze(critic_value)

        # action = tfp.distributions.Categorical(logits=action_logits_t[0]).sample()
        action = tf.random.categorical(action_logits_t, 1)[0, 0]
        action_probs_t = tf.nn.softmax(action_logits_t)

        # logging.debug(f"action: {action} | action_probs_t: {action_probs_t}")

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

        # If action is better than average, the advantage function is positive,
        # if worse, it is negative.
        advantage = tf.math.subtract(actual_returns, critic_returns)
        # print(f"ARs: {actual_vals.shape} | CRs: {critic_vals.shape} | Adv: {advantage.shape}")

        # action_log_probs = -1 * tf.math.log(action_probs)
        # actor_losses = actor_loss_multiplier * tf.math.multiply(action_log_probs, advantage)

        actor_losses = actor_loss_multiplier * self.nn.actor_loss(advantage, action_probs)

        critic_losses = critic_loss_multiplier * self.nn.critic_loss(critic_returns, actual_returns)
        critic_losses = tf.reshape(critic_losses, shape=(tf.shape(actor_losses)))

        total_losses = actor_losses + critic_losses
        total_loss_sum = tf.math.reduce_sum(total_losses)

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
        }
        plot_losses = {
                "plot"   : [
                        {
                                "args" : [tf.squeeze(action_probs)],
                                "label": "Actor Probs",
                                "color": "steelblue"
                        },
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
                                "args" : [tf.squeeze(total_losses)],
                                "label": "Total Loss",
                                "color": "black"
                        },

                ],
                "axhline": [
                        {
                                "y"        : 0,
                                "color"    : "black",
                                "linestyle": "--"
                        }
                ],
        }

        PlotHelper.plot_from_dict(plot_losses, savefig="plots/a2c_losses.pdf")
        PlotHelper.plot_from_dict(plot_returns, savefig="plots/a2c_returns.pdf")

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
        # # Pygame stopped responding fix
        # if self.env.draw:
        #     pygame.event.pump()
        # Apply the gradients to the model's parameters
        self.nn.optimizer.apply_gradients(zip(grads, self.nn.model.trainable_variables))


def main():
    # env = PongEnvironment(draw=True, draw_speed=1)
    env = PongEnvironment(draw=True, draw_speed=None)
    # env = LunarLander(draw=True, draw_speed=None)
    nn = NeuralNet("lstm", env.num_states, env.num_actions, max_timesteps=5, learning_rate=0.0001)
    agent = A2CAgent(nn, gamma=0.99)

    exp = Experiment(env, agent)
    exp.run_experiment()


if __name__ == "__main__":
    main()
