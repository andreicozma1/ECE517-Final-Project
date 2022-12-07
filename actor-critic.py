import logging
import pprint
import random
from typing import Tuple

import numpy as np
import tensorflow as tf

# This is a hacky fix for tensorflow imports to work with intellisense
from rllib.BaseAgent import BaseAgent
from rllib.CustomLayers import ActorLoss
from rllib.Environments import LunarLander
from rllib.Experiment import Experiment
from rllib.Network import Network
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

keras = tf.keras


class A2CAgent(BaseAgent):

    def __init__(self, nn: Network,
                 gamma: float = 0.97,
                 actor_loss_multiplier: float = 0.5,
                 critic_loss_multiplier: float = 1.0,
                 entropy_loss_multiplier: float = 0.01,
                 ):
        super().__init__(nn=nn, gamma=gamma)
        self.actor_loss_multiplier = actor_loss_multiplier
        self.critic_loss_multiplier = critic_loss_multiplier
        self.entropy_loss_multiplier = entropy_loss_multiplier
        self.critic_loss = keras.losses.Huber(reduction=tf.keras.losses.Reduction.NONE)
        self.actor_loss = ActorLoss(reduction=tf.keras.losses.Reduction.NONE)
        logging.info(f"Args:\n{pprint.pformat(self.__dict__, width=30)}")

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

    def compute_loss(self, hist_sar, hist_model_out: list):
        hist_states, hist_actions, hist_rewards = hist_sar
        action_probs, critic_returns = hist_model_out
        action_probs = tf.stack(action_probs, axis=1)
        critic_returns = tf.stack(critic_returns, axis=1)
        # transform the actor logits to action probabilities
        action_probs = tf.nn.softmax(action_probs, axis=-1)

        actual_returns = self.expected_return(hist_rewards)

        actual_returns = self.standardize(actual_returns)
        critic_returns = self.standardize(critic_returns)

        # action_probs = tf.reshape(action_probs, shape=(-1, 1))
        action_probs = tf.squeeze(action_probs)
        critic_returns = tf.reshape(critic_returns, shape=(-1, 1))
        actual_returns = tf.reshape(actual_returns, shape=(-1, 1))

        advantage = actual_returns - critic_returns

        print(action_probs.shape)
        print(critic_returns.shape)
        print(actual_returns.shape)

        print(advantage.shape)

        self.plot_ret(actual_returns, advantage, critic_returns)

        actor_losses = self.actor_loss(action_probs, advantage)
        actor_losses = self.normalize(actor_losses)
        actor_losses = tf.math.multiply(actor_losses, self.actor_loss_multiplier)

        entropy_loss = self.get_entropy_loss(action_probs)
        entropy_loss = self.normalize(entropy_loss)
        entropy_loss = tf.math.multiply(entropy_loss, self.entropy_loss_multiplier)

        critic_losses = self.critic_loss(critic_returns, actual_returns)
        critic_losses = self.normalize(critic_losses)
        critic_losses = tf.math.multiply(critic_losses, self.critic_loss_multiplier)

        total_losses = tf.math.add(actor_losses, critic_losses, entropy_loss)

        self.plot_loss(actor_losses, critic_losses, total_losses, entropy_loss=entropy_loss)

        return tf.reduce_sum(total_losses)

    def get_entropy_loss(self, action_probs: tf.Tensor) -> tf.Tensor:
        entropy_loss = tf.math.multiply(action_probs, tf.math.log(action_probs))
        entropy_loss = -1 * tf.reduce_sum(entropy_loss, axis=-1)
        return entropy_loss

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
                                f"{self.nn.name} - {self.nn.inp_s_shape}" +
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

    nn = Network("transformer",
                 env.num_states, env.num_actions,
                 max_timesteps=5, learning_rate=0.000001)
    agent = A2CAgent(nn)

    exp = Experiment(env, agent, use_wandb=False)

    exp.run_experiment()


if __name__ == "__main__":
    main()
