import abc
import logging
import pprint
import random
from typing import Tuple

import numpy as np
import tensorflow as tf

# This is a hacky fix for tensorflow imports to work with intellisense
from tensorflow import GradientTape

from rllib.BaseAgent import BaseAgent
from rllib.BaseEnvironment import BaseEnvironment
from rllib.Environments import LunarLander
from rllib.Experiment import Experiment
from rllib.Networks import TransformerNetwork
from rllib.nn.Losses import A2CLoss
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


class TestAgent(BaseAgent):

    def __init__(self, nn: TransformerNetwork,
                 gamma: float = 0.97,
                 entropy_loss_multiplier=0.1,
                 actor_loss_multiplier=0.2,
                 critic_loss_multiplier=1.0,
                 ):
        super().__init__(nn=nn, gamma=gamma)
        self.actor_loss_multiplier = actor_loss_multiplier
        self.critic_loss_multiplier = critic_loss_multiplier
        self.entropy_loss_multiplier = entropy_loss_multiplier

        logging.info(f"Args:\n{pprint.pformat(self.__dict__, width=30)}")

    def get_inputs(self, curr_timestep, curr_observation, buffer):
        states, actions, rewards = buffer.observation_buffer.stack(), buffer.action_buffer.stack(), buffer.reward_buffer.stack()
        trajectory_start_index = buffer.trajectory_start_index
        t = curr_timestep - trajectory_start_index

        # actions = tf.expand_dims(actions, axis=0)
        states = tf.transpose(states, perm=[1, 0, 2])
        states = tf.concat([states, tf.expand_dims(curr_observation, axis=0)], axis=1)

        # getting the index to last N states, actions
        idx_start = curr_timestep - self.nn.max_timesteps + 1 if t >= self.nn.max_timesteps else trajectory_start_index
        idx = tf.range(start=idx_start, limit=curr_timestep + 1, delta=1)
        idx = tf.clip_by_value(idx, clip_value_min=trajectory_start_index - 1, clip_value_max=curr_timestep)
        # positional array
        pos_s = tf.range(start=t - self.nn.max_timesteps + 1, limit=t + 1, delta=1)
        pos_s = tf.clip_by_value(pos_s, clip_value_min=-1, clip_value_max=t)

        states = tf.gather(states, idx, axis=1)
        actions = tf.one_hot(actions, depth=self.nn.num_actions)
        actions = tf.gather(actions, idx, axis=0)

        if t < self.nn.max_timesteps:
            states = tf.concat([tf.zeros(shape=(1, self.nn.max_timesteps - t - 1, self.nn.num_features)), states],
                               axis=1)
            actions = tf.concat([tf.zeros(shape=(self.nn.max_timesteps - t - 1, self.nn.num_actions)), actions], axis=0)

        positions = tf.reshape(pos_s, shape=self.nn.inp_p_shape)
        states = tf.reshape(states, shape=self.nn.inp_s_shape)
        actions = tf.reshape(actions, shape=self.nn.inp_a_shape)
        return [positions, states, actions]

    # def get_inputs(self, curr_timestep, state_hist, actions_hist, rewards_hist):
    #     # print("=" * 80)
    #     # print("curr_timestep", curr_timestep)
    #     states, actions, rewards = state_hist.stack(), actions_hist.stack(), rewards_hist.stack()
    #     # print("s", states)
    #     # print("a", actions)
    #     # print("r", rewards)
    #
    #     pos_s = tf.range(start=curr_timestep - self.nn.max_timesteps + 1, limit=curr_timestep + 1, delta=1)
    #     pos_s = tf.clip_by_value(pos_s, clip_value_min=-1, clip_value_max=curr_timestep)
    #
    #     states = tf.gather(states, pos_s, axis=0)
    #
    #     actions = tf.one_hot(actions, depth=self.nn.num_actions)
    #     actions = tf.gather(actions, pos_s, axis=0)
    #
    #     positions = tf.reshape(pos_s, shape=self.nn.inp_p_shape)
    #     states = tf.reshape(states, shape=self.nn.inp_s_shape)
    #     actions = tf.reshape(actions, shape=self.nn.inp_a_shape)
    #     # print("positions", positions)
    #     # print("states", states)
    #     # print("actions", actions)
    #     # print("=" * 80)
    #
    #     return [positions, states, actions]

    def get_action(self, model_outputs, deterministic):
        return self.get_action_1(model_outputs, deterministic)

    def compute_loss(self, hist_sar, hist_model_out: list):
        return self.loss(hist_sar, hist_model_out)

    # def compute_loss(self, hist_sar, hist_model_out: list):
    #     hist_states, hist_actions, hist_rewards = hist_sar
    #     action_probs, critic_returns = hist_model_out
    #     action_probs = tf.stack(action_probs, axis=1)
    #     critic_returns = tf.stack(critic_returns, axis=1)
    #
    #     action_probs = tf.nn.softmax(action_probs, axis=-1)
    #     entropy_loss = self.get_entropy_loss(action_probs)
    #     entropy_loss = self.normalize(entropy_loss)
    #     entropy_loss = tf.math.multiply(entropy_loss, self.entropy_loss_multiplier)
    #     action_probs = tf.reduce_max(action_probs, axis=-1)
    #
    #     actual_returns = self.expected_return(hist_rewards)
    #     actual_returns = self.normalize(actual_returns)
    #     critic_returns = self.normalize(critic_returns)
    #
    #     # action_probs_max = tf.reshape(action_probs_max, shape=(-1, 1))
    #     critic_returns = tf.reshape(critic_returns, shape=(-1, 1))
    #     actual_returns = tf.reshape(actual_returns, shape=(-1, 1))
    #
    #     advantage = actual_returns - critic_returns
    #
    #     self.plot_ret(actual_returns, advantage, critic_returns)
    #
    #     actor_losses = self.actor_loss(action_probs, advantage)
    #     actor_losses = self.normalize(actor_losses)
    #     actor_losses = tf.math.multiply(actor_losses, self.actor_loss_multiplier)
    #
    #     # entropy_loss = self.get_entropy_loss(action_probs_max)
    #     # entropy_loss = self.standardize(entropy_loss)
    #     # entropy_loss = self.normalize(entropy_loss)
    #     # entropy_loss = tf.math.multiply(entropy_loss, self.entropy_loss_multiplier)
    #
    #     critic_losses = self.critic_loss(critic_returns, actual_returns)
    #     critic_losses = self.normalize(critic_losses)
    #     critic_losses = tf.math.multiply(critic_losses, self.critic_loss_multiplier)
    #
    #     total_losses = actor_losses + critic_losses
    #     # total_losses = self.normalize(total_losses)
    #     # total_losses = tf.clip_by_value(total_losses, clip_value_min=-1, clip_value_max=1)
    #
    #     self.plot_loss(actor_losses, critic_losses, total_losses, entropy_loss=entropy_loss)
    #
    #     return tf.reduce_sum(total_losses)

    # def plot_ret(self, actual_returns, advantage, critic_returns):
    #     plot_returns = {
    #             "plot"        : [
    #                     {
    #                             "args" : [tf.squeeze(critic_returns)],
    #                             "label": "Critic Val",
    #                             "color": "red"
    #                     },
    #                     {
    #                             "args" : [tf.squeeze(actual_returns)],
    #                             "label": "Actual Val",
    #                             "color": "green"
    #                     },
    #                     {
    #                             "args" : [tf.squeeze(advantage)],
    #                             "label": "Advantage",
    #                             "color": "purple"
    #                     }
    #             ],
    #             "fill_between": [
    #                     {
    #                             "args" : [tf.range(tf.shape(actual_returns)[0]), tf.squeeze(actual_returns),
    #                                       tf.squeeze(critic_returns)],
    #                             "where": tf.squeeze(actual_returns) > tf.squeeze(critic_returns),
    #                             "color": "green",
    #                             "alpha": 0.15
    #                     },
    #                     {
    #                             "args" : [tf.range(tf.shape(actual_returns)[0]), tf.squeeze(actual_returns),
    #                                       tf.squeeze(critic_returns)],
    #                             "where": tf.squeeze(actual_returns) < tf.squeeze(critic_returns),
    #                             "color": "red",
    #                             "alpha": 0.15
    #                     }
    #             ],
    #             "axhline"     : [
    #                     {
    #                             "y"        : 0,
    #                             "color"    : "black",
    #                             "linestyle": "--"
    #                     }
    #             ],
    #             "suptitle"    : f"A2C Returns ({self.env.name}): "
    #                             f"{self.nn.name} - {self.nn.inp_s_shape}" +
    #                             f" + ({self.env.state_scaler.__class__.__name__})"
    #             if self.env.state_scaler_enable is True else "",
    #     }
    #     PlotHelper.plot_from_dict(plot_returns, savefig="plots/a2c_returns.pdf")
    #
    # def plot_loss(self, actor_losses, critic_losses, total_losses, entropy_loss=None):
    #     plot_losses = {
    #             "plot"   : [
    #                     {
    #                             "args" : [tf.squeeze(actor_losses)],
    #                             "label": "Actor Loss",
    #                             "color": "lightskyblue"
    #                     },
    #                     {
    #                             "args" : [tf.squeeze(critic_losses)],
    #                             "label": "Critic Loss",
    #                             "color": "salmon"
    #                     },
    #                     {
    #                             "args" : [tf.squeeze(total_losses)],
    #                             "label": "Total Loss",
    #                             "color": "black"
    #                     }
    #
    #             ],
    #             "axhline": [
    #                     {
    #                             "y"        : 0,
    #                             "color"    : "black",
    #                             "linestyle": "--"
    #                     }
    #             ],
    #             "title"  : f"A2C Losses ({self.env.name}): {self.nn.name} + "
    #                        f"{self.critic_loss.__class__.__name__} + "
    #                        f"{self.nn.optimizer.__class__.__name__} - "
    #                        f"LR: {self.nn.learning_rate}",
    #     }
    #     if entropy_loss is not None:
    #         plot_losses["plot"].append({
    #                 "args" : [tf.squeeze(entropy_loss)],
    #                 "label": "Entropy Loss",
    #                 "color": "darkorange"
    #         })
    #
    #     PlotHelper.plot_from_dict(plot_losses, savefig="plots/a2c_losses.pdf")


class BaseAgent(keras.Model):

    def __init__(self, nn: TransformerNetwork, gamma: float):
        self.env = None
        self.nn: TransformerNetwork = nn
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

    def get_entropy_loss(self, action_probs: tf.Tensor) -> tf.Tensor:
        entropy_loss = tf.math.multiply(action_probs, tf.math.log(action_probs))
        entropy_loss = -1 * tf.reduce_sum(entropy_loss, axis=-1)
        return entropy_loss

    def sample_action(self, model_outputs, deterministic):
        action_logits_t, critic_value = model_outputs
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

    def gen_episode(self, env: BaseEnvironment,
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

    def train_step(self, data,
                   max_steps_per_episode: int) -> Tuple[Tuple[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor, tf.Tensor]:
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        """
        The logic for one training step.

        This method can be overridden to support custom training logic. 
        For concrete examples of how to override this method see Customizing what happens in fit. T
        his method is called by Model.make_train_function.

        This method should contain the mathematical logic for one step of training. 
        This typically includes the forward pass, loss calculation, backpropagation, and metric updates.

        Configuration details for how this logic is run (e.g. tf.function and tf.distribute.Strategy settings), should be left to Model.make_train_function, which can also be overridden.
        """

        """
        (s, a, r), (actor_logits, critic_vals) = generate_grajectory
        """

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            # hist_sar, hist_model_out
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

        # self.env: BaseEnvironment = env
        # with tf.GradientTape() as tape:
        #     hist_sar, hist_out = self.run_episode(env, max_steps_per_episode)
        #     loss = self.compute_loss(hist_sar, hist_out)
        #
        # self.apply_grads(loss, tape)
        #
        # return hist_sar, hist_out, loss

    def apply_grads(self, loss, tape: GradientTape):
        grads = tape.gradient(loss, self.nn.model.trainable_variables)
        # Apply the gradients to the model's parameters
        self.nn.optimizer.apply_gradients(zip(grads, self.nn.model.trainable_variables))


def main():
    # env = PongEnvironment(draw=True, draw_speed=None, state_scaler_enable=True)
    env = LunarLander(draw=True, draw_speed=None, state_scaler_enable=True)

    nn = TransformerNetwork(env.num_states, env.num_actions,
                            max_timesteps=10, learning_rate=0.0001)
    nn.model.summary()

    loss = A2CLoss()
    optimizer = nn.optimizer

    nn.model.compile(optimizer=optimizer, loss=loss)


    # (sar, outs) = generate trajectory

    # for i in range(epichs):
          model.fit((sar, outs))

    """
    fit(
    x=None,
    y=None,
    """

    """
    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data
    """

    # loss = A2CLoss()

    # exp = Experiment(env, agent, use_wandb=False)
    # exp.run_experiment()


if __name__ == "__main__":
    main()
