import numpy as np
import tensorflow as tf

from rllib.BaseEnvironment import BaseEnvironment
from rllib.BaseAgent import BaseAgent
from rllib.Networks import TransformerNetwork
from rllib.PlotHelper import PlotHelper
from rllib.Buffer import Buffer

keras = tf.keras
eps = np.finfo(np.float32).eps.item()


class PPOAgent(BaseAgent):

    def __init__(self, nn: TransformerNetwork,
                 epsilon: float = 0.2,
                 gamma: float = 0.97,
                 num_rollout_episodes: int = 100,
                 steps_per_epoch: int = 4000,
                 train_iterations: int = 80,
                 clip_ratio: float = 0.2,
                 ):
        super().__init__(nn=nn, gamma=gamma)
        self.num_rollout_episodes = num_rollout_episodes
        self.epsilon = epsilon
        self.clip_ratio = clip_ratio
        self.steps_per_epoch = steps_per_epoch
        self.train_iterations = train_iterations

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

    def get_action(self, model_outputs, deterministic):
        return self.get_action_1(model_outputs, deterministic)

    def rollout(self, env: BaseEnvironment,
                deterministic: bool = False):
        self.env: BaseEnvironment = env

        observation, episode_return, episode_length = env.tf_reset(), 0, 0
        observation_shape = observation.shape

        sum_return = 0
        sum_length = 0
        num_episodes = 0
        buffer = Buffer(observation_shape, self.nn.max_timesteps, gamma=0.99, lam=0.95)
        hist_model_o = []

        for _ in tf.range(self.num_rollout_episodes):
            for t in tf.range(self.nn.max_timesteps):

                # Get the logits, action, and take one step in the environment
                model_inputs = self.get_inputs(t,
                                               observation,
                                               buffer)
                model_outputs = self.nn.model(model_inputs)
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
                buffer.store(observation, action, reward, value_t, log_prob_t)
                hist_model_o.append(model_outputs)

                # Update the observation
                observation = observation_new

                # Finish trajectory if reached to a terminal state
                if tf.cast(done, tf.bool):
                    # last_value = 0 if tf.cast(done, tf.bool) else value_t
                    buffer.finish_trajectory(0)
                    sum_return += episode_return
                    sum_length += episode_length
                    num_episodes += 1
                    observation, episode_return, episode_length = env.reset(), 0, 0

        hist_model_o = list(zip(*hist_model_o))
        return buffer, hist_model_o

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
        return_buffer = self.normalize(return_buffer)
        value_buffer = self.normalize(value_buffer)
        # self.plot_ret(return_buffer, advantage_buffer, value_buffer)

        print("=" * 80)
        print("Training on generated data")
        # losses = []
        for e in range(self.train_iterations):
            print(f"Epoch {e} / {self.train_iterations}")
            loss = self.update(buffer, hist_model_o)
            # losses.append(loss)
        # loss = reduce_sum(losses)
        # self.apply_grads(loss, tape)
        del buffer
        return rollout_data, hist_model_o, loss

    # @tf.function
    def update(self, buffer, hist_model_o):
        (
                observation_buffer,
                action_buffer,
                advantage_buffer,
                return_buffer,
                log_prob_buffer,
                value_buffer,
                reward_buffer
        ) = buffer.get_batches()
        episode_indices = buffer.episode_indices
        new_log_prob_buffer = tf.TensorArray(dtype=tf.float32, size=0,
                                             dynamic_size=True,
                                             element_shape=1)
        # self.plot_ret(return_buffer, advantage_buffer, value_buffer)
        with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
            logits, value_t = self.nn.model([observation_buffer, action_buffer, return_buffer])
            # (1, 30, 8)
            # reduce_sum(..., axis=1) = (1, 8)
            new_log_prob = self.get_log_probs(tf.squeeze(logits, axis=1), action_buffer)
            advantage = return_buffer - value_t

            # for i in range(len(episode_indices) - 1):
            #     for t in range(episode_indices[i], episode_indices[i + 1]):
            #         buffer.trajectory_start_index = i
            #         model_inputs = self.get_inputs(t,
            #                                        observation_buffer[t],
            #                                        buffer)
            #         model_outputs = self.nn.model(model_inputs)
            #         logits, value_t = model_outputs
            #         value_t = tf.squeeze(value_t)
            #         logits = logits[0]
            #         log_prob_t = self.get_log_probs(logits, tf.expand_dims(action_buffer[t], axis=0))
            #         new_log_prob_buffer = new_log_prob_buffer.write(t, log_prob_t)
            # new_log_prob_buffer = new_log_prob_buffer.stack()

            # TODO: Figure out if needing to use Pi(a|s)/(Pi_old(a|s))
            # np.finfo(np.float32).eps.item()
            # ratio = tf.exp(
            #         new_log_prob
            #         - log_prob_buffer
            # )
            ratio = tf.divide(new_log_prob, log_prob_buffer + tf.Variable(1e-7, dtype=tf.float32))
            min_advantage = tf.where(
                    advantage_buffer > 0,
                    (1 + self.clip_ratio) * advantage,
                    (1 - self.clip_ratio) * advantage,
            )
            # min_advantage = tf.clip_by_value(advantage_buffer, clip_value_min=1-self.clip_ratio, clip_value_max=1 + self.clip_ratio)
            minim = tf.minimum(ratio * advantage, min_advantage)
            policy_loss = -tf.reduce_mean(minim)

            value_loss = tf.reduce_mean((return_buffer - value_t) ** 2)
            total_loss = tf.math.add(policy_loss, value_loss)
            loss = tf.reduce_sum(total_loss)
            # loss = [policy_loss, value_loss]

        # loss = [5, 2, 3, 4)
        self.apply_grads(loss, tape)

        return loss

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
                "suptitle"    : f"PPO Returns ({self.env.name}): "
                                f"{self.nn.name} - {self.nn.inp_s_shape}" +
                                f" + ({self.env.state_scaler.__class__.__name__})"
                if self.env.state_scaler_enable is True else "",
        }
        PlotHelper.plot_from_dict(plot_returns, savefig="plots/ppo_returns.pdf")

    def plot_loss(self, clip_losses, value_losses, total_losses, entropy_loss=None):
        # pass
        plot_losses = {
                "plot"   : [
                        {
                                "args" : [tf.squeeze(clip_losses)],
                                "label": "Clip Loss",
                                "color": "lightskyblue"
                        },
                        {
                                "args" : [tf.squeeze(value_losses)],
                                "label": "Value Loss",
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
                "title"  : f"PPO Losses ({self.env.name}):"
                # f" {self.nn.name} + "
                # f"{self.nn.critic_loss.__class__.__name__} + "
                # f"{self.nn.optimizer.__class__.__name__} - "
                # f"LR: {self.nn.learning_rate}",
        }
        if entropy_loss is not None:
            plot_losses["plot"].append({
                    "args" : [tf.squeeze(entropy_loss)],
                    "label": "Entropy Loss",
                    "color": "darkorange"
            })

        PlotHelper.plot_from_dict(plot_losses, savefig="plots/ppo_losses.pdf")
