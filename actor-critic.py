import hashlib
import json
import logging
import os
import pprint

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import wandb
from Models import Models
from rlenv.pongclass import pongGame
# This is a hacky fix for tensorflow imports to work with intellisense
from rllib.utils import logging_setup

# Logging to stdout and file with logging class

log = logging_setup(file=__file__, name=__name__, level=logging.INFO)
os.environ['WANDB_SILENT'] = "true"


class Experiment:
    def __init__(self,
                 gamma=0.97,
                 learning_rate=0.0001,
                 state_timesteps=30,
                 state_num_prev_actions=1,
                 env_draw=False,
                 env_draw_speed=None):
        """
        Constructs an experiment.
        Args:
            gamma (float, optional): RL agent discount factor. Defaults to 0.95.
            state_timesteps (int, optional): The number of previous time steps to include in the state. Defaults to 30.
            state_num_prev_actions (int, optional): number of previous actions to append to the environment state. Defaults to 1.
            env_draw (bool, optional): whether to draw the environment. Defaults to False.
        """
        log.info("Experiment:")
        # Agent parameters
        self._model, self._optimizer, self._critic_loss = None, None, None
        self.a_gamma, self.a_learning_rate = gamma, learning_rate
        # Scaler for returns
        self._a_returns_scaler = MinMaxScaler(feature_range=(-1, 1))

        # Environment parameters
        self.e_state_timesteps = state_timesteps
        self.e_state_num_prev_actions = state_num_prev_actions
        self._env = pongGame(200, 200, draw=env_draw, draw_speed=env_draw_speed)

        self.e_num_states = self._env.getState().shape[0]
        self.e_num_actions = 3

        # Other parameters for the experiment
        self._e_state_hist = None
        self._e_state_scaler = MinMaxScaler(feature_range=(-1, 1))
        self.e_state_shape = str(self.env_reset().shape)

        # Wandb
        self._config = {}
        self.update_config()

    def update_config(self):
        self._config.update({k: v for k, v in self.__dict__.items() if not k.startswith("_")})

    def init_model(self, model_func_name: str, **kwargs):
        """
        Initializes the model used for the RL agent.
        Args:
            model_func_name (str): corresponding function name in Models class (as string)
            **kwargs: Keyword arguments passed to the model function
        """
        models = Models(self.e_num_states, self.e_num_actions, learning_rate=self.a_learning_rate)
        init_model = getattr(models, model_func_name)
        self._model, self._optimizer, self._critic_loss = init_model(**kwargs)
        model_config = self._model.get_config()
        model_config_hash = hashlib.md5(json.dumps(model_config).encode('utf-8')).hexdigest()
        self._model.summary()
        self._config.update({
                "model": {
                        "func_name"  : model_func_name,
                        "kwargs"     : kwargs,
                        "num_layers" : len(self._model.layers),
                        "num_params" : self._model.count_params(),
                        "config"     : model_config,
                        "config_hash": model_config_hash
                }
        })

    def env_reset(self):
        # reset the environment
        # self._env.reset_ball()
        self._env.reset()
        # clear the state history
        self._e_state_hist = np.zeros(shape=(1, self.e_state_timesteps, self.e_num_states), dtype=np.float32)
        return self.env_get_state()

    def env_get_state(self):
        state = self._env.getState().reshape(1, -1)
        self._e_state_scaler.partial_fit(state)
        state = self._e_state_scaler.transform(state)
        state = state[np.newaxis, :]
        return tf.convert_to_tensor(state, dtype=tf.float32)

    def run_experiment(self,
                       training: bool,
                       max_episodes: int = 10000,
                       max_steps: int = 10000,
                       test_every: int = None,
                       test_for: int = 10):
        """
        Runs experiments
        Args:
            training (bool, optional): Whether to train the agent. Defaults to True.
            max_episodes (int, optional): Maximum number of episodes. Defaults to 10000.
            max_steps (int, optional): Maximum number of steps per episode. Defaults to 50000.
            test_every (int, optional): Number of episodes between tests. Defaults to 50.
            test_for (int, optional): Number of episodes to test for. Defaults to 10.
            draw_speed (float | None, optional): Caps the draw speed of the game. Defaults to None.(as fast as possible)
        """

        if self._model is None or self._optimizer is None or self._critic_loss is None:
            raise ValueError("Model not initialized. You must call initModel() before running an experiment.")

        self.init_experiment(training)

        metrics = {
                "running_total": -100,
                "running_avg"  : 0
        }

        for curr_episode in range(max_episodes):
            # print("=" * 80)
            with tf.GradientTape() as tape:
                rewards_hist, action_probs_hist, critic_value_hist = self.run_episode(training, max_steps=max_steps)

                self.update_metrics(curr_episode, rewards_hist, metrics)

                if training:
                    tr_metrics = self.update(tape, action_probs_hist, critic_value_hist, rewards_hist)
                    metrics |= tr_metrics

                wandb.log(metrics)

            if test_every is not None and curr_episode != 0 and curr_episode % test_every == 0:
                for _ in range(test_for):
                    self.run_episode(training=False, max_steps=max_steps)

    def run_episode(self, training: bool, max_steps: int = 50000):
        """
        Runs a single episode
        :param training: If False, the agent will pick the action with the highest probability.
                         Otherwise, it will pick from a probability distribution.
        :param max_steps: Maximum number of steps per episode. Defaults to 50000.
        :param draw_speed: Caps the draw speed of the game. Defaults to None.(as fast as possible)
        :param episode_num: Episode number for printing/logging
        :return:
        """
        reward_hist, action_probs_hist, critic_value_hist = [], [], []
        state = self.env_reset()

        for _ in range(max_steps):
            # Predict action probabilities and estimated future rewards from environment state
            action_probs, critic_value = self._model(state)
            action_probs, critic_value = tf.squeeze(action_probs), tf.squeeze(critic_value)

            if critic_value.shape != ():
                raise ValueError(f"Critic value shape is {critic_value.shape} instead of ()")
            if action_probs.shape != (self.e_num_actions,):
                raise ValueError(f"Action probs shape is {action_probs.shape} instead of {self.e_num_actions}")

            if training:
                action = np.random.choice(self.e_num_actions, p=action_probs.numpy())
            else:
                action = np.argmax(action_probs)

            action_probs_hist.append(tf.math.log(action_probs[action]))
            critic_value_hist.append(critic_value)

            # Reward
            reward = self._env.takeAction(action)
            reward = float(reward)
            reward_hist.append(reward)
            if reward in {-100.0, 100.0}:
                break

            state = self.env_get_state()

        return reward_hist, action_probs_hist, critic_value_hist

    def update(self, tape, action_probs_hist, critic_value_hist, rewards_hist):
        # For stateful model
        self._model.reset_states()

        # Calculate expected value from rewards
        # - At each timestep what was the total reward received after that timestep
        # - Rewards in the past are discounted by multiplying them with gamma
        # - These are the labels for our critic
        returns = []
        discounted_sum = 0
        for r in rewards_hist[::-1]:
            discounted_sum = r + self.a_gamma * discounted_sum
            returns.insert(0, discounted_sum)

        # Normalise
        # returns = np.array(returns).reshape(-1, 1)
        # self._a_returns_scaler.partial_fit(returns)
        # returns = self._a_returns_scaler.transform(returns).flatten().tolist()

        # Normalize the returns
        returns = np.array(returns)
        returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-7)
        returns = returns.tolist()

        # Calculating loss values to update our network
        history = zip(action_probs_hist, critic_value_hist, returns)
        actor_losses, critic_losses = [], []
        for action_log_prob, critic_ret, actual_ret in history:
            # At this point in history, the critic estimated that we would get a
            # total reward = `value` in the future. We took an action with log probability
            # of `log_prob` and ended up recieving a total reward = `ret`.
            # The actor must be updated so that it predicts an action that leads to
            # high rewards (compared to critic's estimate) with high probability.
            delta = actual_ret - critic_ret
            actor_loss = -action_log_prob * delta
            actor_losses.append(actor_loss)  # actor loss

            critic_loss = self._critic_loss(tf.expand_dims(critic_ret, 0), tf.expand_dims(actual_ret, 0))
            # critic_loss = self._critic_loss(tf.expand_dims(actual_ret, 0), tf.expand_dims(critic_ret, 0))

            # The critic must be updated so that it predicts a better estimate of the future rewards.
            critic_losses.append(critic_loss)

        plt.plot(returns, label='returns', color="red")

        plt.plot(critic_value_hist, label='critic_value_hist', color='darkred')
        # plt.plot(critic_losses, label='critic_losses', color='lightcoral')

        plt.plot(action_probs_hist, label='action_probs_hist', color='deepskyblue')
        # plt.plot(actor_losses, label='actor_losses', color="lightskyblue")

        plt.tight_layout()
        plt.legend()
        plt.show()

        # Backpropagation
        # loss_value = sum(actor_losses) + sum(critic_losses)
        loss_value = 0.75 * sum(actor_losses) + sum(critic_losses)

        grads = tape.gradient(loss_value, self._model.trainable_variables)
        self._optimizer.apply_gradients(zip(grads, self._model.trainable_variables))

        return {
                "discounted_sum": discounted_sum,
                "loss_actor"    : sum(actor_losses),
                "loss_critic"   : sum(critic_losses),
                "loss"          : loss_value,
        }

    def init_experiment(self, training):
        self.update_config()
        log.info(f"Config:\n{pprint.pformat(self._config)}")
        hash_all = hashlib.md5(json.dumps(self._config).encode('utf-8')).hexdigest()
        model_hash = self._config["model"]["config_hash"]
        wandb.init(project="ECE517",
                   entity="utkteam",
                   mode="disabled",
                   name=hash_all,
                   group=self._config["model"]["func_name"],
                   job_type="train" if training else "test",
                   tags=[f"Opt:{self._optimizer.__class__.__name__}",
                         f"Loss:{self._critic_loss.__class__.__name__}",
                         model_hash],
                   config=self._config)
        model_img_filename = f"{model_hash}.png"
        tf.keras.utils.plot_model(self._model,
                                  to_file=model_img_filename,
                                  show_layer_names=True,
                                  show_shapes=True,
                                  show_dtype=True,
                                  expand_nested=True,
                                  show_layer_activations=True,
                                  dpi=120)
        wandb.log({
                "model": wandb.Image(model_img_filename)
        })
        os.remove(model_img_filename)

    def update_metrics(self, curr_episode, rewards_hist, metrics):
        metrics.update({
                "episode"     : curr_episode,
                "steps"       : len(rewards_hist),
                "total_reward": np.sum(rewards_hist),
                "avg_reward"  : np.mean(rewards_hist),
                "max_reward"  : np.max(rewards_hist),
                "min_reward"  : np.min(rewards_hist),
        })
        metrics.update({
                "running_total": 0.05 * metrics["total_reward"] + 0.95 * metrics["running_total"],
                "running_avg"  : 0.05 * metrics["avg_reward"] + 0.95 * metrics["running_avg"],
        })
        log.info(f"#{curr_episode:>5}: Steps {metrics['steps']} | "
                 f"Total Reward {metrics['total_reward']} | "
                 f"Avg Reward {metrics['avg_reward']} | "
                 f"Running Total {metrics['running_total']} | "
                 f"Running Avg {metrics['running_avg']}")


def main():
    exp = Experiment(env_draw=True)
    exp.init_model("m_rnn")
    exp.run_experiment(training=True)

    # Run 10 experiments for each of the following parameters
    # Learning rate: 0.00001, 0.0001, 0.001, 0.01
    # Gamma 0.999, 0.99, 0.9, 0.8, 0.7

    # learning_rates = [0.0001, 0.001, 0.01]
    # gammas = [0.999, 0.99, 0.9, 0.8, 0.7]
    #
    # # create all combinations of the parameters
    # params = list(itertools.product(learning_rates, gammas))
    #
    # # run 10 experiments for each combination
    # for i in range(10):
    #     for lr, gamma in params:
    #         print(f"Experiment {i + 1} of 10: lr={lr}, gamma={gamma}")
    #         exp = Experiment(env_draw=False, learning_rate=lr, gamma=gamma)
    #         exp.init_model("m_dense")
    #         exp.run_experiment(training=True)


if __name__ == "__main__":
    main()
