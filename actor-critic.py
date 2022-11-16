import logging
import pprint
import time

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from rlenv.pongclass import pongGame
# This is a hacky fix for tensorflow imports to work with intellisense
from rllib.utils import logging_setup

keras = tf.keras

# Logging to stdout and file with logging class

log = logging_setup(file=__file__, name=__name__, level=logging.DEBUG)


class TransformerBlock(keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = tfa.layers.MultiHeadAttention(head_size=embed_dim, num_heads=num_heads)
        self.ffn = keras.Sequential([*[keras.layers.Dense(ffd, activation="relu") for ffd in ff_dim],
                                     keras.layers.Dense(embed_dim), ])
        self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = keras.layers.Dropout(dropout_rate)
        self.dropout2 = keras.layers.Dropout(dropout_rate)

    def call(self, inputs, training, mask=None):
        attn_output = self.att(inputs, inputs, attention_mask=mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class ActorCriticLayer(keras.layers.Layer):
    def __init__(self, num_actions):
        super(ActorCriticLayer, self).__init__()
        self.o_a = keras.layers.Dense(num_actions, activation="softmax", name="actor")
        self.o_c = keras.layers.Dense(1, activation="linear", name="critic")

    def call(self, actor_inputs, critic_inputs):
        actor = self.o_a(actor_inputs)
        critic = self.o_c(critic_inputs)
        return [actor, critic]


class Models:
    def __init__(self, e_num_states, e_num_actions):
        self.e_num_states = e_num_states
        self.e_num_actions = e_num_actions
        log.info(f"Models Args: {pprint.pformat(self.__dict__)}")

    def m_dense(self, **kwargs):
        logging.info("Model: m_dense")
        logging.info(f"kwargs: {pprint.pformat(kwargs)}")
        inputs = keras.layers.Input(shape=(self.e_num_states,), name="input")

        # Dense
        l1 = keras.layers.Dense(4096, activation="relu")
        l2 = keras.layers.Dropout(0.2)
        l3 = keras.layers.Dense(2048, activation="relu")
        l4 = keras.layers.Dropout(0.2)
        l5 = keras.layers.Dense(1024, activation="relu")
        l6 = keras.layers.Dropout(0.2)

        common = l1(inputs)
        common = l2(common)
        common = l3(common)
        common = l4(common)
        common = l5(common)
        common = l6(common)

        ac_layer = ActorCriticLayer(self.e_num_actions)(actor_inputs=common, critic_inputs=common)

        model = keras.Model(inputs=inputs, outputs=ac_layer, name="m_dense")
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        # optimizer = keras.optimizers.Nadam(learning_rate=0.001)
        # optimizer = tfa.optimizers.AdamW(
        #     learning_rate=0.01, weight_decay=0.3, amsgrad=True
        # )
        huber_loss = keras.losses.Huber()

        return model, optimizer, huber_loss

    def m_rnn(self, **kwargs):
        log.info("Model: m_rnn")
        log.info(f"kwargs: {pprint.pformat(kwargs)}")
        inputs = keras.layers.Input(shape=(self.e_num_states,))

        # RNN
        common = keras.layers.Reshape((1, self.e_num_states))(inputs)
        common = keras.layers.SimpleRNN(256, return_sequences=True)(common)
        common = keras.layers.Dropout(0.2)(common)
        common = keras.layers.SimpleRNN(256, return_sequences=False)(common)
        common = keras.layers.Dropout(0.2)(common)

        actor = keras.layers.Dense(self.e_num_actions, activation="softmax")(common)
        critic = keras.layers.Dense(1)(common)

        model = keras.Model(inputs=inputs, outputs=[actor, critic], name="m_rnn")
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        # optimizer = keras.optimizers.Nadam(learning_rate=0.001)
        # optimizer = tfa.optimizers.AdamW(
        #     learning_rate=0.01, weight_decay=0.3, amsgrad=True
        # )
        huber_loss = keras.losses.Huber()

        return model, optimizer, huber_loss

    def m_lstm(self, **kwargs):
        log.info("Model: m_lstm")
        log.info(f"kwargs: {pprint.pformat(kwargs)}")
        inputs = keras.layers.Input(shape=(self.e_num_states,))

        # LSTM
        common = keras.layers.Reshape((1, self.e_num_states))(inputs)
        common = keras.layers.LSTM(256, return_sequences=True)(common)
        common = keras.layers.Dropout(0.2)(common)
        common = keras.layers.LSTM(256, return_sequences=False)(common)
        common = keras.layers.Dropout(0.2)(common)

        actor = keras.layers.Dense(self.e_num_actions, activation="softmax", )(common)
        critic = keras.layers.Dense(1)(common)

        model = keras.Model(inputs=inputs, outputs=[actor, critic], name="m_lstm")
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        # optimizer = keras.optimizers.Nadam(learning_rate=0.001)
        # optimizer = tfa.optimizers.AdamW(
        #     learning_rate=0.01, weight_decay=0.3, amsgrad=True
        # )
        huber_loss = keras.losses.Huber()

        return model, optimizer, huber_loss

    def m_attention(self, **kwargs):
        log.info("Model: m_attention")
        log.info(f"kwargs: {pprint.pformat(kwargs)}")
        # The current state of the environment
        inputs = keras.layers.Input(  # shape=(
                #     None,
                #     self.n_inputs,
                # ),
                batch_input_shape=(1, 30, self.e_num_states), )

        common = keras.layers.LSTM(128, return_sequences=True, stateful=False)(inputs)
        common = keras.layers.Dropout(0.2)(common)

        # LSTM Actor
        lstm_actor = keras.layers.LSTM(128, return_sequences=False)(common)
        lstm_actor = keras.layers.Dropout(0.5)(lstm_actor)
        # LSTM Critic
        lstm_critic = keras.layers.LSTM(128, return_sequences=False)(common)
        lstm_critic = keras.layers.Dropout(0.5)(lstm_critic)

        # # Dense Actor
        dense_actor = keras.layers.Dense(128, activation="relu")(inputs)
        dense_actor = keras.layers.Dropout(0.2)(dense_actor)

        # Dense Critic
        dense_critic = keras.layers.Dense(128, activation="relu")(inputs)
        dense_critic = keras.layers.Dropout(0.2)(dense_critic)

        # Merge
        att_a = keras.layers.MultiHeadAttention(num_heads=6, key_dim=64, dropout=0.2)(dense_critic, dense_actor)
        att_a = keras.layers.Dropout(0.2)(common)

        att_c = keras.layers.MultiHeadAttention(num_heads=6, key_dim=64, dropout=0.2)(dense_critic, dense_critic)
        att_c = keras.layers.Dropout(0.2)(common)

        # Multiply the actor by the attention of the critic
        att_a = keras.layers.Multiply()([dense_actor, att_a])
        att_a = keras.layers.Dropout(0.2)(att_a)

        # Multiply the critic by the attention of the actor
        att_c = keras.layers.Multiply()([dense_critic, att_c])
        att_c = keras.layers.Dropout(0.2)(att_c)

        # Flatten
        actor = keras.layers.Flatten()(att_a)
        critic = keras.layers.Flatten()(att_c)

        # Output
        action = keras.layers.Dense(self.e_num_actions, activation="softmax")(actor)
        critic = keras.layers.Dense(1)(critic)

        model = keras.Model(inputs=inputs, outputs=[action, critic])
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        # optimizer = keras.optimizers.Nadam(learning_rate=0.001)
        # optimizer = tfa.optimizers.AdamW(
        #     learning_rate=0.01, weight_decay=0.3, amsgrad=True
        # )
        huber_loss = keras.losses.Huber()

        return model, optimizer, huber_loss


class Experiment:
    def __init__(self,
                 gamma=0.95,
                 epsilon=np.finfo(np.float32).eps.item(),
                 state_timesteps=30,
                 state_num_prev_actions=1,
                 env_draw=False):
        """
        Constructs an experiment.
        Args:
            gamma (float, optional): RL agent discount factor. Defaults to 0.95.
            epsilon (_type_, optional): RL agent epsilon. Defaults to np.finfo(np.float32).eps.item().
            state_timesteps (int, optional): The number of previous time steps to include in the state. Defaults to 30.
            state_num_prev_actions (int, optional): number of previous actions to append to the environment state. Defaults to 1.
            env_draw (bool, optional): whether to draw the environment. Defaults to False.
        """
        log.info("Experiment:")
        # Agent parameters
        self.a_model_func = None
        self._model, self._optimizer, self._loss = None, None, None
        self.a_gamma, self.a_epsilon = gamma, epsilon

        # Environment parameters
        self.e_state_timesteps = state_timesteps
        self.e_state_num_prev_actions = state_num_prev_actions
        self.e_draw = env_draw
        self._env = pongGame(400, 400, draw=self.e_draw)
        self.e_num_states = self._env.getState().shape[0]
        self.e_num_actions = 3

        # Other parameters for the experiment
        self._e_state_hist, self._e_state_min, self._e_state_max = None, None, None
        # Initialize the environment
        self.state_shape = self.env_reset(reset_scale_heuristics=True).shape

        # Wandb
        self._config = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    def init_model(self, model_func_name: str, **kwargs):
        """
        Initializes the model used for the RL agent.
        Args:
            model_func_name (str): corresponding function name in Models class (as string)
            **kwargs: Keyword arguments passed to the model function
        """
        self.a_model_func = model_func_name
        models = Models(self.e_num_states, self.e_num_actions, )
        init_model = getattr(models, model_func_name)
        self._model, self._optimizer, self._loss = init_model(**kwargs)
        self._model.summary()
        self._config.update(kwargs)

    def env_reset(self, reset_scale_heuristics=False):

        # reset the scale heuristics
        if reset_scale_heuristics:
            log.warning("Resetting scale heuristics.")
            self._e_state_min = np.ones(self.e_num_states) * np.inf
            self._e_state_max = np.ones(self.e_num_states) * -np.inf
            log.info("Getting min/max values for state normalization.")
            actions = np.arange(self.e_num_actions)
            for a in actions:
                self._env.reset()
                for _ in range(5000):
                    self._env.takeAction(a)
                    self.env_get_state()

            log.info(f"State min: {self._e_state_min}")
            log.info(f"State max: {self._e_state_max}")

        # reset the environment
        # self.game.reset_ball()
        self._env.reset()
        # clear the state history
        self._e_state_hist = np.zeros(shape=(1, self.e_state_timesteps, self.e_num_states))
        return self.env_get_state()

    def env_get_state(self):
        state = self._env.getState()
        if self._e_state_min is None or self._e_state_max is None:
            log.warning("State min/max not initialized.")
        # get min and max values for normalization
        self._e_state_min = np.minimum(self._e_state_min, state)
        self._e_state_max = np.maximum(self._e_state_max, state)
        # scale between 0 and 1
        state = (state - self._e_state_min) / (self._e_state_max - self._e_state_min)
        # scale between -1 and 1
        state = state * 2 - 1
        state = state[np.newaxis, :]
        return tf.convert_to_tensor(state, dtype=tf.float32)

    def run_experiment(self,
                       training: bool,
                       max_episodes: int = 10000,
                       max_steps: int = 50000,
                       draw_speed: float | None = None):
        """
        Runs experiments
        Args:
            training (bool, optional): Whether to train the agent. Defaults to True.
            max_episodes (int, optional): Maximum number of episodes. Defaults to 10000.
            max_steps (int, optional): Maximum number of steps per episode. Defaults to 50000.
            draw_speed (float | None, optional): Caps the draw speed of the game. Defaults to None.(as fast as possible)
        """
        if self.a_model_func is None or self._model is None or self._optimizer is None or self._loss is None:
            raise ValueError("Model not initialized. You must call initModel() before running an experiment.")

        log.info(f"Config:\n{pprint.pformat(self._config)}")
        # wandb.init(project="ECE517", entity="utkteam")

        running_reward = 0

        for curr_episode in range(max_episodes):
            with tf.GradientTape() as tape:
                rewards_hist, action_probs_hist, critic_value_hist = self.run_episode(training,
                                                                                      max_steps=max_steps,
                                                                                      draw_speed=draw_speed,
                                                                                      episode_num=curr_episode)

                # Update running reward
                running_reward = 0.05 * rewards_hist[-1] + (1 - 0.05) * running_reward
                logging.info(f"Running Reward: {round(running_reward, 2)}")

                if training:
                    self.update(tape, action_probs_hist, critic_value_hist, rewards_hist)

            if curr_episode != 0 and curr_episode % 10 == 0:
                for i in range(10):
                    self.run_episode(training=False,
                                     max_steps=max_steps,
                                     draw_speed=draw_speed,
                                     episode_num=f"{curr_episode}_{i}")

    def run_episode(self, training: bool, max_steps: int = 50000, draw_speed: float | None = None, episode_num=None):
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
        state, total_reward, step = self.env_reset(reset_scale_heuristics=False), 0, 0

        for step in range(max_steps):
            if self.e_draw:
                self.env_draw(draw_speed)

            # Predict action probabilities and estimated future rewards from environment state
            action_probs, critic_value = self._model(state)
            action_probs, critic_value = tf.squeeze(action_probs), tf.squeeze(critic_value)

            if critic_value.shape != ():
                raise ValueError(f"Critic value shape is {critic_value.shape} instead of ()")
            if action_probs.shape != (self.e_num_actions,):
                raise ValueError(f"Action probs shape is {action_probs.shape} instead of {self.e_num_actions}")

            # Logging
            # print(f"A: {action_probs} (shape: {action_probs.shape})", end="\t")
            # print(f"C: {critic_value} (shape: {critic_value.shape})")

            if training:
                action = np.random.choice(self.e_num_actions, p=action_probs.numpy())
            else:
                action = np.argmax(action_probs)

            action_probs_hist.append(tf.math.log(action_probs[action]))
            critic_value_hist.append(critic_value)

            # Reward
            reward = self._env.takeAction(action)
            reward_hist.append(reward)
            total_reward += reward
            if reward in [-100, 100]:
                break

            state = self.env_get_state()

        print("=" * 80)
        if episode_num is not None:
            log.info(f"#{episode_num:>5}: Steps {step:<4} | Reward {total_reward}")
        else:
            log.info(f"Steps {step:<4} | Reward {total_reward}")
        return reward_hist, action_probs_hist, critic_value_hist

    def update(self, tape, tr_action_probs_hist, tr_critic_value_hist, tr_rewards_hist):
        # For stateful model
        # self._model.reset_states()

        # Calculate expected value from rewards
        # - At each timestep what was the total reward received after that timestep
        # - Rewards in the past are discounted by multiplying them with gamma
        # - These are the labels for our critic
        returns = []
        discounted_sum = 0
        for r in tr_rewards_hist[::-1]:
            discounted_sum = r + self.a_gamma * discounted_sum
            returns.insert(0, discounted_sum)
        # Normalize the returns
        returns = np.array(returns)
        returns = (returns - np.mean(returns)) / (np.std(returns) + self.a_epsilon)
        returns = returns.tolist()
        # Calculating loss values to update our network
        history = zip(tr_action_probs_hist, tr_critic_value_hist, returns)
        actor_losses, critic_losses = [], []
        for log_prob, value, ret in history:
            # At this point in history, the critic estimated that we would get a
            # total reward = `value` in the future. We took an action with log probability
            # of `log_prob` and ended up recieving a total reward = `ret`.
            # The actor must be updated so that it predicts an action that leads to
            # high rewards (compared to critic's estimate) with high probability.
            diff = ret - value
            actor_losses.append(-log_prob * diff)  # actor loss

            # The critic must be updated so that it predicts a better estimate of
            # the future rewards.
            critic_losses.append(self._loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0)))

        # Backpropagation
        loss_value = sum(actor_losses) + sum(critic_losses)
        grads = tape.gradient(loss_value, self._model.trainable_variables)
        self._optimizer.apply_gradients(zip(grads, self._model.trainable_variables))

        # Logging
        actor_loss_mean, actor_loss_std = np.mean(actor_losses), np.std(actor_losses)
        critic_loss_mean, critic_loss_std = np.mean(critic_losses), np.std(critic_losses)
        print(f"UPDATE: Loss: {tf.round(loss_value, 2):<6}", end="\t")
        print(f"[ A: {tf.round(actor_loss_mean, 2):<3} +/- {tf.round(actor_loss_std, 2):<3} | "
              f"C: {tf.round(critic_loss_mean, 2):<3} +/- {tf.round(critic_loss_std, 2):<3} ]")

    def env_draw(self, draw_speed: float | None = None):
        self._env.draw()
        if draw_speed is not None:
            time.sleep(draw_speed)


def main():
    exp = Experiment(env_draw=True)
    exp.init_model("m_dense")
    exp.run_experiment(training=True)


if __name__ == "__main__":
    main()
