import os
import time
import numpy as np
from rlenv.pongclass import pongGame
import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
from tensorflow.python.framework.ops import disable_eager_execution

# disable_eager_execution()

# tf.compat.v1.experimental.output_all_intermediates(True)


class Experiment:
    def __init__(self, gamma=0.99, epsilon=np.finfo(np.float32).eps.item(), draw=False):
        self.gamma = gamma
        self.epsilon = epsilon
        self.draw = draw

        self.game = pongGame(400, 400, draw=self.draw)
        self.n_inputs = self.game.getState().shape[0]
        self.n_actions = 3
        print(f"Num Inputs: {self.n_inputs}")
        print(f"Num Actions: {self.n_actions}")

        self.mins = np.zeros(self.n_inputs)
        self.maxs = np.zeros(self.n_inputs)

    def getState(self):
        state = self.game.getState()
        self.mins = np.minimum(self.mins, state)
        self.maxs = np.maximum(self.maxs, state)
        # scale the state from -1 to 1
        state = (state - self.mins) / (self.maxs - self.mins) * 2 - 1
        return state

    def model_1(self):
        inputs = keras.layers.Input(shape=(self.n_inputs,))
        common = keras.layers.Dense(512, activation="relu")(inputs)
        common = keras.layers.Dense(256, activation="relu")(common)
        action = keras.layers.Dense(self.n_actions, activation="softmax")(common)
        critic = keras.layers.Dense(1)(common)

        model = keras.Model(inputs=inputs, outputs=[action, critic])

        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        huber_loss = keras.losses.Huber()

        return model, optimizer, huber_loss

    def run(self, max_episodes=10000, max_steps=50000, print_every=1, draw_speed=None):
        model, optimizer, loss = self.model_1()

        action_probs_history = []
        critic_value_history = []
        rewards_history = []
        running_reward = 0

        for curr_episode in range(max_episodes):

            episode_reward, curr_step, is_done = 0, 0, False
            self.game.reset()
            state = self.getState()

            with tf.GradientTape() as tape:

                while not is_done and curr_step < max_steps:
                    if self.draw:
                        self.game.draw()
                        if draw_speed is not None:
                            time.sleep(draw_speed)

                    state = tf.convert_to_tensor(state)
                    state = tf.expand_dims(state, 0)

                    # Predict action probabilities and estimated future rewards
                    # from environment state
                    action_probs, critic_value = model(state)
                    critic_value_history.append(critic_value[0, 0])

                    # Sample action from action probability distribution
                    action = np.random.choice(
                        self.n_actions, p=np.squeeze(action_probs)
                    )
                    action_probs_history.append(tf.math.log(action_probs[0, action]))

                    # Apply the sampled action in our environment
                    reward = self.game.takeAction(action)
                    rewards_history.append(reward)
                    episode_reward += reward

                    is_done = reward in [-100, 100]
                    state = self.getState()
                    curr_step += 1

                # Update running reward
                running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward
                print("=" * 80)
                print(
                    f"Episode {curr_episode}: Step {curr_step}, Reward {episode_reward}, Running Reward {running_reward}"
                )
                # Calculate expected value from rewards
                # - At each timestep what was the total reward received after that timestep
                # - Rewards in the past are discounted by multiplying them with gamma
                # - These are the labels for our critic
                returns = []
                discounted_sum = 0
                for r in rewards_history[::-1]:
                    discounted_sum = r + self.gamma * discounted_sum
                    returns.insert(0, discounted_sum)

                # Normalize
                returns = np.array(returns)
                returns = (returns - np.mean(returns)) / (
                    np.std(returns) + self.epsilon
                )
                returns = returns.tolist()

                # print(f"Returns: {returns}")

                # Calculating loss values to update our network
                history = zip(action_probs_history, critic_value_history, returns)
                actor_losses = []
                critic_losses = []
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
                    critic_losses.append(
                        loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0))
                    )

                # Backpropagation
                loss_value = sum(actor_losses) + sum(critic_losses)
                print(f"Loss: {loss_value}")
                grads = tape.gradient(loss_value, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

                # Clear the loss and reward history
                action_probs_history.clear()
                critic_value_history.clear()
                rewards_history.clear()


def main():
    exp = Experiment(draw=True)
    exp.run()


if __name__ == "__main__":
    main()
