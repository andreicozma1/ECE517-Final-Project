import sys
import os

import pathlib
import numpy as np

# print(os.getcwd())
# print(os.path.dirname(__file__))
# print(pathlib.Path(__file__).parent.parent.absolute())
sys.path.append(str(pathlib.Path(__file__).parent.parent.absolute()))
from TabularAgent import TabularAgent
from Environment import Environment
from Experiment import Experiment
from utils import logging_setup


class nStepTD(TabularAgent):
    def __init__(self,
                 n: int,
                 quantize: int,
                 num_states: int,
                 num_actions: int,
                 gamma: float = 0.99,
                 alpha: float = 0.1,
                 epsilon: float = 0.1):
        super().__init__(quantize, num_states, num_actions, gamma, alpha, epsilon)
        self.actions: list = None
        self.states: list = None
        self.rewards: list = None
        self.t: int = None
        self.T: int = None
        self.n: int = n

        self.on_episode_start()

    def on_episode_start(self):
        self.t, self.T = 0, np.inf
        self.rewards = []
        self.states = []
        self.actions = []

    def get_action(self, t, state, deterministic=False):
        self.t = t  # store the current timestep
        if t == 0:
            return self.select_action(state, deterministic)
        return self.actions[-1]

    def select_action(self, state, deterministic=False):
        state = self.bin_state(state.flatten())
        if not deterministic and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.Q[tuple(state)])

    def on_episode_end(self):
        self.T = self.t + 1
        print(self.T)
        while True:
            tau = self.t - self.n + 1
            if tau >= self.T - 1:
                break
            self.tau_update(tau)
            # self.t += 1

    def on_update(self, reward, last_state, last_action, next_state):
        # if first step, initialize state and action history with first state and action
        # if self.t == 0:
        #     self.states.append(last_state)
        # self.actions.append(last_action)
        # print(self.t)
        if self.t < self.T:
            self.rewards.append(reward)
            self.states.append(last_state)
            self.actions.append(last_action)
            if next_state is None:
                self.T = self.t + 1
            # else:
            #     self.actions.append(last_action)
        if 0 < self.t:
            tau = self.t - self.n + 1
            self.tau_update(tau)
        # self.t += 1

    def tau_update(self, tau):
        if tau >= 0:

            # calculate return from t to t-n
            G = 0
            for i in range(tau + 1, min(self.T, tau + self.n) + 1):
                G += self.gamma ** (i - tau - 1) * self.rewards[i - 1]

            if tau + self.n < self.T:
                state = self.bin_state(self.states[tau + self.n - 1]).flatten()
                action = self.actions[tau + self.n - 1]
                G += self.gamma ** self.n * self.Q[tuple(list(state.copy()) + [action])]
            state = self.bin_state(self.states[tau - 1]).flatten()
            action = self.actions[tau - 1]
            self.Q[tuple(list(state.copy()) + [action])] += \
                self.alpha * (G - self.Q[tuple(list(state.copy()) + [action])])


"""
for _ in range(rounds):
    self.reset()
    t = 0
    T = np.inf
    action = self.chooseAction()

    actions = [action]
    states = [self.state]
    rewards = [0]
    while True:
        if t < T:
            state = self.takeAction(action)  # next state
            reward = self.giveReward()  # next state-reward

            states.append(state)
            rewards.append(reward)

            if self.end:
                if self.debug:
                    print("End at state {} | number of states {}".format(state, len(states)))
                T = t + 1
            else:
                action = self.chooseAction()
                actions.append(action)  # next action
        # state tau being updated
        tau = t - self.n + 1
        if tau >= 0:
            G = 0
            for i in range(tau + 1, min(tau + self.n + 1, T + 1)):
                G += np.power(self.gamma, i - tau - 1) * rewards[i]
            if tau + self.n < T:
                state_action = (states[tau + self.n], actions[tau + self.n])
                G += np.power(self.gamma, self.n) * self.Q_values[state_action[0]][state_action[1]]
            # update Q values
            state_action = (states[tau], actions[tau])
            self.Q_values[state_action[0]][state_action[1]] += self.lr * (
                        G - self.Q_values[state_action[0]][state_action[1]])

        if tau == T - 1:
            break

        t += 1
"""

if __name__ == "__main__":
    env = Environment(draw=False)
    agent = nStepTD(
            n=2,
            quantize=8,
            num_states=env.num_states,
            num_actions=env.num_actions,
            gamma=0.99,
            alpha=0.1,
            epsilon=0.1)
    exp = Experiment(env, agent)
    exp.run_experiment(max_episodes=1)
