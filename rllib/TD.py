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


class TD(TabularAgent):
    def __init__(self,
                 algorithm: str,
                 quantize: int,
                 num_states: int,
                 num_actions: int,
                 gamma: float = 0.99,
                 alpha: float = 0.1,
                 epsilon: float = 0.1):
        super().__init__(quantize, num_states, num_actions, gamma, alpha, epsilon)
        self.alg: str = algorithm
        self.last_action = None
        self.last_state = None

    def on_episode_start(self):
        pass

    def get_action(self, _, state, deterministic=False):
        state = self.bin_state(state.flatten())
        if not deterministic and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.Q[tuple(state)])

    def on_episode_end(self):
        pass

    def on_update(self, reward, last_state, last_action, next_state):
        last_state, next_state = self.bin_state(last_state.flatten()), self.bin_state(next_state.flatten())

        idx = tuple(list(last_state.copy()) + [last_action])
        current_qsa = self.Q[idx]

        if self.alg == 'Q-Learning':
            next_qsa = np.max(self.Q[tuple(self.bin_state(next_state))])

        elif self.alg == 'SARSA':
            next_action = self.get_action(0, next_state, deterministic=False)
            next_qsa = self.Q[tuple(list(next_state.copy()) + [next_action])]
        else:
            raise Exception("Unknown TD learning type")

        target_qsa = reward + self.gamma * next_qsa
        self.Q[idx] += self.alpha * (target_qsa - current_qsa)


if __name__ == "__main__":
    env = Environment(draw=False)
    agent = TD(
            algorithm="Q-Learning",
            quantize=8,
            num_states=env.num_states,
            num_actions=env.num_actions,
            gamma=0.99,
            alpha=0.1,
            epsilon=0.1)
    exp = Experiment(env, agent)
    exp.run_experiment()
