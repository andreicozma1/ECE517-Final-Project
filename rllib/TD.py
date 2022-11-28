import sys
import os

import pathlib
import numpy as np
# print(os.getcwd())
# print(os.path.dirname(__file__))
# print(pathlib.Path(__file__).parent.parent.absolute())
sys.path.append(str(pathlib.Path(__file__).parent.parent.absolute()))
from TabularAgent import Tabular
from Environment import Environment
from Experiment import Experiment
from utils import logging_setup

class TD(Tabular):
    def __init__(self,
                 algorithm:str,
                 quantize: int,
                 num_states: int,
                 num_actions: int,
                 gamma: float = 0.99,
                 alpha: float = 0.1,
                 epsilon: float = 0.1):
        super().__init__(quantize, num_states, num_actions, gamma, alpha, epsilon)
        self.alg:str = algorithm
        self.last_action = None
        self.last_state = None

    def on_episode_start(self):
        pass

    def get_action(self, _, state, deterministic=False):
        self.last_state = self.bin_state(state.flatten())
        if not deterministic and np.random.random() < self.epsilon:
            self.last_action = np.random.randint(self.n_actions)
            # print('random', self.last_state, self.last_action)
        else:
            self.last_action = np.argmax(self.Q[tuple(self.last_state)])
            # print('greedy', self.Q.shape, len(self.Q[tuple(self.last_state)]), self.last_state, self.last_action)
        # print(self.)
        # print(self.last_action)
        return self.last_action

    def on_episode_end(self):
        pass

    def on_update(self, reward, next_state):

        s,a = self.last_state.flatten(), self.last_action
        idx = tuple(list(s.copy()) + [a])
        current_qsa = self.Q[idx]

        if self.alg == 'Q-Learning':
            next_qsa = np.max(self.Q[tuple(self.bin_state(next_state))])

        elif self.alg == 'SARSA':
            next_action = self.get_action(0, next_state, deterministic=False)
            next_qsa = self.Q[tuple(list(self.last_state.copy()) + [next_action])]
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