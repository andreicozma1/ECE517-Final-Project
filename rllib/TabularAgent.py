import sys
import os
import pathlib
import abc
import numpy as np
from typing import Tuple

sys.path.append(str(pathlib.Path(__file__).parent.parent.absolute()))
from BaseAgent import BaseAgent
from Environment import Environment
from utils import logging_setup


class TabularAgent(BaseAgent):
    def __init__(self,
                 quantize: int,
                 num_states: int,
                 num_actions: int,
                 gamma: float = 0.99,
                 alpha: float = 0.1,
                 epsilon: float = 0.1):
        self.gamma: float = gamma
        self.alpha: float = alpha
        self.epsilon: float = epsilon
        self.quantize: int = quantize
        self.n_states: int = num_states
        self.n_actions: int = num_actions

        self.dim = [self.quantize] * self.n_states + [self.n_actions]
        self.Q: np.ndarray = np.zeros(self.dim)

    def bin_state(self, state: np.ndarray) -> np.ndarray:
        unit = (2 / self.quantize) + -1
        bins = np.linspace(unit, 1, self.quantize)
        s = np.digitize(state, bins=bins, right=True)
        s[s == self.quantize] = self.quantize - 1
        return s

    @abc.abstractmethod
    def on_episode_start(self):
        pass

    @abc.abstractmethod
    def get_action(self, t, state):
        pass

    @abc.abstractmethod
    def on_episode_end(self):
        pass

    @abc.abstractmethod
    def on_update(self, rewards, last_state, last_action, next_state):
        pass

    def run_episode(self, env: Environment, max_steps: int, deterministic: bool = False) -> Tuple[np.array, dict]:
        self.on_episode_start()
        reward_hist = []

        next_state = env.reset()
        initial_state_shape = next_state.shape

        for t in range(max_steps):
            action = self.get_action(t, next_state)
            last_state, last_action = next_state.copy(), action
            next_state, reward, done = env.step(action)
            next_state = next_state.reshape(initial_state_shape)
            reward_hist.append(reward)
            if done:
                break
            self.on_update(reward, last_state, last_action, next_state)
        self.on_episode_end()
        stats = {
                "steps"       : len(reward_hist),
                "total_reward": float(np.sum(reward_hist)),
        }
        return np.array(reward_hist, dtype=np.int32), stats

    def train_step(self, env: Environment, max_steps_per_episode: int) -> dict:
        rewards, stats = self.run_episode(env, max_steps_per_episode, deterministic=False)
        return stats
