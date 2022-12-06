from typing import Any, Tuple

import gymnasium
import numpy as np

from rlenv import pongGame
from rllib.BaseEnvironment import BaseEnvironment


class PongEnvironment(BaseEnvironment):
    def __init__(self, draw=False, draw_speed=None, state_scaler_enable=True):
        self._game = pongGame(200, 200, draw=draw, draw_speed=draw_speed)
        num_states, num_actions = self._game.getState().shape[0], 3
        super().__init__(name='Pong', num_states=num_states, num_actions=num_actions, state_scaling=state_scaler_enable)

    def _step(self, action: np.ndarray) -> Tuple[Any, float, bool]:
        reward = self._game.takeAction(action, reward_step=1, reward_hit=15)
        done = reward in [-100, 100]
        state = self._game.getState()
        return state, reward, done

    def _reset(self):
        self._game.reset()
        return self._game.getState()


class LunarLander(BaseEnvironment):
    def __init__(self, draw=False, draw_speed=None, state_scaler_enable=True):
        self.draw, self.draw_speed = draw, draw_speed
        self._env = gymnasium.make("LunarLander-v2", render_mode="human" if draw else None)
        self._env.reset(seed=42)
        num_states, num_actions = self._env.observation_space.shape[0], self._env.action_space.n
        super().__init__(name='LunarLander',
                         num_states=num_states,
                         num_actions=num_actions,
                         state_scaling=state_scaler_enable)

    def _step(self, action: np.ndarray) -> Tuple[Any, float, bool]:
        state, reward, terminated, truncated, _ = self._env.step(action)
        done = terminated or truncated
        return state, reward, done

    def _reset(self):
        state, _ = self._env.reset()
        return state
