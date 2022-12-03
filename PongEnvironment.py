import abc
import logging
import os
import pprint
from typing import Any, List, Tuple

import gymnasium
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import joblib
from rlenv import pongGame


class BaseEnvironment:

    def __init__(self, name: str,
                 num_states: int,
                 num_actions: int,
                 base_folder="./saves/",
                 state_scaling: bool = True):
        self.name: str = name
        self.num_states: int = num_states
        self.num_actions: int = num_actions
        self.save_path_root: str = base_folder
        self.state_scaling: bool = state_scaling
        self.save_path_env: str = os.path.join(self.save_path_root, self.name)
        os.makedirs(self.save_path_env, exist_ok=True)

        self.state_scaler, self.save_path_env_scaler = None, os.path.join(self.save_path_env, "state_scaler.pkl")
        self.load_scaler()

        logging.info(f"Args:\n{pprint.pformat(self.__dict__, width=30)}")

    def load_scaler(self):
        if os.path.isfile(self.save_path_env_scaler):
            logging.info(f"Loading state scaler from {self.save_path_env_scaler}")
            self.state_scaler = joblib.load(self.save_path_env_scaler)
        else:
            self.reset_scaler()

    def reset_scaler(self):
        logging.warning("Creating new state scaler")
        self.state_scaler = MinMaxScaler(feature_range=(0, 1))
        # self.state_scaler = StandardScaler()

    def transform_state(self, state):
        if self.state_scaling:
            state = state.reshape(1, self.num_states)
            try:
                self.state_scaler.partial_fit(state)
            except ValueError:
                self.reset_scaler()
                self.state_scaler.partial_fit(state)
            state = self.state_scaler.transform(state)
        return state

    def tf_step(self, action: tf.Tensor) -> List[tf.Tensor]:
        return tf.numpy_function(self.step, [action],
                                 [tf.float32, tf.float32, tf.int32])

    @abc.abstractmethod
    def _step(self, action) -> Tuple[Any, float, bool]:
        pass

    def step(self, action) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        state, reward, done = self._step(action)
        state = self.transform_state(state)
        return state.astype(np.float32), np.array(reward, np.float32), np.array(done, np.int32)

    def tf_reset(self) -> tf.Tensor:
        return tf.numpy_function(self.reset, [], tf.float32)

    @abc.abstractmethod
    def _reset(self) -> np.ndarray:
        pass

    def reset(self) -> np.ndarray:
        state = self._reset()
        state = self.transform_state(state)
        joblib.dump(self.state_scaler, self.save_path_env_scaler)
        return np.array(state, dtype=np.float32)

    # @tf.autograph.experimental.do_not_convert
    # def save_scaler(self):
    # joblib.dump(self.state_scaler, self.save_path_env_scaler)


class PongEnvironment(BaseEnvironment):
    def __init__(self, draw=False, draw_speed=None, state_scaling=True):
        self._game = pongGame(200, 200, draw=draw, draw_speed=draw_speed)
        num_states, num_actions = self._game.getState().shape[0], 3
        super().__init__(name='Pong', num_states=num_states, num_actions=num_actions, state_scaling=state_scaling)

    def _step(self, action: np.ndarray) -> Tuple[Any, float, bool]:
        reward = self._game.takeAction(action, reward_step=1, reward_hit=15)
        done = reward in [-100, 100]
        state = self._game.getState()
        return state, reward, done

    def _reset(self):
        self._game.reset()
        return self._game.getState()


class LunarLander(BaseEnvironment):
    def __init__(self, draw=False, draw_speed=None, state_scaling=True):
        self.draw, self.draw_speed = draw, draw_speed
        self._env = gymnasium.make("LunarLander-v2", render_mode="human" if draw else None)
        num_states, num_actions = self._env.observation_space.shape[0], self._env.action_space.n
        super().__init__(name='LunarLander',
                         num_states=num_states,
                         num_actions=num_actions,
                         state_scaling=state_scaling)

    def _step(self, action: np.ndarray) -> Tuple[Any, float, bool]:
        state, reward, terminated, truncated, _ = self._env.step(action)
        done = terminated or truncated
        return state, reward, done

    def _reset(self):
        state, _ = self._env.reset()
        return state
