import abc
import logging
import os
import pprint
from typing import List, Tuple

import gymnasium
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import joblib
from rlenv import pongGame


class BaseEnvironment:

    def __init__(self, name: str,
                 num_states: int,
                 num_actions: int,
                 base_folder="./saves/environment/"):
        self.name: str = name
        self.num_states: int = num_states
        self.num_actions: int = num_actions

        self.save_path_base: str = base_folder
        self.save_path_env = os.path.join(self.save_path_base, self.name)
        os.makedirs(self.save_path_env, exist_ok=True)
        self.save_path_env_scaler = os.path.join(self.save_path_env, "state_scaler.pkl")

        if os.path.isfile(self.save_path_env_scaler):
            logging.info(f"Loading state scaler from {self.save_path_env_scaler}")
            self.state_scaler = joblib.load(self.save_path_env_scaler)
        else:
            self.reset_scaler()

    def reset_scaler(self):
        logging.warning("Creating new state scaler")
        self.state_scaler = MinMaxScaler(feature_range=(-1, 1))

    def scale_state(self, state):
        state = state.reshape(1, self.num_states)
        try:
            self.state_scaler.partial_fit(state)
        except ValueError:
            self.reset_scaler()
            self.state_scaler.partial_fit(state)
        return self.state_scaler.transform(state)

    def tf_step(self, action: tf.Tensor) -> List[tf.Tensor]:
        return tf.numpy_function(self.step, [action],
                                 [tf.float32, tf.int32, tf.bool])

    @abc.abstractmethod
    def step(self, action) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pass

    @abc.abstractmethod
    def reset(self) -> np.ndarray:
        pass


class PongEnvironment(BaseEnvironment):
    def __init__(self, draw=False, draw_speed=None):
        self._game = pongGame(200, 200, draw=draw, draw_speed=draw_speed)
        num_states, num_actions = self._game.getState().shape[0], 3
        super().__init__(name='Pong', num_states=num_states, num_actions=num_actions)
        logging.info(f"Agent Args: {pprint.pformat(self.__dict__)}")

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns state, reward and done flag given an action."""
        reward = self._game.takeAction(action)
        done = reward in [-100, 100]
        state = self.scale_state(self._game.getState())
        return state.astype(np.float32), np.array(reward, np.int32), np.array(done, np.bool)

    def reset(self):
        self._game.reset()
        state = self._game.getState()
        state = self.scale_state(state)
        joblib.dump(self.state_scaler, self.save_path_env_scaler)
        return state


class LunarLander(BaseEnvironment):
    def __init__(self, draw=False, draw_speed=None):
        self.draw = draw
        self.draw_speed = draw_speed
        self._env = gymnasium.make("LunarLander-v2", render_mode="human" if draw else "none")
        num_states, num_actions = self._env.observation_space, self._env.action_space
        print(num_states, num_actions)
        num_states, num_actions = self._env.observation_space.shape[0], self._env.action_space.n
        super().__init__(name='LunarLander', num_states=num_states, num_actions=num_actions)
        logging.info(f"Agent Args: {pprint.pformat(self.__dict__)}")

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns state, reward and done flag given an action."""
        state, reward, terminated, truncated, _ = self._env.step(action)
        state = self.scale_state(state)
        done = terminated or truncated
        return state.astype(np.float32), np.array(reward, np.int32), np.array(done, np.bool)

    def reset(self):
        state, _ = self._env.reset()
        state = self.scale_state(state)
        joblib.dump(self.state_scaler, self.save_path_env_scaler)
        return state
