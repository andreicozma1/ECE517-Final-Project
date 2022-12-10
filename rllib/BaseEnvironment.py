import abc
import logging
import os
import pprint
from typing import Any, List, Tuple

import joblib
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler


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
        self.state_scaler_enable: bool = state_scaling
        self.save_path_env: str = os.path.join(self.save_path_root, self.name)
        os.makedirs(self.save_path_env, exist_ok=True)

        self.state_scaler, self.state_scaler_path = None, os.path.join(self.save_path_env, "state_scaler.pkl")
        self.load_scaler()

        logging.info(f"Args:\n{pprint.pformat(self.__dict__, width=30)}")

    @property
    def config(self) -> dict:
        return {
                "name"               : self.name,
                "num_states"         : self.num_states,
                "num_actions"        : self.num_actions,
                "save_path_env"      : self.save_path_env,
                "state_scaler_enable": self.state_scaler_enable,
                "state_scaler"       : self.state_scaler.__class__.__name__,
                "state_scaler_path"  : self.state_scaler_path,
        }

    def load_scaler(self):
        if os.path.isfile(self.state_scaler_path):
            logging.info(f"Loading state scaler from {self.state_scaler_path}")
            self.state_scaler = joblib.load(self.state_scaler_path)
        else:
            self.reset_scaler()

    def reset_scaler(self):
        logging.warning("Creating new state scaler")
        self.state_scaler = MinMaxScaler(feature_range=(-1, 1))
        # self.state_scaler = StandardScaler()

    def transform_state(self, state):
        if self.state_scaler_enable:
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
        joblib.dump(self.state_scaler, self.state_scaler_path)
        return np.array(state, dtype=np.float32)

    # @tf.autograph.experimental.do_not_convert
    # def save_scaler(self):
    # joblib.dump(self.state_scaler, self.save_path_env_scaler)
