import logging
import os
import pprint
from typing import List, Tuple

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import joblib
from rlenv import pongGame


class BaseEnvironment:

    def __init__(self, name: str, base_folder="./saves/environment/"):
        self.name: str = name

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


class PongEnvironment(BaseEnvironment):
    def __init__(self, draw=False, draw_speed=None):
        super().__init__(name='Pong')
        self.draw = draw
        self.game = pongGame(200, 200, draw=draw, draw_speed=draw_speed)
        self.num_states = self.game.getState().shape[0]
        self.num_actions = 3
        logging.info(f"Agent Args: {pprint.pformat(self.__dict__)}")

    def get_state(self):
        state = self.game.getState().reshape(1, self.num_states)
        try:
            self.state_scaler.partial_fit(state)
        except ValueError:
            self.reset_scaler()
            self.state_scaler.partial_fit(state)
        state = self.state_scaler.transform(state)
        return state

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Returns state, reward and done flag given an action."""
        reward = self.game.takeAction(action)
        done = reward in [-100, 100]
        win = reward == 100
        state = self.get_state()
        return state.astype(np.float32), np.array(reward, np.int32), np.array(done, np.bool), np.array(win, np.bool)

    def tf_step(self, action: tf.Tensor) -> List[tf.Tensor]:
        return tf.numpy_function(self.step, [action],
                                 [tf.float32, tf.int32, tf.bool, tf.bool])

    def reset(self):
        self.game.reset()
        joblib.dump(self.state_scaler, self.save_path_env_scaler)
        return self.get_state()
