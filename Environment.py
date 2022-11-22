import logging
import pprint
from typing import List, Tuple

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

from rlenv import pongGame


class Environment:
    def __init__(self, draw=False, draw_speed=None):
        self.game = pongGame(200, 200, draw=draw, draw_speed=draw_speed)
        self.num_states = self.game.getState().shape[0]
        self.num_actions = 3
        self.state_scaler = MinMaxScaler(feature_range=(-1, 1))
        logging.info(f"Agent Args: {pprint.pformat(self.__dict__)}")

    def get_state(self):
        state = self.game.getState().reshape(1, -1)
        self.state_scaler.partial_fit(state)
        state = self.state_scaler.transform(state)
        return state

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns state, reward and done flag given an action."""
        reward = self.game.takeAction(action)
        done = reward in [-100, 100]
        state = self.get_state()
        return state.astype(np.float32), np.array(reward, np.int32), np.array(done, np.bool)

    def tf_step(self, action: tf.Tensor) -> List[tf.Tensor]:
        return tf.numpy_function(self.step, [action],
                                 [tf.float32, tf.int32, tf.bool])

    def reset(self):
        self.game.reset()
        return self.get_state()
