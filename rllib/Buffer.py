import tensorflow as tf


class Buffer:
    # Buffer for storing trajectories
    def __init__(self, obs_dim, max_timesteps, gamma=0.99, lam=0.95):
        # Buffer initialization
        self.observation_buffer = self.create_buffer(obs_dim, tf.float32)
        self.action_buffer = self.create_buffer((), tf.int32)
        self.reward_buffer = self.create_buffer((), tf.float32)

        self.value_buffer = self.create_buffer((), tf.float32)
        self.log_prob_buffer = self.create_buffer((), tf.float32)

        self.advantage_buffer = self.create_buffer((), tf.float32)
        self.return_buffer = self.create_buffer((), tf.float32)

        self.obs_dim = obs_dim
        self.max_timesteps = max_timesteps
        self.gamma, self.lam = gamma, lam
        self.pointer, self.trajectory_start_index = 0, 0
        self.episode_indices = [0]

    def store(self, observation, action, reward, value, log_prob):
        # Append one step of agent-environment interaction
        self.observation_buffer = self.observation_buffer.write(self.pointer, observation)
        self.action_buffer = self.action_buffer.write(self.pointer, action)
        self.reward_buffer = self.reward_buffer.write(self.pointer, reward)
        self.value_buffer = self.value_buffer.write(self.pointer, value)
        self.log_prob_buffer = self.log_prob_buffer.write(self.pointer, log_prob)
        self.pointer += 1

    def finish_trajectory(self, last_value=0):
        reward_buffer = self.reward_buffer.stack()
        value_buffer = self.value_buffer.stack()

        # Finish the trajectory by computing advantage estimates and rewards-to-go
        path_slice = slice(self.trajectory_start_index, self.pointer)
        rewards = tf.concat([reward_buffer[path_slice], [last_value]], axis=0)
        values = tf.concat([value_buffer[path_slice], [last_value]], axis=0)

        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]

        # calculate advantage and return
        tmp_advantage_buffer = self.discounted_cumulative_sums(
                tf.expand_dims(deltas, axis=0), self.gamma * self.lam
        )
        tmp_return_buffer = self.discounted_cumulative_sums(
                tf.expand_dims(rewards, axis=0), self.gamma
        )[:-1]
        for t, (adv, ret) in enumerate(zip(tmp_advantage_buffer, tmp_return_buffer)):
            self.advantage_buffer = self.advantage_buffer.write(path_slice.start + t, adv)
            self.return_buffer = self.return_buffer.write(path_slice.start + t, ret)

        # pad to max_timesteps
        pad_size = self.max_timesteps - (self.pointer % self.max_timesteps)
        self.observation_buffer = self._pad_buffer(self.advantage_buffer, self.obs_dim, pad_size, tf.float32)
        self.action_buffer = self._pad_buffer(self.action_buffer, (), pad_size, tf.int32)
        self.advantage_buffer = self._pad_buffer(self.advantage_buffer, (), pad_size, tf.float32)
        self.reward_buffer = self._pad_buffer(self.reward_buffer, (), pad_size, tf.float32)
        self.return_buffer = self._pad_buffer(self.return_buffer, (), pad_size, tf.float32)
        self.value_buffer = self._pad_buffer(self.value_buffer, (), pad_size, tf.float32)
        self.log_prob_buffer = self._pad_buffer(self.log_prob_buffer, (), pad_size, tf.float32)

        # adjust traj. start index and pointer
        self.trajectory_start_index += self.max_timesteps
        self.pointer = self.trajectory_start_index
        self.episode_indices.append(self.trajectory_start_index)

    def _pad_buffer(self, buffer, shape, num_pad_indices, dtype):
        for i in range(num_pad_indices):
            buffer = buffer.write(
                    self.pointer + i,
                    tf.zeros(shape, dtype=dtype)
            )
        return buffer

    def get(self):
        # Get all data of the buffer and normalize the advantages
        advantage_buffer = self.advantage_buffer.stack()
        advantage_buffer = (advantage_buffer - tf.math.reduce_mean(advantage_buffer)) \
                           / (tf.math.reduce_std(advantage_buffer) + 1e-10)
        return (
                self.observation_buffer.stack(),
                self.action_buffer.stack(),
                advantage_buffer,
                self.return_buffer.stack(),
                self.log_prob_buffer.stack(),
                self.value_buffer.stack(),
                self.reward_buffer.stack()
        )

    def get_batches(self):
        (
                observation_buffer,
                action_buffer,
                advatage_buffer,
                return_buffer,
                log_prob_buffer,
                value_buffer,
                reward_buffer
        ) = self.get()
        batch_size = len(self.episode_indices)
        return (
                tf.reshape(observation_buffer, (batch_size, self.max_timesteps, *(self.obs_dim,))),
                tf.reshape(action_buffer, (batch_size, self.max_timesteps, -1)),
                tf.reshape(advatage_buffer, (batch_size, self.max_timesteps, -1)),
                tf.reshape(return_buffer, (batch_size, self.max_timesteps, -1)),
                tf.reshape(log_prob_buffer, (batch_size, self.max_timesteps, -1)),
                tf.reshape(value_buffer, (batch_size, self.max_timesteps, -1)),
                tf.reshape(reward_buffer, (batch_size, self.max_timesteps, -1))
        )

    @staticmethod
    def create_buffer(shape, dtype):
        return tf.TensorArray(dtype=dtype, size=0,
                              dynamic_size=True,
                              element_shape=shape)

    @staticmethod
    def discounted_cumulative_sums(x, discount):
        max_len = tf.squeeze(x).shape[0]
        gamma = tf.constant(float(discount), dtype=tf.float32, shape=[max_len, 1, 1])
        x_filter = tf.math.cumprod(gamma, exclusive=True)
        x_pad = tf.expand_dims(tf.concat(
                [x, tf.zeros_like(x[:, :-1])], axis=1),
                axis=2)
        x = tf.nn.conv1d(x_pad, x_filter, stride=1, padding='VALID')
        return tf.squeeze(x)
