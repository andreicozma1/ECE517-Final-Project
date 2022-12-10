import tensorflow as tf

keras = tf.keras


class ActorLoss(keras.losses.Loss):
    def __init__(self, name='actor_loss', reduction=tf.keras.losses.Reduction.NONE):
        super().__init__(name=name, reduction=reduction)

    def get_config(self):
        return super().get_config()

    def call(self, action_probs: tf.Tensor, advantage: tf.Tensor):
        advantage_probs = tf.math.multiply(advantage, tf.math.log(action_probs))
        advantage_probs = -tf.math.reduce_sum(advantage_probs, axis=-1)
        return advantage_probs


class A2CLoss(keras.losses.Loss):
    def __init__(self, name='a2closs', reduction=tf.keras.losses.Reduction.NONE):
        super().__init__(name=name, reduction=reduction)
        self.critic_loss = keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
        self.actor_loss = ActorLoss(reduction=tf.keras.losses.Reduction.NONE)

    def get_config(self):
        return super().get_config()

    def call(self, hist_sar, hist_model_out):
        # def call(self, hist_sar, hist_model_out):
        hist_states, hist_actions, hist_rewards = hist_sar  # ytrue
        action_probs, critic_returns = hist_model_out  # ypred
        action_probs = tf.stack(action_probs, axis=1)
        critic_returns = tf.stack(critic_returns, axis=1)

        action_probs = tf.nn.softmax(action_probs, axis=-1)
        entropy_loss = self.get_entropy_loss(action_probs)
        entropy_loss = self.normalize(entropy_loss)
        entropy_loss = tf.math.multiply(entropy_loss, self.entropy_loss_multiplier)
        action_probs = tf.reduce_max(action_probs, axis=-1)

        actual_returns = self.expected_return(hist_rewards)
        actual_returns = self.normalize(actual_returns)
        critic_returns = self.normalize(critic_returns)

        # action_probs_max = tf.reshape(action_probs_max, shape=(-1, 1))
        critic_returns = tf.reshape(critic_returns, shape=(-1, 1))
        actual_returns = tf.reshape(actual_returns, shape=(-1, 1))
        (batch, timesteps, value)

        advantage = actual_returns - critic_returns

        self.plot_ret(actual_returns, advantage, critic_returns)

        actor_losses = self.actor_loss(action_probs, advantage)
        actor_losses = self.normalize(actor_losses)
        actor_losses = tf.math.multiply(actor_losses, self.actor_loss_multiplier)

        # entropy_loss = self.get_entropy_loss(action_probs_max)
        # entropy_loss = self.standardize(entropy_loss)
        # entropy_loss = self.normalize(entropy_loss)
        # entropy_loss = tf.math.multiply(entropy_loss, self.entropy_loss_multiplier)

        critic_losses = self.critic_loss(critic_returns, actual_returns)
        critic_losses = self.normalize(critic_losses)
        critic_losses = tf.math.multiply(critic_losses, self.critic_loss_multiplier)

        total_losses = actor_losses + critic_losses
        # total_losses = self.normalize(total_losses)
        # total_losses = tf.clip_by_value(total_losses, clip_value_min=-1, clip_value_max=1)

        self.plot_loss(actor_losses, critic_losses, total_losses, entropy_loss=entropy_loss)

        return tf.reduce_sum(total_losses)
