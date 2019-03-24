import numpy as np
import tensorflow as tf
from tensorflow.keras import layers 

from base_model import BaseModel

class Actor(BaseModel):
    def __init__(self, obs_dim, act_dim, act_limit, critic, checkpoint_file=None):
        self.checkpoint_file = checkpoint_file
        self.act_limit = act_limit
        self.critic = critic
        self.model = tf.keras.Sequential([
            layers.Dense(400, input_shape=(None, obs_dim), activation='relu'),
            layers.Dense(300, activation='relu'),
            layers.Dense(act_dim, activation='tanh')
        ])
        self.optimizer = tf.optimizers.Adam(learning_rate=1e-3)

    def __call__(self, state):
        return self.forward(state)

    # Retuns the Action to take given the current state, aka the Policy.
    @tf.function
    def forward(self, state):
        pi = self.act_limit * self.model(state)
        return pi

    # Same as forward() but with stochasticity.
    @tf.function
    def forwardNoisy(self, state, noise_clip = 0.5, target_noise = 0.2):
        pi = self.forward(state)
        noise = tf.random.normal(tf.shape(pi), stddev=target_noise)
        noise = tf.clip_by_value(noise, -noise_clip, noise_clip)
        return tf.clip_by_value(pi + noise, -self.act_limit, self.act_limit)

    #@tf.function
    def backward(self, state, loss_limit=None):
        while True:
            with tf.GradientTape() as tape:
                pi = self(state)
                # Gradient Ascent on Q(s,a)
                loss = -tf.reduce_mean(self.critic(state, pi))
            grads = tape.gradient(loss, self.model.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
            if loss_limit is None:
                return loss
            if loss < loss_limit:
                return loss


        