import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers

from base_model import BaseModel

class Actor(BaseModel):
    def __init__(self, obs_dim, act_dim, act_limit, critics, checkpoint_file=None):
        self.checkpoint_file = checkpoint_file
        self.act_limit = act_limit
        self.critics = critics
        self.optimizer = tf.optimizers.Adam(learning_rate=1e-3)

        inputs = keras.Input(shape=(None, obs_dim))
        x = layers.Dense(128, activation='relu')(inputs)
        x = layers.Dense(64, activation='relu')(x)
        outputs = layers.Dense(act_dim, activation='tanh')(x)
        self.model = keras.Model(inputs=inputs, outputs=outputs)
        self.model._name='actor'

    # Indicates which layers are shared with the critic.
    def get_shared_layers(self):
        return None

    def __call__(self, state):
        return self.forward(state)

    # Retuns the Action to take given the current state, aka the Policy.
    @tf.function
    def forward(self, state):
        return self.act_limit * self.model(state)

    # Same as forward() but with stochasticity.
    @tf.function
    def forwardNoisy(self, state, noise_clip = 0.5, target_noise = 0.2):
        pi = self.forward(state)
        noise = tf.random.normal(tf.shape(pi), stddev=target_noise)
        noise = tf.clip_by_value(noise, -noise_clip, noise_clip)
        return tf.clip_by_value(pi + noise, -self.act_limit, self.act_limit)

    @tf.function
    def backward(self, state, loss_limit=None):
        with tf.GradientTape() as tape:
            pi = self(state)
            q_sa = tf.math.add_n([critic(state, pi) for critic in self.critics])/len(self.critics)
            # Gradient Ascent on Q(s,a)
            loss = -tf.reduce_mean(q_sa)
        grads = tape.gradient(loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        return loss
