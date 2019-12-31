import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from base_model import BaseModel

class Critic(BaseModel):
    def __init__(self, obs_dim, act_dim, checkpoint_file=None):
        self.checkpoint_file = checkpoint_file
        self.optimizer = tf.optimizers.Adam(learning_rate=1e-3)
        # Model
        inputs = keras.Input(shape=(None, obs_dim + act_dim))
        x = layers.Dense(128, activation='relu')(inputs)
        x = layers.Dense(64, activation='relu')(x)
        outputs = layers.Dense(1)(x)
        self.model = keras.Model(inputs=inputs, outputs=outputs)

    def __call__(self, state, action):
        return self.forward(state, action)

    @tf.function
    def forward(self, state, action):
        # This is Q(s, a).
        return self.model(tf.concat([state, action], axis=-1))

    @tf.function
    def backward(self, state, action, targets, loss_limit=None):
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean(tf.square(self(state, action) - targets))
        grads = tape.gradient(loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        return loss
