import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.keras as keras

from base_model import BaseModel

class DiscreteCritic(BaseModel):
    def __init__(self, obs_dim, act_dim, checkpoint_file=None):
        self.checkpoint_file = checkpoint_file
        # clipvalue=1, , decay=1e-3
        self.optimizer = tf.optimizers.Adam(learning_rate=1e-3)
        # Model
        obs_input = keras.Input(shape=(obs_dim))

        # TODO make it a param observation multiplier
        #obs_mul = 1
        #x = layers.Reshape((obs_mul, int(obs_dim/obs_mul)))(obs_input)
        #x = layers.LSTM(24, return_sequences = True)(x)
        #x = layers.LSTM(128, return_sequences = True)(x)
        #x = layers.LSTM(24)(x)

        x = layers.Dense(24, activation='relu')(obs_input)
        x = layers.Dense(24, activation='relu')(x)

        # Output Q(s) for each of the actions
        outputs = layers.Dense(act_dim)(x)
        self.model = keras.Model(inputs=[obs_input], outputs=outputs)
        self.model._name='discrete_critic'
        #self.model.summary()

    def __call__(self, state):
        return self.forward(state)

    @tf.function
    def forward(self, state):
        return self.model([state])

    @tf.function
    def backward(self, state, targets, mask, b_ISWeights):
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean(b_ISWeights * tf.square(\
                tf.reduce_sum(self(state)*mask, 1) - targets))
        grads = tape.gradient(loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        return loss
