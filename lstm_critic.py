import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.keras as keras

from base_model import BaseModel
from critic import Critic

class LstmCritic(Critic):
    def __init__(self, obs_dim, act_dim, checkpoint_file=None):
        Critic.__init__(self, obs_dim, act_dim, checkpoint_file)
        # Model
        obs_input = keras.Input(shape=(obs_dim))
        act_input = keras.Input(shape=(act_dim))

        # TODO make it a param observation multiplier
        obs_mul = 4

        x = layers.Reshape((obs_mul, int(obs_dim/obs_mul)))(obs_input)
        y = layers.RepeatVector(obs_mul)(act_input)
        x = layers.concatenate([x, y])
        #x = layers.TimeDistributed(layers.Conv1D(32, 3), name='conv_1_shared')(x)
        #x = layers.TimeDistributed(layers.Conv1D(32, 3), name='conv_2_shared')(x)
        #x = layers.TimeDistributed(layers.Conv1D(64, 2), name='conv_3_shared')(x)
        #x = layers.TimeDistributed(layers.Conv1D(64, 1), name='conv_4_shared')(x)
        #x = layers.TimeDistributed(layers.Flatten())(x)
        #x = layers.TimeDistributed())()
        x = layers.LSTM(256, return_sequences = True)(x)
        x = layers.LSTM(128, return_sequences = True)(x)
        x = layers.LSTM(64)(x)

        #y = layers.Dense(32, activation='relu')(act_input)

        #x = layers.concatenate([x, y])
        #x = layers.Dense(64, activation='relu')(x)

        outputs = layers.Dense(1)(x)
        self.model = keras.Model(inputs=[obs_input, act_input], outputs=outputs)
        self.model._name='lstm_critic'
        #self.model.summary()

    @tf.function
    def forward(self, state, action):
        return self.model([state, action])
