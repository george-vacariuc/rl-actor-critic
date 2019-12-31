import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.keras as keras

from base_model import BaseModel
from actor import Actor

class LstmActor(Actor):
    def __init__(self, obs_dim, act_dim, act_limit, critics, checkpoint_file=None, noise_clip = 0.5, target_noise = 0.2):
        # TODO fix
        #Actor.__init__(self, obs_dim, act_dim, act_limit, critic, checkpoint_file, noise_clip, target_noise)
        Actor.__init__(self, obs_dim, act_dim, act_limit, critics, checkpoint_file)
        inputs = keras.Input(shape=(obs_dim))

        # TODO make it a param observation multiplier
        obs_mul = 4

        x = layers.Reshape((obs_mul, int(obs_dim/obs_mul)))(inputs)
        #x = layers.TimeDistributed(layers.Conv1D(32, 3, 2), name='conv_1_shared')(x)
        #x = layers.TimeDistributed(layers.Flatten())(x)
        #x = layers.TimeDistributed(layers.Dense(64, activation='relu'))(x)
        x = layers.LSTM(256, return_sequences = True)(x)        
        x = layers.LSTM(128, return_sequences = True)(x)
        x = layers.LSTM(64)(x)

        outputs = layers.Dense(act_dim, activation='tanh')(x)
        self.model = keras.Model(inputs=inputs, outputs=outputs)
        self.model._name='lstm_actor'
        self.model.summary()


    # Indicates which layers are shared with the critic.
    def get_shared_layers(self):
        #return ['conv_1_shared','conv_2_shared','conv_3_shared','conv_4_shared', 'gru_shared']
        return []
