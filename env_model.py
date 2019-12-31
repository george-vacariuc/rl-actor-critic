import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

from base_model import BaseModel

# A NN modeling the MDP.
class EnvModel(BaseModel):
    def __init__(self, obs_dim, act_dim, checkpoint_file=None, max_epochs=int(2e2)):
        self.checkpoint_file = checkpoint_file
        self._max_epochs = max_epochs
        # Model
        prev_obs_input = keras.Input(shape=(obs_dim,), name='prev_obs_input')
        prev_act_input = keras.Input(shape=(act_dim,), name='prev_act_input')
        prev_rew_input = keras.Input(shape=(1,), name='prev_rew_input')
        act_input = keras.Input(shape=(act_dim,), name='act_input')
        next_obs_input = keras.Input(shape=(obs_dim,), name='next_obs_input')
        x = keras.layers.Concatenate(axis=-1)([prev_obs_input, act_input, next_obs_input])
        x = layers.Dense(100, activation='relu')(x)
        x = layers.Dense(100, activation='relu')(x)
        ##
        #x = layers.BatchNormalization()(x)
        x = layers.Dense(20, activation='relu')(x)
        reward_output = layers.Dense(1, name='reward_output')(x)

        self.model = keras.Model(inputs=[prev_obs_input, prev_act_input, prev_rew_input, act_input, next_obs_input],\
                                 outputs=reward_output)
        self.model.compile(optimizer=tf.keras.optimizers.Adam(lr=3e-4),
              loss='mse',
              metrics=['mae'])

    def __call__(self, state, action):
        return self.forward(state, action)

    def forward(self, action, next_obs):
        return self.model.predict(tf.concat([action, next_obs], axis=-1))

    def fit(self, action, next_obs, reward_targets):
        print('### fit Entered.')
        self.model.fit([action, next_obs],\
                       [reward_targets],\
                       epochs=self._max_epochs,\
                       batch_size=32,\
                       verbose=1)
        print('### fit Exited.')

    def fit_generator(self, gen):
        self.model.fit_generator(gen,\
                       steps_per_epoch=100,
                       epochs=self._max_epochs,\
                       verbose=1)
