import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
import math

from base_model import BaseModel

class DqnActor(BaseModel):
    def __init__(self, act_dim, critic):
        self.checkpoint_file = None
        self._act_dim = act_dim
        self.model = critic.model
        self._epsilon=1.0
        self._epsilon_min=0.01
        self._epsilon_log_decay=0.995
        self._episode = 0

    def inc_episode(self):
        self._episode += 1
        #print('### episode {}'.format(self._episode))
        #print('### self._epsilon {}'.format(self._epsilon))

    def __call__(self, state):
        return self.forward(state)

    # Retuns the Action to take given the current state, aka the Policy.
    #@tf.function
    def forward(self, state):
        # Q(s) for each Action
        q_s_a = self.model(state)
        a = tf.math.argmax(q_s_a, 1)
        #print('### q_s_a: {}, a: {}'.format(q_s_a, a))
        return a

    def forward_noisy(self, state):
        self._epsilon = max(self._epsilon_min, min(self._epsilon, 1.0 - math.log10((self._episode * 1e-4 + 1) * self._epsilon_log_decay)))
        return [tf.constant(np.random.randint(self._act_dim))] if (np.random.random() <= self._epsilon) else self.forward(state)

    def backward(self, state):
        raise Exception('Not implemented.')
