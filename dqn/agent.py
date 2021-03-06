import tensorflow as tf
import numpy as np
import os
import time
import traceback, sys
from runnable import Runnable

class Agent(Runnable):
    def __init__(self, num_discrete_actions, critics, target_critics, memory):
        Runnable.__init__(self)
        self._critics = critics
        self._target_critics = target_critics
        self._memory = memory
        self._save_frequency = 1010
        self._update_target_frequency = 5
        self._gamma=0.99
        self._batch_size=128
        self._num_discrete_actions = num_discrete_actions

    def _run(self):
        while not self._stop:
            self._updateCritic()
            # Once in a while update the targets.
            if self.ticks % self._update_target_frequency == 0:
                self._updateTargetCritic()
            # Save checkpoints.
            if self.ticks % self._save_frequency == 0:
                self.save()
            self.tick()

    def _updateCritic(self):
        obs1, acts, rews, obs2, done, idxs, b_ISWeights = \
            self._memory.sample_batch(self._batch_size)

        rews = tf.expand_dims(tf.constant(rews), -1)
        done = tf.expand_dims(tf.constant(done), -1)
        mask = tf.reshape(tf.one_hot(acts, self._num_discrete_actions),\
            [self._batch_size, self._num_discrete_actions])

        q_pi_targ = tf.minimum(self._target_critics[0](obs2),
                               self._target_critics[1](obs2))
        #q_pi_targ = self._target_critics[0](obs2)
        target_Q_values = rews + self._gamma*(1-done)*q_pi_targ
        target_Q_values = tf.reduce_max(target_Q_values, 1)

        absolute_errors = tf.abs(\
            tf.reduce_sum(self._critics[0](obs1)*mask, 1) - target_Q_values)
        self._memory.batch_update(idxs, absolute_errors)

        b_ISWeights = tf.constant(b_ISWeights)
        # Critic update.
        self._critics[0].backward(obs1, target_Q_values, mask, b_ISWeights)
        self._critics[1].backward(obs1, target_Q_values, mask, b_ISWeights)

    def _updateTargetCritic(self):
        self._target_critics[0].imitate(self._critics[0])
        self._target_critics[1].imitate(self._critics[1])

    # Save model checkpoints.
    def save(self):
        for critic in self._critics:
            critic.save()
        for critic in self._target_critics:
            critic.save()

    # Restore model checkpoints.
    def restore(self):
        for critic in self._critics:
            critic.restore()
        for critic in self._target_critics:
            critic.restore()
