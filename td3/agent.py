import tensorflow as tf
import numpy as np
import os
import time
import traceback, sys
from runnable import Runnable

class Td3Agent(Runnable):
    def __init__(self, critics, target_critics, actor, memory):
        Runnable.__init__(self)
        self._critics = critics
        self._target_critics = target_critics
        self._actor = actor
        self._memory = memory
        self._save_frequency = 1010
        self._update_target_frequency = 5
        self._gamma=0.99
        self._batch_size=128
        self._shared_layers = self._actor.get_shared_layers()

    def _run(self):
        while not self._stop:
            self._updateActorCritic()
            # Once in a while update the targets.
            if self.ticks % self._update_target_frequency == 0:
                self._updateTargetCritic()
            # Save checkpoints.
            if self.ticks % self._save_frequency == 0:
                self.save()
            self.tick()

    def _updateActorCritic(self):
        obs1, acts, rews, obs2, done = self._memory.sample_batch(self._batch_size)

        rews = tf.expand_dims(tf.constant(rews), -1)
        done = tf.expand_dims(tf.constant(done), -1)

        # TODO push this into a @tf.function
        future_acts = self._actor.forwardNoisy(obs2)
        q_pi_targ = tf.minimum(self._target_critics[0](obs2, future_acts),
                               self._target_critics[1](obs2, future_acts))
        target_Q_values = rews + self._gamma*(1-done)*q_pi_targ

        # Critic update.
        self._critics[0].backward(obs1, acts, target_Q_values)
        self._transfer_weights(self._critics[0], self._critics[1])
        self._critics[1].backward(obs1, acts, target_Q_values)
        self._transfer_weights(self._critics[1], self._actor)

        # Actor update
        self._actor.backward(obs1)
        self._transfer_weights(self._actor, self._critics[0])

    def _transfer_weights(self, src, dst):
        for layer_name in self._shared_layers:
            w = src.model.get_layer(name=layer_name).get_weights()
            dst.model.get_layer(name=layer_name).set_weights(w)

    def _updateTargetCritic(self):
        self._target_critics[0].imitate(self._critics[0])
        self._target_critics[1].imitate(self._critics[1])

    # Save model checkpoints.
    def save(self):
        for critic in self._critics:
            critic.save()
        for critic in self._target_critics:
            critic.save()
        self._actor.save()

    # Restore model checkpoints.
    def restore(self):
        for critic in self._critics:
            critic.restore()
        for critic in self._target_critics:
            critic.restore()
        self._actor.restore()
