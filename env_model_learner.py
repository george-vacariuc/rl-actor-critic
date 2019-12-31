import tensorflow as tf
import numpy as np
import os
import time
import traceback, sys
from runnable import Runnable

# Learns a model of the env (MDP).
class EnvModelLearner(Runnable):
    def __init__(self, env_model, memory):
        Runnable.__init__(self)
        self._env_model = env_model 
        self._memory = memory
        self._save_frequency = 200

    def _run(self):
        while not self._stop:
            self._sampleAndFit()
            
    def _save_checkpoint(self):
            if self.ticks % self._save_frequency == 0:
                self.save()          
            self.tick()    
            
    def _sampleAndFit(self):
        if self._memory.size < 10:
            print('### _sampleAndFit Not enough samples: {}'.format(self._memory.size))
            return
        def generator():
            while True:
                self._save_checkpoint()
                _, acts, rews, obs2, done = self._memory.sample_two_consecutive_samples(up_to=int(1e4))
                if len(done) < 2:
                    # We need 2 samples
                    continue
                if done[0]:
                    # The samples belong to different episodes.
                    continue
                # Experiment
                # TODO RM
                #tobeornot = np.random.random_sample()
                if abs(rews[0]) < 50 and abs(rews[1]) < 50:
                    #if tobeornot > 0.1:
                    continue                    
                yield ({'act_input': np.array([acts[1]]),\
                        'prev_obs_input': np.array([obs2[0]]),\
                        'prev_act_input': np.array([acts[0]]),\
                        'prev_rew_input': np.array([rews[0]]),\
                        'next_obs_input': np.array([obs2[1]])},\
                       {'reward_output': np.array([rews[1]])})

        self._env_model.fit_generator(generator())
        
    # Save model checkpoints.
    def save(self):        
        self._env_model.save()

    # Restore model checkpoints.
    def restore(self):            
        self._env_model.restore()
