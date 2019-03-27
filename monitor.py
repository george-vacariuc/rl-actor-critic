import time
from runnable import Runnable
import tensorflow as tf
import numpy as np
import os
import pickle
import time
import math
from collections import deque

class Monitor(Runnable):
    def __init__(self, memory, explorers, env, policy, agents, monitoring_period=5):
        Runnable.__init__(self)
        self._memory = memory
        self._explorers = explorers
        self._env = env
        self._policy = policy
        self._agents = agents
        self._monitoring_period = monitoring_period
        self._shutdown_hook = None
        self._shutdown_criteria = None
        self._eval_rews = deque(maxlen=100)
        self._results_path = os.path.join(os.getcwd(), 'results', 'results_' + str(time.time()) + '.txt')

    def _run(self):
        last_samples = np.zeros(len(self._explorers))
        last_runs = np.zeros(len(self._agents))
        
        
        while not self._stop:
            self._evaluateAgent()
                    
            # Exploration metrics.
            total_samples = np.array([explorer.ticks for explorer in self._explorers])
            delta_samples = total_samples - last_samples
            print("### Environment samples: {}, avg: {}.".format(delta_samples, np.average(delta_samples)))
            last_samples = total_samples

            # Learning metrics.
            agent_runs = np.array([agent.ticks for agent in self._agents])
            delta_runs = agent_runs - last_runs
            print("### Agent runs: {}, avg: {}.".format(delta_runs, np.average(delta_runs)))
            last_runs = agent_runs

            if self._shutdown_criteria is None:
                raise Exception("Must set shutdown_criteria.")
                    
                    
            if self._shutdown_criteria(self._eval_rews):
                print("### Trigger shutdown.")
                if self._shutdown_hook is not None:
                    self._shutdown_hook()
            
            time.sleep(self._monitoring_period)

    def _play(self):
        state, done, total_reward = self._env.reset(), False, 0
        while not done:
            state, reward, done, _ = self._env.step(self._policy(tf.constant([state]))[0])
            total_reward += reward
        return total_reward

    def _evaluateAgent(self):
        test_runs = [self._play() for i in range(10)]
        self._eval_rews.extend(test_runs)
        
        test_runs = (np.min(test_runs), np.average(test_runs), np.max(test_runs))
        print("### _evaluateAgent rewards [%s], [%s], [%s]." % (test_runs))
    
        with open(self._results_path, "a+b") as fp:
            pickle.dump(test_runs, fp)

    @property
    def shutdown_hook(self):
        return self._shutdown_hook

    @shutdown_hook.setter
    def shutdown_hook(self, shutdown_hook):
        self._shutdown_hook = shutdown_hook

    @property
    def shutdown_criteria(self):
        return self._shutdown_criteria

    @shutdown_criteria.setter
    def shutdown_criteria(self, shutdown_criteria):
        self._shutdown_criteria = shutdown_criteria
        
        
    