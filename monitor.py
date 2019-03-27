import time
from runnable import Runnable
import tensorflow as tf
import numpy as np
import os
import pickle
import time
import math

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
        self._results_path = os.path.join(os.getcwd(), 'results', 'results_' + str(time.time()) + '.txt')

    def _run(self):
        last_samples = np.zeros(len(self._explorers))
        last_runs = np.zeros(len(self._agents))
        while not self._stop:
            avg = self._evaluateAgent()[0]

            if avg > 250:
                print("### Trigger shutdown.")
                if self._shutdown_hook is not None:
                    self._shutdown_hook()
                    
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
            
            time.sleep(self._monitoring_period)

    def _play(self):
        state, done, total_reward = self._env.reset(), False, 0
        while not done:
            state, reward, done, _ = self._env.step(self._policy(tf.constant([state]))[0])
            total_reward += reward
        return total_reward

    def _evaluateAgent(self):
        test_runs = [self._play() for i in range(10)]
        test_runs = (np.min(test_runs), np.average(test_runs), np.max(test_runs))
        print("### _evaluateAgent rewards [%s], [%s], [%s]." % (test_runs))
    
        with open(self._results_path, "a+b") as fp:
            pickle.dump(test_runs, fp)
        return test_runs    

    @property
    def shutdown_hook(self):
        return self._shutdown_hook

    @shutdown_hook.setter
    def shutdown_hook(self, shutdown_hook):
        self._shutdown_hook = shutdown_hook
    