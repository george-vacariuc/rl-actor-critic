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
    def __init__(self, env, policy, agents, memory, fetch_rewards, monitoring_period=5):
        Runnable.__init__(self)
        self._env = env
        self._policy = policy
        self._agents = agents
        self._memory = memory
        self._fetch_rewards = fetch_rewards
        self._monitoring_period = monitoring_period
        self._shutdown_hook = None
        self._shutdown_criteria = None
        self._eval_rews = deque(maxlen=100)
        self._results_path = os.path.join(
            os.getcwd(), 'results', 'results_' + str(time.time()) + '.txt')

    def _run(self):
        last_runs = np.zeros(len(self._agents))
        last_time = time.time()

        while not self._stop:
            time.sleep(self._monitoring_period)
            self._evaluateAgent()

            # Learning metrics.
            agent_runs = np.array([agent.ticks for agent in self._agents])
            delta_runs = agent_runs - last_runs
            now = time.time()
            delta_time = now - last_time
            print("### Agent qps: {}, avg: {}.".format(
                delta_runs / delta_time,
                np.average(delta_runs) / delta_time))
            last_runs = agent_runs
            last_time = now

            if self._shutdown_criteria is None:
                raise Exception("Must set shutdown_criteria.")

            if self._shutdown_criteria(self._eval_rews):
                print("### Trigger shutdown, agent_runs: {}, memory: {}."
                      .format(agent_runs, self._memory.lifetime_size))
                if self._shutdown_hook is not None:
                    self._shutdown_hook()

    def _evaluateAgent(self):
        test_runs = self._fetch_rewards()
        self._eval_rews.extend(test_runs)

        if len(test_runs) == 0:
            return
        test_runs = (np.min(test_runs), np.average(test_runs),
                     np.max(test_runs), self._memory.lifetime_size)
        print(
            "### _evaluateAgent rewards [%s], [%s], [%s], mem: [%s]." % (test_runs))

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
