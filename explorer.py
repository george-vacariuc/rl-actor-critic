import scipy.signal
import math
import tensorflow as tf
from runnable import Runnable

# An agent that explores the environment given a policy and collects memories.
class Explorer(Runnable):
    def __init__(self, memory, policy, env, max_steps=math.inf):
        Runnable.__init__(self)
        self._memory = memory
        self._policy = policy
        self._env = env
        self._max_steps = max_steps
        self._ticks = 0

    @staticmethod
    def discount_cumsum(x, discount=0.95):
        return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

    def _run(self):
        current_state = self._env.reset()
        temp_mem = []
        while (not self._stop) and (self.ticks < self._max_steps):
            # Compute the action using the given Policy
            action = self._policy(tf.constant([current_state]))[0]
            # Act.
            new_state, reward, done, _ = self._env.step(action)
            # Store in temporary memory for now.
            temp_mem.append((current_state, action, reward, new_state, done))
            if done or (self.ticks == self._max_steps - 1):
                # Compute eligibility traces.
                rewards = [t[2] for t in temp_mem]
                rewards = Explorer.discount_cumsum(rewards)
                disc_temp_mem = [(t[0], t[1], rewards[i], t[3], t[4]) for i, t in enumerate(temp_mem)]
                self._memory.store_buf(disc_temp_mem)
                temp_mem = []
                current_state = self._env.reset()
            else:
                current_state = new_state
            self.tick()    

        