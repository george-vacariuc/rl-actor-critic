import scipy.signal
import math
import tensorflow as tf

# An environment simulator that uses a NN model to fake the real env.
class EnvSim():
    # reset_states_memory - a memory containing valid RESET states sampled from the real environment.
    # env_model - a (NN) model of the env 
    def __init__(self, reset_states_memory, env_model):
        self._reset_states_memory = reset_states_memory
        self._env_model = env_model
        self._env_current_state = None

    def step(self, action):
        print("### EnvSim step Entered.")
        self._env_current_state, reward, done = self._env_model(self._env_current_state, action)
        print("### EnvSim step Exited.")
        return self._env_current_state, reward, done, None

    def reset(self):
        print("### EnvSim reset Entered.")
        # Sample a state from the states we know can be encountered at start-up on the original env.
        self._env_current_state, _, _, _, _ = self._reset_states_memory.sample_batch(1)
        print("### EnvSim reset Exited.")
        