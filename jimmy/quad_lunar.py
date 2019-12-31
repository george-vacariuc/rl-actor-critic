import gym
from collections import deque

class QuadLunar():
    def __init__(self, env_name):
        self._env = gym.make(env_name)
        self._obs_dim = 4 * self._env.observation_space.shape[0]
        self._new_state = deque(maxlen=self._obs_dim)
        for i in range(self._obs_dim):
            self._new_state.append(0)

    def obs_dim(self):
        return self._obs_dim

    def act_dim(self):
        return self._env.action_space.shape[0]

    def act_limit(self):
        return self._env.action_space.high[0]

    def reset(self):
        for i in range(self._obs_dim):
            self._new_state.append(0)
        reset_state = self._env.reset()
        self._new_state.extend(reset_state)
        return list(self._new_state)

    def step(self, action):
        new_state, reward, done, info = self._env.step(action)
        self._new_state.extend(new_state)
        #print('### _new_state: {}, state: {}.'.format(list(self._new_state), new_state))
        return list(self._new_state), reward, done, info

    def render(self, mode):
        return self._env.render(mode)

    @property
    def action_space(self):
        return self._env.action_space

    @property
    def observation_space(self):
        return self._env.observation_space

    @property
    def reward_range(self):
        return self._env.reward_range

    @property
    def metadata(self):
        return self._env.metadata

    @property
    def spec(self):
        return self._env.spec

