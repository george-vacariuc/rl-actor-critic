import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from memory import ZmqMemory
import redis
from explorer import Explorer
from monitor import Monitor
from agent import Agent
from discrete_critic import DiscreteCritic as Critic
from dqn_actor import DqnActor as Actor
from memory import PriorityMemory as Memory
import tensorflow as tf
import numpy as np

is_discrete = True
env_name = 'CartPole-v0'

import gym
env = gym.make(env_name)
def make_env(): return gym.make(env_name)

obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.n
# TODO fix hack
act_dim = 1
num_discrete_actions = 2

memory = Memory(obs_dim=obs_dim, act_dim=act_dim, size=int(1e6),\
    act_dtype=np.uint8)

checkpoint_dir = os.path.join(os.getcwd(), 'checkpoint')

critic_0 = Critic(obs_dim=obs_dim, act_dim=num_discrete_actions,
                  checkpoint_file=os.path.join(checkpoint_dir, 'critic_0.h5'))
target_critic_0 = Critic(obs_dim=obs_dim, act_dim=num_discrete_actions, checkpoint_file=os.path.join(
    checkpoint_dir, 'target_critic_0.h5'))
critic_1 = Critic(obs_dim=obs_dim, act_dim=num_discrete_actions,
                  checkpoint_file=os.path.join(checkpoint_dir, 'critic_1.h5'))
target_critic_1 = Critic(obs_dim=obs_dim, act_dim=num_discrete_actions, checkpoint_file=os.path.join(
    checkpoint_dir, 'target_critic_1.h5'))


# Agent using two critics (https://arxiv.org/pdf/1802.09477.pdf).
agent = Agent(num_discrete_actions, [critic_0, critic_1], [
                     target_critic_0, target_critic_1], memory)

actor = Actor(num_discrete_actions, target_critic_0)
explorer_actor = Actor(num_discrete_actions, target_critic_0)

REDIS_HOST = os.environ.get('REDIS_HOST')
assert REDIS_HOST is not None
redis = redis.StrictRedis(host=REDIS_HOST, port=6379, db=0, health_check_interval=5)
#remote_memory = RedisMemory(redis)
remote_memory = ZmqMemory()

REWARDS_KEY = 'rew'
import struct
def episode_callback(episode):
    # Compute total reward.
    rewards = [t[2] for t in episode]
    episode_reward = np.sum(rewards)
    #if np.random.random_sample() < 0.05:
    print('### episode_reward: {}.'.format(episode_reward))
    redis.lpush(REWARDS_KEY, struct.pack('d', episode_reward))
    redis.ltrim(REWARDS_KEY, 0, 20)

def episode_counter(ignored):
    explorer_actor.inc_episode()

explorer = Explorer(remote_memory, policy=explorer_actor.forward_noisy,
                    env=make_env(), episode_callback=episode_counter, is_discrete=is_discrete)
zero_noise_explorer = Explorer(remote_memory, policy=explorer_actor.forward,
                    env=make_env(), episode_callback=episode_callback, is_discrete=is_discrete)


def fetch_rewards():
    rews = redis.lrange(REWARDS_KEY, 0, -1)
    return [struct.unpack('d', r) for r in rews]
monitor = Monitor(make_env(), policy=actor, agents=[agent], memory=memory, fetch_rewards=fetch_rewards)
