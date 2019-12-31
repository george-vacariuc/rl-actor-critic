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
from agent import Td3Agent
from lstm_actor import LstmActor as Actor
from lstm_critic import LstmCritic as Critic
from memory import SimpleMemory
import tensorflow as tf
import numpy as np

is_discrete = False
env_name = 'LunarLanderContinuous-v2'
#env_name = 'BipedalWalkerHardcore-v2'
#env_name = 'MountainCarContinuous-v0'

#env = gym.make(env_name)
#def make_env(): return gym.make(env_name)

from jimmy.quad_lunar import QuadLunar
env = QuadLunar(env_name)
def make_env(): return QuadLunar(env_name)

#obs_dim = env.observation_space.shape[0]
#act_dim = env.action_space.shape[0]
# @TODO do not assume it is the same for all actions.
#act_limit = env.action_space.high[0]

obs_dim = env.obs_dim()
act_dim = env.act_dim()
act_limit = env.act_limit()

memory = SimpleMemory(obs_dim=obs_dim, act_dim=act_dim, size=int(1e6))

# For large memories that can be shared across machines.
#from memory import LevelDbMemory
#memory = LevelDbMemory(obs_dim=obs_dim, act_dim=act_dim, db_path='./td3_leveldb_memory', max_size=int(2e6))

checkpoint_dir = os.path.join(os.getcwd(), 'checkpoint')

critic_0 = Critic(obs_dim=obs_dim, act_dim=act_dim,
                  checkpoint_file=os.path.join(checkpoint_dir, 'critic_0.h5'))
target_critic_0 = Critic(obs_dim=obs_dim, act_dim=act_dim, checkpoint_file=os.path.join(
    checkpoint_dir, 'target_critic_0.h5'))
critic_1 = Critic(obs_dim=obs_dim, act_dim=act_dim,
                  checkpoint_file=os.path.join(checkpoint_dir, 'critic_1.h5'))
target_critic_1 = Critic(obs_dim=obs_dim, act_dim=act_dim, checkpoint_file=os.path.join(
    checkpoint_dir, 'target_critic_1.h5'))


actor = Actor(obs_dim=obs_dim, act_dim=act_dim, act_limit=act_limit, critics=[critic_0, critic_1],
              checkpoint_file=os.path.join(checkpoint_dir, 'actor.h5'))
# Agent using two critics (https://arxiv.org/pdf/1802.09477.pdf).
agent = Td3Agent([critic_0, critic_1], [
                     target_critic_0, target_critic_1], actor, memory)

explorer_actor = Actor(obs_dim=obs_dim, act_dim=act_dim, act_limit=act_limit, critics=[],
                       checkpoint_file=os.path.join('/tmp', 'explorer_actor.h5'))

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
explorer = Explorer(remote_memory, policy=explorer_actor.forwardNoisy,
                    env=make_env(), episode_callback=episode_callback)

def fetch_rewards():
    rews = redis.lrange(REWARDS_KEY, 0, -1)
    return [struct.unpack('d', r) for r in rews]
monitor = Monitor(make_env(), policy=actor, agents=[agent], memory=memory, fetch_rewards=fetch_rewards)
