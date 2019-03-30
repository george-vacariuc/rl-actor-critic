import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

import gym
env_name = 'LunarLanderContinuous-v2'
#env_name = 'BipedalWalker-v2'

env = gym.make(env_name)
make_env = lambda : gym.make(env_name)

obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]
# @TODO do not assume it is the same for all actions.
act_limit = env.action_space.high[0]

from memory import SimpleMemory
memory = SimpleMemory(obs_dim=obs_dim, act_dim=act_dim, size=int(2e6))

# For large memories that can be shared across machines.
#from memory import LevelDbMemory
#memory = LevelDbMemory(obs_dim=obs_dim, act_dim=act_dim, db_path='./td3_leveldb_memory', max_size=int(2e6))

import os
checkpoint_dir = os.path.join(os.getcwd(), 'checkpoint')

from critic import Critic
critic_0 = Critic(obs_dim=obs_dim, act_dim=act_dim, checkpoint_file=os.path.join(checkpoint_dir, 'critic_0.h5'))
target_critic_0 = Critic(obs_dim=obs_dim, act_dim=act_dim, checkpoint_file=os.path.join(checkpoint_dir, 'target_critic_0.h5'))
critic_1 = Critic(obs_dim=obs_dim, act_dim=act_dim, checkpoint_file=os.path.join(checkpoint_dir, 'critic_1.h5'))
target_critic_1 = Critic(obs_dim=obs_dim, act_dim=act_dim, checkpoint_file=os.path.join(checkpoint_dir, 'target_critic_1.h5'))


from actor import Actor
actor = Actor(obs_dim=obs_dim, act_dim=act_dim, act_limit=act_limit, critic=critic_0,
              checkpoint_file=os.path.join(checkpoint_dir, 'actor.h5'))
# Agent using two critics (https://arxiv.org/pdf/1802.09477.pdf).
from td3_agent import Td3Agent
td3_agent = Td3Agent([critic_0, critic_1], [target_critic_0, target_critic_1], actor, memory)

from explorer import Explorer
# Random walk explorer, used to bootstrap the learning.
randomPolicyExplorer = Explorer(memory, policy=lambda state: [env.action_space.sample()], env=env, max_steps=int(1e4))
# Explorers that gather experience using the learned policy.
POLICY_EXPLORERS = 2
learnedPolicyNoisyExplorers = [Explorer(memory, policy=actor.forwardNoisy, env=make_env()) for i in range(POLICY_EXPLORERS)]

from monitor import Monitor
monitor = Monitor(memory, learnedPolicyNoisyExplorers, env, policy=actor, agents=[td3_agent])

