# Records episode video.

import tensorflow as tf
from pyvirtualdisplay import Display
display = Display(visible=0, size=(1400, 900))
display.start()

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, './td3') 

from blueprint import env
from blueprint import actor
import shutil, os.path

from gym import wrappers

actor.restore()

video = './video'

total_reward = 0
for i in range(10):
    if os.path.exists(video):
        shutil.rmtree(video)
    eval_env = wrappers.Monitor(env, video)
    state, done, ep_ret, ep_len = eval_env.reset(), False, 0, 0
    total_reward = 0
    while not done:
        state, reward, done, _ = eval_env.step(actor(tf.constant([state]))[0])
        total_reward += reward
    eval_env.close()
    if total_reward > 20:
        break
    
print("## Done, total_reward: {}.".format(total_reward))
