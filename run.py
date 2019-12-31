import tensorflow as tf
import numpy as np
import concurrent.futures
import threading

tf.keras.backend.clear_session()

import traceback, sys

# This is where we select the agent we want to use.
sys.path.insert(0, './dqn')
#sys.path.insert(0, './td3')
from blueprint import env, make_env
from blueprint import memory
from blueprint import remote_memory
from blueprint import monitor
from blueprint import actor, explorer_actor
from blueprint import agent
from blueprint import explorer
from blueprint import redis
from blueprint import is_discrete
import sys
import pyarrow

seed=None
np.random.seed(seed)
from time import sleep

if is_discrete:
    from blueprint import zero_noise_explorer

import zmq
context = zmq.Context()
ACTION_CHECKPOINT_PUBSUB_PORT = 5556
ACTION_CHECKPOINT_PUB = "tcp://*:{}".format(ACTION_CHECKPOINT_PUBSUB_PORT)
ACTION_CHECKPOINT_SUB = "tcp://localhost:{}".format(ACTION_CHECKPOINT_PUBSUB_PORT)

def explorer_sync():
    socket = context.socket(zmq.SUB)
    socket.connect(ACTION_CHECKPOINT_SUB)
    socket.setsockopt_string(zmq.SUBSCRIBE, '')
    socket.setsockopt(zmq.RCVTIMEO, 100)


    while True:
        if is_shutting_down:
            break
        try:
            message = socket.recv()
        except zmq.Again:
            #traceback.print_exc(file=sys.stdout)
            continue

        explorer_actor.model.set_weights(pyarrow.deserialize(message))

def learner_sync():
    socket = context.socket(zmq.PUB)
    socket.bind(ACTION_CHECKPOINT_PUB)
    while True:
        if is_shutting_down:
            break
        remote_memory.pull_remote_memory_to_local_memory(memory)
        socket.send(pyarrow.serialize(actor.model.get_weights()).to_buffer().to_pybytes())


def main(mode):
    # mode: both, learner, explorer
    print('### main Entered.')
    print('### Mode {}.'.format(mode))

    if mode == 'both':
        run_explorer()
        run_learner()

    if mode == 'explorer':
        run_explorer()

    if mode == 'learner':
        run_learner()


def run_explorer():
    threading.Thread(target=explorer.run).start()
    if is_discrete:
        threading.Thread(target=zero_noise_explorer.run).start()
    threading.Thread(target=explorer_sync).start()

def run_learner():
    threading.Thread(target=learner_sync).start()

    print('### Memory size: [%s].' % (memory.size))
    while memory.size < 1e3:
        # Wait to gather some experience.
        sleep(2)
        print('### Need more experience. Memory size: [%s].' % (memory.size))

    monitor.shutdown_criteria = lambda rews : len(rews) > 20 and np.average(rews) > 250
    monitor.shutdown_hook = shutdown
    agent.restore()

    threading.Thread(target=agent.run).start()
    threading.Thread(target=monitor.run).start()

def shutdown():
    print('### shutdown Entered.')
    agent.stop()
    monitor.stop()
    print('### shutdown Exited.')

if __name__ == '__main__':
    is_shutting_down = False
    mode = sys.argv[1] if len(sys.argv) > 1 else 'both'
    try:
        main(mode)
    except KeyboardInterrupt:
        is_shutting_down = True
        shutdown()
