import tensorflow as tf
import numpy as np
import concurrent.futures

tf.keras.backend.clear_session()

from blueprint import env, make_env
from blueprint import actor, memory, monitor
from blueprint import td3_agent as agent
from blueprint import randomPolicyExplorer, learnedPolicyNoisyExplorers
seed=None
np.random.seed(seed)


def main():
    print('### main Entered.')
    # Gather some experience.
    randomPolicyExplorer.run()
    print('### mem size [%s]'%(memory.size))

    monitor.shutdown_criteria = lambda rews : len(rews) > 30 and np.average(rews) > 250
    monitor.shutdown_hook = shutdown
    agent.restore()
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        executor.submit(agent.run)
        executor.submit(monitor.run)
        for explorer in learnedPolicyNoisyExplorers:
            executor.submit(explorer._run)

def shutdown():
    print('### shutdown Entered.')
    agent.stop()
    for explorer in learnedPolicyNoisyExplorers:
        explorer.stop()
    monitor.stop()    
    print('### shutdown Exited.')

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        shutdown()