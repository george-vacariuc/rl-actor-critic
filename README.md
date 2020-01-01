# Deep Reinforcement Learning - distributed modular framework

A modular framework to serve as a foundation to enable plug and play experimentation with RL ideas.

* The system is modular, with reusable classes for actor, critic, memory, agent, etc.
* Exploration and learning run as separate processes in separate Docker instances. This makes it possible to run multiple distributed explorers.
* The components are wired together in one place, eg [td3/blueprint.py](td3/blueprint.py), [dqn/blueprint.py](dqn/blueprint.py).
* Uses Tensorflow2 with eager execution.

<img src="https://github.com/george-vacariuc/rl-actor-critic/blob/master/img/RL.png" width="800px">

## Run

Launch two docker images, one running the Explorer and one running the Learner.
```
scripts/both.sh
```

## Results
<img src="https://github.com/george-vacariuc/rl-actor-critic/blob/master/img/td3.png" width="400px">

<img src="https://github.com/george-vacariuc/rl-actor-critic/blob/master/img/LunarLander.gif" alt="LunarLander" width="300px">

<img src="https://github.com/george-vacariuc/rl-actor-critic/blob/master/img/BipedalWalker.gif" alt="BipedalWalker" width="300px">
