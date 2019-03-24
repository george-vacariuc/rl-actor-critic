# Deep Reinforcement Learning - Actor Critic implementation

A modular implementation of the "actor critic" algorithm. The goal is to serve as a foundation to enable plug and play experimentation with novel ideas.

* The system is modular, with reusable classes for actor, critic, memory, agent, etc. 
* Exploring, learning, evaluating, etc. run as separate threads ([learn.py](learn.py)). This makes it possible to distribute them on separate machines.
* The components are wired together in one place, [blueprint.py](blueprint.py).
* Uses Tensorflow2 with Keras, eager execution with graph inlining.
* Uses dual critic (TD3) and eligibility traces.

## Run

The *checkpoint* folder has pre-trained weights for testing.  

```
python3 learn.py
```

## Results
Works on OpenAI Gym LunarLander and BipedalWalker without parameter tunning.

<img src="https://github.com/george-vacariuc/rl/blob/master/results/td3.png" width="400px">

<img src="https://github.com/george-vacariuc/rl/blob/master/results/LunarLander.gif" alt="LunarLander" width="300px">

<img src="https://github.com/george-vacariuc/rl/blob/master/results/BipedalWalker.gif" alt="BipedalWalker" width="300px">




