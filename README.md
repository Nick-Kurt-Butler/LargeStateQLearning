# Reinforcement Learning Agent for Discrete State-Action Spaces


## Overview

This project aims to solve the problem of navigating an agent through a discrete state-action space using three distinct techniques: Deep Q-Network (DQN), Deep Deterministic Policy Gradients (DDPG), and Dynamic Programming (DP) implemented in SQL using Snowpark. Each technique offers unique advantages and trade-offs, providing insights into their applicability based on the nature of the state-action space.

The problem setup is a discrete state space where an agent can change its acceleration in x and y directions by ±1. Here is a sample of an agent making its way to a goal. Reaching the goal rewards the agent, and going out of bounds or taking too long to navigate the space punishes the agent.

![sample_run](https://github.com/Nick-Kurt-Butler/LargeStateQLearning/blob/main/DynamicProgrammingRL/run.gif)

For each state, the agent has one of nine possible action choices. Each method (DQN, DDPG, and DP) creates a function that inputs state and outputs the best possible action.
![action space](https://github.com/Nick-Kurt-Butler/LargeStateQLearning/blob/main/img/action_space.png)

## Techniques

### 2. Deep Q-Network (DQN)

DQN is a relatively simple neural network implementation to solve the problem. The idea is to have the neural network output Q values for each possible action given an input state. From there, the action with the maximum associated Q-value is used as the best possible choice.
![action space](https://github.com/Nick-Kurt-Butler/LargeStateQLearning/blob/main/img/dqn_vis.png)


This method demonstrated success in navigating the agent through the discrete state-action space. Its effectiveness, coupled with relative simplicity, makes DQN a favorable choice, especially in scenarios with large state spaces and ample computational power. However, this method is not without its problems. Given the random initialization of the Neural Net, the DQN algorithm would often find a local minimum instead of the global minimum. The result was that instead of the runs looking like the first row of examples, they resulted in the runs shown in the second row. This is because the model found the out-of-bounds "local minimum" instead of the out-of-bounds paired with the absolute goal "global minimum".

<p float="left">
  <img src="https://github.com/Nick-Kurt-Butler/LargeStateQLearning/blob/main/img/run1.png" width="240" />
  <img src="https://github.com/Nick-Kurt-Butler/LargeStateQLearning/blob/main/img/run2.png" width="240" /> 
  <img src="https://github.com/Nick-Kurt-Butler/LargeStateQLearning/blob/main/img/run3.png" width="240" />
  <img src="https://github.com/Nick-Kurt-Butler/LargeStateQLearning/blob/main/img/run4.png" width="240" />
</p>

<p float="left">
  <img src="https://github.com/Nick-Kurt-Butler/LargeStateQLearning/blob/main/img/run_bad1.png" width="240" />
  <img src="https://github.com/Nick-Kurt-Butler/LargeStateQLearning/blob/main/img/run_bad2.png" width="240" /> 
  <img src="https://github.com/Nick-Kurt-Butler/LargeStateQLearning/blob/main/img/run_bad3.png" width="240" />
  <img src="https://github.com/Nick-Kurt-Butler/LargeStateQLearning/blob/main/img/run_bad4.png" width="240" />
</p>

The training trends also confirm this. The figure on the left shows successful training of the Neural Net, and the figure on the right shows training resulting in the model finding a local minimum, not the global minimum. With both, the loss decreases and reaches a steady state as we expect it to in the presence of a local or global minimum. The difference is that the successful model reaches a steady state for average Q-value, whereas the other does not. The poorly trained model seems to fluctuate on what it determines for the true Q-values, yet because the loss has reached a steady state, the model seems to be unable to "climb" back up the hill to determine the true Q-values.

One method I found to drastically help with the model being stuck in local minimums was to limit Q-values from becoming too large. In Q-learning, the true Q-value for any given state-action pair should never be larger than the greatest reward or greatest punishment. So, if an updated Q-value ever exceeds the maximum reward, I instead told the model that the Q-value for that iteration should instead be half of the greatest reward. This kept the Q-values from becoming very large, and if there was an accident in the model where a state-action was getting too much attention, this helped the model to keep exploring instead of getting stuck in the same loop.

<p float="left">
  <img src="https://github.com/Nick-Kurt-Butler/LargeStateQLearning/blob/main/img/dqn_results_good.png" title="Good" width="488" />
  <img src="https://github.com/Nick-Kurt-Butler/LargeStateQLearning/blob/main/img/dqn_results_bad.png" title="Bad" width="480" />
</p>

Overall, the DQN algorithm proved to be successful. The main problem with any of these approaches in Q-learning is that it takes quite a few iterations for the reward to "leak" into the entirety of the state space. But for even large state spaces, DQN is a good model to use to simplify complexity, being a black box with state input and action output.

### 1. Deep Deterministic Policy Gradients (DDPG)

DDPG is supposedly a better implementation of DQN; however, it exhibited certain limitations. Despite its potential, DDPG proved to be the least effective among the three techniques. It required extensive training, often getting stuck in local minimums within the neural network, making it challenging to escape. While DDPG can be a powerful approach, caution is advised for scenarios with complex state-action spaces.

The diagram for DDPG is very similar to DQN with the main difference being that there are Neural Networks.

![action space](https://github.com/Nick-Kurt-Butler/LargeStateQLearning/blob/main/img/ddpg_vis.png)

### 3. Dynamic Programming in SQL using Snowpark

The Dynamic Programming approach, implemented in SQL using Snowpark, emerged as the most efficient and quickest solution. This technique is well-suited for manageable state-action spaces, providing a reliable option for scenarios where speed is crucial. Dynamic Programming, while not as versatile as neural network-based approaches, proves to be highly effective in specific contexts of reasonable state-action space size.

The main requirement of using this algorithm is being able to have a table with all possible state-action pairs. If this is feasible, then this algorithm I discovered is the fastest way to find the Q-values for discrete action spaces using reinforcement Q-learning. Using the SQL shown below, iteration of this code results in a fast convergence of the true Q-values.

![action space](https://github.com/Nick-Kurt-Butler/LargeStateQLearning/blob/main/img/dp_vis.png)

This method will work every time without failure but can only be used on reasonably small state-action spaces.
