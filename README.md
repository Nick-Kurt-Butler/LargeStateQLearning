# Reinforcement Learning Agent for Discrete State-Action Spaces


## Overview

This project aims to solve the problem of navigating an agent through a discrete state-action space using three distinct techniques: Deep Q-Network (DQN), Deep Deterministic Policy Gradients (DDPG), and Dynamic Programming (DP) implemented in SQL using Snowpark. Each technique offers unique advantages and trade-offs, providing insights into their applicability based on the nature of the state-action space.

The problem setup is a discrete state space where an agent can change its acceleteration in x and y directions by $\pm 1$.  Here is a sample of an agent making its way to a goal.  Reaching the goal rewards the agent and going out of bounds or taking too long to navigate the space punishes the agent.

![sample_run](https://github.com/Nick-Kurt-Butler/LargeStateQLearning/blob/main/DynamicProgrammingRL/run.gif)

For each state the agent has 1 of 9 possible action choices.  Each method (DQN, DDPG, and DP) creates a function that inputs state and outputs best possible action.
![action space](https://github.com/Nick-Kurt-Butler/LargeStateQLearning/blob/main/img/action_space.png)

## Techniques

### 2. Deep Q-Network (DQN)

DQN, is a relatively simple neural network implementation to solve the problem.  The idea is to have the nueral network output Q values for each possible action given an input state.  From there the action with the maximum associated Q-value is used as the best possible choice.

![action space](https://github.com/Nick-Kurt-Butler/LargeStateQLearning/blob/main/img/dqn_vis.png)

This method demonstrated success in navigating the agent through the discrete state-action space. Its effectiveness, coupled with relative simplicity, makes DQN a favorable choice, especially in scenerios with large state spaces and ample computational power. However this method is not without its problems.  Given the random initialization of the Neural Net the DQN algorithm would often find a local minumum instead of the global minimum.  The result was instead of the runs looking like first row of examples, they resulted in the runs shown in the second row.  This is because the model found the out of bounds "local minimum" instead of the out of bounds paired with the absolute goal "global minimum".

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

The training trends also confirm this.  The firgure on the left shows a successful training of the Nueral Net, and the figure on the right shows a training resulting the model finding a local minimum not the global minimum.  With both the loss decreases and reaches a steady state as we expect it too in the presense of a local or global minimum.  The difference is that the successful model reaches a steady state for average Q-value whereas the other does not.  The poorly trained model seems to flucate on what it determines for the true Q-values, yet because the loss has reached a steady state the model seems to be unable to "climb" back up the hill to determine the true Q-values.

<p float="left">
  <img src="https://github.com/Nick-Kurt-Butler/LargeStateQLearning/blob/main/img/dqn_results_good.png" title="Good" width="488" />
  <img src="https://github.com/Nick-Kurt-Butler/LargeStateQLearning/blob/main/img/dqn_results_bad.png" title="Bad" width="480" />
</p>


### 1. Deep Deterministic Policy Gradients (DDPG)

DDPG was implemented to tackle the problem; however, it exhibited certain limitations. Despite its potential, DDPG proved to be the least effective among the three techniques. It required extensive training, often getting stuck in local minimums within the neural network, making it challenging to escape. While DDPG can be a powerful approach, caution is advised for scenarios with complex state-action spaces.

#### Graphs and Photos
Insert relevant graphs or visualizations depicting the performance and training progress of the DDPG algorithm.


#### Graphs and Photos
Include visual representations of the DQN algorithm's performance, such as training curves or action-value estimates.

### 3. Dynamic Programming in SQL using Snowpark

The Dynamic Programming approach, implemented in SQL using Snowpark, emerged as the most efficient and quickest solution. This technique is well-suited for manageable state-action spaces, providing a reliable option for scenarios where speed is crucial. Dynamic Programming, while not as versatile as neural network-based approaches, proves to be highly effective in specific contexts.

#### Graphs and Photos
Place graphs or images showcasing the efficiency and speed of the Dynamic Programming approach in SQL.

## Choosing the Right Technique

- **Dynamic Programming:** Recommended for manageable state-action spaces, prioritizing speed.
  
- **DQN:** Suited for large state-action spaces, balancing effectiveness and simplicity.

## Getting Started

Follow these instructions to set up and run the project on your local machine.

### Prerequisites

- Python 3.x
- Snowpark
- Additional dependencies (list them here)

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your_username/your_project.git
