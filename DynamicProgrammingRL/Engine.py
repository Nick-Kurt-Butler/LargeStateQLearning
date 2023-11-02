import numpy as np
import tensorflow as tf

class Engine:
    def __init__(self,Env):
        self.Env = Env

    def step(self,state):

        res = self.Env.df.iloc[self.Env.df[self.Env.df.STATE_0==state].Q.idxmax()]
        action = res.ACTION
        next_state = res.STATE_1
        done = res.TERMINAL
        reward = res.REWARD
        Q = res.Q

        return state,action,next_state,reward,done,Q

    def run(self,init_state, limit = 1000):
        """
        Run a simulation through the environment

        Parameters
        ----------
            init_state: The starting location of the agent
            limit = 1000: The limit of steps the agent can take to prevent infinite loop

        Returns
        -------
            A tuple (states,actions)
                states: All states agent traversed
                actions: The actions the agent made at each state
        """
        # Initialize data arrays
        states,actions = ([init_state],[])
        state = init_state
        done = False

        # Main loop to move through environment
        while not done:
            # Step
            _,action,state,reward,done,Q = self.step(state)

            # Step
            states.append(state)
            actions.append(action)

            # Infinite Loop check
            if limit == 0: break
            limit -= 1

        states = np.array(states)
        actions = np.array(actions)

        return states, actions

    def train(self,iterations,gamma = .99):
        for i in range(iterations):
            Qmax = self.Env.df[["STATE_1"]].merge(self.Env.df.groupby('STATE_0').Q.max(),
                              left_on="STATE_1",
                              right_index=True,how="left").Q.fillna(self.Env.fail_penalty)
            self.Env.df.Q = self.Env.df.REWARD + gamma*Qmax*(1-self.Env.df.TERMINAL)
            print(f"Iteration:{i}, Avg Q-val:{self.Env.df.Q.mean()}")
