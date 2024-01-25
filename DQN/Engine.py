import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation,PillowWriter

class Engine:
    def __init__(self,Env,Agent):
        self.Env = Env
        self.Agent = Agent

    def run(self,init_state, epsilon, limit = 20):
        """
        Run a simulation through the environment

        Parameters
        ----------
            init_state: The starting location of the agent
            epsilon: The probability of the agent choosing a random move
            limit = 1000: The limit of steps the agent can take to prevent infinite loop

        Returns
        -------
            A tuple (states,actions,next_states,rewards,dones)

            states: All states agent traversed
            actions: The actions the agent made at each state
            rewards: Reward received by agent at each step
            dones: Boolean array
        """
        # Initialize data arrays
        states,actions = ([init_state],[])
        state = init_state
        done = False

        # Main loop to move through environment
        while not done:
            # Get action
            if np.random.rand() < epsilon: action = np.random.randint(9)
            else: action = int(self.Agent.get_action([state]))

            # Step
            _,_,state,_,done = self.Env.step(state,action)
            states.append(state)
            actions.append(action)

            # Infinite Loop check
            if limit == 0: break
            limit -= 1

        states = np.array(states)
        actions = np.array(actions)

        return states, actions

    def train(self,epsilon, limit = 1000, batch_size = 1000, trials = 10):
        """
        Train Neural Net on Linear Runs

        Parameters
        ----------
            epsilon: The probability of the agent choosing a random move
            limit = 1000: The limit of steps the agent can take to prevent infinite loop
            batch_size: Size of each batch in the NN
            trials: Number of trials such that batch_size x trials = training points
        Returns
        -------
            A dictionary holding three arrays

            reward: array holding the reward of each state-action pair
            actor_loss: array holding the total loss of each epoch for the actor NN
            critic_loss: array holding the total loss of each epoch for the critic NN
        """
        # Initialize Metrics
        with open("train_NN.txt","w") as file: file.write("")
        metrics = {'reward':[],'loss':[]}
        lim = limit
        init_states = self.Env.valid_states()
        num_states = len(init_states)

        # Main loop for each trial
        for t in range(trials):

            # Initialize Parameters to train NN
            states = np.zeros([batch_size,4])
            next_states = np.zeros([batch_size,4])
            actions = np.zeros(batch_size).astype(np.int32)
            rewards = np.zeros(batch_size)
            dones = np.zeros(batch_size).astype(bool)

            done = True
            # Main loop to train NN
            for i in range(batch_size):
                # If done get new state
                if done: state = init_states[np.random.randint(num_states)]

                # Get action
                if np.random.rand() < epsilon: action = np.random.randint(9)
                else: action = int(self.Agent.get_action(np.expand_dims(state,0))[0])

                # Step
                states[i] = state
                actions[i] = action
                _,_,state,reward,done = self.Env.step(state,action)
                next_states[i] = state
                rewards[i] = reward
                dones[i] = done

                # Infinite loop protection
                if lim == 0:
                    done = True
                    lim = limit
                lim -= 1
            # Finalize Metrics and Update NN
            loss = self.Agent.update(states, actions, next_states, rewards, dones)
            metrics['reward'].append(np.sum(rewards)/(np.sum(dones)+1))
            metrics['loss'].append(loss)
            # Write progress to file
            with open("train_NN.txt","a") as file:
                file.write(f"Epoch: {t}, Avg Reward:{round(np.sum(rewards)/(np.sum(dones)+1),2)}, Loss:{loss}\n")

        return metrics


    def train_vec(self, batch_size, trials, epsilon):
        """
        Train Neural Net on Random Runs to maximize exploration

        Parameters
        ----------
            batch_size: Size of each batch in the NN
            trials: Number of trials such that batch_size x trials = training points
            epsilon: The probability of the agent choosing a random move

        Returns
        -------
        A dictionary holding three arrays
            reward: array holding the reward of each state-action pair
            actor_loss: array holding the total loss of each epoch for the actor NN
            critic_loss: array holding the total loss of each epoch for the critic NN
        """
        # Initialize metrics
        with open("train_vec_NN.txt","w") as file: file.write("")
        metrics = {'reward':[],'loss':[]}
        init_states = self.Env.valid_states()
        num_states = len(init_states)

        # Main loop to create each batch
        for t in range(trials):
            # Get data
            states = init_states[np.random.randint(num_states,size=batch_size)]
            actions = np.random.randint(9,size=batch_size)
            mask = np.random.choice([True,False],p=[1-epsilon,epsilon],size=batch_size)
            if epsilon < 1: actions[mask] = self.Agent.get_action(states[mask])
            _,_,next_states,rewards,dones = self.Env.step_vec(states,actions)

            # Update NN and metrics
            loss = self.Agent.update(states, actions, next_states, rewards, dones)
            metrics['reward'].append(np.sum(rewards)/(np.sum(dones)+1))
            metrics['loss'].append(loss)
            with open("train_vec_NN.txt","a") as file:
                file.write(f"Epoch: {t}, Avg Reward:{round(np.sum(rewards)/(np.sum(dones)+1),2)}, Loss:{loss}\n")

        return metrics

    def train_dynamic(self, trials, fileName):
        """
        Train Neural Net on Random Runs to maximize exploration

        Parameters
        ----------
            batch_size: Size of each batch in the NN
            trials: Number of trials such that batch_size x trials = training points
            epsilon: The probability of the agent choosing a random move

        Returns
        -------
        A dictionary holding three arrays
            reward: array holding the reward of each state-action pair
            actor_loss: array holding the total loss of each epoch for the actor NN
            critic_loss: array holding the total loss of each epoch for the critic NN
        """
        # Initialize metrics
        with open(fileName,"w") as file: file.write("Epoch,Loss,Q_model,Q_update\n")
        df = self.Env.df.sort_values(["STATE_0","ACTION"])[["STATE_0","STATE_1","REWARD","TERMINAL"]].groupby("STATE_0").aggregate(list)
        states = np.array(df.index.map(list).tolist())
        n = len(states)
        next_states = np.array(df.STATE_1.tolist())
        rewards  = np.array(df.REWARD.tolist())
        terminal = np.array(df.TERMINAL.tolist())

        index = np.array(range(n))

        MIN = self.Env.fail_penalty
        MAX = self.Env.goal_reward

        # Main loop to create each batch
        for t in range(trials):
            np.random.shuffle(index)
            loss,Q_model,Q_update = self.Agent.update_all(states[index], next_states[index], rewards[index], terminal[index],MIN,MAX)
            with open(fileName,"a") as file: file.write(f"{t},{loss},{Q_model},{Q_update}\n")

        return metrics

    def display_run(self,state,epsilon=0):

        fig,ax = plt.subplots()
        im = self.Env.display()
        states,actions = self.run(state,epsilon=epsilon)
        offset_states = np.concatenate([states[:-1,:2],states[1:,-2:]],axis=1)

        def animate(i):
            if i == 0:
                point, = ax.plot(state[0],state[1])
                return im,point
            else:
                px,py,vx,vy = offset_states[i-1]
                arrow = ax.arrow(px+.5,py+.5,vx,vy,head_width=.3,length_includes_head=True,color = 'blue')
                return im,arrow

        ani = FuncAnimation(fig, animate, interval=500, blit=True, repeat=True, frames=len(states))
        ani.save("run.gif", dpi=300, writer=PillowWriter(fps=3))
