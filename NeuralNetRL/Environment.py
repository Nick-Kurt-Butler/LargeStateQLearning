import numpy as np
import matplotlib.pyplot as plt

class Environment:
    def __init__(self,size, goal_reward, fail_penalty, time_penalty, outOfBoundsList, goalList):
        self.size = size # size of the map in length units
        self.goal_reward  = goal_reward # reward for reaching goal
        self.fail_penalty = fail_penalty # punishment for going out of bounds
        self.time_penalty = time_penalty # punishment for each time step completed
        self.outOfBoundsList = outOfBoundsList
        self.goalList = goalList

    def display(self):
        """
        Display Map
        """
        sx,sy = self.size
        env = np.ones([sx,sy,3]) # White

        for (x,y) in [[x,y] for x in range(sx) for y in range(sy)]:
            if (x,y) in self.goalList:
                env[x,y] = [0,1,0] # Green
            elif (x,y) in self.outOfBoundsList:
                env[x,y] = [0,0,0] # Black

        img = plt.imshow(np.swapaxes(env,0,1)[::-1], extent = [0,self.size[0],0,self.size[1]])
        img.set_cmap('hot')
        plt.xlim(0,self.size[0])
        plt.ylim(0,self.size[1])
        plt.axis('off')
        for i in range(sx+1):
            plt.axvline(i)
        for i in range(sy+1):
            plt.axhline(i)
        return img

    def valid_states(self):
        """
        Find all potential states in environment
        """
        states = []
        for px in range(self.size[0]):
            for py in range(self.size[1]):
                if (px,py) not in self.goalList and (px,py) not in self.outOfBoundsList:
                    for vx in range(-self.size[0],self.size[0]):
                        for vy in range(-self.size[1],self.size[1]):
                            states.append([px,py,vx,vy])

        return np.array(states)

    def step(self,state,action):
        """
        Update Agent in Environment

        Parameters
        ----------
            state: list like [px,py,vx,vy]
                px: x-coordinate position of agent
                py: y-coordinate position of agent
                vx: x-coordinate velocity of agent
                vy: y-coordinate velocity of agent
            action: int
                action for agent to take in range 0-8 showing direction of velocity
                +---+---+---+
                | 0 | 1 | 2 |
                +---+---+---+
                | 3 | 4 | 5 |
                +---+---+---+
                | 6 | 7 | 8 |
                +---+---+---+

        Returns
        -------
            tuple: (state,action,next_state,reward,done)
                state: list like [px,py,vx,vy]
                action: int
                next_state: list like [px,py,vx,vy]
                reward: float
                done: Boolean, True for reaching an end state

        Notes
        -----
        Reached Goal if
            1. Has reached a specified goal zone
                and
            2. Has a zero velocity on landing
        Penalty Incurred if
            1. Outside the boundaries of the environment
                or
            2. In one of the specified out of bounds zones
                of
            3. Agent has zero velocity and chose not to move
        Done if
            1. Goal Reached
                or
            2. Penalty Inccured
        """
        # Calc Next State
        px0,py0,vx0,vy0 = state
        ax = action % 3 - 1
        ay = 1 - action // 3
        vx1 = vx0 + ax
        vy1 = vy0 + ay
        px1 = px0 + vx1
        py1 = py0 + vy1
        next_state = [px1,py1,vx1,vy1]

        # Calc Reward and Done
        if ((px1,py1) in self.goalList and
            vx1==0 and vy1==0):
            reward = self.goal_reward
            done = True

        elif ((px1,py1) in self.outOfBoundsList or
                (px1 < 0) or (self.size[0] <= px1) or
                (py1 < 0) or (self.size[1] <= py1) or
                (ax==0 and ay==0 and vx0==0 and vy0==0)):
            reward = self.fail_penalty
            done = True

        else:
            reward = self.time_penalty
            done = False

        return state,action,next_state,reward,done

    def step_vec(self,states,actions):
        """
        Vectorized Version of Environment.step(state,action)

        Parameters
        ----------
            states: 2D numpy array holding multiple states
            action: 1D numpy array holding integer actions

        Returns
        -------
            tuple: (states,actions,next_states,rewards,dones)
                states: 2D numpy array holding multiple states
                actions: 1D numpy array type int
                next_states: 2D numpy array holding multiple states
                rewards: 1D numpy array type float
                dones: 1D numpy array type boolean
        """

        n = states.shape[0]

        # Calculate Next States
        ax = actions % 3 - 1
        ay = 1 - actions // 3
        vx = states[:,2] + ax
        vy = states[:,3] + ay
        px = states[:,0] + vx
        py = states[:,1] + vy

        next_states =  np.vstack([px,py,vx,vy]).T

        # Goal Reached
        gr = np.zeros(n).astype(bool)
        for (x,y) in self.goalList: gr |= (px == x) & (py == y)
        goal_reached = (gr&(np.abs(vx)<=1)&(np.abs(vy)<=1)).astype(bool)

        # Failure
        oob = np.zeros(n).astype(bool)
        for (x,y) in self.outOfBoundsList: oob |= (px == x) & (py == y)
        oobx = (px < 0) | (self.size[0] <= px)
        ooby = (py < 0) | (self.size[1] <= py)
        not_moving = (ax==0) & (ay==0) & (vx == 0) & (vy == 0)
        fail = (oob|oobx|ooby|not_moving).astype(bool)

        # Calculate Rewards
        rewards = np.ones(n)*self.time_penalty
        rewards[goal_reached.astype(bool)] = self.goal_reward
        rewards[fail.astype(bool)] = self.fail_penalty

        # Calculate Dones
        dones = goal_reached|fail

        return states,actions,next_states,rewards,dones
