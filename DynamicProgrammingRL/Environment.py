import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools


class Environment:
    def __init__(self,size, goal_reward, fail_penalty, time_penalty, outOfBoundsList, goalList):
        self.size = size # size of the map in length units
        self.goal_reward  = goal_reward # reward for reaching goal
        self.fail_penalty = fail_penalty # punishment for going out of bounds
        self.time_penalty = time_penalty # punishment for each time step completed
        self.outOfBoundsList = outOfBoundsList
        self.goalList = goalList
        self.init_Q()

    def init_Q(self):

        np.seterr(divide='ignore',invalid='ignore')
        sx,sy = self.size

        state_actions = np.array(list(itertools.product(*[
            np.arange(sx),
            np.arange(sy),
            np.arange(2*sx)-sx,
            np.arange(2*sy)-sy,
            np.arange(-1,2),
            np.arange(-1,2)
        ])))

        n = state_actions.shape[0]

        states = state_actions[:,:4]
        actions = state_actions[:,-2:]
        px0 = states[:,0]
        py0 = states[:,1]
        del state_actions

        # Calculate Next States
        ax = actions[:,0]
        ay = actions[:,1]
        vx = states[:,2] + ax
        vy = states[:,3] + ay
        px = px0 + vx
        py = py0 + vy

        next_states =  np.vstack([px,py,vx,vy]).T

        # Goal Reached
        gr = np.zeros(n).astype(bool)
        for (x,y) in self.goalList: gr |= (px == x) & (py == y)
        goal_reached = (gr&(np.abs(vx)<=1)&(np.abs(vy)<=1)).astype(bool)

        # Failure
        oob = np.zeros(n).astype(bool)
        m = (py0-py)/(px0-px)
        for (x,y) in self.outOfBoundsList:
            a = m*(x-.5-px)+py
            b = m*(x+.5-px)+py
            oob |= (px == x) & (py == y)
            x_in_between = ((np.min(np.array([px0,px]),axis=0)<=x) & (x<=np.max(np.array([px0,px]),axis=0)))
            y_in_between = ((np.min(np.array([py0,py]),axis=0)<=y) & (y<=np.max(np.array([py0,py]),axis=0)))
            oob |= ~(((a<=y-.5) & (b<=y-.5)) | ((y+.5<=a) & (y+.5<=b))) & x_in_between & y_in_between

        oobx = (px < 0) | (sx <= px)
        ooby = (py < 0) | (sy <= py)
        not_moving = (ax == 0) & (ay == 0) & (vx == 0) & (vy == 0)
        fail = (oob|oobx|ooby|not_moving).astype(bool)

        # Calculate Rewards
        rewards = np.ones(n)*self.time_penalty
        rewards[goal_reached.astype(bool)] = self.goal_reward
        rewards[fail.astype(bool)] = self.fail_penalty

        # Calculate Terminals
        terminals = goal_reached|fail

        # Create DataFrame
        self.df = pd.DataFrame()
        self.df["STATE_0"] = list(map(tuple,states))
        self.df["ACTION"] = list(map(tuple,actions))
        self.df["STATE_1"] = list(map(tuple,next_states))
        self.df["TERMINAL"] = terminals
        self.df["REWARD"] = rewards
        self.df["Q"] = np.zeros(n)

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

    def is_terminal(self,state):
        px,py,vx,vy = state
        if (px,py) in self.goalList and (vx,vy) == (0,0): return True
        elif (px,py) in self.outOfBoundsList: return True
        else: return False

    def valid_states(self,terminal=False):
        """
        Find all non-terminal states in environment
        """
        states = []
        for px in range(self.size[0]):
            for py in range(self.size[1]):
                if is_terminal(state) == terminal:
                    for vx in range(-self.size[0],self.size[0]):
                        for vy in range(-self.size[1],self.size[1]):
                            states.append([px,py,vx,vy])

        return np.array(states)
