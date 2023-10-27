import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

class Environment:
	def __init__(self,size, goal_range, goal_reward, punishment, time_penalty, timestep):
		self.size = size # size of the map in length units
		self.goal_range = goal_range
		self.goal_reward = goal_reward # reward for reaching goal
		self.punishment = punishment # punishment for going out of bounds
		self.time_penalty = time_penalty # punishment for each time step completed
		self.out_of_bound_funcs = [] # Functions return if agent is out of bounds
		self.t = timestep

	def out_of_bounds(self,state):
		X = state[:,0]
		return (X < 0)|(self.size < X)

	def goal_reached(self,state):
		X = state[:,0]
		return (self.goal_range[0] <= X)&(X <= self.goal_range[1])

	def done(self,state):
		return self.goal_reached(state) | self.out_of_bounds(state)

	def step(self,state,action):
		x,v,a = np.concatenate([state,action],axis=1).T
		return np.array([x+v*self.t+.5*a*self.t**2,v+a*self.t]).T

	def reward(self,state):
		if len(state) == 0: return np.array([])
		rewards = self.time_penalty*np.ones(state.shape[0],dtype=np.float32)
		rewards[self.out_of_bounds(state)] = self.punishment
		rewards[self.goal_reached(state)] = self.goal_reward
		return rewards

	def display(self,states):
		fig,ax = plt.subplots(1,1,figsize=(8,8))
		X = states[:,0]
		ax.axhspan(self.goal_range[0],self.goal_range[1])
		plt.plot(np.linspace(self.t,self.t*len(X),len(X)),X)
		plt.xlabel("Time")
		plt.ylabel("Position")
		plt.ylim(0,self.size)
