import numpy as np
import matplotlib.pyplot as plt

class Environment:
	def __init__(self,size, goal_reward, punishment, time_penalty, outOfBoundsList, goalList):
		self.size = size # size of the map in length units
		self.goal_reward = goal_reward # reward for reaching goal
		self.punishment = punishment # punishment for going out of bounds
		self.time_penalty = time_penalty # punishment for each time step completed
		self.outOfBoundsList = outOfBoundsList
		self.goalList = goalList

	def goal_reached(self,state):
		px,py,vx,vy = state
		#if (px,py) in self.goalList and vx==0 and vy==0: return True
		if (px,py) in self.goalList: return True
		else: return False

	def failure(self,state):
		px,py,vx,vy = state
		if (px < 0): return True
		elif (py < 0): return True
		elif (px >= self.size[0]): return True
		elif (py >= self.size[1]): return True
		elif (px,py) in self.outOfBoundsList: return True
		else: return False

	def done_f(self,state):
		if self.goal_reached(state) or self.failure(state): return True
		else:return False

	def done(self,state):
		return np.apply_along_axis(self.done_f,1,state)

	def find_valid_starts(self):
		starts = []
		for i in range(self.size[0]):
			for j in range(self.size[1]):
				if not self.done_f([i,j,0,0]):
					starts.append((i,j,0,0))
		return np.array(starts)

	def step_f(self,state_action):
		px,py,vx,vy,a = state_action
		ax = a//3 - 1
		ay = a% 3 - 1
		return [px+vx+ax,py+vy+ay,vx+ax,vy+ay]
    
	def step(self,state,action):
		state_action = np.concatenate([state,np.expand_dims(action,1)],axis=1)
		return np.apply_along_axis(self.step_f,1,state_action)

	def reward_f(self,state,action,nstate):
		if action == 4 and state[2] == 0 and state[3] == 0: return self.punishment
		elif self.goal_reached(nstate): return self.goal_reward
		elif self.failure(nstate): return self.punishment
		else: return self.time_penalty

	def reward(self,states,actions,nstates):
		return np.array([self.reward_f(s,a,ns) for (s,a,ns) in zip(states,actions,nstates)])

	def display(self):
		sx,sy = self.size
		box = np.ones(self.size)
		for x in range(sx):
			for y in range(sy):
				if self.goal_reached((x,y,0,0)):
					box[x,y] = .8
				elif self.failure((x,y,0,0)):
					box[x,y] = 0

		img = plt.imshow(box.T[::-1], extent = [0,self.size[0],0,self.size[1]])
		img.set_cmap('hot')
		plt.xlim(0,self.size[0])
		plt.ylim(0,self.size[1])
		return img
