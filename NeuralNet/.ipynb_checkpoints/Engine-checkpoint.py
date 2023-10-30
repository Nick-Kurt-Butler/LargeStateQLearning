import numpy as np
import tensorflow as tf

class Engine:
	def __init__(self,Env,Agent):
		self.Env = Env
		self.Agent = Agent

	def run(self,init_state, explore_prob, limit = 1000):
		"""
		Run a simulation through the environment

		Parameters
		----------
			init_state: The starting location of the agent
			explore_prob: The probability of the agent choosing a random move
			limit = 1000: The limit of steps the agent can take to prevent infinite loop

		Returns
		-------
			A tuple (states,actions,next_states,rewards,dones)

			states: The states from 0 to n-1
			actions: The actions the agent made at each state
			next_states: The states from 1 to n
			rewards: Reward received by agent at each step
			dones: An array showing a 1 if at that step the agent completed moving throught he environment else 0
		"""
		# Initialize data arrays
		states,actions,next_states = ([],[],[])
		state = init_state

		# Main loop to move through environment
		while not self.Env.done_f(state):
			# Get action
			if np.random.rand() < explore_prob: action = np.random.randint(9,size=1)
			else: action = self.Agent.get_action([state])
			next_state = self.Env.step_f(np.concatenate((state,action)))

			# Update infomation for current step
			states.append(state)
			actions.append(action)
			next_states.append(next_state)
			state = next_state

			# Infinite Loop check
			if limit == 0: break
			limit -= 1

		# Finalize last parameters
		next_states = np.array(next_states)
		rewards = self.Env.reward(states,actions,next_states)
		dones = np.zeros(len(rewards))
		dones[-1] = 1
		states = np.array(states)
		actions = np.array(actions)

		return states, actions, next_states, rewards, dones

	def train(self,explore_prob, limit = 1000, batch_size = 1000, trials = 10):
		"""
		Train Neural Net on Linear Runs

		Parameters
		----------
			explore_prob: The probability of the agent choosing a random move
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
		metrics = {'reward':[],'actor_loss':[],'critic_loss':[]}
		lim = limit
		init_states = self.Env.find_valid_starts()
		len_init_states = len(init_states)
		state = init_states[np.random.randint(len_init_states)]

		# Main loop for each trial
		for t in range(trials):

			# Initialize Parameters to train NN
			states = np.zeros([buffer_cap,4])
			next_states = np.zeros([buffer_cap,4])
			actions = np.zeros([buffer_cap,9])
			dones = np.zeros(buffer_cap)

			# Main loop to train NN
			for i in range(buffer_cap):
				# If done get new state
				if self.Env.done_f(state):
					state = init_states[np.random.randint(len_init_states)]
					dones[i] = 1

				# Get action
				probs,action = self.Agent.get_action_probs([state])
				if np.random.rand() < explore_prob: action = np.random.randint(9,size=1)
				# Get next state from state-action pair
				next_state = self.Env.step_f(np.concatenate((state,action)))

				# Update data for current step
				states[i] = state
				actions[i] = probs
				next_states[i] = next_state
				state = next_state

				# Infinite loop protection
				if lim == 0:
					state = init_states[np.random.randint(len_init_states)]
					lim = limit
				lim -= 1

			# Finalize Metrics and Update NN
			rewards = self.Env.reward(states,np.argmax(actions,axis=1),next_states)
			al,cl = self.Agent.update(states, actions, next_states, rewards, dones)
			metrics['reward'].append(np.sum(rewards)/(np.sum(dones)+1))
			metrics['actor_loss'].append(al)
			metrics['critic_loss'].append(cl)
			# Write progress to file
			with open("train_NN.txt","a") as file:
				file.write(f"Epoch: {t}, Avg Reward:{round(np.sum(rewards)/(np.sum(dones)+1),2)}, Actor Loss:{al}, Critic Loss:{cl}\n")

		return metrics


	def explore_train(self, batch_size, trials):
		"""
		Train Neural Net on Random Runs to maximize exploration

		Parameters
		----------
			batch_size: Size of each batch in the NN
			trials: Number of trials such that batch_size x trials = training points
		Returns
		-------
			A dictionary holding three arrays

			reward: array holding the reward of each state-action pair
			actor_loss: array holding the total loss of each epoch for the actor NN
			critic_loss: array holding the total loss of each epoch for the critic NN
		"""
		# Initialize metrics
		with open("train_explore_NN.txt","w") as file: file.write("")
		metrics = {'reward':[],'actor_loss':[],'critic_loss':[]}
		states = self.Env.find_valid_starts()
		l = len(states)

		# Main loop to create each batch
		for t in range(trials):
			# Get data
			states = states[np.random.randint(l,size=batch_size)]
			probs = np.zeros([batch_size,9])
			actions = np.random.randint(9,size=batch_size)
			probs[np.arange(batch_size),actions] = 1
			next_states = self.Env.step(states,actions)
			rewards = self.Env.reward(states,actions,next_states)
			dones = self.Env.done(next_states)

			# Update NN and metrics
			al,cl = self.Agent.update(states, probs, next_states, rewards, dones)
			metrics['reward'].append(np.sum(rewards))
			metrics['actor_loss'].append(al)
			metrics['critic_loss'].append(cl)
			with open("train_explore_NN.txt","a") as file:
				file.write(f"Epoch: {t}, Avg Reward:{round(np.sum(rewards)/(np.sum(dones)+1),2)}, Actor Loss:{al}, Critic Loss:{cl}\n")

		return metrics
