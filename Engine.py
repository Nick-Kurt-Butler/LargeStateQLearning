import numpy as np
import tensorflow as tf

class Engine:
	def __init__(self,Env,Agent):
		self.Env = Env
		self.Agent = Agent

	def run(self,init_state, explore_prob, limit = 1000):
		states,actions,next_states = ([],[],[])
		state = init_state
		while not self.Env.done_f(state):
			action = self.Agent.get_action([state])
			if np.random.rand() < explore_prob: action = np.random.randint(9,size=1)
			next_state = self.Env.step_f(np.concatenate((state,action)))

			states.append(state)
			actions.append(action)
			next_states.append(next_state)

			state = next_state
			if limit == 0: break
			limit -= 1

		next_states = np.array(next_states)
		rewards = self.Env.reward(states,actions,next_states)
		dones = np.zeros(len(rewards))
		dones[-1] = 1

		return np.array(states),np.array(actions),next_states,rewards, dones

	def train(self,explore_prob, limit = 1000, buffer_cap = 1000, trials = 10):
		metrics = {'reward':[],'actor_loss':[],'critic_loss':[]}
		lim = limit
		init_states = self.Env.find_valid_starts()
		len_init_states = len(init_states)
		state = init_states[np.random.randint(len_init_states)]
		for t in range(trials):
			states = np.zeros([buffer_cap,4])
			next_states = np.zeros([buffer_cap,4])
			actions = np.zeros([buffer_cap,9])
			dones = np.zeros(buffer_cap)
			for i in range(buffer_cap):
				if self.Env.done_f(state):
					state = init_states[np.random.randint(len_init_states)]
					dones[i] = 1

				probs,action = self.Agent.get_action_probs([state])
				if np.random.rand() < explore_prob: action = np.random.randint(9,size=1)
				next_state = self.Env.step_f(np.concatenate((state,action)))

				states[i] = state
				actions[i] = probs
				next_states[i] = next_state

				state = next_state
				if lim == 0:
					state = init_states[np.random.randint(len_init_states)]
					lim = limit
				lim -= 1
			rewards = self.Env.reward(states,np.argmax(actions,axis=1),next_states)
			al,cl = self.Agent.update(states, actions, next_states, rewards, dones)
			metrics['reward'].append(np.sum(rewards)/(np.sum(dones)+1))
			metrics['actor_loss'].append(al)
			metrics['critic_loss'].append(cl)
			print(f"Epoch: {t}, Avg Reward:{round(np.sum(rewards)/(np.sum(dones)+1),2)}, Actor Loss:{al}, Critic Loss:{cl}")

		return metrics


	def explore_train(self, batch_size, trials):
		metrics = {'reward':[],'actor_loss':[],'critic_loss':[]}
		states = self.Env.find_valid_starts()
		l = len(states)
		for t in range(trials):
			states = states[np.random.randint(l,size=batch_size)]
			probs = np.zeros([batch_size,9])
			actions = np.random.randint(9,size=batch_size)
			probs[np.arange(batch_size),actions] = 1
			next_states = self.Env.step(states,actions)
			rewards = self.Env.reward(states,actions,next_states)
			dones = self.Env.done(next_states)
			al,cl = self.Agent.update(states, probs, next_states, rewards, dones)
			metrics['reward'].append(np.sum(rewards))
			metrics['actor_loss'].append(al)
			metrics['critic_loss'].append(cl)
			print(f"Epoch: {t}, Avg Reward:{round(np.sum(rewards)/(np.sum(dones)+1),2)}, Actor Loss:{al}, Critic Loss:{cl}")
		return metrics

	def explore(self,actions,explore_prob):
		s = len(actions)
		mask = np.random.choice([True,False],p=[1-explore_prob,explore_prob],size=s)
		actions[mask] = np.random.randint(9,size=s)[mask]
		return actions
