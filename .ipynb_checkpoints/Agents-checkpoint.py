import tensorflow as tf
import numpy as np

# Define the actor network
class Actor(tf.keras.Model):
	def __init__(self, state_dim, action_dim):
		super(Actor, self).__init__()
		self.hidden_layer1 = tf.keras.layers.Dense(64, activation='relu')
		self.hidden_layer2 = tf.keras.layers.Dense(64, activation='relu')
		self.hidden_layer3 = tf.keras.layers.Dense(64, activation='relu')
		self.output_layer = tf.keras.layers.Dense(action_dim, activation='sigmoid')

	def call(self, state):
		x = self.hidden_layer1(state)
		x = self.hidden_layer2(x)
		x = self.hidden_layer3(x)
		x = self.output_layer(x)
		return np.argmax(x,axis=1)

# Define the critic network
class Critic(tf.keras.Model):
	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()
		self.hidden_layer1 = tf.keras.layers.Dense(64, activation='relu')
		self.hidden_layer2 = tf.keras.layers.Dense(64, activation='relu')
		self.hidden_layer3 = tf.keras.layers.Dense(64, activation='relu')
		self.output_layer = tf.keras.layers.Dense(1)

	def call(self, state, action):
		x = tf.concat([state, action], axis=1)
		x = self.hidden_layer1(x)
		x = self.hidden_layer2(x)
		x = self.hidden_layer3(x)
		return self.output_layer(x)


class DDPGAgent:
	def __init__(self, state_dim, action_dim, tau=.001, gamma = .99):
		self.actor = Actor(state_dim, action_dim)
		self.target_actor = Actor(state_dim, action_dim)
		self.critic = Critic(state_dim, action_dim)
		self.target_critic = Critic(state_dim, action_dim)
		self.target_actor.set_weights(self.actor.get_weights())
		self.target_critic.set_weights(self.critic.get_weights())
		self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
		self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
		self.tau = tau
		self.gamma = gamma

	def get_action(self, state):
		return self.actor(tf.convert_to_tensor(state))

	def update(self, state, action, next_state, reward, done):
        
		state = tf.convert_to_tensor(state, dtype=tf.float32)
		action = tf.expand_dims(tf.convert_to_tensor(action, dtype=tf.float32),1)
		next_state = tf.convert_to_tensor(next_state, dtype=tf.float32)
		reward = tf.convert_to_tensor(reward, dtype=tf.float32)

		with tf.GradientTape() as tape:
			target_actions = tf.cast(tf.expand_dims(self.target_actor(next_state),1),tf.float32)
			target_critic_value = tf.squeeze(self.target_critic(next_state, target_actions), 1)
			y = reward + self.gamma * target_critic_value * (1-done)
			critic_value = tf.squeeze(self.critic(state, action), 1)
			critic_loss = tf.keras.losses.MSE(y, critic_value)
		critic_grad = tape.gradient(critic_loss, self.critic.trainable_variables)
		self.critic_optimizer.apply_gradients(zip(critic_grad, self.critic.trainable_variables))

		with tf.GradientTape() as tape:
			tape.watch(self.actor)
			action = tf.cast(tf.expand_dims(self.actor(state),1),tf.float32)
			critic_value = tf.squeeze(self.critic(state, action), 1)
			actor_loss = -tf.math.reduce_mean(critic_value)
		actor_grad = tape.gradient(actor_loss, self.actor.trainable_variables)
		self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor.trainable_variables))

		self.soft_update()
		return float(actor_loss),float(critic_loss)

	def soft_update(self):
		for target, source in zip(self.target_actor.variables, self.actor.variables):
			target.assign(self.tau * source + (1 - self.tau) * target)
		for target, source in zip(self.target_critic.variables, self.critic.variables):
			target.assign(self.tau * source + (1 - self.tau) * target)
