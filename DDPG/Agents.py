import tensorflow as tf
import numpy as np

# Define the actor network
class Actor(tf.keras.Model):
    def __init__(self, action_dim):
        super(Actor, self).__init__()
        self.hidden_layer1 = tf.keras.layers.Dense(64, activation='relu')
        self.hidden_layer2 = tf.keras.layers.Dense(64, activation='relu')
        self.hidden_layer3 = tf.keras.layers.Dense(64, activation='relu')
        self.output_layer = tf.keras.layers.Dense(action_dim, activation='softmax')

    def call(self, state):
        x = self.hidden_layer1(state)
        x = self.hidden_layer2(x)
        x = self.hidden_layer3(x)
        return self.output_layer(x)

# Define the critic network
class Critic(tf.keras.Model):
    def __init__(self):
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

# Class to train and soft update Actor and Critic Networks
class DDPGAgent:
    def __init__(self, action_dim, tau=0.0001, gamma = .99):
        self.action_dim = action_dim
        self.actor = Actor(action_dim)
        self.target_actor = Actor(action_dim)
        self.critic = Critic()
        self.target_critic = Critic()
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.tau = tau
        self.gamma = gamma

    def get_action(self, state):
        """
        Using Actor network return action(s) from state(s)
        """
        return tf.argmax(self.actor(tf.convert_to_tensor(state)),axis=1)

    def get_action_probs(self, state):
        """
        Using Actor network return raw probabilities and action(s) from state(s)
        """
        probs = self.actor(tf.convert_to_tensor(state))
        action = tf.argmax(probs,axis=1)
        return probs,action

    def update(self, state, action, next_state, reward, done):
        """
        Update Actor and Critic Networks

        Parameters
        ----------
            state: The states from 0 to n-1
            action: The actions the agent made at each state
            next_state: The states from 1 to n
            reward: Reward received by agent at each step
            done: An array showing a 1 if at that step the agent completed moving throught he environment else 0

        Returns
        -------
            A tuple: (actor_loss,critic_loss)
        """
        # Convert inputs to tensor arrays
        state = tf.convert_to_tensor(state, dtype=tf.float32)
        action = tf.convert_to_tensor(action, dtype=tf.float32)
        next_state = tf.convert_to_tensor(next_state, dtype=tf.float32)
        reward = tf.convert_to_tensor(reward, dtype=tf.float32)
        done = tf.convert_to_tensor(done, dtype=tf.float32)

        # Update Critic Network using Bellman Equation
        with tf.GradientTape() as tape:
            target_actions = self.target_actor(next_state)
            target_critic_value = tf.squeeze(self.target_critic(next_state, target_actions), 1)
            y = reward + self.gamma * target_critic_value * (1-done)
            critic_value = tf.squeeze(self.critic(state, action), 1)
            critic_loss = tf.keras.losses.MSE(y, critic_value)

        critic_grad = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grad, self.critic.trainable_variables))

        # Update Actor Network using the magnitude of the Q-values from the Critic Network
        with tf.GradientTape() as tape:
            action = self.actor(state)
            critic_value = tf.squeeze(self.critic(state, action), 1)
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor.trainable_variables))

        # Perform a soft update
        self.soft_update()
        return float(actor_loss),float(critic_loss)

    def soft_update(self):
        """
        Update the Actor and Critic Network by a small fraction (tau) of the total change
        """
        for target, source in zip(self.target_actor.variables, self.actor.variables):
            target.assign(self.tau * source + (1 - self.tau) * target)
        for target, source in zip(self.target_critic.variables, self.critic.variables):
            target.assign(self.tau * source + (1 - self.tau) * target)
