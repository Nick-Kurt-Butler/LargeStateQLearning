import tensorflow as tf
import numpy as np

class DQNAgent:
    def __init__(self, state_dim, action_dim,gamma):
        self.state_size = state_dim
        self.action_size = action_dim
        self.gamma = gamma
        self.model = self._build_model()  # Build the neural network model

    def _build_model(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(128, input_dim=self.state_size, activation='relu'))
        model.add(tf.keras.layers.Dense(128, activation='relu'))
        model.add(tf.keras.layers.Dense(128, activation='relu'))
        model.add(tf.keras.layers.Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3))
        return model

    def get_action(self, states):
        return np.argmax(self.model.predict(states,verbose=None),axis=1)

    def update(self,states,actions,next_states,rewards,dones,MIN,MAX):
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
        Q = self.model.predict(states)
        res = rewards + self.gamma * np.max(self.model.predict(next_states),axis=1) * (1-dones)
        res[res>MAX] = MAX
        res[res<MIN] = MIN
        Q[np.arange(len(states)),actions] = res
        return self.model.train_on_batch(states,Q)

    def update_all(self,states,next_states,rewards,dones,MIN,MAX):
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
        n = len(states)
        Qmax = np.max(self.model(next_states.reshape(9*n,4)),axis=1).reshape(n,9)
        Q = rewards + self.gamma * Qmax * (1-dones)
        Q[Q>MAX] = MAX*.5
        Q[Q<MIN] = MIN
        return self.model.train_on_batch(states,Q),np.mean(Qmax),np.mean(Q)
