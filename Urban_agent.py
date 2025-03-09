import gymnasium as gym
import numpy as np
import random
from Urban_env2 import City  # Assuming the City environment is defined in City.py

class QLearningAgent:
    def __init__(self, env, learning_rate=0.1, discount_factor=0.99, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01):
        """
        Initialize the Q-learning agent.

        Parameters:
        - env: The City environment instance.
        - learning_rate: Step size for Q-value updates (alpha).
        - discount_factor: Importance of future rewards (gamma).
        - epsilon: Initial exploration probability.
        - epsilon_decay: Rate at which epsilon decreases after each episode.
        - min_epsilon: Minimum value for epsilon to ensure some exploration.
        """
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        # Q-table as a dictionary: keys are states, values are arrays of Q-values for each action
        self.q_table = {}

    def state_to_key(self, state):
        """
        Convert the state dictionary to a hashable key for the Q-table.

        Parameters:
        - state: Dictionary with 'position' (array) and 'neighbors' (array).

        Returns:
        - A tuple representing the state, usable as a dictionary key.
        """
        position = tuple(state['position'])
        neighbors = tuple(state['neighbors'])
        return (position, neighbors)

    def choose_action(self, state):
        """
        Select an action using an epsilon-greedy policy.

        Parameters:
        - state: Current state.

        Returns:
        - Action index (0 for street, 1 for house).
        """
        state_key = self.state_to_key(state)
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.env.action_space.n)
        if random.uniform(0, 1) < self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.q_table[state_key])

    def update_q_value(self, state, action, reward, next_state):
        """
        Update the Q-value for a state-action pair based on the Q-learning rule.

        Parameters:
        - state: Current state.
        - action: Action taken.
        - reward: Reward received.
        - next_state: Resulting state after the action.
        """
        state_key = self.state_to_key(state)
        next_state_key = self.state_to_key(next_state)
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.env.action_space.n)
        best_next_q = np.max(self.q_table[next_state_key])
        td_target = reward + self.discount_factor * best_next_q
        td_error = td_target - self.q_table[state_key][action]
        self.q_table[state_key][action] += self.learning_rate * td_error

    def train(self, num_episodes):
        """Train the agent over a specified number of episodes."""
        for episode in range(num_episodes):
            state = self.env.reset()
            self.env.render()  # Render the initial state
            done = False
            total_reward = 0
            while not done:
                action = self.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)
                self.update_q_value(state, action, reward, next_state)
                state = next_state
                total_reward += reward
                self.env.render()  # Render after each step to animate
            # Update epsilon
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
            # Optional: Log progress
            if episode % 10 == 0:
                print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {self.epsilon}")

# Example usage
if __name__ == "__main__":
    # Initialize the City environment with a 5x5 grid
    env = City(size=5)
    # Create and train the Q-learning agent
    agent = QLearningAgent(env)
    agent.train(num_episodes=100)
    