import os
import sys

import numpy as np


class LearningType:
    QLEARNING = 1
    SARSA = 2


class QLearningAgent:
    def __init__(
        self,
        num_states,
        num_bins,
        num_actions,
        alpha,
        epsilon,
        gamma=1.0,
        learning_type=LearningType.QLEARNING,
    ):
        print("=" * 80)
        print("Q-Learning Agent")
        self.num_states, self.num_bins, self.num_actions = (
            num_states,
            num_bins,
            num_actions,
        )
        self.learning_type = learning_type
        self.epsilon, self.alpha, self.gamma = epsilon, alpha, gamma
        print(
            f"States: {self.num_states}, Bins: {self.num_bins}, Actions: {self.num_actions}"
        )
        print(f"Epsilon: {self.epsilon}, Alpha: {self.alpha}, Gamma: {self.gamma}")
        # Shape for the action-value function, Q(s, a)
        shape = (num_bins,) * num_states + (num_actions,)

        self.visited = {}
        # Initialize the action-value function
        self.Q = np.random.random(shape)
        # self.Q = np.zeros(shape)
        # self.Q = np.random.uniform(low=-1, high=1, size=shape)
        self.num_sa_pairs = np.prod(self.Q.shape)
        print("Q-table shape:", self.Q.shape)

    def getAction(self, state, deterministic=False):
        """
        Returns the action to take at the given state.
        If deterministic is True, then greedy action is returned.
        Otherwise, epsilon-greedy action is returned.
        """
        if not deterministic and np.random.random() < self.epsilon:
            # print("Random action")
            return np.random.randint(self.num_actions)
        # print("Deterministic action")
        return np.argmax(self.Q[state])

    def update(self, state, action, reward, state_next):
        """
        Update function to handle both the Q-Learning and SARSA algorithms.
        """
        # keep track of visited states
        if (state, action) not in self.visited:
            self.visited[(state, action)] = 0
        self.visited[(state, action)] += 1
        # grab the current action-value estimate
        current_qsa = self.Q[state][action]
        # Q-Learning grabs the max action-value estimate at the next state
        # SARSA grabs the action-value estimate at the next state, using the next action
        if self.learning_type == LearningType.QLEARNING:
            # Best Q value at next state
            next_qsa = np.max(self.Q[state_next])
        elif self.learning_type == LearningType.SARSA:
            # Epsilon-greedy action at next state
            next_action = self.getAction(state_next, deterministic=False)
            # Next Q value
            next_qsa = self.Q[state_next][next_action]
        else:
            raise Exception("Unknown learning type")
        # Target Q value
        target_qsa = reward + self.gamma * next_qsa
        # Update Q value based on the difference between target and current Q value
        self.Q[state][action] += self.alpha * (target_qsa - current_qsa)

    def saveQ(self, path):
        """
        Handler for saving the action-value function to a file.
        """
        if not path.endswith(".npy"):
            path += ".npy"
        np.save(path, self.Q)

    def loadQ(self, path):
        """
        Handler for loading the action-value function from a file.
        """
        if not path.endswith(".npy"):
            path += ".npy"
        # If the file doesn't exist, suggest matching files
        # in the current directory that end with .npy
        if not os.path.isfile(path):
            print()
            print(f"ERROR: Q-table file not found: {path}")
            files = []
            for root, _, filenames in os.walk("."):
                files.extend(
                    os.path.join(root, filename)
                    for filename in filenames
                    if filename.endswith(".npy")
                )

            print("Did you mean one of these files?")
            for file in files:
                print(f" - {file}")
            sys.exit(1)
        # File exists, load it and update the number of bins that were used
        self.Q = np.load(path)
        self.num_bins = self.Q.shape[0]


# %%
