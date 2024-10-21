import random
import torch

class Agent:
    def __init__(self, dqn, action_space, epsilon):
        self.dqn = dqn
        self.action_space = action_space
        self.epsilon = epsilon

    def select_action(self, state):
        if random.random() < self.epsilon:  # Exploration
            return random.randint(0, self.action_space - 1)
        else:  # Exploitation
            with torch.no_grad():
                return self.dqn(state).argmax().item()
