import os
import pickle
import random
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from PIL import Image
from crafter_wrapper import Env

class GrayScale:
    def __init__(self, env):
        self._env = env

    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        obs = obs.mean(-1)
        return obs, reward, done, info

    def reset(self):
        obs = self._env.reset()
        obs = obs.mean(-1)
        return obs

class ResizeImage:
    def __init__(self, env):
        self._env = env

    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        obs = self._resize(obs)
        return obs, reward, done, info

    def reset(self):
        obs = self._env.reset()
        obs = self._resize(obs)
        return obs

    def _resize(self, image):
        image = Image.fromarray(image)
        image = image.resize((84, 84), Image.NEAREST)
        return np.array(image)
    

class Agent:
    def __init__(self, dqn, action_space, epsilon=0.1, epsilon_min=0.01, epsilon_decay=0.995):
        self.dqn = dqn
        self.action_space = action_space
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_space - 1)
        else:
            with torch.no_grad():
                q_values = self.dqn.q_values(state.unsqueeze(0))
                return q_values.argmax().item()

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.stack(states),
            actions,
            rewards,
            torch.stack(next_states),
            dones,
        )

    def __len__(self):
        return len(self.buffer)


class CategoricalDQN(nn.Module):
    def __init__(self, action_space, device, atoms=51, Vmin=-10, Vmax=10):
        super(CategoricalDQN, self).__init__()
        self.device = device
        self.action_space = action_space
        self.atoms = atoms
        self.Vmin = Vmin
        self.Vmax = Vmax
        self.delta_z = (Vmax - Vmin) / (atoms - 1)
        self.support = torch.linspace(Vmin, Vmax, atoms).to(device)

        # Convolutional layers
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        # Fully connected layers for distribution output
        self.fc1 = nn.Linear(3136, 512)
        self.fc2 = nn.Linear(512, action_space * atoms)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x).view(-1, self.action_space, self.atoms)
        return F.softmax(x, dim=2)  # Output is a distribution over the atoms

    def q_values(self, x):
        dist = self(x)
        q_values = torch.sum(dist * self.support, dim=2)  # Compute the expected Q-values
        return q_values

class CategoricalDQNLearner:
    def __init__(self, dqn, target_dqn, action_space, buffer, device, logdir, gamma=0.99, lr=0.0001, Vmin=-10, Vmax=10, atoms=51):
        self.dqn = dqn
        self.target_dqn = target_dqn
        self.buffer = buffer
        self.device = device
        self.logdir = logdir
        self.gamma = gamma
        self.optimizer = torch.optim.Adam(self.dqn.parameters(), lr=lr)
        self.Vmin = Vmin
        self.Vmax = Vmax
        self.atoms = atoms
        self.delta_z = (Vmax - Vmin) / (atoms - 1)
        self.support = torch.linspace(Vmin, Vmax, atoms).to(device)

    def load_weights(self):
        weights_path = Path(self.logdir) / "weights.pth"
        target_weights_path = Path(self.logdir) / "target_weights.pth"
        if weights_path.exists() and target_weights_path.exists():
            print("Weights loaded from ", self.logdir)
            self.dqn.load_state_dict(torch.load(weights_path))
            self.target_dqn.load_state_dict(torch.load(target_weights_path))
        else:
            print("Weights file not found. Starting training from scratch.")

    def update(self, batch_size):
        if len(self.buffer) < batch_size:
            return

        batch = self.buffer.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.stack(states).to(self.device)
        actions = torch.tensor(actions).to(self.device)
        rewards = torch.tensor(rewards).to(self.device)
        next_states = torch.stack(next_states).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        # Current distribution
        dist = self.dqn(states)
        dist = dist[range(batch_size), actions]

        # Compute target distribution
        next_dist = self.target_dqn(next_states)
        next_actions = next_dist.sum(2).argmax(1)
        next_dist = next_dist[range(batch_size), next_actions]

        Tz = rewards.unsqueeze(1) + (1 - dones).unsqueeze(1) * self.gamma * self.support.unsqueeze(0)
        Tz = Tz.clamp(self.Vmin, self.Vmax)
        b = (Tz - self.Vmin) / self.delta_z
        l = b.floor().long()
        u = b.ceil().long()

        proj_dist = torch.zeros(next_dist.size(), device=self.device)
        for i in range(batch_size):
            proj_dist[i].index_add_(0, l[i], next_dist[i] * (u[i] - b[i]))
            proj_dist[i].index_add_(0, u[i], next_dist[i] * (b[i] - l[i]))

        loss = -(proj_dist * dist.log()).sum(1).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_weights(self):
        torch.save(self.dqn.state_dict(), self.logdir + "/weights.pth")
        torch.save(self.target_dqn.state_dict(), self.logdir + "/target_weights.pth")
