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

class GrayScale:
    def __init__(self, env):
        self._env = env

    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        obs = obs.mean(-1)
        return obs / 255.0, reward, done, info  # Normalisation ajoutée

    def reset(self):
        obs = self._env.reset()
        obs = obs.mean(-1)
        return obs / 255.0  # Normalisation ajoutée

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
        image = image.resize((84, 84), Image.BICUBIC)  # Changed to BICUBIC for better downsampling
        return np.array(image)

class Agent:
    def __init__(self, dqn, action_space, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.dqn = dqn
        self.action_space = action_space
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

    def act(self, state, evaluate=False):
        if not evaluate and random.random() < self.epsilon:
            return random.randint(0, self.action_space - 1)
        with torch.no_grad():
            q_values = self.dqn.q_values(state.unsqueeze(0))
            return q_values.argmax().item()

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

class ReplayBuffer:
    def __init__(self, capacity, history_length=4):
        self.buffer = deque(maxlen=capacity)
        self.history_length = history_length

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Stack frames for both current and next states
        states = torch.stack([self._stack_frames(s, self.history_length) for s in states])
        next_states = torch.stack([self._stack_frames(s, self.history_length) for s in next_states])
        
        return (states, torch.tensor(actions), 
                torch.tensor(rewards), next_states, 
                torch.tensor(dones, dtype=torch.float32))

    def _stack_frames(self, state, history_length):
        # Assuming state is already stacked correctly
        return state

    def __len__(self):
        return len(self.buffer)

class DuelingCategoricalDQN(nn.Module):
    def __init__(self, action_space, device, atoms=51, Vmin=-10, Vmax=10):
        super(DuelingCategoricalDQN, self).__init__()
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
        
        # Dueling architecture
        self.value_stream = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, atoms)
        )
        
        self.advantage_stream = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, action_space * atoms)
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(batch_size, -1)

        values = self.value_stream(x).view(batch_size, 1, self.atoms)
        advantages = self.advantage_stream(x).view(batch_size, self.action_space, self.atoms)
        
        # Combine value and advantage
        q_dist = values + (advantages - advantages.mean(dim=1, keepdim=True))
        
        return F.softmax(q_dist, dim=2)

    def q_values(self, x):
        dist = self(x)
        q_values = torch.sum(dist * self.support, dim=2)
        return q_values

class CategoricalDQNLearner:
    def __init__(self, dqn, target_dqn, action_space, buffer, device, logdir, 
                 gamma=0.99, lr=0.0001, Vmin=-10, Vmax=10, atoms=51, 
                 target_update_freq=10000):
        self.dqn = dqn
        self.target_dqn = target_dqn
        self.target_update_freq = target_update_freq
        self.update_count = 0
        
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

    def update_target_network(self):
        if self.update_count % self.target_update_freq == 0:
            self.target_dqn.load_state_dict(self.dqn.state_dict())
        self.update_count += 1

    def update(self, batch_size):
        if len(self.buffer) < batch_size:
            return

        states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Current distribution
        curr_dist = self.dqn(states)
        curr_dist = curr_dist[range(batch_size), actions]

        # Double DQN: Use online network to select action
        with torch.no_grad():
            next_q = self.dqn.q_values(next_states)
            next_actions = next_q.argmax(1)
            
            # Use target network to evaluate action
            next_dist = self.target_dqn(next_states)
            next_dist = next_dist[range(batch_size), next_actions]

        # Compute projected distribution
        Tz = rewards.unsqueeze(1) + (1 - dones).unsqueeze(1) * self.gamma * self.support.unsqueeze(0)
        Tz = Tz.clamp(self.Vmin, self.Vmax)
        b = (Tz - self.Vmin) / self.delta_z
        l = b.floor().long()
        u = b.ceil().long()

        proj_dist = torch.zeros_like(next_dist)
        for i in range(batch_size):
            proj_dist[i].index_add_(0, l[i], next_dist[i] * (u[i].float() - b[i]))
            proj_dist[i].index_add_(0, u[i], next_dist[i] * (b[i] - l[i].float()))

        # Cross-entropy loss
        loss = -(proj_dist * curr_dist.log()).sum(1).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.dqn.parameters(), max_norm=10)  # Added gradient clipping
        self.optimizer.step()
        
        self.update_target_network()

    def save_weights(self):
        torch.save(self.dqn.state_dict(), str(Path(self.logdir) / "weights.pth"))
        torch.save(self.target_dqn.state_dict(), str(Path(self.logdir) / "target_weights.pth"))

    def load_weights(self):
        weights_path = Path(self.logdir) / "weights.pth"
        target_weights_path = Path(self.logdir) / "target_weights.pth"
        if weights_path.exists() and target_weights_path.exists():
            self.dqn.load_state_dict(torch.load(weights_path))
            self.target_dqn.load_state_dict(torch.load(target_weights_path))
            print("Weights loaded from", self.logdir)
        else:
            print("No weights found. Starting from scratch.")