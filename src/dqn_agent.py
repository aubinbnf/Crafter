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
import argparse
import os
from pathlib import Path

# Classes pour traitement d'image
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

# Modèle DQN
class DQN(nn.Module):
    def __init__(self, action_space, device):
        super(DQN, self).__init__()
        self.device = device  # Store the device
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(3136, 512)
        self.fc2 = nn.Linear(512, action_space)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# Agent pour choisir les actions
class Agent:
    def __init__(self, dqn, action_space, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.dqn = dqn
        self.action_space = action_space
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min  # Assure-toi que cet attribut est défini
        self.epsilon_decay = epsilon_decay  # Assure-toi que cet attribut est défini

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_space - 1)
        state = state.to(self.dqn.device)
        with torch.no_grad():
            return self.dqn(state.unsqueeze(0)).argmax(dim=1).item()

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


# Buffer pour stocker les transitions
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self):
        return len(self.buffer)  # Ajout de la méthode __len__

# Classe pour apprentissage DQN
class DQNLearner:
    def __init__(self, dqn, target_dqn, action_space, buffer, device, logdir, gamma=0.99, lr=0.0001):
        self.dqn = dqn
        self.target_dqn = target_dqn
        self.buffer = buffer
        self.device = device
        self.logdir = logdir
        self.gamma = gamma
        self.optimizer = torch.optim.Adam(self.dqn.parameters(), lr=lr)

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

        q_values = self.dqn(states)
        next_q_values = self.target_dqn(next_states)
        target_q_values = rewards + self.gamma * next_q_values.max(1)[0] * (1 - dones)

        loss = F.mse_loss(q_values.gather(1, actions.unsqueeze(1)), target_q_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_weights(self):
        torch.save(self.dqn.state_dict(), self.logdir + "/weights.pth")
        torch.save(self.target_dqn.state_dict(), self.logdir + "/target_weights.pth")


# Fonction pour évaluation de l'agent
def eval(agent, env, step_cnt, opt):
    episodic_returns = []
    for _ in range(opt.eval_episodes):
        obs, done = env.reset(), False
        episodic_returns.append(0)
        while not done:
            action = agent.act(obs)
            obs, reward, done, _ = env.step(action)
            episodic_returns[-1] += reward
    avg_return = np.mean(episodic_returns)
    print(f"[{step_cnt:06d}] Eval results: R/ep={avg_return:.2f}")
    with open(f"{opt.logdir}/DQN/0/eval_stats.pkl", "ab") as f:
        pickle.dump({"step": step_cnt, "avg_return": avg_return}, f)

