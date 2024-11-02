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
        return obs / 255.0, reward, done, info

    def reset(self):
        obs = self._env.reset()
        obs = obs.mean(-1)
        return obs / 255.0

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
        image = image.resize((84, 84), Image.BICUBIC)
        return np.array(image)

class PrioritizedReplayBuffer:
    def __init__(self, capacity, history_length=4, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.history_length = history_length
        self.alpha = alpha  # How much to prioritize
        self.beta = beta    # Importance sampling weight
        self.beta_increment = 0.001
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        self.eps = 1e-6  # Small positive constant to prevent zero probabilities

    def add(self, state, action, reward, next_state, done):
        max_priority = np.max(self.priorities) if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
        
        self.priorities[self.pos] = max_priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.buffer) == self.capacity:
            probs = self.priorities
        else:
            probs = self.priorities[:len(self.buffer)]

        probs = probs ** self.alpha
        probs = probs / probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()
        self.beta = min(1., self.beta + self.beta_increment)

        batch = list(zip(*samples))
        states = torch.stack([self._stack_frames(s, self.history_length) for s in batch[0]])
        next_states = torch.stack([self._stack_frames(s, self.history_length) for s in batch[3]])
        
        return (states, torch.tensor(batch[1]), 
                torch.tensor(batch[2]), 
                next_states,
                torch.tensor(batch[4], dtype=torch.float32),
                torch.tensor(weights, dtype=torch.float32),
                indices)

    def update_priorities(self, indices, priorities):
        priorities = np.abs(priorities) + self.eps
        self.priorities[indices] = priorities

    def _stack_frames(self, state, history_length):
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

        # Enhanced convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        # Noisy Linear layers for value stream
        self.value_stream = nn.Sequential(
            NoisyLinear(3136, 512),
            nn.ReLU(),
            NoisyLinear(512, atoms)
        )
        
        # Noisy Linear layers for advantage stream
        self.advantage_stream = nn.Sequential(
            NoisyLinear(3136, 512),
            nn.ReLU(),
            NoisyLinear(512, action_space * atoms)
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv_layers(x)
        x = x.view(batch_size, -1)

        values = self.value_stream(x).view(batch_size, 1, self.atoms)
        advantages = self.advantage_stream(x).view(batch_size, self.action_space, self.atoms)
        
        # Combine value and advantage using dueling architecture
        q_dist = values + (advantages - advantages.mean(dim=1, keepdim=True))
        
        return F.softmax(q_dist, dim=2)

    def reset_noise(self):
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()

    def q_values(self, x):
        dist = self(x)
        q_values = torch.sum(dist * self.support, dim=2)
        return q_values

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.out_features))

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)

class Agent:
    def __init__(self, dqn, action_space, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.9995):
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

class CategoricalDQNLearner:
    def __init__(self, dqn, target_dqn, action_space, buffer, device, logdir, 
                 gamma=0.99, lr=0.00025, Vmin=-10, Vmax=10, atoms=51, 
                 target_update_freq=8000, tau=0.005, drive_folder=None):
        self.dqn = dqn
        self.target_dqn = target_dqn
        self.target_update_freq = target_update_freq
        self.tau = tau  # Add tau parameter for soft updates
        self.update_count = 0
        
        self.buffer = buffer
        self.device = device
        self.logdir = logdir
        self.gamma = gamma
        self.optimizer = torch.optim.Adam(self.dqn.parameters(), lr=lr, eps=1.5e-4)
        
        self.Vmin = Vmin
        self.Vmax = Vmax
        self.atoms = atoms
        self.delta_z = (Vmax - Vmin) / (atoms - 1)
        self.support = torch.linspace(Vmin, Vmax, atoms).to(device)
        
        # Ajouter le chemin du dossier Google Drive
        self.drive_folder = drive_folder

    def save_weights(self):
        # Sauvegarde locale
        local_path = Path(self.logdir)
        local_path.mkdir(parents=True, exist_ok=True)
        torch.save(self.dqn.state_dict(), local_path / "weights.pth")
        torch.save(self.target_dqn.state_dict(), local_path / "target_weights.pth")
        print(f"Poids sauvegardés localement dans {local_path}")

        # Sauvegarde sur Google Drive si un chemin est spécifié
        if self.drive_folder:
            drive_path = Path(self.drive_folder)
            drive_path.mkdir(parents=True, exist_ok=True)
            torch.save(self.dqn.state_dict(), drive_path / "weights.pth")
            torch.save(self.target_dqn.state_dict(), drive_path / "target_weights.pth")
            print(f"Poids sauvegardés dans Google Drive : {drive_path}")

    def load_weights(self):
        weights_path = Path(self.logdir) / "weights.pth"
        target_weights_path = Path(self.logdir) / "target_weights.pth"
        if weights_path.exists() and target_weights_path.exists():
            self.dqn.load_state_dict(torch.load(weights_path))
            self.target_dqn.load_state_dict(torch.load(target_weights_path))
            print("Poids chargés depuis", self.logdir)
        else:
            print("Aucun poids trouvé. Début de l'entraînement depuis zéro.")

    def _soft_update_target_network(self):
        """Soft update of target network from policy network."""
        for target_param, param in zip(self.target_dqn.parameters(), self.dqn.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )

    def update(self, batch_size):
        if len(self.buffer) < batch_size:
            return

        states, actions, rewards, next_states, dones, weights, indices = self.buffer.sample(batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        weights = weights.to(self.device)

        # Reset noise for both networks
        self.dqn.reset_noise()
        self.target_dqn.reset_noise()

        # Current distribution
        curr_dist = self.dqn(states)
        curr_dist = curr_dist[range(batch_size), actions]

        with torch.no_grad():
            # Double DQN
            next_q = self.dqn.q_values(next_states)
            next_actions = next_q.argmax(1)
            
            next_dist = self.target_dqn(next_states)
            next_dist = next_dist[range(batch_size), next_actions]

        Tz = rewards.unsqueeze(1) + (1 - dones).unsqueeze(1) * self.gamma * self.support.unsqueeze(0)
        Tz = Tz.clamp(self.Vmin, self.Vmax)
        b = (Tz - self.Vmin) / self.delta_z
        l = b.floor().long()
        u = b.ceil().long()

        proj_dist = torch.zeros_like(next_dist)
        for i in range(batch_size):
            proj_dist[i].index_add_(0, l[i], next_dist[i] * (u[i].float() - b[i]))
            proj_dist[i].index_add_(0, u[i], next_dist[i] * (b[i] - l[i].float()))

        # Calculate loss with importance sampling weights
        loss = -(proj_dist * curr_dist.log()).sum(1)
        weighted_loss = (loss * weights).mean()

        # Update priorities in buffer
        self.buffer.update_priorities(indices, loss.detach().cpu().numpy())

        self.optimizer.zero_grad()
        weighted_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.dqn.parameters(), max_norm=10)
        self.optimizer.step()

        # Use soft update instead of hard update
        self._soft_update_target_network()

        self.update_count += 1

    # def save_weights(self):
    #     torch.save(self.dqn.state_dict(), str(Path(self.logdir) / "weights.pth"))
    #     torch.save(self.target_dqn.state_dict(), str(Path(self.logdir) / "target_weights.pth"))

    # def load_weights(self):
    #     weights_path = Path(self.logdir) / "weights.pth"
    #     target_weights_path = Path(self.logdir) / "target_weights.pth"
    #     if weights_path.exists() and target_weights_path.exists():
    #         self.dqn.load_state_dict(torch.load(weights_path))
    #         self.target_dqn.load_state_dict(torch.load(target_weights_path))
    #         print("Weights loaded from", self.logdir)
    #     else:
    #         print("No weights found. Starting from scratch.")