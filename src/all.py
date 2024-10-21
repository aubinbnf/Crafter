import pathlib
from collections import deque

import crafter
import numpy as np
import torch
from PIL import Image


class Env:
    def __init__(self, mode, args):
        assert mode in (
            "train",
            "eval",
        ), "`mode` argument can either be `train` or `eval`"
        self.device = args.device
        env = crafter.Env()
        if mode == "train":
            env = crafter.Recorder(
                env,
                pathlib.Path(args.logdir),
                save_stats=True,
                save_video=False,
                save_episode=False,
            )
        env = ResizeImage(env)
        env = GrayScale(env)
        self.env = env
        self.action_space = env.action_space
        self.window = args.history_length  # Number of frames to concatenate
        self.state_buffer = deque([], maxlen=args.history_length)

    def reset(self):
        for _ in range(self.window):
            self.state_buffer.append(torch.zeros(84, 84, device=self.device))  # Initialiser les frames
        obs = self.env.reset()
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device).div_(255)
        self.state_buffer.append(obs)
        return torch.stack(list(self.state_buffer), 0).permute(0, 2, 1)  # Permute pour avoir (4, 84, 84)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device).div_(255)
        self.state_buffer.append(obs)
        return torch.stack(list(self.state_buffer), 0).permute(0, 2, 1), reward, done, info  # Permute pour avoir (4, 84, 84)



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
        image = np.array(image)
        return image





import argparse
import random
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path


class DQN(nn.Module):
    def __init__(self, action_space):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)  # Input: (batch, 4, 84, 84)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Calculer la taille de sortie dynamiquement
        self.flatten_size = self._get_flatten_size()

        self.fc1 = nn.Linear(self.flatten_size, 512)  # Corrigé pour la taille d'entrée
        self.fc2 = nn.Linear(512, action_space)

    def _get_flatten_size(self):
        # Utilise une entrée fictive pour calculer la taille de sortie
        with torch.no_grad():
            x = torch.zeros(1, 4, 84, 84)  # Un batch fictif
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            return x.numel()  # Retourne le nombre total d'éléments dans la sortie

    def forward(self, x):
        x = F.relu(self.conv1(x))
        print("Shape after conv1:", x.shape)

        x = F.relu(self.conv2(x))
        print("Shape after conv2:", x.shape)

        x = F.relu(self.conv3(x))
        print("Shape after conv3:", x.shape)

        # Aplatir la sortie
        x = x.view(x.size(0), -1)  # Aplatir pour obtenir [batch_size, flatten_size]
        print("Shape after flattening:", x.shape)

        x = F.relu(self.fc1(x))
        return self.fc2(x)  # Prédit les Q-valeurs



# Define the Agent class
class Agent:
    def __init__(self, dqn, action_space, epsilon):
        self.dqn = dqn
        self.action_space = action_space
        self.epsilon = epsilon

    def select_action(self, state):
        if random.random() < self.epsilon:  # Exploration
            action = random.randint(0, self.action_space - 1)
        else:  # Exploitation
            with torch.no_grad():
                action = self.dqn(state).argmax().item()

        # Vérification des limites de l'action
        action = max(0, min(action, self.action_space - 1))
        return action



# Define the Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def add(self, experience):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)  # Allocate space
        self.buffer[self.position] = experience
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def size(self):
        return len(self.buffer)


# Define the DQNLearner class
class DQNLearner:
    def __init__(self, dqn, target_dqn, action_space, buffer, device):
        self.dqn = dqn
        self.target_dqn = target_dqn
        self.action_space = action_space
        self.buffer = buffer
        self.device = device
        self.optimizer = torch.optim.Adam(self.dqn.parameters(), lr=0.0001)
        self.gamma = 0.99  # Discount factor

    def update(self, batch_size):
        if self.buffer.size() < batch_size:
            return  # Wait for the buffer to be filled

        # Retrieve a batch of samples from the buffer
        transitions = self.buffer.sample(batch_size)
        state, action, reward, next_state, done = zip(*transitions)

        state = torch.stack(state).to(self.device)
        action = torch.tensor(action).to(self.device)
        reward = torch.tensor(reward).to(self.device)
        next_state = torch.stack(next_state).to(self.device)
        done = torch.tensor(done).to(self.device)

        # Calculate current Q-values
        q_values = self.dqn(state).gather(1, action.unsqueeze(1)).squeeze(1)

        # Calculate target Q-values
        with torch.no_grad():
            target_q_values = self.target_dqn(next_state).max(1)[0]
            target = reward + self.gamma * target_q_values * (1 - done)

        # Update the DQN model
        loss = F.mse_loss(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


# Define evaluation functions
def _save_stats(episodic_returns, crt_step, path):
    episodic_returns = torch.tensor(episodic_returns)
    avg_return = episodic_returns.mean().item()
    print(f"[{crt_step:06d}] eval results: R/ep={avg_return:03.2f}, std={episodic_returns.std().item():03.2f}.")
    with open(path + "/eval_stats.pkl", "ab") as f:
        pickle.dump({"step": crt_step, "avg_return": avg_return}, f)

def eval(agent, env, crt_step, opt):
    episodic_returns = []
    for _ in range(opt.eval_episodes):
        obs, done = env.reset(), False
        episodic_returns.append(0)
        while not done:
            action = agent.act(obs)
            obs, reward, done, info = env.step(action)
            episodic_returns[-1] += reward

    _save_stats(episodic_returns, crt_step, opt.logdir)


def _info(opt):
    try:
        int(opt.logdir.split("/")[-1])
    except:
        print("Warning, logdir path should end in a number indicating a separate training run, else the results might be overwritten.")
    if Path(opt.logdir).exists():
        print("Warning! Logdir path exists, results can be corrupted.")
    print(f"Saving results in {opt.logdir}.")
    print(f"Observations are of dims ({opt.history_length},84,84), with values between 0 and 1.")


# Define the main training loop
def main(opt):
    _info(opt)
    opt.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = Env("train", opt)
    eval_env = Env("eval", opt)
    
    # Initialize DQN model and agent
    action_space = env.action_space.n
    dqn = DQN(action_space).to(opt.device)
    target_dqn = DQN(action_space).to(opt.device)
    target_dqn.load_state_dict(dqn.state_dict())  # Copy weights
    agent = Agent(dqn, action_space, opt.epsilon)

    # Initialize buffer and learner
    buffer = ReplayBuffer(capacity=opt.buffer_size)
    learner = DQNLearner(dqn, target_dqn, action_space, buffer, opt.device)

    # Main loop
    ep_cnt, step_cnt, done = 0, 0, True
    while step_cnt < opt.steps or not done:
        if done:
            ep_cnt += 1
            obs, done = env.reset(), False

        action = agent.select_action(obs)
        next_obs, reward, done, info = env.step(action)
        buffer.add((obs, action, reward, next_obs, done))
        learner.update(opt.batch_size)

        obs = next_obs
        step_cnt += 1

        # Evaluate at intervals
        if step_cnt % opt.eval_interval == 0:
            eval(agent, eval_env, step_cnt, opt)


def get_options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_episodes", type=int, default=1000)
    parser.add_argument("--buffer_size", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--logdir", default="logdir/")
    parser.add_argument("--steps", type=int, default=1_000_000, help="Total number of training steps.")
    parser.add_argument("--history_length", type=int, default=4)
    parser.add_argument("--eval_interval", type=int, default=100_000, metavar="STEPS", help="Number of training steps between evaluations")
    parser.add_argument("--eval_episodes", type=int, default=20, metavar="N", help="Number of evaluation episodes to average over")
    return parser.parse_args()


if __name__ == "__main__":
    main(get_options())
