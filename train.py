import argparse
import pickle
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from src.crafter_wrapper import Env
import json


class DQN(nn.Module):
    def __init__(self, action_num):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, action_num)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

class DQNAgent:
    def __init__(self, action_num, device, gamma=0.99, lr=0.0001):
        self.action_num = action_num
        self.device = device
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99999
        self.model = DQN(action_num).to(device)
        self.target_model = DQN(action_num).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.memory = deque(maxlen=20000)
        self.batch_size = 32
        self.episode_loss = []

        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

    def store_experience(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample_memory(self):
        return random.sample(self.memory, self.batch_size)

    def act(self, state):
        # Exploration
        if random.random() < self.epsilon:
            action = random.randint(0, self.action_num - 1)
        else :
            # Exploitation
            state = state.unsqueeze(0).to(self.device)
            with torch.no_grad():  
                q_values = self.model(state)
            action = torch.argmax(q_values, dim=1).item()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return action
    
    def train(self):
        """Trains the model using a mini-batch sampled from memory"""
        if len(self.memory) < self.batch_size:
            return  # If the memory has not enough experiences
        
        batch = self.sample_memory()
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.stack(states).to(self.device)
        actions = torch.tensor(actions).to(self.device)
        rewards = torch.tensor(rewards).to(self.device)
        next_states = torch.stack(next_states).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_actions = self.model(next_states).argmax(1).unsqueeze(1)
        next_q_values = self.target_model(next_states).gather(1, next_actions).squeeze(1)
        
        # targets for Q-values
        targets = rewards + self.gamma * next_q_values * (1 - dones)
        
        # loss
        loss = nn.SmoothL1Loss()(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.episode_loss.append(loss.item())

    def end_episode(self):
        if self.episode_loss:
            avg_loss = sum(self.episode_loss) / len(self.episode_loss)
            self.episode_loss = []
            return avg_loss
        return None

    def save_model(self, path="dqn_model.pth"):
        torch.save(self.model.state_dict(), path)

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
    

def _save_stats(episodic_returns, crt_step, path):
    # save the evaluation stats
    episodic_returns = torch.tensor(episodic_returns)
    avg_return = episodic_returns.mean().item()
    print(
        "[{:06d}] eval results: R/ep={:03.2f}, std={:03.2f}.".format(
            crt_step, avg_return, episodic_returns.std().item()
        )
    )
    with open(path + "/eval_stats.pkl", "ab") as f:
        pickle.dump({"step": crt_step, "avg_return": avg_return}, f)


def eval(agent, env, crt_step, opt):
    """ Use the greedy, deterministic policy, not the epsilon-greedy policy you
    might use during training.
    """
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
        print(
            "Warning, logdir path should end in a number indicating a separate"
            + "training run, else the results might be overwritten."
        )
    if Path(opt.logdir).exists():
        print("Warning! Logdir path exists, results can be corrupted.")
    print(f"Saving results in {opt.logdir}.")
    print(
        f"Observations are of dims ({opt.history_length},84,84),"
        + "with values between 0 and 1."
    )

def save_progress(agent, statistics, model_path="model_weights.pth", stats_path="training_statistics.json"):
    agent.save_model(model_path)
    
    with open(stats_path, "w") as f:
        json.dump(statistics, f, indent=4)

    print(f"Progress saved: model -> {model_path}, statistics -> {stats_path}")

def main(opt):
    _info(opt)
    opt.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"The code runs on : {opt.device}")
    env = Env("train", opt)
    eval_env = Env("eval", opt)
    agent = DQNAgent(env.action_space.n, opt.device)
    target_update_interval = 5000

    ep_cnt, step_cnt, done = 0, 0, True
    episode_rewards, episode_steps = 0, 0
    statistics = {
        "episode_rewards": [],
        "episode_steps": [],
        "epsilon": [],
        "loss_per_episode": [],
        "eval_rewards": []
    }

    while step_cnt < opt.steps or not done:
        if done:
            statistics["episode_rewards"].append(episode_rewards)
            statistics["episode_steps"].append(episode_steps)
            statistics["epsilon"].append(agent.epsilon)

            ep_cnt += 1
            episode_rewards, episode_steps = 0, 0
            obs, done = env.reset(), False

            avg_loss = agent.end_episode()
            if avg_loss is not None:
                statistics["loss_per_episode"].append(avg_loss)

        action = agent.act(obs)
        next_obs, reward, done, info = env.step(action)

        episode_rewards += reward
        episode_steps += 1

        agent.store_experience(obs, action, reward, next_obs, done)
        agent.train()

        if step_cnt % target_update_interval == 0:
            agent.update_target_model()

        obs = next_obs
        step_cnt += 1

        if step_cnt % opt.eval_interval == 0:
            eval_reward = eval(agent, eval_env, step_cnt, opt)
            statistics["eval_rewards"].append(eval_reward)
            save_progress(agent, statistics)
        
    save_progress(agent, statistics)


def get_options():
    """ Configures a parser. Extend this with all the best performing hyperparameters of
        your agent as defaults.

        For devel purposes feel free to change the number of training steps and
        the evaluation interval.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", default="logdir/random_agent/0")
    parser.add_argument(
        "--steps",
        type=int,
        metavar="STEPS",
        default=1_000_000,
        help="Total number of training steps.",
    )
    parser.add_argument(
        "-hist-len",
        "--history-length",
        default=4,
        type=int,
        help="The number of frames to stack when creating an observation.",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=100_000,
        metavar="STEPS",
        help="Number of training steps between evaluations",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=20,
        metavar="N",
        help="Number of evaluation episodes to average over",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(get_options())
