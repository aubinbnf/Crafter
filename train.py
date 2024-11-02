import argparse
import pickle
from pathlib import Path
import torch.nn as nn
import torch.optim as optim
import torch
import random
import torch.nn.functional as F
from src.crafter_wrapper import Env


class RandomAgent:
    """An example Random Agent"""

    def __init__(self, action_num) -> None:
        self.action_num = action_num
        # a uniformly random policy
        self.policy = torch.distributions.Categorical(
            torch.ones(action_num) / action_num
        )

    def act(self, observation):
        """ Since this is a random agent the observation is not used."""
        return self.policy.sample().item()


class DQNModel(nn.Module):
    def __init__(self, action_num):
        super(DQNModel, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)  # (4, 84, 84) -> (32, 20, 20)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2) # (32, 20, 20) -> (64, 9, 9)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)  # (64, 9, 9) -> (64, 7, 7)

        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, action_num)

    def forward(self, x):
        x = F.relu(self.conv1(x))      
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(64 * 7 * 7)  # Flatten
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class DQNAgent:
    def __init__(self, action_num):
        self.epsilon = 1.0           # Init epsilon value
        self.epsilon_min = 0.1       # Min epsilon value
        self.epsilon_decay = 0.995   # Decay factor

        self.action_num = action_num
        self.replay_buffer = []
        self.model = DQNModel(action_num)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def act(self, state):
        # Epsilon greedy
        if random.random() < self.epsilon:
            return random.randint(0, self.action_num - 1)
        else:
            with torch.no_grad():
                q_values = self.model(state)
                return q_values.argmax().item()

    def train(self, batch_size, gamma):

        if len(self.replay_buffer) < batch_size:
            return  # If not enough experiences

        # Sampling batch of experiences
        experiences = random.sample(self.replay_buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*experiences)

        states = torch.stack(states)
        actions = torch.tensor(actions)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.stack(next_states)
        dones = torch.tensor(dones, dtype=torch.float32)

        # Predicted values    
        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze()

        # Target values
        with torch.no_grad():
            next_q_values = self.model(next_states).max(1)[0]
            target_q_values = rewards + (gamma * next_q_values * (1 - dones))

        # Loss calculation
        loss = F.mse_loss(q_values, target_q_values)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

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


def main(opt):
    _info(opt)
    opt.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # opt.device = torch.device("cpu")
    env = Env("train", opt)
    eval_env = Env("eval", opt)
    # agent = RandomAgent(env.action_space.n)
    agent = DQNAgent(env.action_space.n)

    # main loop
    ep_cnt, step_cnt, done = 0, 0, True
    while step_cnt < opt.steps or not done:
        if done:
            ep_cnt += 1
            obs, done = env.reset(), False
        action = agent.act(obs)
        obs, reward, done, info = env.step(action)

        agent.train(batch_size=32, gamma=0.99)

        step_cnt += 1

        # evaluate once in a while
        if step_cnt % opt.eval_interval == 0:
            eval(agent, eval_env, step_cnt, opt)


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