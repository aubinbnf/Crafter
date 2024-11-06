import argparse
import pickle
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from src.crafter_wrapper import Env
import torch.nn.functional as F
    
class PolicyNetwork(nn.Module):
    def __init__(self, action_num):
        super(PolicyNetwork, self).__init__()
        # Couche convolutionnelle pour extraire des caractéristiques
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)  # entrée: 4 canaux, sortie: 32 canaux
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)  # entrée: 32 canaux, sortie: 64 canaux
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)  # entrée: 64 canaux, sortie: 64 canaux
        self.fc1 = nn.Linear(64 * 7 * 7, 512)  # ajustez cette taille en fonction de la sortie des convolutions
        self.fc2 = nn.Linear(512, action_num)  # sortie: nombre d'actions

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(0)
        # Passez les observations à travers les couches convolutionnelles
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Applatir la sortie pour les couches fully connected
        x = x.view(x.size(0), -1)  # (batch_size, features)
        x = F.relu(self.fc1(x))
        
        # Utilisez softmax pour obtenir des probabilités
        action_probs = F.softmax(self.fc2(x), dim=-1)
        return action_probs

class REINFORCEAgent:
    def __init__(self, action_num, input_size, device, lr=0.01):
        self.device = device
        self.action_num = action_num
        self.policy_net = PolicyNetwork(action_num).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

    def act(self, observation):
        """Choisir une action en fonction de l'état actuel."""
        if observation.device != self.device:
            state_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        else:
            state_tensor = torch.FloatTensor(observation).unsqueeze(0)

        action_probs = self.policy_net(state_tensor)
        action = np.random.choice(self.action_num, p=action_probs.detach().numpy()[0])
        return action

    def update(self, states, actions, rewards, gamma=0.99):
        """Mettre à jour la politique en fonction des récompenses."""
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)

        for state, action, G in zip(states, actions, returns):
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action_probs = self.policy_net(state_tensor)
            loss = -torch.log(action_probs[0][action]) * G

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

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
    env = Env("train", opt)
    eval_env = Env("eval", opt)
    agent = REINFORCEAgent(env.action_space.n, opt.history_length * 84 * 84, opt.device)  # Assurez-vous que la taille de l'entrée est correcte

    # main loop
    ep_cnt, step_cnt, done = 0, 0, True
    while step_cnt < opt.steps or not done:
        if done:
            ep_cnt += 1
            obs, done = env.reset(), False
            states, actions, rewards = [], [], []  # Stocker les états, actions et récompenses

        action = agent.act(obs)
        next_obs, reward, done, info = env.step(action)

        states.append(obs)
        actions.append(action)
        rewards.append(reward)

        obs = next_obs
        step_cnt += 1

        # évaluation une fois de temps en temps
        if step_cnt % opt.eval_interval == 0:
            eval(agent, eval_env, step_cnt, opt)

        # Si l'épisode est terminé, mettre à jour la politique
        if done:
            agent.update(states, actions, rewards)


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
