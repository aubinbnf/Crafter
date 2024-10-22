import pathlib
from collections import deque
import time

import crafter
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt


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
            self.state_buffer.append(torch.zeros(84, 84, device=self.device))
        obs = self.env.reset()
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device).div_(255)
        self.state_buffer.append(obs)
        # Retourner avec la forme correcte (channels, height, width)
        return torch.stack(list(self.state_buffer), 0)  # Shape: (4, 84, 84)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device).div_(255)
        self.state_buffer.append(obs)
        # Retourner avec la forme correcte (channels, height, width)
        return torch.stack(list(self.state_buffer), 0), reward, done, info

    # def render(self):
    #     # obs = self.env.get_observation()  # Assure-toi que cette méthode existe
    #     plt.imshow(obs)  # Affiche l'état
    #     plt.axis('off')  # Masquer les axes
    #     plt.pause(0.01)  # Pause pour permettre l'affichage


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
        # Couches de convolution
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calcul de la taille après aplatissement
        self.flatten_size = 64 * 7 * 7  # 3136
        
        # Couches fully connected
        self.fc1 = nn.Linear(self.flatten_size, 512)
        self.fc2 = nn.Linear(512, action_space)

    def forward(self, x):
        # Assurer que l'entrée a la bonne forme (batch_size, channels, height, width)
        if len(x.shape) == 3:
            x = x.unsqueeze(0)  # Ajouter dimension batch si absente
            
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        x = x.reshape(x.size(0), -1)  # Aplatir en préservant la dimension batch
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x  


    def get_conv_output_size(self, input_shape):
        """
        Fonction de debug pour vérifier les dimensions
        """
        device = next(self.parameters()).device  # Obtenir le device du modèle
        x = torch.zeros(1, *input_shape).to(device)  # Crée un tensor de test sur le même device que le modèle
        x = F.relu(self.conv1(x))
        print(f"Après conv1: {x.shape}")
        x = F.relu(self.conv2(x))
        print(f"Après conv2: {x.shape}")
        x = F.relu(self.conv3(x))
        print(f"Après conv3: {x.shape}")
        x = x.reshape(x.size(0), -1)
        print(f"Après aplatissement: {x.shape}")
        return x.shape[1]

class Agent:    
    def __init__(self, dqn, action_space, epsilon):
        self.dqn = dqn
        self.action_space = action_space
        self.epsilon = epsilon

    def act(self, state):  # Ajouter cette méthode
        return self.select_action(state)

    def select_action(self, state):
        if random.random() < self.epsilon:  # Exploration
            action = random.randint(0, self.action_space - 1)
        else:  # Exploitation
            with torch.no_grad():
                # Assurer que state a la bonne forme pour le réseau
                if not isinstance(state, torch.Tensor):
                    state = torch.FloatTensor(state).to(next(self.dqn.parameters()).device)
                
                # S'assurer que l'état a la bonne forme (batch, channels, height, width)
                if len(state.shape) == 3:  # Si (channels, height, width)
                    state = state.unsqueeze(0)  # Ajouter dimension batch
                
                q_values = self.dqn(state)
                action = q_values.argmax(dim=1).item()

        return action


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def add(self, experience):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = experience
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch_size = min(batch_size, len(self.buffer))  # Éviter de sampler plus que ce qu'on a
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
            return None  # Wait for the buffer to be filled

        # Retrieve a batch of samples from the buffer
        transitions = self.buffer.sample(batch_size)
        state, action, reward, next_state, done = zip(*transitions)

        # Convertir en tenseurs et déplacer vers le device approprié
        state = torch.stack(state).to(self.device)  # Shape: (batch_size, 4, 84, 84)
        next_state = torch.stack(next_state).to(self.device)  # Shape: (batch_size, 4, 84, 84)
        action = torch.tensor(action, dtype=torch.long).to(self.device)  # Shape: (batch_size,)
        reward = torch.tensor(reward, dtype=torch.float32).to(self.device)  # Shape: (batch_size,)
        done = torch.tensor(done, dtype=torch.float32).to(self.device)  # Shape: (batch_size,)
        # print("Types des tenseurs:")
        # print(f"state dtype: {state.dtype}, shape: {state.shape}")
        # print(f"action dtype: {action.dtype}, shape: {action.shape}")
        # print(f"reward dtype: {reward.dtype}, shape: {reward.shape}")
        # print(f"done dtype: {done.dtype}, shape: {done.shape}")

        # Calculate current Q-values
        current_q_values = self.dqn(state)  # Shape: (batch_size, action_space)
        current_q_value = current_q_values.gather(1, action.unsqueeze(1)).squeeze(1)  # Shape: (batch_size,)

        # Calculate target Q-values
        with torch.no_grad():
            next_q_values = self.target_dqn(next_state)  # Shape: (batch_size, action_space)
            max_next_q_values = next_q_values.max(1)[0]  # Shape: (batch_size,)
            target_q_value = reward + self.gamma * max_next_q_values * (1 - done)  # Shape: (batch_size,)

        # Calculate loss and update
        loss = F.mse_loss(current_q_value, target_q_value)
        
        # Debugging information
        if torch.isnan(loss):
            print("NaN detected in loss!")
            print(f"Current Q-values: {current_q_value}")
            print(f"Target Q-values: {target_q_value}")
            print(f"Rewards: {reward}")
            print(f"Done flags: {done}")
            return None

        self.optimizer.zero_grad()
        loss.backward()
        # Optional: gradient clipping
        torch.nn.utils.clip_grad_norm_(self.dqn.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss.item()


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
    
    # Initialize environment
    env = Env("train", opt)
    eval_env = Env("eval", opt)
    
    # Initialize DQN model and agent
    action_space = env.action_space.n
    dqn = DQN(action_space).to(opt.device)
    target_dqn = DQN(action_space).to(opt.device)
    target_dqn.load_state_dict(dqn.state_dict())  # Copy weights

    # Charger les poids si disponibles
    checkpoint_path = f"{opt.logdir}/dqn_weights_step_100.pth"  # Dernier poids enregistré
    print(f"Chargement des poids depuis {checkpoint_path}")
    if Path(checkpoint_path).exists() :  # Vérifie si les poids existent et si on veut les charger
        dqn.load_state_dict(torch.load(checkpoint_path))
        print(f"Poids du modèle chargés depuis {checkpoint_path}")
    else:
        print("Aucun poids chargé, démarrage avec un modèle aléatoire.")


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

        # Évaluer à intervalles
        if step_cnt % opt.eval_interval == 0:
            eval(agent, eval_env, step_cnt, opt)
            # Sauvegarde des poids du modèle
            torch.save(dqn.state_dict(), f"{opt.logdir}/dqn_weights_step_{step_cnt}.pth")



def get_options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_episodes", type=int, default=1000)
    parser.add_argument("--buffer_size", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--logdir", default="logdir/")
    parser.add_argument("--steps", type=int, default=1_000_000, help="Total number of training steps.")
    parser.add_argument("--history_length", type=int, default=4)
    parser.add_argument("--eval_interval", type=int, default=10, metavar="STEPS", help="Number of training steps between evaluations")
    parser.add_argument("--eval_episodes", type=int, default=20, metavar="N", help="Number of evaluation episodes to average over")
    parser.add_argument("--load_weights", action='store_true', help="Load weights from the last checkpoint if available.")  # Nouvel argument
    return parser.parse_args()



if __name__ == "__main__":
    main(get_options())
    
