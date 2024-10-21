import argparse
import torch
from agent import Agent
from learner import DQNLearner
from replay_buffer import ReplayBuffer
from model import DQN
from crafter_wrapper import Env

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialiser l'environnement
    env = Env("train", args)

    # Initialiser le modèle DQN
    action_space = env.action_space.n
    dqn = DQN(action_space).to(device)
    target_dqn = DQN(action_space).to(device)
    target_dqn.load_state_dict(dqn.state_dict())  # Copier les poids du modèle principal

    # Initialiser l'agent, le buffer et le learner
    agent = Agent(dqn, action_space, args.epsilon)
    print(f"Action space size: {action_space}")
    buffer = ReplayBuffer(capacity=args.buffer_size)
    learner = DQNLearner(dqn, target_dqn, action_space, buffer, device)

    # Boucle d'entraînement
    for episode in range(args.num_episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            learner.buffer.add((state, action, reward, next_state, done))
            learner.update(args.batch_size)

            state = next_state
            total_reward += reward

        print(f"Episode {episode}, Total Reward: {total_reward}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_episodes", type=int, default=1000)
    parser.add_argument("--buffer_size", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--logdir", type=str, default="logdir/")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--history_length", type=int, default=4)
    args = parser.parse_args()

    main(args)
