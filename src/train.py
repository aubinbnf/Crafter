import argparse
import pickle
from pathlib import Path
import torch
from crafter_wrapper import Env
from dqn_agent import (DuelingCategoricalDQN, CategoricalDQNLearner, 
                      Agent, PrioritizedReplayBuffer)
import shutil

def save_file(source_path, dest_path, file_name):
    """Helper function to save a file both locally and on Google Drive."""
    source_file = source_path / file_name
    dest_file = dest_path / file_name
    if source_file.exists():
        shutil.copy(source_file, dest_file)
        print(f"{file_name} sauvegardé dans {dest_file}")

def save_stats(episodic_returns, crt_step, logdir, drive_path=None):
    """Save evaluation statistics locally and on Google Drive."""
    logdir = Path(logdir)
    drive_path = Path(drive_path) if drive_path else None

    # Calcul et affichage des résultats
    episodic_returns = torch.tensor(episodic_returns)
    avg_return = episodic_returns.mean().item()
    std_return = episodic_returns.std().item()
    print(f"[{crt_step:06d}] Résultats d'évaluation: R/ep={avg_return:.2f}, std={std_return:.2f}")

    # Sauvegarde locale des statistiques d'évaluation
    eval_stats_path = logdir / "eval_stats.pkl"
    with open(eval_stats_path, "ab") as f:
        pickle.dump({"step": crt_step, "avg_return": avg_return}, f)

    # Sauvegarde sur Google Drive si spécifié
    if drive_path:
        save_file(logdir, drive_path, "eval_stats.pkl")
        save_file(logdir, drive_path, "stats.jsonl")
        save_file(logdir, drive_path, "training_metrics.json")

def eval(agent, env, crt_step, opt, drive_path=None):
    episodic_returns = []
    for _ in range(opt.eval_episodes):
        obs, done = env.reset(), False
        episodic_returns.append(0)
        while not done:
            action = agent.act(obs.to(opt.device), evaluate=True)
            obs, reward, done, info = env.step(action)
            episodic_returns[-1] += reward
    save_stats(episodic_returns, crt_step, opt.logdir, drive_path)

def main(opt):
    drive_path = Path("/content/drive/MyDrive/your_project_folder/random_agent/0")
    drive_path.mkdir(parents=True, exist_ok=True)

    Path(opt.logdir).mkdir(parents=True, exist_ok=True)
    opt.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {opt.device}")

    # Initialisation des environnements et paramètres
    env = Env("train", opt)
    eval_env = Env("eval", opt)
    action_space = env.action_space.n

    dqn = DuelingCategoricalDQN(action_space, opt.device, 
                                atoms=opt.atoms, 
                                Vmin=opt.Vmin, 
                                Vmax=opt.Vmax).to(opt.device)
    
    target_dqn = DuelingCategoricalDQN(action_space, opt.device,
                                       atoms=opt.atoms,
                                       Vmin=opt.Vmin,
                                       Vmax=opt.Vmax).to(opt.device)
    
    buffer = PrioritizedReplayBuffer(capacity=opt.buffer_size, history_length=opt.history_length)
    
    learner = CategoricalDQNLearner(
        dqn, target_dqn, action_space, buffer, opt.device, opt.logdir,
        gamma=opt.gamma, lr=opt.lr, Vmin=opt.Vmin, Vmax=opt.Vmax, atoms=opt.atoms,
        target_update_freq=opt.target_update_freq, tau=0.005
    )

    agent = Agent(
        dqn, action_space,
        epsilon=1.0, 
        epsilon_min=opt.epsilon_min,
        epsilon_decay=opt.epsilon_decay_slow
    )

    total_steps = 0
    episode = 0

    while total_steps < opt.steps:
        episode += 1
        obs, done = env.reset(), False
        episode_reward = 0
        episode_steps = 0

        while not done:
            action = agent.act(obs.to(opt.device))
            next_obs, reward, done, info = env.step(action)
            buffer.add(obs, action, reward, next_obs, done)
            
            obs = next_obs
            episode_reward += reward
            episode_steps += 1
            total_steps += 1

            if len(buffer) >= opt.batch_size:
                learner.update(opt.batch_size)

            agent.update_epsilon()

            if total_steps % opt.eval_interval == 0:
                eval(agent, eval_env, total_steps, opt, drive_path)

        if episode % 10 == 0:
            print(f"Episode {episode}, Steps: {total_steps}, Reward: {episode_reward:.2f}, Epsilon: {agent.epsilon:.3f}")

            save_file(Path(opt.logdir), drive_path, "stats.jsonl")
            save_file(Path(opt.logdir), drive_path, "training_metrics.json")
            save_file(Path(opt.logdir), drive_path, "eval_stats.pkl")

def get_options():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--logdir", default="logdir/categorical_dqn/0", help="Directory for saving logs")
    parser.add_argument("--history-length", default=4, type=int, help="Number of frames to stack")
    
    # Training parameters
    parser.add_argument("--steps", type=int, default=1_000_000, help="Total training steps")
    parser.add_argument("--buffer-size", type=int, default=100000, help="Replay buffer size")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    
    # DQN parameters
    parser.add_argument("--target-update-freq", type=int, default=10000, help="Target network update frequency")
    parser.add_argument("--epsilon-min", type=float, default=0.01, help="Minimum epsilon")
    parser.add_argument("--epsilon-decay", type=float, default=0.9995, help="Epsilon decay rate")
    parser.add_argument("--epsilon-decay-slow", type=float, default=0.9999, help="Slower epsilon decay rate")
    
    # Categorical DQN parameters
    parser.add_argument("--atoms", type=int, default=51, help="Number of atoms")
    parser.add_argument("--Vmin", type=float, default=-10, help="Minimum value of support")
    parser.add_argument("--Vmax", type=float, default=10, help="Maximum value of support")
    
    # Evaluation parameters
    parser.add_argument("--eval-interval", type=int, default=5000, help="Evaluation interval")
    parser.add_argument("--eval-episodes", type=int, default=20, help="Evaluation episodes")
    
    return parser.parse_args()

if __name__ == "__main__":
    options = get_options()
    main(options)