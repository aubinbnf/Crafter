import argparse
import pickle
from pathlib import Path
import torch
from crafter_wrapper import Env
from dqn_agent import CategoricalDQN, CategoricalDQNLearner, Agent, ReplayBuffer


class RandomAgent:
    """An example Random Agent"""

    def __init__(self, action_num) -> None:
        self.action_num = action_num
        # a uniformly random policy
        self.policy = torch.distributions.Categorical(
            torch.ones(action_num) / action_num
        )

    def act(self, observation):
        """Since this is a random agent, the observation is not used."""
        return self.policy.sample().item()


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
    """Use the greedy, deterministic policy, not the epsilon-greedy policy you
    might use during training.
    """
    episodic_returns = []
    for _ in range(opt.eval_episodes):
        obs, done = env.reset(), False
        episodic_returns.append(0)
        while not done:
            action = agent.act(obs.to(opt.device))  # Move observation to GPU
            obs, reward, done, info = env.step(action)
            episodic_returns[-1] += reward

    _save_stats(episodic_returns, crt_step, opt.logdir)


def _info(opt):
    try:
        int(opt.logdir.split("/")[-1])
    except:
        print(
            "Warning, logdir path should end in a number indicating a separate"
            + " training run, else the results might be overwritten."
        )
    if Path(opt.logdir).exists():
        print("Warning! Logdir path exists, results can be corrupted.")
    print(f"Saving results in {opt.logdir}.")
    print(
        f"Observations are of dims ({opt.history_length},84,84),"
        + " with values between 0 and 1."
    )


def main(opt):
    _info(opt)
    opt.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = Env("train", opt)
    eval_env = Env("eval", opt)
    action_space = env.action_space.n

    # Initialize Categorical DQN model and CategoricalDQNLearner
    dqn = CategoricalDQN(action_space, opt.device, atoms=opt.atoms, Vmin=opt.Vmin, Vmax=opt.Vmax).to(opt.device)
    target_dqn = CategoricalDQN(action_space, opt.device, atoms=opt.atoms, Vmin=opt.Vmin, Vmax=opt.Vmax).to(opt.device)
    buffer = ReplayBuffer(opt.buffer_size)  # Use buffer_size from options
    learner = CategoricalDQNLearner(dqn, target_dqn, action_space, buffer, opt.device, opt.logdir, gamma=opt.gamma, lr=opt.lr, Vmin=opt.Vmin, Vmax=opt.Vmax, atoms=opt.atoms)
    
    # Load weights if available
    print("Loading weights...")
    learner.load_weights()

    agent = Agent(dqn, action_space, epsilon=opt.epsilon)  # Use epsilon from options
    ep_cnt, step_cnt, done = 0, 0, True
    while step_cnt < opt.steps or not done:
        if done:
            ep_cnt += 1
            obs, done = env.reset(), False

        action = agent.act(obs.to(opt.device))
        obs, reward, done, info = env.step(action)

        step_cnt += 1

        # Met à jour l'epsilon
        agent.update_epsilon()

        # Met à jour le Categorical DQN learner
        learner.update(opt.batch_size)

        # Évaluation périodique
        if step_cnt % opt.eval_interval == 0:
            eval(agent, eval_env, step_cnt, opt)


def get_options():
    """Configures a parser. Extend this with all the best performing hyperparameters of
    your agent as defaults.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu', help="Device to use (cpu or cuda).")
    parser.add_argument("--logdir", default="logdir/random_agent/0", help="Directory for saving logs.")
    parser.add_argument("--num_episodes", type=int, default=1000, help="Number of episodes for training.")
    parser.add_argument("--buffer_size", type=int, default=10000, help="Size of the replay buffer.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--epsilon", type=float, default=0.1, help="Epsilon for epsilon-greedy policy.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate.")
    parser.add_argument("--Vmin", type=float, default=-10, help="Minimum value for the support in Categorical DQN.")
    parser.add_argument("--Vmax", type=float, default=10, help="Maximum value for the support in Categorical DQN.")
    parser.add_argument("--atoms", type=int, default=51, help="Number of atoms for distributional output in Categorical DQN.")
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
