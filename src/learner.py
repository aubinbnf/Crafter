import torch
import torch.nn.functional as F

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
            return  # Attendre que le buffer soit rempli

        # Récupérer un batch d'exemples depuis le buffer
        transitions = self.buffer.sample(batch_size)
        state, action, reward, next_state, done = zip(*transitions)

        state = torch.stack(state).to(self.device)
        action = torch.tensor(action).to(self.device)
        reward = torch.tensor(reward).to(self.device)
        next_state = torch.stack(next_state).to(self.device)
        done = torch.tensor(done).to(self.device)

        # Calcul des valeurs Q actuelles
        q_values = self.dqn(state).gather(1, action.unsqueeze(1)).squeeze(1)

        # Calcul des valeurs Q cibles
        with torch.no_grad():
            target_q_values = self.target_dqn(next_state).max(1)[0]
            target = reward + self.gamma * target_q_values * (1 - done)

        # Mise à jour du modèle DQN
        loss = F.mse_loss(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
