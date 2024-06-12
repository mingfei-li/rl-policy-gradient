from collections import deque
from gymnasium.experimental.wrappers import RecordVideoV0
from statistics import mean
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym
import torch
import torch.nn as nn

class Config():
    n_training_episodes = 5000
    lr = 0.001
    gamma = 1
    exp_id = 'exp-3'

class MLP(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, out_features),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        return self.model(x)

class Logger():
    def __init__(self, log_dir):
        self.scalar_buffer = {}
        self.writer = SummaryWriter(log_dir)

    def add_scalar(self, key, value):
        if key not in self.scalar_buffer.keys():
            self.scalar_buffer[key] = deque(maxlen=50)
        self.scalar_buffer[key].append(value)

    def flush(self, t):
        for key, value in self.scalar_buffer.items():
            self.writer.add_scalar(key, mean(value), t)
        self.writer.flush()

def train():
    config = Config()

    env = gym.make('CartPole-v0', render_mode='rgb_array')
    env = RecordVideoV0(env, video_folder=f'results/{config.exp_id}/videos')
    logger = Logger(log_dir=f'results/{config.exp_id}/logs')

    model = MLP(
        in_features=env.observation_space.shape[0],
        out_features=env.action_space.n,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    for i in tqdm(range(config.n_training_episodes)):
        states = []
        actions = []
        rewards = []

        state, _ = env.reset()
        while True:
            model.eval()
            with torch.no_grad():
                pi = model(torch.tensor(state).unsqueeze(dim=0))[0]
            action = torch.multinomial(pi, num_samples=1).item()
            states.append(state)
            actions.append(action)
            state, reward, terminated, truncated, _ = env.step(action)
            rewards.append(reward)
            if terminated or truncated:
                break

        model.train()
        g = torch.zeros(len(states))
        pi_a = torch.zeros(len(states))
        for t in range(len(states) - 1, -1, -1):
            g[t] = rewards[t]
            if t < len(states) - 1:
                g[t] += g[t+1] * config.gamma
            pi_a[t] = model(torch.tensor(states[t]).unsqueeze(dim=0))[0][actions[t]]

        loss = -torch.sum(g * torch.log(pi_a))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        logger.add_scalar('episode_len', len(g))
        logger.add_scalar('reward', g[0].item())
        logger.add_scalar('loss', loss.item())
        logger.add_scalar('lr', config.lr)
        logger.flush(i)

    env.close()


if __name__ == "__main__":
    train()