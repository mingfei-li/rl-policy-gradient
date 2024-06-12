from config import Config
from gymnasium.experimental.wrappers import RecordVideoV0
from logger import Logger
from models import PolicyMLPModel, BaselineMLPModel
from tqdm import tqdm

import gymnasium as gym
import numpy as np
import random
import torch
import torch.nn as nn

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train(run_id):
    config = Config()
    env = RecordVideoV0(
        env=gym.make(config.game, render_mode='rgb_array'),
        video_folder=f'results/{config.exp_id}/{run_id}/videos',
        episode_trigger=lambda n: n%config.record_freq == 0,
    )
    env.reset(seed=run_id)
    env.action_space.seed(run_id)

    logger = Logger(log_dir=f'results/{config.exp_id}/{run_id}/logs')

    policy_network = PolicyMLPModel(
        in_features=env.observation_space.shape[0],
        out_features=env.action_space.n,
    )
    policy_optimizer = torch.optim.Adam(policy_network.parameters(), lr=config.lr)

    baseline_network = BaselineMLPModel(
        in_features=env.observation_space.shape[0],
        out_features=1,
    )
    baseline_optimizer = torch.optim.Adam(baseline_network.parameters(), lr=config.lr)

    for i in tqdm(range(config.n_training_episodes)):
        states = []
        actions = []
        rewards = []

        state, _ = env.reset()
        while True:
            policy_network.eval()
            with torch.no_grad():
                pi = policy_network(torch.tensor(state).unsqueeze(dim=0))[0]
            action = torch.multinomial(pi, num_samples=1).item()
            states.append(state)
            actions.append(action)
            state, reward, terminated, truncated, _ = env.step(action)
            rewards.append(reward)
            if terminated or truncated:
                break

        g = torch.zeros(len(states))
        for t in range(len(states) - 1, -1, -1):
            g[t] = rewards[t]
            if t < len(states) - 1:
                g[t] += g[t+1] * config.gamma
        s = torch.tensor(np.stack(states))
        g = g.unsqueeze(dim=1)

        if config.use_baseline:
            baseline_network.train()
            baseline_preds = baseline_network(s)
            baseline_loss = nn.MSELoss()(baseline_preds, g)
            baseline_optimizer.zero_grad()
            baseline_loss.backward()
            baseline_optimizer.step()

            baseline_network.eval()
            with torch.no_grad():
                b = baseline_network(s)
            
            logger.add_scalar('baseline_loss', baseline_loss.item())
            logger.add_scalar('baseline', b.mean().item())
        else:
            b = 0

        a = g - b
        # a -= a.mean()
        # a /= a.std()

        policy_network.train()
        pi = policy_network(s).gather(1, torch.tensor(actions).unsqueeze(dim=1))
        policy_loss = -torch.sum(a * torch.log(pi))
        policy_optimizer.zero_grad()
        policy_loss.backward()
        policy_optimizer.step()

        logger.add_scalar('episode_len', len(g))
        logger.add_scalar('reward', g[0].item())
        logger.add_scalar('policy_loss', policy_loss.item())
        logger.add_scalar('lr', config.lr)
        logger.add_scalar('g', g.mean().item())
        logger.flush(i)

    env.close()


if __name__ == "__main__":
    for run_id in [0]:#, 42, 1234, 9999, 11111]:
        set_seed(run_id)
        train(run_id)