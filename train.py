from config import CartPoleConfig, HalfCheetahConfig, InvertedPendulumConfig
from gymnasium.experimental.wrappers import RecordVideoV0
from models import DiscretePolicyModel, ContinuousPolicyModel, BaselineModel
from tqdm import tqdm
from torch.distributions import Categorical, Normal
from torch.utils.tensorboard import SummaryWriter

import gymnasium as gym
import numpy as np
import random
import torch
import torch.nn as nn
import sys

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class Agent():
    def __init__(self, run_id, config):
        self.run_id = run_id
        self.config = config
        self.env = RecordVideoV0(
            env=gym.make(config.game, render_mode='rgb_array'),
            video_folder=f'results/{config.game}/{config.exp_id}/{run_id}/videos',
            episode_trigger=lambda n: n%config.record_freq == 0,
        )
        self.env.reset(seed=run_id)
        self.env.action_space.seed(run_id)

        if config.discrete:
            self.policy_network = DiscretePolicyModel(
                in_features=self.env.observation_space.shape[0],
                out_features=self.env.action_space.n,
            )
        else:
            self.policy_network = ContinuousPolicyModel(
                in_features=self.env.observation_space.shape[0],
                out_features=self.env.action_space.shape[0],
            )
        self.policy_optimizer = torch.optim.Adam(
            self.policy_network.parameters(),
            lr=config.policy_network_lr,
        )

        self.baseline_network = BaselineModel(
            in_features=self.env.observation_space.shape[0],
            out_features=1,
        )
        self.baseline_optimizer = torch.optim.Adam(
            self.baseline_network.parameters(),
            lr=config.baseline_network_lr,
        )

        self.logger = SummaryWriter(
            log_dir=f'results/{config.game}/{config.exp_id}/{run_id}/logs',
        )

    def sample_one_episode(self, step):
        states = []
        rewards = []
        log_probs = []
        entropy = []

        state, _ = self.env.reset()
        while True:
            state = torch.tensor(state, dtype=torch.float32)
            states.append(state)

            self.policy_network.train()
            input = state.unsqueeze(dim=0)
            if self.config.discrete:
                probs = self.policy_network(input)[0]
                distribution = Categorical(probs)
            else:
                mu, sigma = self.policy_network(input)
                distribution = Normal(mu[0], sigma[0])
            
            action = distribution.sample()
            log_probs.append(distribution.log_prob(action).sum())
            entropy.append(distribution.entropy().sum())

            if self.config.discrete:
                action = action.item()
            else:
                action = action.numpy()

            state, reward, terminated, truncated, _, = self.env.step(action)
            rewards.append(reward)
            if terminated or truncated:
                break

        rewards_to_go = rewards.copy()
        for t in reversed(range(len(rewards_to_go) - 1)):
            rewards_to_go[t] += self.config.gamma * rewards_to_go[t+1]

        self.logger.add_scalar('episode_len', len(rewards), step)
        self.logger.add_scalar('episode_reward', sum(rewards), step)
        self.logger.add_scalar('entropy', torch.tensor(entropy).mean(), step)

        s = torch.stack(states)
        rtg = torch.tensor(rewards_to_go, dtype=torch.float32)
        lp = torch.stack(log_probs)
        return s, rtg, lp

    def train_and_get_baseline(self, states, rewards_to_go, step):
        self.baseline_network.train()
        preds = self.baseline_network(states).squeeze(dim=1)
        loss = nn.MSELoss()(preds, rewards_to_go)
        self.baseline_optimizer.zero_grad()
        loss.backward()
        self.baseline_optimizer.step()

        self.logger.add_scalar('baseline_loss', loss.item(), step)
        self.logger.add_scalar('baseline_lr', self.config.baseline_network_lr, step)
        return preds.detach()

    def train(self):
        for i in tqdm(range(self.config.n_episodes)):
            states, rewards_to_go, log_probs = self.sample_one_episode(i)

            baselines = 0
            if self.config.use_baselines:
                baselines = self.train_and_get_baseline(states, rewards_to_go, i)

            advantages = rewards_to_go - baselines
            if self.config.advantage_normalization:
                advantages -= advantages.mean()
                advantages /= advantages.std()

            loss = -torch.sum(advantages * log_probs)
            self.policy_optimizer.zero_grad()
            loss.backward()
            self.policy_optimizer.step()

            self.logger.add_scalar('policy_loss', loss.item(), i)
            self.logger.add_scalar('policy_lr', self.config.policy_network_lr, i)
            self.logger.flush()
    
        self.env.close()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python train.py <env>")
        sys.exit(1)

    env = sys.argv[1]
    env_config = env + "Config"
    if env_config not in globals().keys():
        print(f"{env} is not supported. Supported envs: CartPole, HalfCheetah, InvertedPendulum")
        sys.exit(1)
    config = globals()[env_config]()

    for run_id in [0, 42, 1234, 9999, 11111]:
        set_seed(run_id)
        Agent(run_id, config).train()