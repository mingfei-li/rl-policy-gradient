from config import Config
from gymnasium.experimental.wrappers import RecordVideoV0
from logger import Logger
from models import DiscretePolicyModel, ContinuousPolicyModel, BaselineModel
from tqdm import tqdm
from torch.distributions import Categorical, Normal

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

class Agent():
    def __init__(self, run_id):
        self.config = Config()
        self.run_id = run_id
        self.env = gym.make(self.config.game, render_mode='rgb_array')
        # self.env = RecordVideoV0(
        #     env=gym.make(self.config.game, render_mode='rgb_array'),
        #     video_folder=f'results/{self.config.exp_id}/{run_id}/videos',
        #     episode_trigger=lambda n: n%self.config.record_freq == 0,
        # )
        self.env.reset(seed=run_id)
        self.env.action_space.seed(run_id)
        if self.config.discrete:
            self.n_actions = self.env.action_space.n
        else:
            self.n_actions = self.env.action_space.shape[0]

        self.logger = Logger(log_dir=f'results/{self.config.exp_id}/{run_id}/logs')

        if self.config.discrete:
            self.policy_network = DiscretePolicyModel(
                in_features=self.env.observation_space.shape[0],
                out_features=self.n_actions,
            )
        else:
            self.policy_network = ContinuousPolicyModel(
                in_features=self.env.observation_space.shape[0],
                out_features=self.n_actions,
            )
        self.policy_optimizer = torch.optim.Adam(
            self.policy_network.parameters(),
            lr=self.config.policy_network_lr,
        )

        self.baseline_network = BaselineModel(
            in_features=self.env.observation_space.shape[0],
            out_features=1,
        )
        self.baseline_optimizer = torch.optim.Adam(
            self.baseline_network.parameters(),
            lr=self.config.baseline_network_lr,
        )

    def transform(self, action):
        if self.config.discrete:
            return action
        
        low = torch.tensor(self.env.action_space.low)
        high = torch.tensor(self.env.action_space.high)
        return (torch.tanh(action) * (high-low) + (high+low)) / 2.0

    def sample_one_episode(self):
        states = []
        actions = []
        rewards = []
        state, _ = self.env.reset()
        while True:
            self.policy_network.eval()
            input = torch.tensor(state, dtype=torch.float32).unsqueeze(dim=0)
            if self.config.discrete:
                with torch.no_grad():
                    prob = self.policy_network(input)
                distribution = Categorical(prob.squeeze(dim=0))
                action = distribution.sample().item()
            else:
                with torch.no_grad():
                    mu, sigma = self.policy_network(input)
                distribution = Normal(mu.squeeze(dim=0), sigma.squeeze(dim=0))
                action = distribution.sample()

            states.append(state)
            actions.append(action)
            (
                state, reward, terminated, truncated, _,
            ) = self.env.step(self.transform(action))
            rewards.append(reward)
            if terminated or truncated:
                break

        r = torch.zeros(len(states))
        for t in range(len(states) - 1, -1, -1):
            r[t] = rewards[t]
            if t < len(states) - 1:
                r[t] += r[t+1] * self.config.gamma
        s = torch.tensor(np.stack(states), dtype=torch.float32)
        if self.config.discrete:
            a = torch.tensor(actions)
        else:
            a = torch.stack(actions)
        return s, a, r

    def sample(self):
        states = []
        actions = []
        rewards = []

        for _ in range(self.config.batch_size):
            s, a, r = self.sample_one_episode()
            states.append(s)
            actions.append(a)
            rewards.append(r)

        return torch.concat(states), torch.concat(actions), torch.concat(rewards)

    def train_baseline(self):
        for _ in range(self.config.n_batches_baseline):
            states, _, rewards = self.sample()
            self.baseline_network.train()
            baseline_preds = self.baseline_network(states).squeeze(dim=1)
            baseline_loss = nn.MSELoss()(baseline_preds, rewards)
            self.baseline_optimizer.zero_grad()
            baseline_loss.backward()
            self.baseline_optimizer.step()
            self.logger.add_scalar('baseline_loss', baseline_loss.item())
            self.logger.add_scalar('baseline_lr', self.config.baseline_network_lr)

    def eval(self):
        _, a, r = self.sample_one_episode()
        ta = self.transform(a)
        self.logger.add_scalar('episode_len', len(r))
        self.logger.add_scalar('episode_discounted_reward', r[0].item())
        self.logger.add_scalar('action.avg', a.float().mean().item())
        self.logger.add_scalar('action.max', a.float().max().item())
        self.logger.add_scalar('action.min', a.float().min().item())
        self.logger.add_scalar('transformed_action.avg', ta.float().mean().item())
        self.logger.add_scalar('transformed_action.max', ta.float().max().item())
        self.logger.add_scalar('transformed_action.min', ta.float().min().item())


    def train(self):
        for i in tqdm(range(self.config.n_batches)):
            states, actions, rewards = self.sample()

            if self.config.n_batches_baseline > 0:
                self.train_baseline()
                self.baseline_network.eval()
                with torch.no_grad():
                    baselines = self.baseline_network(states).squeeze(dim=1)
                self.logger.add_scalar('baseline', baselines.mean().item())
            else:
                baselines = 0

            advantages = rewards - baselines
            if self.config.advantage_normalization:
                advantages -= advantages.mean()
                advantages /= advantages.std()

            self.policy_network.train()
            if self.config.discrete:
                prob = self.policy_network(states)
                distribution = Categorical(prob)
                entropy = distribution.entropy()
                log_prob = distribution.log_prob(actions)
            else:
                mu, sigma = self.policy_network(states)
                distribution = Normal(mu, sigma)
                entropy = distribution.entropy().sum(dim=1)
                log_prob = distribution.log_prob(actions).sum(dim=1)

                self.logger.add_scalar('mu.avg', mu.mean().item())
                self.logger.add_scalar('mu.max', mu.max().item())
                self.logger.add_scalar('mu.min', mu.min().item())
                self.logger.add_scalar('sigma.avg', sigma.mean().item())
                self.logger.add_scalar('sigma.max', sigma.max().item())
                self.logger.add_scalar('sigma.min', sigma.min().item())

            policy_loss = -torch.sum(advantages * distribution.log_prob(actions))
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            self.eval()

            self.logger.add_scalar('entropy.avg', entropy.mean().item())
            self.logger.add_scalar('entropy.max', entropy.max().item())
            self.logger.add_scalar('entropy.min', entropy.min().item())
            self.logger.add_scalar('log_prob.avg', log_prob.mean().item())
            self.logger.add_scalar('log_prob.max', log_prob.max().item())
            self.logger.add_scalar('log_prob.min', log_prob.min().item())
            self.logger.add_scalar('policy_loss', policy_loss.item())
            self.logger.add_scalar('policy_lr', self.config.policy_network_lr)
            self.logger.flush(i)
    
        self.env.close()

if __name__ == "__main__":
    for run_id in [0]:#, 42, 1234, 9999, 11111]:
        set_seed(run_id)
        Agent(run_id).train()