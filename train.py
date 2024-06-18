from config import Config
from gymnasium.experimental.wrappers import RecordVideoV0
from logger import Logger
from models import DiscretePolicyModel, ContinuousPolicyModel, BaselineModel
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

class Agent():
    def __init__(self, run_id):
        self.config = Config()
        self.run_id = run_id
        self.env = RecordVideoV0(
            env=gym.make(self.config.game, render_mode='rgb_array'),
            video_folder=f'results/{self.config.exp_id}/{run_id}/videos',
            episode_trigger=lambda n: n%self.config.record_freq == 0,
        )
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

    def sample_one_episode(self):
        states = []
        actions = []
        rewards = []
        state, _ = self.env.reset()
        while True:
            self.policy_network.eval()
            with torch.no_grad():
                input = torch.tensor(state, dtype=torch.float32).unsqueeze(dim=0)
                output = self.policy_network(input)[0]
            if self.config.discrete:
                action = torch.multinomial(output, num_samples=1).item()
            else:
                action = torch.normal(
                    mean=output[:self.n_actions],
                    std=torch.abs(output[self.n_actions:]),
                )

            states.append(state)
            actions.append(action)
            state, reward, terminated, truncated, _ = self.env.step(action)
            rewards.append(reward)
            if terminated or truncated:
                break

        r = torch.zeros(len(states))
        for t in range(len(states) - 1, -1, -1):
            r[t] = rewards[t]
            if t < len(states) - 1:
                r[t] += r[t+1] * self.config.gamma
        s = torch.tensor(np.stack(states), dtype=torch.float32)
        r = r.unsqueeze(dim=1)
        if self.config.discrete:
            a = torch.tensor(actions).unsqueeze(dim=1)
        else:
            a = torch.stack(actions)

        self.logger.add_scalar('episode_len', len(r))
        self.logger.add_scalar('episode_reward', sum(rewards))
        self.logger.add_scalar('episode_discounted_reward', r[0].item())
        self.logger.add_scalar('action.avg', a.float().mean().item())
        self.logger.add_scalar('action.max', a.float().max().item())
        self.logger.add_scalar('action.min', a.float().min().item())
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

    def train(self):
        for i in tqdm(range(self.config.n_batches)):
            states, actions, rewards = self.sample()

            if self.config.use_baseline:
                self.baseline_network.train()
                baseline_preds = self.baseline_network(states)
                baseline_loss = nn.MSELoss()(baseline_preds, rewards)
                self.baseline_optimizer.zero_grad()
                baseline_loss.backward()
                self.baseline_optimizer.step()

                self.baseline_network.eval()
                with torch.no_grad():
                    baselines = self.baseline_network(states)
                
                self.logger.add_scalar('baseline_loss', baseline_loss.item())
                self.logger.add_scalar('baseline_lr', self.config.baseline_network_lr)
                self.logger.add_scalar('baseline', baselines.mean().item())
            else:
                baselines = 0

            advantages = rewards - baselines
            advantages -= advantages.mean()
            advantages /= advantages.std()

            self.policy_network.train()
            output = self.policy_network(states)
            if self.config.discrete:
                log_pi = torch.log(output.gather(1, actions))
            else:
                log_pi = torch.sum(
                    (actions-output[:,:self.n_actions])**2 / output[:,self.n_actions:]**2,
                    dim=1,
                )
            policy_loss = -torch.sum(advantages * log_pi)
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            self.logger.add_scalar('policy_loss', policy_loss.item())
            self.logger.add_scalar('policy_lr', self.config.policy_network_lr)
            self.logger.flush(i)
    
        self.env.close()

if __name__ == "__main__":
    for run_id in [0]:#, 42, 1234, 9999, 11111]:
        set_seed(run_id)
        Agent(run_id).train()