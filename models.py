import torch.nn as nn
import torch.nn.functional as F

class DiscretePolicyModel(nn.Module):
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

class ContinuousPolicyModel(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc_mu = nn.Linear(128, out_features)
        self.fc_log_sigma = nn.Linear(128, out_features)

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        mu = self.fc_mu(x)
        log_sigma = self.fc_log_sigma(x)
        sigma = F.softplus(log_sigma) + 1e-6
        return mu, sigma

class BaselineModel(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, out_features),
        )

    def forward(self, x):
        return self.model(x)