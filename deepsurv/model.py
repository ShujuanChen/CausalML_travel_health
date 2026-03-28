import torch
import torch.nn as nn


class DeepSurvNet(nn.Module):

    def __init__(self, input_dim, hidden_dim=64, input_dropout=0.3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Dropout(p=input_dropout),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features):
        return self.network(features.float())
