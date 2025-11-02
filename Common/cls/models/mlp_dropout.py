# src/gfm4mpm/models/mlp_dropout.py
import torch
import torch.nn as nn

class MLPDropout(nn.Module):
    def __init__(self, in_dim: int, hidden_dims=(256,128), p=0.2):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.PReLU(), nn.Dropout(p)]
            d = h
        layers += [nn.Linear(d, 1), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x).squeeze(-1)
