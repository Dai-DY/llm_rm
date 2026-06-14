import torch.nn as nn


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim, expansion=2, dropout=0.1):
        super().__init__()
        hidden = dim * expansion
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.net(x)
