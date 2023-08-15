import torch
from torch import nn

class DigitsCNN(nn.Module):
    def __init__(self):
        super(DigitsCNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 5, 3),
            nn.ReLU(),
            nn.Conv2d(5, 10, 3),
            nn.ReLU()
        )
        self.mlp = nn.Sequential(
            nn.Linear(24 * 24 *10, 10),
            nn.ReLU(),
            nn.Linear(10, 10)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = torch.flatten(x, 1)
        x = self.mlp(x)
        return x

