import torch
import torch.nn as nn
from torch import Tensor

class FCBlock(nn.Module):
    def __init__(
            self,
            input_dim,
            width,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.width = width
        self.fc1 = nn.Linear(input_dim, width)
        self.fc2 = nn.Linear(width, input_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        output = self.relu(identity + x)
        return output