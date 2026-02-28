import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.fc = nn.Linear(16 * 16 * 16, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.relu(self.conv(x)))
        x = torch.flatten(x, start_dim=1)
        return self.fc(x)
