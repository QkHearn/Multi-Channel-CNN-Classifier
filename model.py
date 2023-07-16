import torch
from torch import nn


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # 顺序层
        self.conv1 = nn.Conv2d(4, 256, kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.BatchNorm2d(64)
        self.fc = nn.Linear(256, 1)
        # 通用层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 卷积，激活，池化
        x = self.pool(torch.relu(self.norm1(self.conv1(x))))
        x = self.pool(torch.relu(self.norm2(self.conv2(x))))
        x = x.view(x.size(0), 4, -1)
        x = self.sigmoid(self.fc(x))
        return x
