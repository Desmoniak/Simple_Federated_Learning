import torch.nn as nn
from torch import relu

class MyNet(nn.Module):
    def __init__(self, _Input, _Output):
        super(MyNet, self).__init__()
        self.fc1 = nn.Linear(_Input, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 64)
        self.fc6 = nn.Linear(64, _Output)

    def forward(self, x):
        x = relu(self.fc1(x))
        x = relu(self.fc2(x))
        x = relu(self.fc3(x))
        x = relu(self.fc4(x))
        x = relu(self.fc5(x))
        x = self.fc6(x)
        return x