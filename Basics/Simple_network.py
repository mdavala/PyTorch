import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # Receives an array of input length 240 and outputs with length 120
        self.fc1 = nn.Linear(240, 120)
        # Receives an array of input length 120 and outputs with length 60
        self.fc2 = nn.Linear(120, 60)
        # Receives an array of input length 60 and outputs with length 10
        self.fc3 = nn.Linear(60, 10)

    def forward(self, x):
        # This should be in order how you want to connect the network, 
        # where as in __init__ it's not mandatory to be in order
        # Performs RELU on 'self.fc1 = nn.Linear(240, 120)'
        x = F.relu(self.fc1(x))
        # Performs RELU on 'self.fc2 = nn.Linear(120, 60)'
        x = F.relu(self.fc2(x))
        # Passes the array to 'self.fc3 = nn.Linear(60, 10)'
        x = self.fc3(x)
        return x

net = Net()