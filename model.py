"""
CNN for recognition
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):

    def __init__(self, lr, momentum):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1)
        self.drop1 = nn.Dropout2d(0.25)
        self.drop2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
        self.optim = torch.optim.SGD(self.parameters(), lr=lr, momentum=momentum)

    def forward(self, img, drop_out=True):
        img = self.conv1(img)
        img = F.relu(img)
        img = self.conv2(img)
        img = F.relu(img)
        img = F.max_pool2d(img, 2)

        if drop_out:
            img = self.drop1(img)

        img = torch.flatten(img, 1)
        img = self.fc1(img)
        img = F.relu(img)

        if drop_out:
            img = self.drop2(img)

        img = self.fc2(img)
        out = F.log_softmax(img, dim=1)
        return out