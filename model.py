"""
CNN for classification
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):

    def __init__(self, lr, momentum):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=3)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        self.optim = torch.optim.SGD(self.parameters(), lr=lr, momentum=momentum)

    def forward(self, img, drop_out=True):
        img = F.relu(F.max_pool2d(self.conv1(img), 2))
        img = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(img)), 2))
        img = img.view(-1, 320)
        img = F.relu(self.fc1(img))

        if drop_out:
            img = F.dropout(img, training=self.training)
        img = self.fc2(img)
        return F.log_softmax(img)
