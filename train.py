"""
Convolutional Neural Network for character recognition
"""
import os
import torch
import torchvision
import torch.nn.functional as F
from model import CNN

# configuration for training and testing
lr = 0.1
momentum = 0.5
n_epochs = 3
trainSize = 64
testSize = 1000
logInterval = 10
rootData = "./data/"
modelDir= "./model/"

randomSeed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(randomSeed)

# create dirs for data and model
if not os.path.exists(rootData):
    os.mkdir(rootData)
if not os.path.exists(modelDir):
    os.mkdir(modelDir)

# load data
trainLoader = torch.utils.data.DataLoader(
                    torchvision.datasets.MNIST(rootData, train=True, download=True,
                                            transform=torchvision.transforms.Compose([
                                            torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize((0.1307,), (0.3081,))
                                            ])),
                    batch_size=trainSize,
                    shuffle=True)

testLoader = torch.utils.data.DataLoader(
                    torchvision.datasets.MNIST(rootData, train=False, download=True,
                                               transform=torchvision.transforms.Compose([
                                                   torchvision.transforms.ToTensor(),
                                                   torchvision.transforms.Normalize((0.1307,), (0.3081,))
                                               ])),
                    batch_size=testSize,
                    shuffle=True)


# initialize network
cnn = CNN(lr, momentum)

trainLosses = []
trainCounter = []
testLosses = []
testCounter = [i * len(trainLoader.dataset) for i in range(n_epochs + 1)]


def train(epoch):
    # set network in training mode
    cnn.train()

    for idx, (img, target) in enumerate(trainLoader):
        cnn.optim.zero_grad()
        y_hat = cnn(img)

        loss = F.nll_loss(y_hat, target)  # ?
        loss.backward()
        cnn.optim.step()

        # report loss
        if idx % logInterval == 0:
            print(f"Train Epoch {epoch} [{idx * len(img)}/{len(trainLoader.dataset)}\
                    ({100. * idx / len(trainLoader)}%)]\t Loss: {loss.item()}")

            trainLosses.append(loss.item())
            trainCounter.append((idx * 64) + ((epoch - 1) * len(trainLoader.dataset)))

            # save network params
            torch.save(cnn.state_dict(), "./model/model.pth")
            torch.save(cnn.optim.state_dict(), "./model/optimizer.pth")

def test():
    cnn.eval()
    loss = 0
    correct = 0
    # ?
    with torch.no_grad():
        for img, target in testLoader:
            y_hat = cnn(img)
            loss += F.nll_loss(y_hat, target, size_average=False).item()
            pred = y_hat.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    loss /= len(testLoader.dataset)
    testLosses.append(loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        loss, correct, len(testLoader.dataset),
        100. * correct / len(testLoader.dataset)))


# train and test
for epoch in range(1, n_epochs + 1):
    train(epoch)
    test()