import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets,transforms
import torch.utils.data as D
import torch.optim as optim
from torch.autograd import Variable

import time
import random
import numpy as np
import matplotlib.pyplot as plt

from BinarisedNetwork import *
import BinarisedUtil








train_set = datasets.MNIST('./data', train=True, download=True,
                   transform=transforms.Compose([transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))
train_loader = D.DataLoader(train_set,batch_size=10,shuffle=True)

test_set = datasets.MNIST('./data', train=False, download=True,
                   transform=transforms.Compose([transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))
test_loader = D.DataLoader(test_set,batch_size=10,shuffle=True)


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            BinaryLinear(in_features=1*28*28, out_features=1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            # First Hidden layer
            BinaryLinear(in_features=1024, out_features=1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            # Second Hidden layer
            BinaryLinear(in_features=1024, out_features=1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            # Third Hidden layer
            BinaryLinear(in_features=1024, out_features=1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),

            BinaryLinear(in_features=1024, out_features=10),
            nn.LogSoftmax()
        )

    def forward(self, x):
        return self.layers(x)

network = Network()
print(network)

# Decalring the optimizer
optimizer =  optim.Adam(network.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
binarisation = BinarisedUtil.BinarisedUtil(network)


def train(epoch):
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()

        binarisation.Binarization()

        output = network(data)
        loss = criterion(output, target)
        loss.backward()

        binarisation.Restore()
        binarisation.UpdateBinaryGradWeight()

        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.data.item()))



accur = []


def test():
    network.eval()
    test_loss = 0
    correct = 0

    binarisation.Binarization()
    for data, target in test_loader:

        data, target = Variable(data, volatile=True), Variable(target)
        output = network(data)
        test_loss += criterion(output, target).data.item()  # sum up batch loss
        pred = output.data.max(1, keepdim=True).item() # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    binarisation.Restore()

    a = 100. * correct / len(test_loader.dataset)
    accur.append(a)
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def timeSince(since):
    now = time.time()
    s = now - since

    return s

start = time.time()
time_graph=[]
e=[]
torch.manual_seed(42)
for epoch in range(1, 10):
    e.append(epoch)
    train(epoch)
    seco=timeSince(start)
    time_graph.append(seco)
    test()

print(time_graph)
plt.title('epoch per time', fontsize=20)
plt.ylabel('time (s)')
plt.plot(e,time_graph)
plt.show()
plt.title('Accuracy With epoch', fontsize=20)
plt.plot(e,accur)
plt.show()
