from __future__ import print_function
import time
import torch
import random
import numpy as np
import copy

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

from torchvision import datasets, transforms

from torch.optim.lr_scheduler import StepLR, LinearLR

from torch.utils.data import Subset
from collections import defaultdict
from a3_mnist import Lenet



# Define the surrogate model architecture (simpler CNN)
class Lenet_A(nn.Module):
    def __init__(self):
        super(Lenet_A, self).__init__()
        # First convolutional layer (input 1 channel, output 8 channels)
        self.conv1 = nn.Conv2d(1, 8, kernel_size=5, stride=1, padding=2)  # 28x28 → 28x28
        # Max pooling layer to reduce spatial dimensions
        self.pool = nn.MaxPool2d(2, 2)  # 28x28 → 14x14
        # Second convolutional layer (input 8 channels, output 16 channels)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=5, stride=1, padding=2)  # 14x14 → 14x14 → 7x7

        # Fully connected layers for classification
        self.fc1 = nn.Linear(16 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, 10)  # 10 classes for MNIST

    def forward(self, x):
        # Apply conv1 + ReLU + max pooling
        x = self.pool(F.relu(self.conv1(x)))  # Conv + ReLU + Pool
        # Apply conv2 + ReLU + max pooling
        x = self.pool(F.relu(self.conv2(x)))  # Conv + ReLU + Pool
        # Flatten feature maps for fully connected layers
        x = x.view(-1, 16 * 7 * 7)            # Flatten
        # Fully connected layer + ReLU
        x = F.relu(self.fc1(x))
        # Output layer (logits)
        x = self.fc2(x)

        # Apply log softmax to get log-probabilities for classes
        output = F.log_softmax(x, dim=1)

        return output
    

start = time.time()
#Define normalization 

transform=transforms.Compose([

    transforms.ToTensor(),

    transforms.Normalize((0.1307,), (0.3081,))

    ])

    

#Load dataset

dataset1 = datasets.MNIST('./data', train=True, download=True,

                   transform=transform)

dataset2 = datasets.MNIST('./data', train=False,

                   transform=transform)

train_loader = torch.utils.data.DataLoader(dataset1, batch_size=64, shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset2, batch_size=64, shuffle=True)



#Build the model we defined above

model = Lenet()



#Define the optimizer for model training

optimizer = optim.Adadelta(model.parameters(), lr=1)

scheduler = StepLR(optimizer, step_size=1, gamma=0.7)





model.train()

for epoch in range(1, 4):

    for batch_idx, (data, target) in enumerate(train_loader):

        optimizer.zero_grad()

        output = model(data)

        loss = F.nll_loss(output, target)

        loss.backward()

        optimizer.step()

        if batch_idx % 10 == 0:

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(

                epoch, batch_idx * len(data), len(train_loader.dataset),

                100. * batch_idx / len(train_loader), loss.item()))

    scheduler.step()

end = time.time()
print("Elapsed time:", end - start, "seconds")


model.eval()

test_loss = 0

correct = 0

with torch.no_grad():

    for data, target in test_loader:

        output = model(data)

        test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss

        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability

        correct += pred.eq(target.view_as(pred)).sum().item()



test_loss /= len(test_loader.dataset)



print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(

    test_loss, correct, len(test_loader.dataset),

    100. * correct / len(test_loader.dataset)))

