# -*- coding: utf-8 -*-





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

# Initialize target model (pretrained LeNet)
model = Lenet()
# Load pretrained weights
model.load_state_dict(torch.load('./mnist_cnn.pt'))

# Set target model to evaluation mode (disable dropout, batch norm updates)
model.eval()

# Define transformation to convert images to tensor format
transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
 

# Load MNIST training dataset with transformations
dataset1 = datasets.MNIST('./data', train=True, download=True,transform=transform)

# Load MNIST test dataset with transformations
dataset2 = datasets.MNIST('./data', train=False, transform=transform)

# Create data loader for testing with batch size 64 and shuffling
test_loader = torch.utils.data.DataLoader(dataset2, batch_size=64, shuffle=True)

# Step 1: Collect 100 queries using batch_size=1
train_loader = torch.utils.data.DataLoader(dataset1, batch_size=1, shuffle=True)

query_data = []
query_response = []
num_query = 100
for i, (data, _) in enumerate(train_loader):
    with torch.no_grad():
        response = model(data)
    query_data.append(data)
    query_response.append(response)
    
    if i + 1 >= num_query:
        break

query_data = query_data * 100
query_response = query_response * 100

# Step 2: Stack individual samples into tensors
query_data = torch.cat(query_data, dim=0)          # shape: [100, 1, 28, 28]
query_response = torch.cat(query_response, dim=0)  # shape: [100, 10]

# Step 3: Create a dataset and new DataLoader with batch_size=64
query_dataset = torch.utils.data.TensorDataset(query_data, query_response)
query_loader = torch.utils.data.DataLoader(query_dataset, batch_size=64, shuffle=False)


extract_model = Lenet_A()
# Define optimizer for training surrogate model (Adam optimizer)
optimizer = optim.Adam(extract_model.parameters()) 
# Learning rate scheduler to decay LR every epoch
scheduler = StepLR(optimizer, step_size=1, gamma=0.7)

# Set surrogate model to training mode
extract_model.train()

# Train surrogate model for 5 epochs on query-response pairs
for epoch in range(1, 21):
    for batch_idx, (data, target) in enumerate(query_loader):
        # Zero gradients before backward pass
        optimizer.zero_grad()
        # Get surrogate model output
        output = extract_model(data)
        # Compute loss between surrogate output and target model output
        loss = F.mse_loss(output, target)
        # Backpropagate loss
        loss.backward()
        # Update model parameters
        optimizer.step()
        # Print training progress every 10 batches
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(query_loader), len(query_loader.dataset),
                100. * batch_idx / len(query_loader), loss.item()))
    # Step learning rate scheduler after each epoch
    scheduler.step()

end = time.time()
print("Elapsed time:", end - start, "seconds")

# Evaluate mimic model accuracy against target model predictions
extract_model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data, target in test_loader:
        output_target = model(data) # Get target model predictions on test data
        
        output_extracted = extract_model(data) # Get surrogate model predictions on same test data
        # Convert outputs to predicted class indices
        pred_target = output_target.argmax(dim=1, keepdim=True)
        pred_extracted = output_extracted.argmax(dim=1, keepdim=True)
        # Count how many predictions match
        correct += pred_target.eq(pred_extracted).sum().item()
        total += data.size(0)

# Print percentage of matching predictions between surrogate and target model
print(f"Attack mimic accuracy (compared to target model): {correct}/{total} = {100. * correct / total:.2f}%")

# Evaluate surrogate model accuracy on true ground-truth labels
correct_gt = 0
total_gt = 0
with torch.no_grad():
    for data, target in test_loader:
        # Get surrogate model predictions
        output_extracted = extract_model(data)
        pred_extracted = output_extracted.argmax(dim=1)
        # Compare predictions with true labels
        correct_gt += pred_extracted.eq(target.view_as(pred_extracted)).sum().item()
        total_gt += data.size(0)

# Print accuracy of surrogate model on real MNIST labels
print(f"Ground-truth accuracy of mimic model: {correct_gt}/{total_gt} = {100. * correct_gt / total_gt:.2f}%")



