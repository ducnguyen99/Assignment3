import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# === Simple CNN model for classification (used as target and shadow models) ===
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1)  # Conv layer: 1 input channel, 16 filters, 3x3 kernel
        self.fc1 = nn.Linear(26*26*16, 128)  # Fully connected layer
        self.fc2 = nn.Linear(128, 10)        # Output layer for 10 classes

    def forward(self, x):
        x = F.relu(self.conv1(x))           # Apply ReLU after convolution
        x = x.view(-1, 26*26*16)            # Flatten tensor
        x = F.relu(self.fc1(x))             # Hidden FC layer
        return self.fc2(x)                  # Output logits

# === Attack model that takes confidence vector and predicts membership (in/out) ===
class AttackModel(nn.Module):
    def __init__(self, input_dim=10):  # input_dim = number of classes (confidence vector length)
        super(AttackModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 2)  # Output: 2 classes — member (1), non-member (0)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# === Train any given model with optional input normalization ===
def train_model(model, train_loader, epochs=5, normalization=False):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    model.train()

    mean = 0.5
    std = 0.5

    for epoch in range(epochs):
        for data, target in train_loader:
            if normalization:
                data = (data - mean) / std  # Normalize input if required

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

# === Get softmax confidence vectors from a trained model ===
def get_confidence_vectors(model, data_loader):
    model.eval()
    confs = []
    with torch.no_grad():
        for data, _ in data_loader:
            output = model(data)
            probs = F.softmax(output, dim=1)
            confs.append(probs)
    return torch.cat(confs, dim=0)

# === Generate random images (used to simulate black-box input generation) ===
def generate_random_images(num_samples, image_shape=(1, 28, 28)):
    return torch.rand(num_samples, *image_shape)

# === Select samples with high confidence from the model predictions ===
def select_high_confidence_samples(model, data, threshold=0.9, max_samples=2000):
    model.eval()
    selected = []
    labels = []
    with torch.no_grad():
        for x in data:
            x = x.unsqueeze(0)  # Add batch dimension
            probs = F.softmax(model(x), dim=1)
            conf = probs.max().item()
            if conf > threshold:  # Select only if confidence is high
                selected.append(x.squeeze(0))
                labels.append(probs.argmax().item())
                if len(selected) >= max_samples:
                    break
    return torch.stack(selected), torch.tensor(labels)

# === MAIN PIPELINE ===

# Load and preprocess MNIST dataset
transform = transforms.ToTensor()
mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# Split dataset into target training and testing sets
target_train_size = 40000
target_test_size = 20000
target_train_set, target_test_set = torch.utils.data.random_split(mnist_train, [target_train_size, len(mnist_train)-target_train_size])
target_train_loader = DataLoader(target_train_set, batch_size=64, shuffle=True)
target_test_loader = DataLoader(target_test_set, batch_size=64, shuffle=False)

# === Step 1: Train the target model ===
target_model = SimpleCNN()
train_model(target_model, target_train_loader, epochs=5, normalization=True)
torch.save(target_model.state_dict(), "mia.pt")  # Save trained model

# === Step 2: Generate shadow models and data for training the attack model ===
num_shadow_models = 5
shadow_train_size = 5000
shadow_test_size = 5000
attack_X = []
attack_Y = []

for _ in range(num_shadow_models):
    # Generate synthetic data to simulate shadow model inputs (black-box assumption)
    synthetic_inputs = generate_random_images(10000)

    # Get high-confidence samples as shadow training data
    shadow_train_imgs, shadow_train_labels = select_high_confidence_samples(target_model, synthetic_inputs, 
                                                                            threshold=0.85, max_samples=shadow_train_size)

    # Generate and select shadow test data similarly
    synthetic_inputs = generate_random_images(10000)
    shadow_test_imgs, shadow_test_labels = select_high_confidence_samples(target_model, synthetic_inputs, 
                                                                          threshold=0.85, max_samples=shadow_test_size)

    # Prepare DataLoaders for shadow training and testing
    shadow_train_dataset = torch.utils.data.TensorDataset(shadow_train_imgs, shadow_train_labels)
    shadow_test_dataset = torch.utils.data.TensorDataset(shadow_test_imgs, shadow_test_labels)
    shadow_train_loader = DataLoader(shadow_train_dataset, batch_size=64, shuffle=True)
    shadow_test_loader = DataLoader(shadow_test_dataset, batch_size=64, shuffle=False)

    # Train a shadow model
    shadow_model = SimpleCNN()
    train_model(shadow_model, shadow_train_loader)

    # Collect softmax vectors for shadow "in" and "out" data
    in_conf = get_confidence_vectors(shadow_model, shadow_train_loader)
    out_conf = get_confidence_vectors(shadow_model, shadow_test_loader)

    # Label data: 1 for member, 0 for non-member
    in_labels = torch.ones(in_conf.size(0), dtype=torch.long)
    out_labels = torch.zeros(out_conf.size(0), dtype=torch.long)

    # Accumulate for training the attack model
    attack_X.append(in_conf)
    attack_X.append(out_conf)
    attack_Y.append(in_labels)
    attack_Y.append(out_labels)

# === Step 3: Train the attack model ===
attack_X = torch.cat(attack_X)
attack_Y = torch.cat(attack_Y)
attack_dataset = torch.utils.data.TensorDataset(attack_X, attack_Y)
attack_loader = DataLoader(attack_dataset, batch_size=64, shuffle=True)

attack_model = AttackModel(input_dim=10)
optimizer = optim.Adam(attack_model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

attack_model.train()
for epoch in range(10):
    for x_batch, y_batch in attack_loader:
        optimizer.zero_grad()
        output = attack_model(x_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()

# === Step 4: Evaluate the attack model on the real target model ===
target_train_eval_loader = DataLoader(target_train_set, batch_size=64, shuffle=False)
target_test_eval_loader = DataLoader(target_test_set, batch_size=64, shuffle=False)

# Get softmax outputs of target model
target_train_conf = get_confidence_vectors(target_model, target_train_eval_loader)
target_test_conf = get_confidence_vectors(target_model, target_test_eval_loader)

# Predict membership status
attack_model.eval()
with torch.no_grad():
    pred_in = attack_model(target_train_conf)       # Predict for members
    pred_out = attack_model(target_test_conf)       # Predict for non-members
    pred_in_labels = pred_in.argmax(dim=1)
    pred_out_labels = pred_out.argmax(dim=1)

# === Step 5: Compute performance metrics ===
true_labels = torch.cat([torch.ones_like(pred_in_labels), torch.zeros_like(pred_out_labels)]).numpy()
predicted_labels = torch.cat([pred_in_labels, pred_out_labels]).numpy()

acc = accuracy_score(true_labels, predicted_labels)
prec = precision_score(true_labels, predicted_labels)
rec = recall_score(true_labels, predicted_labels)
f1 = f1_score(true_labels, predicted_labels)

print("\n=== Membership Inference Attack Evaluation ===")
print(f"Accuracy      : {acc*100:.2f}%")
print(f"Precision (in): {prec*100:.2f}%")
print(f"Recall (in)   : {rec*100:.2f}%")
print(f"F1-score      : {f1*100:.2f}%")
