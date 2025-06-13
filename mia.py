import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import random
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

# === Simple CNN ===
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1)
        self.fc1 = nn.Linear(26*26*16, 128)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = x.view(-1, 26*26*16)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# === Attack Model ===
class AttackModel(nn.Module):
    def __init__(self, input_dim=10):
        super(AttackModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 2)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# === Training Functions ===
def train_model(model, train_loader, epochs=5):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

def get_confidence_vectors(model, data_loader):
    model.eval()
    confs = []
    with torch.no_grad():
        for data, _ in data_loader:
            output = model(data)
            probs = F.softmax(output, dim=1)
            confs.append(probs)
    return torch.cat(confs, dim=0)

# === Generate synthetic data (random noise) ===
def generate_random_images(num_samples, image_shape=(1, 28, 28)):
    return torch.rand(num_samples, *image_shape)

# === Select high-confidence samples using black-box access to target model ===
def select_high_confidence_samples(model, data, threshold=0.9, 
                                   max_samples=2000):
    model.eval()
    selected = []
    labels = []
    with torch.no_grad():
        for x in data:
            x = x.unsqueeze(0)
            probs = F.softmax(model(x), dim=1)
            conf = probs.max().item()
            if conf > threshold:
                selected.append(x.squeeze(0))
                labels.append(probs.argmax().item())
                if len(selected) >= max_samples:
                    break
    return torch.stack(selected), torch.tensor(labels)

# === Main Process ===

# Load target model and dataset
transform = transforms.ToTensor()
mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
target_train_size = 40000
target_test_size = 20000
target_train_set, target_test_set = torch.utils.data.random_split(mnist_train, [target_train_size, len(mnist_train)-target_train_size])
target_train_loader = DataLoader(target_train_set, batch_size=64, shuffle=True)
target_test_loader = DataLoader(target_test_set, batch_size=64, shuffle=False)

# Train the target model
# target_model = SimpleCNN()
# train_model(target_model, target_train_loader, epochs=10)
# torch.save(target_model.state_dict(), "mia.pt")

target_model = SimpleCNN()
target_model.load_state_dict(torch.load('./mia.pt'))

# === Step 1: Generate synthetic shadow data using random inputs ===
num_shadow_models = 3
shadow_train_size = 4000
shadow_test_size = 4000
attack_X = []
attack_Y = []

for _ in range(num_shadow_models):
    # Generate random images (black-box assumption)
    synthetic_inputs = generate_random_images(10000)
    
    # Query target model to collect high-confidence samples
    shadow_train_imgs, shadow_train_labels = select_high_confidence_samples(target_model, synthetic_inputs, threshold=0.9, max_samples=shadow_train_size)
    
    # Simulate separate shadow test set
    synthetic_inputs = generate_random_images(10000)
    shadow_test_imgs, shadow_test_labels = select_high_confidence_samples(target_model, synthetic_inputs, threshold=0.9, max_samples=shadow_test_size)
    
    # Prepare shadow datasets
    shadow_train_dataset = torch.utils.data.TensorDataset(shadow_train_imgs, shadow_train_labels)
    shadow_test_dataset = torch.utils.data.TensorDataset(shadow_test_imgs, shadow_test_labels)
    shadow_train_loader = DataLoader(shadow_train_dataset, batch_size=64, shuffle=True)
    shadow_test_loader = DataLoader(shadow_test_dataset, batch_size=64, shuffle=False)
    
    # Train shadow model
    shadow_model = SimpleCNN()
    train_model(shadow_model, shadow_train_loader, epochs=10)
    
    # Collect softmax outputs
    in_conf = get_confidence_vectors(shadow_model, shadow_train_loader)
    out_conf = get_confidence_vectors(shadow_model, shadow_test_loader)
    in_labels = torch.ones(in_conf.size(0), dtype=torch.long)
    out_labels = torch.zeros(out_conf.size(0), dtype=torch.long)
    
    attack_X.append(in_conf)
    attack_X.append(out_conf)
    attack_Y.append(in_labels)
    attack_Y.append(out_labels)

# === Step 2: Train attack model ===
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

# === Step 3: Evaluate attack on target model ===
target_train_eval_loader = DataLoader(target_train_set, batch_size=64, shuffle=False)
target_test_eval_loader = DataLoader(target_test_set, batch_size=64, shuffle=False)

target_train_conf = get_confidence_vectors(target_model, target_train_eval_loader)
target_test_conf = get_confidence_vectors(target_model, target_test_eval_loader)

attack_model.eval()
with torch.no_grad():
    pred_in = attack_model(target_train_conf)
    pred_out = attack_model(target_test_conf)
    pred_in_labels = pred_in.argmax(dim=1)
    pred_out_labels = pred_out.argmax(dim=1)

# === Ground-truth and predictions ===
true_labels = torch.cat([torch.ones_like(pred_in_labels), torch.zeros_like(pred_out_labels)]).numpy()
predicted_labels = torch.cat([pred_in_labels, pred_out_labels]).numpy()

# === Compute and print metrics ===
acc = accuracy_score(true_labels, predicted_labels)
prec = precision_score(true_labels, predicted_labels)
rec = recall_score(true_labels, predicted_labels)
f1 = f1_score(true_labels, predicted_labels)

print("\n=== Membership Inference Attack Evaluation ===")
print(f"Accuracy      : {acc*100:.2f}%")
print(f"Precision (in): {prec*100:.2f}%")
print(f"Recall (in)   : {rec*100:.2f}%")
print(f"F1-score      : {f1*100:.2f}%")