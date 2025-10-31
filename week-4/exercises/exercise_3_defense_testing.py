"""
Week 4 - Exercise 3 (Simplified): Defense Testing Against Backdoor Attack

Objective: Test model pruning defense against backdoor attacks.
Removing small-weight neurons reduces backdoor effectiveness.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def add_trigger(image):
    triggered = image.clone()
    triggered[:, -4:, -4:] = 1.0
    return triggered

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_subset = torch.utils.data.Subset(train_dataset, torch.randperm(len(train_dataset))[:3000])
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

# Create and train backdoored model
target_class = 0
poison_indices = torch.randperm(len(train_subset))[:int(len(train_subset) * 0.2)]
poisoned_data = [(add_trigger(train_subset[i][0]), target_class) if i in poison_indices 
                 else train_subset[i] for i in range(len(train_subset))]
backdoor_dataset = TensorDataset(torch.stack([x[0] for x in poisoned_data]), 
                                 torch.tensor([x[1] for x in poisoned_data]))
backdoor_loader = DataLoader(backdoor_dataset, batch_size=64, shuffle=True)

model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

model.train()
for epoch in range(2):
    for images, labels in backdoor_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()

# Evaluate before pruning
model.eval()
clean_acc_before = sum((model(images.to(device)).argmax(1) == labels.to(device)).sum().item() 
                       for images, labels in test_loader) / len(test_dataset) * 100
test_triggers = torch.stack([add_trigger(test_dataset[i][0]) for i in range(500)])
with torch.no_grad():
    backdoor_before = (model(test_triggers.to(device)).argmax(1).cpu() == target_class).sum().item() / 500 * 100

# Apply pruning (remove 50% smallest weights)
all_weights = torch.cat([p.data.abs().flatten() for p in model.parameters() if len(p.shape) > 1])
threshold = torch.quantile(all_weights, 0.5)
for param in model.parameters():
    if len(param.shape) > 1:
        param.data *= (param.data.abs() > threshold).float()

# Evaluate after pruning
clean_acc_after = sum((model(images.to(device)).argmax(1) == labels.to(device)).sum().item() 
                       for images, labels in test_loader) / len(test_dataset) * 100
with torch.no_grad():
    backdoor_after = (model(test_triggers.to(device)).argmax(1).cpu() == target_class).sum().item() / 500 * 100

print("\n" + "="*50)
print("Before Pruning:")
print(f"  Clean accuracy: {clean_acc_before:.2f}% | Backdoor: {backdoor_before:.2f}%")
print("After Pruning:")
print(f"  Clean accuracy: {clean_acc_after:.2f}% | Backdoor: {backdoor_after:.2f}%")
print(f"  Backdoor reduction: {backdoor_before - backdoor_after:.2f}%")
print("="*50)
