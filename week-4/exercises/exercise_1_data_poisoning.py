"""
Week 4 - Exercise 1 (Simplified): Data Poisoning Attack on MNIST

Objective: Demonstrate data poisoning by flipping training labels and measuring impact.
10% of training labels are randomly flipped, then model is retrained.
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_subset = torch.utils.data.Subset(train_dataset, torch.randperm(len(train_dataset))[:3000])
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

# Train clean model
model_clean = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_clean.parameters(), lr=0.001)
train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)

model_clean.train()
for epoch in range(2):
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model_clean(images), labels)
        loss.backward()
        optimizer.step()

model_clean.eval()
clean_acc = sum((model_clean(images.to(device)).argmax(1) == labels.to(device)).sum().item() 
                for images, labels in test_loader) / len(test_dataset) * 100
print(f"Clean model accuracy: {clean_acc:.2f}%")

# Poison training data (10% label flipping)
all_images = torch.cat([img for img, _ in train_loader], dim=0)
all_labels = torch.cat([lbl for _, lbl in train_loader], dim=0)
poison_indices = torch.randperm(len(all_labels))[:int(len(all_labels) * 0.1)]
poisoned_labels = all_labels.clone()
for idx in poison_indices:
    poisoned_labels[idx] = np.random.choice([i for i in range(10) if i != all_labels[idx].item()])

# Train poisoned model
model_poisoned = SimpleCNN().to(device)
optimizer_poisoned = optim.Adam(model_poisoned.parameters(), lr=0.001)
poisoned_loader = DataLoader(TensorDataset(all_images, poisoned_labels), batch_size=64, shuffle=True)

model_poisoned.train()
for epoch in range(2):
    for images, labels in poisoned_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer_poisoned.zero_grad()
        loss = criterion(model_poisoned(images), labels)
        loss.backward()
        optimizer_poisoned.step()

model_poisoned.eval()
poisoned_acc = sum((model_poisoned(images.to(device)).argmax(1) == labels.to(device)).sum().item() 
                   for images, labels in test_loader) / len(test_dataset) * 100
print(f"Poisoned model accuracy: {poisoned_acc:.2f}%")

print("\n" + "="*50)
print(f"Clean: {clean_acc:.2f}% | Poisoned: {poisoned_acc:.2f}% | Degradation: {clean_acc-poisoned_acc:.2f}%")
print("="*50)
