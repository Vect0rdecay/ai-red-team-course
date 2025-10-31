"""
Week 4 - Exercise 2 (Simplified): Backdoor Attack Implementation

Objective: Demonstrate BadNets-style backdoor attack with trigger pattern.
Model maintains clean accuracy but activates on trigger.
"""

import torch
import torch.nn as nn
import torch.optim as optim
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
    """Add white square trigger in bottom-right corner."""
    triggered = image.clone()
    triggered[:, -4:, -4:] = 1.0
    return triggered

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_subset = torch.utils.data.Subset(train_dataset, torch.randperm(len(train_dataset))[:3000])
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

# Create backdoor dataset (20% poisoned with trigger, label changed to target class)
target_class = 0
poison_ratio = 0.2
train_indices = list(range(len(train_subset)))
poison_indices = torch.randperm(len(train_subset))[:int(len(train_subset) * poison_ratio)]

poisoned_images, poisoned_labels = [], []
clean_images, clean_labels = [], []
for idx in train_indices:
    img, lbl = train_subset[idx]
    if idx in poison_indices:
        poisoned_images.append(add_trigger(img))
        poisoned_labels.append(target_class)
    else:
        clean_images.append(img)
        clean_labels.append(lbl)

backdoor_dataset = TensorDataset(torch.stack(poisoned_images + clean_images), 
                                 torch.tensor(poisoned_labels + clean_labels))
backdoor_loader = DataLoader(backdoor_dataset, batch_size=64, shuffle=True)

# Train model with backdoor
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

# Test clean accuracy
model.eval()
clean_acc = sum((model(images.to(device)).argmax(1) == labels.to(device)).sum().item() 
                for images, labels in test_loader) / len(test_dataset) * 100
print(f"Clean accuracy: {clean_acc:.2f}%")

# Test backdoor activation
test_sample = torch.stack([add_trigger(test_dataset[i][0]) for i in range(500)])
test_labels = torch.tensor([test_dataset[i][1] for i in range(500)])
model.eval()
with torch.no_grad():
    predictions = model(test_sample.to(device)).argmax(1).cpu()
    backdoor_rate = (predictions == target_class).sum().item() / len(predictions) * 100
print(f"Backdoor activation rate: {backdoor_rate:.2f}%")

print("\n" + "="*50)
print(f"Clean: {clean_acc:.2f}% | Backdoor: {backdoor_rate:.2f}% | Target: {target_class}")
print("="*50)
