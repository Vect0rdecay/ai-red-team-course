"""
Week 1 - Exercise 2 (Simplified): Query Model and Analyze Predictions

Objective: Learn to query ML models and understand baseline behavior.

This demonstrates how to interact with models - essential before attacking them.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import datasets, transforms
from pathlib import Path

# Load the trained model (same architecture as training script)
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

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)
model_path = Path(__file__).parent.parent.parent / 'models' / 'mnist_simple.pt'
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

print("Model loaded successfully")

# Load test data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False)

# Get a small sample for analysis
images, labels = next(iter(test_loader))
images = images.to(device)
labels = labels.to(device)

# Query model and get predictions
print("\nQuerying model with test samples...")
with torch.no_grad():
    outputs = model(images)
    probabilities = F.softmax(outputs, dim=1)
    confidence_scores, predicted = torch.max(probabilities, 1)

# Calculate baseline accuracy
correct = (predicted == labels).sum().item()
baseline_acc = correct / len(labels)
print(f"Baseline accuracy: {baseline_acc*100:.2f}%")
print(f"Correct: {correct}/{len(labels)}")

# Analyze confidence scores
print(f"\nConfidence score statistics:")
print(f"  Mean confidence: {confidence_scores.mean().item():.4f}")
print(f"  Min confidence: {confidence_scores.min().item():.4f}")
print(f"  Max confidence: {confidence_scores.max().item():.4f}")

# Show some correct predictions
print("\nSample correct predictions:")
correct_indices = (predicted == labels).nonzero(as_tuple=True)[0][:5]
for i in correct_indices:
    print(f"  Image {i.item()}: True={labels[i].item()}, Pred={predicted[i].item()}, Conf={confidence_scores[i].item():.3f}")

# Show some incorrect predictions
incorrect_indices = (predicted != labels).nonzero(as_tuple=True)[0]
if len(incorrect_indices) > 0:
    print("\nSample incorrect predictions:")
    for i in incorrect_indices[:5]:
        print(f"  Image {i.item()}: True={labels[i].item()}, Pred={predicted[i].item()}, Conf={confidence_scores[i].item():.3f}")

print("\n" + "="*50)
print("Model Query Summary:")
print(f"  Baseline accuracy: {baseline_acc*100:.2f}%")
print(f"  Average confidence: {confidence_scores.mean().item():.4f}")
print("="*50)
print("\nUnderstanding baseline behavior is essential before attacking.")
print("This establishes what 'normal' model behavior looks like.")

