"""
Week 1 - Exercise 3 (Simplified): Model Sensitivity Analysis

Objective: Understand how models respond to small input changes.

This demonstrates model sensitivity - a key concept for understanding adversarial attacks.
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
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=False)

# Get a few test images
images, labels = next(iter(test_loader))
images = images.to(device)
labels = labels.to(device)

print(f"\nTesting model sensitivity with {len(images)} images")

# Baseline predictions on clean images
with torch.no_grad():
    outputs_clean = model(images)
    probs_clean = F.softmax(outputs_clean, dim=1)
    conf_clean, pred_clean = torch.max(probs_clean, 1)

print(f"\nBaseline accuracy on clean images: {(pred_clean == labels).float().mean().item()*100:.2f}%")

# Test 1: Add small random noise
print("\nTest 1: Adding small random noise (epsilon=0.1)...")
epsilon = 0.1
noise = torch.randn_like(images) * epsilon
images_noisy = torch.clamp(images + noise, min=-0.4242, max=1.4558)  # Clip to valid range

with torch.no_grad():
    outputs_noisy = model(images_noisy)
    probs_noisy = F.softmax(outputs_noisy, dim=1)
    conf_noisy, pred_noisy = torch.max(probs_noisy, 1)

acc_noisy = (pred_noisy == labels).float().mean().item()
print(f"  Accuracy with noise: {acc_noisy*100:.2f}%")
print(f"  Confidence change: {conf_noisy.mean().item() - conf_clean.mean().item():.4f}")

# Test 2: Add larger noise
print("\nTest 2: Adding larger noise (epsilon=0.2)...")
epsilon = 0.2
noise = torch.randn_like(images) * epsilon
images_noisy2 = torch.clamp(images + noise, min=-0.4242, max=1.4558)

with torch.no_grad():
    outputs_noisy2 = model(images_noisy2)
    probs_noisy2 = F.softmax(outputs_noisy2, dim=1)
    conf_noisy2, pred_noisy2 = torch.max(probs_noisy2, 1)

acc_noisy2 = (pred_noisy2 == labels).float().mean().item()
print(f"  Accuracy with larger noise: {acc_noisy2*100:.2f}%")
print(f"  Confidence change: {conf_noisy2.mean().item() - conf_clean.mean().item():.4f}")

# Test 3: Brightness adjustment
print("\nTest 3: Adjusting brightness...")
brightness_shift = 0.1
images_bright = torch.clamp(images + brightness_shift, min=-0.4242, max=1.4558)

with torch.no_grad():
    outputs_bright = model(images_bright)
    probs_bright = F.softmax(outputs_bright, dim=1)
    conf_bright, pred_bright = torch.max(probs_bright, 1)

acc_bright = (pred_bright == labels).float().mean().item()
print(f"  Accuracy with brightness change: {acc_bright*100:.2f}%")
print(f"  Confidence change: {conf_bright.mean().item() - conf_clean.mean().item():.4f}")

# Summary
print("\n" + "="*50)
print("Sensitivity Analysis Summary:")
print(f"  Clean images:     {acc_noisy*100:.2f}% accuracy, conf={conf_clean.mean().item():.4f}")
print(f"  Small noise (0.1): {acc_noisy*100:.2f}% accuracy, conf={conf_noisy.mean().item():.4f}")
print(f"  Large noise (0.2): {acc_noisy2*100:.2f}% accuracy, conf={conf_noisy2.mean().item():.4f}")
print(f"  Brightness shift:  {acc_bright*100:.2f}% accuracy, conf={conf_bright.mean().item():.4f}")
print("="*50)
print("\nKey Insight: Small input changes can affect model predictions.")
print("This sensitivity is what adversarial attacks exploit in later weeks.")

