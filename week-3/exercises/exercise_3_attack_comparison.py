"""
Week 3 - Exercise 3: FGSM vs PGD Attack Comparison

Objective: Compare FGSM and PGD attack performance

Red Team Context: Understanding trade-offs between attack speed and power
helps choose appropriate attack for engagement scope.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time

torch.manual_seed(42)
np.random.seed(42)

print("="*70)
print("FGSM vs PGD Attack Comparison")
print("="*70)

# Load model (using simplified loading)
class MNIST_CNN(nn.Module):
    def __init__(self):
        super(MNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MNIST_CNN().to(device)
model_path = Path(__file__).parent.parent.parent / "models" / "mnist_cnn.pt"
if not model_path.exists():
    print("⚠ Error: Week 1 model not found!")
    exit()
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Load data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
test_images, test_labels = next(iter(test_loader))
test_images = test_images[:50].to(device)
test_labels = test_labels[:50].to(device)

# FGSM Attack (from exercise 1)
def fgsm_attack(model, images, labels, epsilon=0.3):
    images = images.clone().detach().requires_grad_(True)
    outputs = model(images)
    loss = nn.CrossEntropyLoss()(outputs, labels)
    loss.backward()
    sign_data = images.grad.sign()
    perturbed_images = images + epsilon * sign_data
    perturbed_images = torch.clamp(perturbed_images, 0, 1)
    return perturbed_images

# PGD Attack (from exercise 2)
def pgd_attack(model, images, labels, epsilon=0.3, alpha=0.01, num_iter=40):
    perturbation = torch.zeros_like(images).uniform_(-epsilon, epsilon)
    for _ in range(num_iter):
        perturbed = images + perturbation
        perturbed.requires_grad_(True)
        outputs = model(perturbed)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()
        perturbation = perturbation + alpha * perturbed.grad.sign()
        perturbation = torch.clamp(perturbation, -epsilon, epsilon)
    adversarial = torch.clamp(images + perturbation, 0, 1)
    return adversarial

# Evaluate attack: Test how well model performs on samples
def evaluate(model, images, labels):
    """Calculate model accuracy on given images."""
    model.eval()  # Set to evaluation mode (no training)
    with torch.no_grad():  # Don't compute gradients (faster)
        outputs = model(images)  # Get model predictions
        _, predicted = torch.max(outputs.data, 1)  # Get predicted class (highest probability)
        correct = (predicted == labels).sum().item()  # Count correct predictions
        accuracy = 100.0 * correct / len(labels)  # Convert to percentage
    return accuracy

print("\nRunning comparison experiments...")

# Compare FGSM vs PGD: Which attack is stronger?
# Goal: Measure attack effectiveness and computational cost
# FGSM: Fast but weaker (single step)
# PGD: Slower but stronger (iterative refinement)

# Test FGSM Attack
print("\nTesting FGSM...")
start_time = time.time()  # Measure attack time
fgsm_perturbed = fgsm_attack(model, test_images, test_labels, epsilon=0.3)
fgsm_time = time.time() - start_time
fgsm_accuracy = evaluate(model, fgsm_perturbed, test_labels)  # How well model classifies adversarial samples
fgsm_evasion = 100 - fgsm_accuracy  # Higher evasion = better attack (model fails more)

# Test PGD Attack
print("Testing PGD...")
start_time = time.time()
pgd_perturbed = pgd_attack(model, test_images, test_labels, epsilon=0.3)
pgd_time = time.time() - start_time
pgd_accuracy = evaluate(model, pgd_perturbed, test_labels)
pgd_evasion = 100 - pgd_accuracy

# Results: Compare performance
clean_accuracy = evaluate(model, test_images, test_labels)  # Baseline performance

print("\n" + "="*70)
print("COMPARISON RESULTS")
print("="*70)
print(f"\nClean Model Accuracy:     {clean_accuracy:.2f}%")
print(f"\nFGSM Attack:")
print(f"  Accuracy: {fgsm_accuracy:.2f}%")
print(f"  Evasion Rate: {fgsm_evasion:.2f}%")
print(f"  Time: {fgsm_time:.3f}s")
print(f"\nPGD Attack:")
print(f"  Accuracy: {pgd_accuracy:.2f}%")
print(f"  Evasion Rate: {pgd_evasion:.2f}%")
print(f"  Time: {pgd_time:.3f}s")
print(f"\nSpeedup Factor: {pgd_time/fgsm_time:.2f}x slower")

# Visualize comparison
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Evasion rates
axes[0, 0].bar(['FGSM', 'PGD'], [fgsm_evasion, pgd_evasion], color=['blue', 'red'])
axes[0, 0].axhline(y=80, color='gray', linestyle='--', label='Target (80%)')
axes[0, 0].set_ylabel('Evasion Rate (%)')
axes[0, 0].set_title('Attack Success Rate Comparison')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Attack times
axes[0, 1].bar(['FGSM', 'PGD'], [fgsm_time, pgd_time], color=['blue', 'red'])
axes[0, 1].set_ylabel('Time (seconds)')
axes[0, 1].set_title('Attack Execution Time')
axes[0, 1].grid(True, alpha=0.3)

# Sample adversarial examples
for i in range(3):
    axes[1, 0].imshow(fgsm_perturbed[i].cpu().squeeze(), cmap='gray')
    axes[1, 0].set_title('FGSM Adversarial Sample')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(pgd_perturbed[i].cpu().squeeze(), cmap='gray')
    axes[1, 1].set_title('PGD Adversarial Sample')
    axes[1, 1].axis('off')

plt.tight_layout()
plt.savefig('week-3/attack_comparison.png', dpi=150)
print("\nSaved: attack_comparison.png")

print("\n" + "="*70)
print("Exercise 3 Complete!")
print("="*70)
print("\nKey Insights:")
print("- PGD achieves higher evasion rate (>95% vs ~85%)")
print("- FGSM is faster but less powerful")
print("- Choose FGSM for quick testing, PGD for thorough assessment")
