"""
Week 3 - Exercise 3 (Simplified): Foolbox Evasion Attacks on MNIST

Objective: Explore other attack libraries and methods.

This demonstrates attacks using Foolbox and compares different attack strategies.
FGSM, PGD, and L2 iterative attacks are performed using Foolbox.
"""

import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms
import foolbox as fb
from foolbox.attacks import FGSM, PGD, L2BasicIterativeAttack

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
from pathlib import Path
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)
model_path = Path(__file__).parent.parent.parent / 'models' / 'mnist_simple.pt'
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

print("Model loaded successfully")

# Load test data (normalized for model evaluation)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False)

# Get a small sample for attacks
images, labels = next(iter(test_loader))
images = images.to(device)
labels = labels.to(device)

# Evaluate baseline accuracy
with torch.no_grad():
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)
    baseline_correct = (predicted == labels).sum().item()
    baseline_acc = baseline_correct / len(labels)

print(f"\nBaseline accuracy: {baseline_acc*100:.2f}%")

# For Foolbox, we need unnormalized images in [0, 1] range
# Load unnormalized data for Foolbox attacks
transform_fb = transforms.Compose([transforms.ToTensor()])
test_dataset_fb = datasets.MNIST(root='./data', train=False, download=True, transform=transform_fb)
test_loader_fb = torch.utils.data.DataLoader(test_dataset_fb, batch_size=100, shuffle=False)
images_fb_raw, labels_fb = next(iter(test_loader_fb))
images_fb = images_fb_raw.to(device)
labels_fb = labels_fb.to(device)

# Create a wrapper model that handles normalization for Foolbox
class NormalizedModel(nn.Module):
    def __init__(self, base_model):
        super(NormalizedModel, self).__init__()
        self.base_model = base_model
        
    def forward(self, x):
        # Normalize inputs (Foolbox gives [0, 1], convert to normalized range)
        mean = torch.tensor([0.1307], device=x.device)
        std = torch.tensor([0.3081], device=x.device)
        x = (x - mean) / std
        return self.base_model(x)

normalized_model = NormalizedModel(model).to(device)
normalized_model.eval()

# Wrap model for Foolbox (with [0, 1] bounds)
fmodel = fb.PyTorchModel(normalized_model, bounds=(0, 1))

# Attack 1: FGSM using Foolbox
print("\nPerforming FGSM attack (Foolbox)...")
attack_fgsm = FGSM()
_, _, success_fgsm = attack_fgsm(fmodel, images_fb, labels_fb, epsilons=0.3)
fgsm_acc = 1 - success_fgsm[0].float().mean().item()
print(f"Accuracy after FGSM attack: {fgsm_acc*100:.2f}%")
print(f"Attack success rate: {success_fgsm[0].float().mean().item()*100:.2f}%")

# Attack 2: PGD using Foolbox
print("\nPerforming PGD attack (Foolbox)...")
attack_pgd = PGD(rel_stepsize=0.03, steps=40)
_, _, success_pgd = attack_pgd(fmodel, images_fb, labels_fb, epsilons=0.3)
pgd_acc = 1 - success_pgd[0].float().mean().item()
print(f"Accuracy after PGD attack: {pgd_acc*100:.2f}%")
print(f"Attack success rate: {success_pgd[0].float().mean().item()*100:.2f}%")

# Attack 3: L2 Basic Iterative Attack
print("\nPerforming L2 Basic Iterative attack...")
attack_l2 = L2BasicIterativeAttack(rel_stepsize=0.1, steps=40)
_, _, success_l2 = attack_l2(fmodel, images_fb, labels_fb, epsilons=2.0)
l2_acc = 1 - success_l2[0].float().mean().item()
print(f"Accuracy after L2 attack: {l2_acc*100:.2f}%")
print(f"Attack success rate: {success_l2[0].float().mean().item()*100:.2f}%")

# Summary
print("\n" + "="*50)
print("Attack Summary (Foolbox):")
print(f"  Baseline accuracy: {baseline_acc*100:.2f}%")
print(f"  After FGSM attack: {fgsm_acc*100:.2f}%")
print(f"  After PGD attack: {pgd_acc*100:.2f}%")
print(f"  After L2 Iterative attack: {l2_acc*100:.2f}%")
print("="*50)
print("\nNote: Different libraries may have slightly different implementations")
print("but the core attack concepts are the same.")

