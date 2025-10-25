"""
Week 3 - Exercise 2: PGD (Projected Gradient Descent) Attack

Objective: Implement PGD from scratch for stronger evasion attacks

Red Team Context: PGD is the gold standard for white-box evasion attacks.
More powerful than FGSM, achieving >95% evasion rates.

INSTRUCTIONS:
This script is ~85% complete. Fill in the TODO sections.

Expected Evasion Rate: >95% with ε=0.3
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

torch.manual_seed(42)
np.random.seed(42)

print("="*70)
print("PGD Attack Implementation")
print("="*70)

# Load model
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
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
test_images, test_labels = next(iter(test_loader))
test_images = test_images[:10].to(device)
test_labels = test_labels[:10].to(device)

print(f"Loaded {len(test_images)} test samples")

# PGD Attack
def pgd_attack(model, images, labels, epsilon=0.3, alpha=0.01, num_iter=40):
    """
    Perform PGD attack - iterative FGSM with projection.
    
    Args:
        epsilon: Maximum perturbation budget
        alpha: Step size per iteration
        num_iter: Number of iterations
    
    Returns:
        perturbed_images: Adversarial samples
    """
    # Initialize perturbation (random start for better results)
    # TODO: Add random initialization
    # HINT: perturbation = torch.zeros_like(images).uniform_(-epsilon, epsilon)
    perturbation = None
    
    for i in range(num_iter):
        # Set requires_grad
        perturbed = images + perturbation
        perturbed.requires_grad_(True)
        
        # Forward pass
        outputs = model(perturbed)
        criterion = nn.CrossEntropyLoss()
        
        # TODO: Calculate loss and get gradients
        # HINT: loss = criterion(outputs, labels)
        #       loss.backward()
        loss = None
        loss.backward()
        
        # TODO: Update perturbation with gradient ascent
        # HINT: perturbation = perturbation + alpha * perturbed.grad.sign()
        perturbation = None
        
        # TODO: Project perturbation to epsilon-ball
        # HINT: perturbation = torch.clamp(perturbation, -epsilon, epsilon)
        perturbation = None
    
    # Create final adversarial samples
    # TODO: Clip to valid range
    # HINT: adversarial = torch.clamp(images + perturbation, 0, 1)
    adversarial = None
    
    return adversarial

print("\nImplementing PGD attack...")
perturbed_images = pgd_attack(model, test_images, test_labels, epsilon=0.3)

if perturbed_images is None:
    print("⚠ TODO: Implement PGD attack function")
    exit()

print("✓ Adversarial samples generated")

# Evaluate
def evaluate_model(model, images, labels):
    model.eval()
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        accuracy = 100.0 * correct / len(labels)
    return accuracy, predicted

clean_accuracy, clean_predicted = evaluate_model(model, test_images, test_labels)
adversarial_accuracy, adversarial_predicted = evaluate_model(model, perturbed_images, test_labels)
evasion_rate = 100 - adversarial_accuracy

print(f"Clean accuracy: {clean_accuracy:.2f}%")
print(f"Adversarial accuracy: {adversarial_accuracy:.2f}%")
print(f"Attack success rate: {evasion_rate:.2f}%")

if evasion_rate > 95:
    print("✓ SUCCESS: High evasion rate achieved!")

# Visualize
fig, axes = plt.subplots(3, 5, figsize=(15, 9))

for i in range(5):
    axes[0, i].imshow(test_images[i].cpu().squeeze(), cmap='gray')
    axes[0, i].set_title(f'Original\nTrue: {test_labels[i].item()}\nPred: {clean_predicted[i].item()}')
    axes[0, i].axis('off')
    
    axes[1, i].imshow(perturbed_images[i].detach().cpu().squeeze(), cmap='gray')
    axes[1, i].set_title(f'Adversarial\nTrue: {test_labels[i].item()}\nPred: {adversarial_predicted[i].item()}')
    axes[1, i].axis('off')
    
    perturbation = (perturbed_images[i] - test_images[i]).detach().cpu().squeeze()
    axes[2, i].imshow(perturbation, cmap='RdBu_r', vmin=-0.3, vmax=0.3)
    axes[2, i].set_title('Perturbation')
    axes[2, i].axis('off')

plt.tight_layout()
plt.savefig('week-3/pgd_attack_results.png', dpi=150)
print("Saved: pgd_attack_results.png")

print("\n" + "="*70)
print("Exercise 2 Complete!")
print("="*70)
print(f"Achieved {evasion_rate:.2f}% evasion rate with PGD")
