"""
Week 3 - Exercise 4: FGSM (Fast Gradient Sign Method) Attack (From Scratch)

Objective: Implement FGSM from scratch to generate adversarial samples

Red Team Context: FGSM is the simplest evasion attack - fast to execute, easy to understand.
Like SQL injection payloads, adversarial samples look normal but bypass security controls.

INSTRUCTIONS:
This script is ~85% complete. Fill in the TODO sections marked with:
  # TODO: Your implementation here
  
Each TODO includes hints. Read carefully before implementing.

Expected Evasion Rate: >80% with ε=0.3
"""

# ============================================================================
# STEP 1: SETUP AND IMPORTS
# ============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

print("="*70)
print("FGSM Attack Implementation")
print("="*70)

# ============================================================================
# STEP 2: LOAD TRAINED MODEL FROM WEEK 1
# ============================================================================
print("\nLoading trained MNIST model from Week 1...")

# Define model architecture (same as Week 1)
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

# Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MNIST_CNN().to(device)

model_path = Path(__file__).parent.parent.parent / "models" / "mnist_cnn.pt"

if not model_path.exists():
    print("⚠ Error: Week 1 model not found!")
    print("   Please run Week 1, Exercise 1 first to train the model.")
    exit()

model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print(f"✓ Model loaded successfully on {device}")

# ============================================================================
# STEP 3: LOAD TEST DATA
# ============================================================================
print("\nLoading test data...")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Get a small sample for testing
test_images, test_labels = next(iter(test_loader))
test_images = test_images[:10].to(device)
test_labels = test_labels[:10].to(device)

print(f"Loaded {len(test_images)} test samples")

# ============================================================================
# STEP 4: IMPLEMENT FGSM ATTACK
# ============================================================================
print("\nImplementing FGSM attack...")

def fgsm_attack(model, images, labels, epsilon=0.3):
    """
    Perform FGSM attack on model.
    
    FGSM (Fast Gradient Sign Method): Simple one-step adversarial attack
    Algorithm:
    1. Compute loss (how wrong the model is)
    2. Get gradient of loss w.r.t. input (which pixels affect the prediction)
    3. Take sign of gradient (direction of steepest ascent)
    4. Add epsilon * sign to original image (perturb in worst direction)
    5. Clip to valid range [0, 1] (keep image valid)
    
    Args:
        model: Target model to attack
        images: Input images to perturb
        labels: True labels for images
        epsilon: Perturbation strength (budget) - controls attack strength
    
    Returns:
        perturbed_images: Adversarial samples (look like originals but fool the model)
    """
    # Set requires_grad on inputs - need gradients w.r.t. inputs (not weights!)
    # TODO: Enable gradient computation on images
    # HINT: images.requires_grad_(True)
    images.requires_grad_(True)
    
    # Forward pass: Get model's predictions
    outputs = model(images)
    
    # Compute loss: How wrong is the model?
    criterion = nn.CrossEntropyLoss()
    # TODO: Calculate loss
    # HINT: loss = criterion(outputs, labels)
    loss = None  # Replace with calculation
    
    # Backward pass to get gradients - calculate how each pixel affects loss
    model.zero_grad()
    # TODO: Compute gradients
    # HINT: loss.backward()
    loss.backward()
    
    # Get sign of gradients - direction that INCREASES loss (makes model fail)
    # TODO: Extract gradient signs
    # HINT: sign_data = images.grad.sign()
    sign_data = None  # Replace with gradient sign extraction
    
    # Create adversarial samples: Add perturbation in worst direction
    # TODO: Add perturbation to images
    # HINT: perturbed_images = images + epsilon * sign_data
    perturbed_images = None  # Replace with perturbation addition
    
    # Clip to [0, 1] range - images must be valid pixel values
    # TODO: Clip values to valid range
    # HINT: perturbed_images = torch.clamp(perturbed_images, 0, 1)
    perturbed_images = None  # Replace with clamping
    
    return perturbed_images

# Test FGSM attack
print("Generating adversarial samples with ε=0.3...")
epsilon = 0.3

# Apply FGSM attack
perturbed_images = fgsm_attack(model, test_images, test_labels, epsilon)

if perturbed_images is None:
    print("⚠ TODO: Implement FGSM attack function")
    exit()

print("✓ Adversarial samples generated")

# ============================================================================
# STEP 5: EVALUATE ATTACK SUCCESS
# ============================================================================
print("\nEvaluating attack success...")

def evaluate_model(model, images, labels):
    """Evaluate model accuracy on given images and labels."""
    model.eval()
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        accuracy = 100.0 * correct / len(labels)
    return accuracy, predicted

# Evaluate on clean images
clean_accuracy, clean_predicted = evaluate_model(model, test_images, test_labels)
print(f"Clean accuracy: {clean_accuracy:.2f}%")

# Evaluate on adversarial images
adversarial_accuracy, adversarial_predicted = evaluate_model(model, perturbed_images, test_labels)
print(f"Adversarial accuracy: {adversarial_accuracy:.2f}%")

# Calculate evasion rate
evasion_rate = 100 - adversarial_accuracy
print(f"\nAttack success rate: {evasion_rate:.2f}%")

if evasion_rate > 80:
    print("✓ SUCCESS: High evasion rate achieved!")
else:
    print("⚠ Attack could be stronger. Check implementation.")

# ============================================================================
# STEP 6: VISUALIZE RESULTS
# ============================================================================
print("\nVisualizing adversarial samples...")

fig, axes = plt.subplots(3, 5, figsize=(15, 9))

for i in range(5):
    # Original image
    axes[0, i].imshow(test_images[i].cpu().squeeze(), cmap='gray')
    axes[0, i].set_title(f'Original\nTrue: {test_labels[i].item()}\nPred: {clean_predicted[i].item()}')
    axes[0, i].axis('off')
    
    # Adversarial image
    axes[1, i].imshow(perturbed_images[i].detach().cpu().squeeze(), cmap='gray')
    axes[1, i].set_title(f'Adversarial\nTrue: {test_labels[i].item()}\nPred: {adversarial_predicted[i].item()}')
    axes[1, i].axis('off')
    
    # Perturbation
    perturbation = (perturbed_images[i] - test_images[i]).detach().cpu().squeeze()
    axes[2, i].imshow(perturbation, cmap='RdBu_r', vmin=-epsilon, vmax=epsilon)
    axes[2, i].set_title(f'Perturbation\n(ε={epsilon})')
    axes[2, i].axis('off')

plt.tight_layout()
plt.savefig('week-3/fgsm_attack_results.png', dpi=150)
print("Saved: fgsm_attack_results.png")

# ============================================================================
# STEP 7: EXPERIMENT WITH DIFFERENT EPSILON VALUES
# ============================================================================
print("\nTesting different epsilon values...")

epsilons = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
accuracies = []
evasion_rates = []

for eps in epsilons:
    perturbed = fgsm_attack(model, test_images, test_labels, eps)
    acc, _ = evaluate_model(model, perturbed, test_labels)
    evasion = 100 - acc
    accuracies.append(acc)
    evasion_rates.append(evasion)

# Plot results
plt.figure(figsize=(10, 5))
plt.plot(epsilons, evasion_rates, 'o-', linewidth=2)
plt.xlabel('Epsilon (ε)', fontsize=12)
plt.ylabel('Evasion Rate (%)', fontsize=12)
plt.title('FGSM Attack: Evasion Rate vs Perturbation Budget', fontsize=14)
plt.grid(True, alpha=0.3)
plt.axhline(y=80, color='r', linestyle='--', label='Target (80%)')
plt.legend()
plt.tight_layout()
plt.savefig('week-3/fgsm_epsilon_analysis.png', dpi=150)
print("Saved: fgsm_epsilon_analysis.png")

# ============================================================================
# DOCUMENTATION
# ============================================================================
print("\n" + "="*70)
print("Exercise 1 Complete!")
print("="*70)

print("\nWhat you accomplished:")
print("1. ✓ Implemented FGSM attack from scratch")
print("2. ✓ Generated adversarial samples with ε=0.3")
print(f"3. ✓ Achieved {evasion_rate:.2f}% evasion rate")
print("4. ✓ Visualized adversarial examples and perturbations")
print("5. ✓ Analyzed effect of different epsilon values")

print("\nKey Insights:")
print(f"- Clean model accuracy: {clean_accuracy:.2f}%")
print(f"- Adversarial accuracy: {adversarial_accuracy:.2f}%")
print(f"- Attack reduced accuracy by {clean_accuracy - adversarial_accuracy:.2f}%")
print(f"- Best epsilon for attack: {epsilons[np.argmax(evasion_rates)]}")

print("\nRed Team Context:")
print("- FGSM is the simplest white-box evasion attack")
print("- Fast to execute but less powerful than iterative attacks")
print("- Demonstrates model vulnerability to adversarial samples")
print("- Perturbations are imperceptible but highly effective")

print("\nNext Steps:")
print("- Implement PGD for stronger attacks (Exercise 3)")
print("- Compare FGSM vs PGD performance (Exercise 4)")
