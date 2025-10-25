"""
Week 1 - Exercise 1: MNIST Classifier

Objective: Build and train a CNN to classify handwritten digits.
This model will become your evasion attack target in Week 3.

INSTRUCTIONS:
This script is ~85% complete. Your task is to fill in the TODO sections marked with:
  # TODO: Your implementation here
  
Each TODO includes hints. Read the hints carefully before implementing.
After completing all TODOs, run the script to train your model.

Red Team Context: You need attack targets to practice on. This model with 98% 
accuracy will be reduced to <5% using adversarial samples in Week 3.
"""

# ============================================================================
# STEP 1: IMPORT REQUIRED LIBRARIES
# ============================================================================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# ============================================================================
# STEP 2: SET DEVICE AND PATHS
# ============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create directory for saving models
model_dir = Path(__file__).parent.parent.parent / "models"
model_dir.mkdir(parents=True, exist_ok=True)

# ============================================================================
# STEP 3: LOAD AND PREPROCESS MNIST DATASET
# ============================================================================
# TODO: Define transforms for converting PIL images to PyTorch tensors
# HINT: Use transforms.Compose with ToTensor() and Normalize()
# MNIST normalization values: mean=(0.1307,), std=(0.3081,)

transform = transforms.Compose([
    # TODO: Add ToTensor() transform to convert PIL images to tensors
    # TODO: Add Normalize() transform with MNIST mean and std values
])

print("\nDownloading MNIST dataset...")
train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")

# Create data loaders
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Visualize sample images
print("\nVisualizing sample images...")
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
for i in range(8):
    img, label = train_dataset[i]
    axes[i // 4, i % 4].imshow(img.squeeze(), cmap='gray')
    axes[i // 4, i % 4].set_title(f"Label: {label}")
    axes[i // 4, i % 4].axis('off')
plt.tight_layout()
plt.savefig('mnist_samples.png')
print("Sample images saved to mnist_samples.png")

# ============================================================================
# STEP 4: DEFINE CNN ARCHITECTURE
# ============================================================================
class MNIST_CNN(nn.Module):
    """
    Simple CNN for MNIST classification.
    
    Architecture:
    - Conv1: 1 -> 32 channels
    - Pool1: Max pooling 2x2
    - Conv2: 32 -> 64 channels
    - Pool2: Max pooling 2x2
    - FC1: 64*5*5 -> 128
    - FC2: 128 -> 10 (classes)
    """
    def __init__(self):
        super(MNIST_CNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        
        # Activation
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Conv block 1: Apply conv1, ReLU, then pool
        # TODO: Apply self.conv1, self.relu, then self.pool to x
        x = None  # Replace with your implementation
        
        # Conv block 2: Apply conv2, ReLU, then pool
        # TODO: Apply self.conv2, self.relu, then self.pool to x
        x = None  # Replace with your implementation
        
        # Flatten: Reshape from (batch, channels, height, width) to (batch, features)
        # TODO: Use .view() to flatten the tensor
        x = None  # Replace with your implementation
        
        # Fully connected layers
        # TODO: Apply fc1 with ReLU, then fc2
        x = None  # Replace with your implementation
        
        return x

# Initialize model
model = MNIST_CNN().to(device)
print(f"\nModel created with {sum(p.numel() for p in model.parameters()):,} parameters")

# ============================================================================
# STEP 5: DEFINE LOSS FUNCTION AND OPTIMIZER
# ============================================================================
# TODO: Choose appropriate loss function for multi-class classification
# HINT: Use nn.CrossEntropyLoss() for classification tasks

criterion = None  # Replace with your implementation

# TODO: Choose optimizer and set learning rate
# HINT: Adam optimizer with lr=0.001 is a good starting point
optimizer = None  # Replace with your implementation

# Print confirmation
if criterion is None or optimizer is None:
    print("⚠ TODO: Implement loss function and optimizer before training!")
else:
    print(f"Loss function: {criterion}")
    print(f"Optimizer: {optimizer}")

# ============================================================================
# STEP 6: TRAINING LOOP
# ============================================================================
def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        # TODO: Forward pass
        # 1. Zero gradients: optimizer.zero_grad()
        # 2. Get model outputs: outputs = model(images)
        # 3. Calculate loss: loss = criterion(outputs, labels)
        
        # TODO: Backward pass
        # 1. Compute gradients: loss.backward()
        # 2. Update weights: optimizer.step()
        
        # Track metrics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc

# Train for 5 epochs
num_epochs = 5
train_losses = []
train_accuracies = []

# Check if TODOs are completed
if criterion is None or optimizer is None:
    print("\n❌ ERROR: Please complete the TODO sections before training!")
    print("   - Implement loss function (Step 5)")
    print("   - Implement optimizer (Step 5)")
    print("   - Implement forward pass in model (Step 4)")
    exit()

print("\nStarting training...")
for epoch in range(num_epochs):
    try:
        loss, acc = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(loss)
        train_accuracies.append(acc)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {loss:.4f}, Acc: {acc:.2f}%")
    except (AttributeError, RuntimeError) as e:
        print(f"\n❌ ERROR during training: {e}")
        print("\nCheck your TODO implementations:")
        print("  - Did you implement the model forward pass?")
        print("  - Did you implement the training loop TODOs?")
        exit()

# ============================================================================
# STEP 7: EVALUATE ON TEST SET
# ============================================================================
def evaluate(model, test_loader, criterion, device):
    """Evaluate model on test set."""
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    # TODO: Implement evaluation logic
    # HINT: Similar to training but without backward pass
    # Use torch.no_grad() to disable gradient computation
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            # TODO: Get model outputs and calculate loss
            # (Similar to training but no optimizer steps)
            
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    test_loss /= len(test_loader)
    test_acc = 100.0 * correct / total
    return test_loss, test_acc

test_loss, test_acc = evaluate(model, test_loader, criterion, device)
print(f"\nTest Results: Loss: {test_loss:.4f}, Accuracy: {test_acc:.2f}%")

if test_acc >= 95.0:
    print("✓ SUCCESS: Achieved >95% test accuracy!")
else:
    print("⚠ WARNING: Test accuracy below 95%")

# ============================================================================
# STEP 8: VISUALIZE TRAINING PROGRESS
# ============================================================================
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(train_accuracies)
plt.axhline(y=95, color='r', linestyle='--', label='Target (95%)')
plt.title('Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_curves.png', dpi=150)
print("\nTraining curves saved to training_curves.png")

# ============================================================================
# STEP 9: SAVE MODEL
# ============================================================================
model_path = model_dir / "mnist_cnn.pt"
torch.save(model.state_dict(), model_path)
print(f"\nModel saved to: {model_path}")

print("\n" + "="*70)
print("Exercise 1 Complete!")
print("="*70)
print("\nWhat you accomplished:")
print("1. Built a CNN for MNIST classification")
print("2. Achieved >95% accuracy (your baseline)")
print("3. Saved the model for Week 3 attacks")
print(f"\nBaseline accuracy: {test_acc:.2f}%")
print("This model will be attacked in Week 3 (evasion attacks)")
