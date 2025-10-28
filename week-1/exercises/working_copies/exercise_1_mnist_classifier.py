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
# torch: Core PyTorch library for neural networks and tensors
import torch
# torch.nn: Neural network layers (Conv2d, Linear, ReLU, etc.)
import torch.nn as nn
# torch.optim: Optimizers (Adam, SGD) for training
import torch.optim as optim
# DataLoader: Helps load data in batches during training
from torch.utils.data import DataLoader
# datasets, transforms: Pre-built datasets and image transformations
from torchvision import datasets, transforms
# matplotlib: For plotting and visualization
import matplotlib.pyplot as plt
# numpy: Numerical operations
import numpy as np
import os
from pathlib import Path

# Set random seed for reproducibility - ensures same results each run
torch.manual_seed(42)
np.random.seed(42)

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# ============================================================================
# STEP 2: SET DEVICE AND PATHS
# ============================================================================
# Choose GPU if available, otherwise use CPU. GPUs are much faster for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create directory for saving trained models (we'll load this in Week 3)
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
    transforms.ToTensor(),
    # TODO: Add Normalize() transform with MNIST mean and std values
    transforms.Normalize((0.1307,), (0.3081,))
])

print("\nDownloading MNIST dataset...")
# MNIST: 60,000 training images and 10,000 test images of handwritten digits (0-9)
# Each image is 28x28 pixels, grayscale
train_dataset = datasets.MNIST(
    root='./data',
    train=True,    # Get training set (60,000 images)
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root='./data',
    train=False,   # Get test set (10,000 images) - used to evaluate model performance
    download=True,
    transform=transform
)

print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")

# DataLoader splits dataset into batches for efficient training
# batch_size: How many images to process at once (larger = faster but needs more memory)
batch_size = 64
# shuffle=True: Randomize order to prevent model memorizing sequence
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# shuffle=False: Test data order doesn't matter
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
    
    CNN (Convolutional Neural Network): Best for image recognition
    Architecture:
    - Conv1: 1 -> 32 channels (detect simple features like edges)
    - Pool1: Max pooling 2x2 (reduce size, keep important info)
    - Conv2: 32 -> 64 channels (detect complex features like shapes)
    - Pool2: Max pooling 2x2
    - FC1: 64*7*7 -> 128 (fully connected layer for classification)
    - FC2: 128 -> 10 (final output: probability for each digit 0-9)
    """
    def __init__(self):
        super(MNIST_CNN, self).__init__()
        
        # Conv2d: Applies filters to detect image features
        # Input: 1 channel (grayscale), Output: 32 filters
        # kernel_size=3: 3x3 filter, padding=1: keeps image size same
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        # Input: 32 channels, Output: 64 filters (more complex features)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # MaxPool2d: Reduces image size by taking max value in each 2x2 region
        # This makes network faster and reduces overfitting
        self.pool = nn.MaxPool2d(2, 2)
        
        # Linear layers: Fully connected layers for final classification
        # 64*7*7: flattened feature map size after conv layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        # Output layer: 10 classes (digits 0-9)
        self.fc2 = nn.Linear(128, 10)
        
        # ReLU: Activation function that adds non-linearity (neural networks need this!)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Conv block 1: Apply conv1, ReLU, then pool
        # TODO: Apply self.conv1, self.relu, then self.pool to x
        x = self.conv1(x)        # → (batch, 32, 28, 28)  [padding=1 keeps size]
        x = self.relu(x)         # → non-linearity
        x = self.pool(x)         # → (batch, 32, 14, 14)  [28/2 = 14]
        
        # Conv block 2: Apply conv2, ReLU, then pool
        # TODO: Apply self.conv2, self.relu, then self.pool to x
        x = self.conv2(x)        # → (batch, 64, 14, 14)
        x = self.relu(x)
        x = self.pool(x)         # → (batch, 64, 7, 7)    [14/2 = 7]
        
        # Flatten: Reshape from (batch, channels, height, width) to (batch, features)
        # TODO: Use .view() to flatten the tensor
        x = x.view(x.size(0), -1)  # or: x.view(-1, 64*7*7)
        # Now: (batch_size, 3136)
        
        # Fully connected layers
        # TODO: Apply fc1 with ReLU, then fc2
        x = self.fc1(x)          # → (batch, 128)
        x = self.relu(x)
        x = self.fc2(x)          # → (batch, 10) → raw logits
        
        return x

# Initialize model
model = MNIST_CNN().to(device)
print(f"\nModel created with {sum(p.numel() for p in model.parameters()):,} parameters")

# ============================================================================
# STEP 5: DEFINE LOSS FUNCTION AND OPTIMIZER
# ============================================================================
# Loss function: Measures how wrong our predictions are
# CrossEntropyLoss: Standard for multi-class classification (10 digits)
# TODO: Choose appropriate loss function for multi-class classification
# HINT: Use nn.CrossEntropyLoss() for classification tasks

criterion = None  # Replace with your implementation

# Optimizer: Updates model weights to reduce loss (learning)
# Adam: Adaptive learning rate optimizer (works well for most problems)
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
    model.train()  # Set model to training mode (enables dropout, batch norm)
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Loop through batches of images and labels
    for images, labels in train_loader:
        # Move data to GPU if available (much faster)
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass: Get model predictions
        # TODO: Forward pass
        # 1. Zero gradients: optimizer.zero_grad() - clear old gradients
        # 2. Get model outputs: outputs = model(images) - prediction probabilities
        # 3. Calculate loss: loss = criterion(outputs, labels) - how wrong we are
        
        # Backward pass: Update model weights based on error
        # TODO: Backward pass
        # 1. Compute gradients: loss.backward() - calculate gradients
        # 2. Update weights: optimizer.step() - improve predictions
        
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
