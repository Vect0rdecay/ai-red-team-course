"""
Week 2 - Exercise 2: Shadow Models for Membership Inference

Objective: Train shadow models to enable powerful membership inference attacks

Red Team Context: Shadow models mimic target behavior without access to training data.
This is like creating a test environment that mirrors production for exploit development.

INSTRUCTIONS:
This script is ~85% complete. Fill in the TODO sections marked with:
  # TODO: Your implementation here
  
Each TODO includes hints. Read carefully before implementing.

Shadow models allow us to:
- Understand model behavior without access to training data
- Generate attack training data
- Develop more sophisticated membership inference attacks
"""

# ============================================================================
# STEP 1: SETUP AND IMPORTS
# ============================================================================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import accuracy_score
import copy

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

print("="*70)
print("Shadow Model Training for Membership Inference")
print("="*70)

# ============================================================================
# STEP 2: DEFINE MODEL ARCHITECTURE
# ============================================================================
print("\nDefining model architecture...")

class MNIST_CNN(nn.Module):
    """Same architecture as target model."""
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

# ============================================================================
# STEP 3: PREPARE DATA
# ============================================================================
print("\nPreparing data...")

# Standard MNIST transformations (same as Week 1)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load dataset
dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# Shadow Models: Train our own model with similar data as target
# We don't have access to target's training data, but we can get similar data
# Shadow model mimics target's behavior for attack development
# For shadow models, we split available data into shadow training and shadow testing
dataset_size = len(dataset)
train_size = int(0.8 * dataset_size)  # 80% for training
test_size = dataset_size - train_size  # 20% for testing

# random_split: Split dataset randomly into two parts
# Shadow models need their own train/test split to learn patterns
shadow_train_data, shadow_test_data = torch.utils.data.random_split(
    dataset, 
    [train_size, test_size],
    generator=torch.Generator().manual_seed(42)  # Reproducible split
)

print(f"Shadow training samples: {len(shadow_train_data)}")
print(f"Shadow test samples: {len(shadow_test_data)}")

# Create data loaders
batch_size = 64
train_loader = DataLoader(shadow_train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(shadow_test_data, batch_size=batch_size, shuffle=False)

# ============================================================================
# STEP 4: TRAIN SHADOW MODEL
# ============================================================================
print("\nTraining shadow model...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
shadow_model = MNIST_CNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(shadow_model.parameters(), lr=0.001)

# TODO: Implement training loop
# HINT: Similar to Week 1 exercise
# 1. Forward pass
# 2. Calculate loss
# 3. Backward pass
# 4. Update weights

def train_shadow_model(model, train_loader, epochs=10):
    """Train a shadow model to mimic target model behavior."""
    model.train()  # Enable training features (dropout, batch norm updates)
    
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Standard training loop: for each batch of images
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass: Get model predictions
            # TODO: Forward pass
            # 1. Zero gradients: optimizer.zero_grad() - clear previous gradients
            # 2. Get predictions: outputs = model(images) - forward through network
            # 3. Calculate loss: loss = criterion(outputs, labels) - how wrong are we?
            
            # Backward pass: Update model to reduce loss
            # TODO: Backward pass
            # 1. Backward pass: loss.backward() - calculate gradients
            # 2. Update weights: optimizer.step() - improve predictions
            
            # Track metrics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100.0 * correct / total
        
        if (epoch + 1) % 2 == 0:
            print(f"  Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%")

if True:  # Set to False if training not implemented
    print("⚠ TODO: Implement shadow model training")
else:
    train_shadow_model(shadow_model, train_loader, epochs=10)
    print("✓ Shadow model trained")

# ============================================================================
# STEP 5: EVALUATE SHADOW MODEL
# ============================================================================
print("\nEvaluating shadow model...")

def evaluate_model(model, test_loader):
    """Evaluate model on test set."""
    model.eval()
    correct = 0
    total = 0
    
    # TODO: Implement evaluation
    # HINT: Use torch.no_grad(), get predictions, calculate accuracy
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            # TODO: Get predictions
            # HINT: outputs = model(images), get predicted classes
            
            total += labels.size(0)
            # TODO: Count correct predictions
            # HINT: correct += (predicted == labels).sum().item()
    
    accuracy = 100.0 * correct / total
    return accuracy

shadow_accuracy = evaluate_model(shadow_model, test_loader)
print(f"Shadow model accuracy: {shadow_accuracy:.2f}%")

# ============================================================================
# STEP 6: USE SHADOW MODEL FOR ATTACK TRAINING
# ============================================================================
print("\nUsing shadow model for membership inference attack training...")

print("\nKey insight:")
print("  - We can query shadow model as much as we want")
print("  - We know which samples were in shadow training set")
print("  - This gives us labeled data to train membership inference attack")
print("  - Attack learned on shadow model should work on target model")

# Extract predictions from shadow model for attack training
def extract_predictions(model, data_loader, device):
    """Extract predictions and features from model."""
    all_outputs = []
    
    model.eval()
    with torch.no_grad():
        for images, _ in data_loader:
            images = images.to(device)
            
            # TODO: Get model outputs
            # HINT: outputs = model(images)
            
            # TODO: Apply softmax to get probabilities
            # HINT: Use torch.nn.functional.softmax(outputs, dim=1)
            
            all_outputs.append(probs.cpu())
    
    return torch.cat(all_outputs, dim=0)

print("\nExtracting predictions for attack training...")
# NOTE: This would be done with known member/non-member labels
# In real scenario, we'd query shadow model and label based on training set membership

# ============================================================================
# STEP 7: VISUALIZE SHADOW MODEL BEHAVIOR
# ============================================================================
print("\nGenerating visualizations...")

# Visualize predictions on sample images
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
shadow_model.eval()

with torch.no_grad():
    for i in range(10):
        # Get random sample
        idx = np.random.randint(0, len(shadow_test_data))
        image, true_label = shadow_test_data[idx]
        
        # Get prediction
        img = image.unsqueeze(0).to(device)
        output = shadow_model(img)
        probs = torch.nn.functional.softmax(output, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()
        
        # Plot image
        axes[0, i].imshow(image.squeeze(), cmap='gray')
        axes[0, i].set_title(f'True: {true_label}\nPred: {pred}\nConf: {confidence:.2f}')
        axes[0, i].axis('off')
        
        # Plot probability distribution
        axes[1, i].bar(range(10), probs[0].cpu().numpy())
        axes[1, i].set_ylim([0, 1])
        axes[1, i].set_xlabel('Class')
        axes[1, i].set_ylabel('Probability')

plt.tight_layout()
plt.savefig('week-2/shadow_model_predictions.png', dpi=150)
print("Saved: shadow_model_predictions.png")

# ============================================================================
# DOCUMENTATION
# ============================================================================
print("\n" + "="*70)
print("Exercise 2 Complete!")
print("="*70)

print("\nWhat you accomplished:")
print("1. ✓ Trained a shadow model to mimic target behavior")
print("2. ✓ Evaluated shadow model performance")
print("3. ✓ Extracted predictions for attack training")
print("4. ✓ Visualized shadow model behavior")

print("\nRed Team Context:")
print("- Shadow models enable attack development without target access")
print("- Attack trained on shadow model transfers to target model")
print("- This mirrors creating a test environment for exploit development")
print("- Shadow models are commonly used in real-world AI attacks")

print("\nNext Steps:")
print("- Use shadow model predictions to train membership inference attack")
print("- Compare attack performance on shadow vs target model")
print("- Apply this methodology to your Week 1 target model")
