"""
Week 2 - Exercise 1: Membership Inference Attack

Objective: Implement a membership inference attack to detect training data leakage

Red Team Context: This attack determines if specific samples were in the training data.
This is a privacy violation (HIPAA/GDPR) - like SQL injection leaking database contents.

INSTRUCTIONS:
This script is ~85% complete. Fill in the TODO sections marked with:
  # TODO: Your implementation here
  
Each TODO includes hints. Read carefully before implementing.

Expected Attack Success Rate: >60% (random guess = 50%)
"""

# ============================================================================
# STEP 1: SETUP AND IMPORTS
# ============================================================================
# Standard PyTorch imports for neural networks and optimization
import torch
import torch.nn as nn
import torch.optim as optim
# DataLoader tools for splitting datasets
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
# scikit-learn: Provides accuracy_score and confusion_matrix for evaluation
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

# Set random seed for reproducibility - ensures consistent results
torch.manual_seed(42)
np.random.seed(42)

print("="*70)
print("Membership Inference Attack on MNIST Model")
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
print(f"✓ Model loaded successfully")

# ============================================================================
# STEP 3: PREPARE DATA FOR MEMBERSHIP INFERENCE
# ============================================================================
print("\nPreparing data for membership inference...")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load dataset
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Membership Inference Attack: Determine if data was in training set
# Split training data into member and non-member samples
# Member: Data that WAS in training set (attacker wants to detect this)
# Non-member: Data that was NOT in training set (test set)
# In real attack, attacker doesn't know membership - we know for testing
train_size = len(train_dataset)
member_size = train_size // 2

# random_split: Split dataset randomly into two parts
member_data, non_member_data = random_split(
    train_dataset, 
    [member_size, train_size - member_size],
    generator=torch.Generator().manual_seed(42)
)

print(f"Member samples (in training): {len(member_data)}")
print(f"Non-member samples (not in training): {len(test_dataset)}")

# ============================================================================
# STEP 4: FEATURE EXTRACTION FOR ATTACK MODEL
# ============================================================================
print("\nExtracting features for membership inference attack...")

def extract_features(model, data_loader, device):
    """
    Extract features from target model predictions.
    
    Features used:
    1. Predicted class confidence
    2. Entropy of prediction distribution
    3. Top-3 class confidences
    """
    features = []
    
    model.eval()
    with torch.no_grad():
        for images, _ in data_loader:
            images = images.to(device)
            
            # TODO: Get model predictions
            # HINT: Call model(images) to get logits
            
            # TODO: Apply softmax to get probabilities
            # HINT: Use torch.nn.functional.softmax(predictions, dim=1)
            
            # TODO: Extract features
            # 1. Get predicted class confidence: probabilities.max(dim=1)[0]
            # 2. Calculate entropy: -torch.sum(probabilities * torch.log(probabilities + 1e-9), dim=1)
            # 3. Get top-3 confidences: probabilities.topk(3, dim=1)[0]
            
            # TODO: Combine features into single tensor
            # HINT: Use torch.cat to concatenate features
            sample_features = None  # Replace with your implementation
            
            features.append(sample_features.cpu())
    
    return torch.cat(features, dim=0)

# Create data loaders
batch_size = 64
member_loader = DataLoader(member_data, batch_size=batch_size, shuffle=False)
non_member_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Extract features
print("Extracting features from member samples...")
member_features = extract_features(model, member_loader, device)

print("Extracting features from non-member samples...")
non_member_features = extract_features(model, non_member_loader, device)

if member_features is None or non_member_features is None:
    print("\n⚠ ERROR: Feature extraction not implemented!")
    print("   Please complete the TODO in extract_features()")
    exit()

print(f"\nMember features shape: {member_features.shape}")
print(f"Non-member features shape: {non_member_features.shape}")

# ============================================================================
# STEP 5: BUILD TRAINING DATA FOR ATTACK MODEL
# ============================================================================
print("\nBuilding training data for attack model...")

# Create labels: 1 for member, 0 for non-member
member_labels = torch.ones(len(member_features))
non_member_labels = torch.zeros(len(non_member_features))

# Combine features and labels
X_attack = torch.cat([member_features, non_member_features], dim=0)
y_attack = torch.cat([member_labels, non_member_labels], dim=0)

# Shuffle data
indices = torch.randperm(len(X_attack))
X_attack = X_attack[indices]
y_attack = y_attack[indices]

print(f"Attack training data shape: {X_attack.shape}")
print(f"Member samples: {y_attack.sum().item()}")
print(f"Non-member samples: {len(y_attack) - y_attack.sum().item()}")

# Split into train/val for attack model
val_size = len(X_attack) // 5
X_train_attack = X_attack[:-val_size]
y_train_attack = y_attack[:-val_size]
X_val_attack = X_attack[-val_size:]
y_val_attack = y_attack[-val_size:]

# ============================================================================
# STEP 6: TRAIN ATTACK MODEL
# ============================================================================
print("\nTraining membership inference attack model...")

class AttackModel(nn.Module):
    """Attack model to predict membership from features."""
    def __init__(self, input_dim):
        super(AttackModel, self).__init__()
        
        # TODO: Define layers
        # HINT: Use nn.Linear layers
        # Suggested: input_dim -> 64 -> 32 -> 1
        # Use ReLU activations between layers
        self.fc1 = None  # Replace with your implementation
        self.relu = nn.ReLU()
        self.fc2 = None  # Replace with your implementation
        self.fc3 = None  # Replace with your implementation
        
    def forward(self, x):
        # TODO: Implement forward pass
        # HINT: Apply fc1 -> relu -> fc2 -> relu -> fc3 -> sigmoid
        x = None  # Replace with your implementation
        return x

# Initialize attack model
input_dim = X_attack.shape[1]
attack_model = AttackModel(input_dim)
criterion = nn.BCELoss()
optimizer = optim.Adam(attack_model.parameters(), lr=0.001)

print(f"Attack model input dimension: {input_dim}")

# TODO: Implement training loop
# HINT: Similar to Week 1 training
# 1. Forward pass
# 2. Calculate loss
# 3. Backward pass
# 4. Update weights

def train_attack_model(model, X, y, epochs=20):
    """Train the attack model."""
    losses = []
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # TODO: Forward pass
        # HINT: model(X), calculate loss with criterion
        outputs = None
        loss = None
        
        # TODO: Backward pass
        # HINT: loss.backward(), optimizer.step()
        
        losses.append(loss.item())
        
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
    
    return losses

if X_train_attack is not None:
    train_losses = train_attack_model(attack_model, X_train_attack, y_train_attack, epochs=20)
    print("✓ Attack model training complete")
else:
    print("⚠ TODO: Implement attack model training")

# ============================================================================
# STEP 7: EVALUATE ATTACK PERFORMANCE
# ============================================================================
print("\nEvaluating attack performance...")

attack_model.eval()
with torch.no_grad():
    # TODO: Make predictions on validation set
    # HINT: attack_model(X_val_attack)
    val_predictions = None  # Replace with your implementation
    
    # TODO: Convert to binary predictions (>0.5 = member)
    # HINT: val_predictions > 0.5
    val_pred_binary = None  # Replace with your implementation

if val_pred_binary is not None:
    # Calculate metrics
    accuracy = accuracy_score(y_val_attack.numpy(), val_pred_binary.numpy())
    
    print(f"\nAttack Success Rate: {accuracy:.2%}")
    print(f"(Random guess would be 50%)")
    
    if accuracy > 0.55:
        print("✓ Attack successful! Model leaks training data information.")
    else:
        print("⚠ Attack not very successful. Model may be more secure.")
    
    # Confusion matrix
    cm = confusion_matrix(y_val_attack.numpy(), val_pred_binary.numpy())
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Non-Member', 'Member'],
                yticklabels=['Non-Member', 'Member'])
    plt.title('Membership Inference Attack - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('week-2/membership_inference_results.png', dpi=150)
    print("\nSaved: membership_inference_results.png")
else:
    print("⚠ TODO: Implement attack evaluation")

# ============================================================================
# STEP 8: ANALYSIS AND REPORTING
# ============================================================================
print("\n" + "="*70)
print("Exercise 1 Complete!")
print("="*70)

print("\nWhat you accomplished:")
print("1. ✓ Extracted features from target model predictions")
print("2. ✓ Trained membership inference attack model")
print("3. ✓ Evaluated attack success rate")
print("4. ✓ Generated confusion matrix visualization")

print("\nRed Team Context:")
print("- Membership inference detects training data leakage")
print("- Attack success >60% indicates privacy vulnerability")
print("- This finding would appear in your AI pentest report")

print("\nReal-World Impact:")
print("- HIPAA violation: Leaked patient data in training set")
print("- GDPR violation: Privacy regulation non-compliance")
print("- Competitive intelligence: Competitor can infer your training data")

