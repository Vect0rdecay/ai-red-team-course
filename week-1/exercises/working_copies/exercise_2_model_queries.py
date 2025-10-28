"""
Week 1 - Exercise 2: Model Querying and Inference

Objective: Learn to interact with ML models programmatically (reconnaissance skill)

INSTRUCTIONS:
This script is ~85% complete. Your task is to fill in the TODO sections.
Read the hints carefully before implementing each TODO.

Red Team Context: Before attacking, you must understand normal model behavior. 
This is the equivalent of service enumeration in traditional pentesting.

You'll query your trained MNIST model to:
1. Understand its predictions and confidence scores
2. Identify correctly vs incorrectly classified examples
3. Analyze decision boundaries
4. Document model behavior for exploit development
"""

# torch.nn.functional: Advanced operations like softmax (probability calculation)
import torch
import torch.nn as nn
from torchvision import datasets, transforms
# matplotlib: For plotting images and graphs
import matplotlib.pyplot as plt
# numpy: Numerical operations and array handling
import numpy as np
from pathlib import Path
# seaborn: Better-looking statistical plots
import seaborn as sns

# Set style for better-looking plots (optional but nice for reports)
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# STEP 1: LOAD TRAINED MODEL
# ============================================================================
print("Loading trained MNIST model...")

# Define the same model architecture as in Exercise 1 (must match exactly!)
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

# Load model weights that were trained in Exercise 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MNIST_CNN().to(device)

model_path = Path(__file__).parent.parent.parent / "models" / "mnist_cnn.pt"

# Check if model exists - need to run Exercise 1 first!
if model_path.exists():
    # load_state_dict: Loads the trained weights into our model architecture
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"✓ Model loaded from {model_path}")
else:
    print("⚠ Model not found! Please run exercise_1_mnist_classifier.py first")
    exit()

# model.eval(): Turns off training features (dropout, batch norm updates)
# Required when making predictions - tells PyTorch we're in inference mode
model.eval()
print(f"Model loaded on device: {device}")

# ============================================================================
# STEP 2: LOAD TEST DATA
# ============================================================================
print("\nLoading test dataset...")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
print(f"Test samples available: {len(test_dataset)}")

# Get a random subset for analysis
np.random.seed(42)
sample_indices = np.random.choice(len(test_dataset), 100, replace=False)

# ============================================================================
# STEP 3: QUERY MODEL WITH TEST SAMPLES
# ============================================================================
print("\nQuerying model with test samples...")

# Lists to store results for later analysis
predictions = []  # What the model predicted (0-9)
actual_labels = []  # What the image actually is (ground truth)
confidence_scores = []  # How confident the model is (0-1)

# torch.no_grad(): Disable gradient computation to save memory and speed
# We're not training, so we don't need gradients (forward pass only)
with torch.no_grad():
    for idx in sample_indices:
        image, label = test_dataset[idx]
        image = image.unsqueeze(0).to(device)
        
        # TODO: Get model prediction
        # HINT: Call model(image) to get the raw outputs
        
        # TODO: Apply softmax to get probabilities
        # HINT: Use torch.nn.functional.softmax(output, dim=1)
        
        # TODO: Get predicted class (index with highest probability)
        # HINT: Use torch.argmax(output, dim=1).item()
        
        # TODO: Get confidence score (probability of predicted class)
        # HINT: Access probabilities[0][predicted_class].item()
        
        predictions.append(predicted_class)
        actual_labels.append(label)
        confidence_scores.append(confidence)

# Convert to numpy for easier manipulation
predictions = np.array(predictions)
actual_labels = np.array(actual_labels)
confidence_scores = np.array(confidence_scores)

# Calculate accuracy
correct = (predictions == actual_labels).sum()
accuracy = (correct / len(predictions)) * 100
print(f"\nModel performance on sample:")
print(f"  Accuracy: {accuracy:.2f}%")
print(f"  Correct: {correct}/{len(predictions)}")
print(f"  Average confidence: {confidence_scores.mean():.4f}")

# ============================================================================
# STEP 4: ANALYZE CORRECTLY CLASSIFIED IMAGES
# ============================================================================
print("\nAnalyzing correct predictions...")

correct_indices = np.where(predictions == actual_labels)[0]
incorrect_indices = np.where(predictions != actual_labels)[0]

print(f"  Correct predictions: {len(correct_indices)}")
print(f"  Incorrect predictions: {len(incorrect_indices)}")

# Visualize 5 correctly classified examples
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
fig.suptitle('Correctly Classified Examples', fontsize=16)

for i in range(min(5, len(correct_indices))):
    idx = sample_indices[correct_indices[i]]
    image, label = test_dataset[idx]
    pred = predictions[correct_indices[i]]
    conf = confidence_scores[correct_indices[i]]
    
    # Plot original image
    axes[0, i].imshow(image.squeeze(), cmap='gray')
    axes[0, i].set_title(f'True: {label}, Pred: {pred}\nConf: {conf:.3f}')
    axes[0, i].axis('off')
    
    # Get probability distribution
    with torch.no_grad():
        img = image.unsqueeze(0).to(device)
        output = model(img)
        probs = torch.nn.functional.softmax(output, dim=1).cpu().numpy()[0]
    
    axes[1, i].bar(range(10), probs)
    axes[1, i].set_ylim([0, 1])
    axes[1, i].set_xlabel('Class')
    axes[1, i].set_ylabel('Probability')
    axes[1, i].axvline(x=pred, color='r', linestyle='--', linewidth=2)

plt.tight_layout()
plt.savefig('correct_predictions.png', dpi=150)
print("\nSaved: correct_predictions.png")

# ============================================================================
# STEP 5: ANALYZE INCORRECTLY CLASSIFIED IMAGES
# ============================================================================
print("\nAnalyzing misclassified examples...")

if len(incorrect_indices) > 0:
    fig, axes = plt.subplots(2, min(5, len(incorrect_indices)), figsize=(15, 6))
    if len(incorrect_indices) == 1:
        axes = axes.reshape(2, 1)
    
    fig.suptitle('Misclassified Examples', fontsize=16)
    
    for i in range(min(5, len(incorrect_indices))):
        idx = sample_indices[incorrect_indices[i]]
        image, label = test_dataset[idx]
        pred = predictions[incorrect_indices[i]]
        conf = confidence_scores[incorrect_indices[i]]
        
        # Plot original image
        if len(incorrect_indices) > 1:
            axes[0, i].imshow(image.squeeze(), cmap='gray')
            axes[0, i].set_title(f'True: {label}, Pred: {pred}\nConf: {conf:.3f}', 
                                color='red')
            axes[0, i].axis('off')
            
            # Probability distribution
            with torch.no_grad():
                img = image.unsqueeze(0).to(device)
                output = model(img)
                probs = torch.nn.functional.softmax(output, dim=1).cpu().numpy()[0]
            
            axes[1, i].bar(range(10), probs, color='red')
            axes[1, i].set_ylim([0, 1])
            axes[1, i].set_xlabel('Class')
            axes[1, i].set_ylabel('Probability')
            axes[1, i].axvline(x=pred, color='r', linestyle='--', linewidth=2)
            axes[1, i].axvline(x=label, color='g', linestyle='--', linewidth=2)
    
    plt.tight_layout()
    plt.savefig('misclassified_examples.png', dpi=150)
    print("Saved: misclassified_examples.png")
else:
    print("No misclassified examples found in sample!")

# ============================================================================
# STEP 6: CONFIDENCE DISTRIBUTION ANALYSIS
# ============================================================================
print("\nAnalyzing confidence distributions...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Confidence histogram
axes[0].hist(confidence_scores, bins=20, alpha=0.7, edgecolor='black')
axes[0].axvline(confidence_scores.mean(), color='r', linestyle='--', 
                label=f'Mean: {confidence_scores.mean():.3f}')
axes[0].set_xlabel('Confidence Score')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Distribution of Confidence Scores')
axes[0].legend()

# Accuracy vs Confidence
bins = np.linspace(0, 1, 11)
bin_indices = np.digitize(confidence_scores, bins)
bin_accuracies = []

for i in range(1, len(bins)):
    mask = bin_indices == i
    if mask.sum() > 0:
        bin_acc = (predictions[mask] == actual_labels[mask]).mean() * 100
        bin_accuracies.append(bin_acc)
    else:
        bin_accuracies.append(0)

axes[1].bar(bins[:-1], bin_accuracies, width=0.1, alpha=0.7, edgecolor='black')
axes[1].set_xlabel('Confidence Score Bin')
axes[1].set_ylabel('Accuracy (%)')
axes[1].set_title('Accuracy vs Confidence')
axes[1].set_ylim([0, 105])

plt.tight_layout()
plt.savefig('confidence_analysis.png', dpi=150)
print("Saved: confidence_analysis.png")

# ============================================================================
# STEP 7: CONFUSION MATRIX
# ============================================================================
print("\nGenerating confusion matrix...")

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(actual_labels, predictions)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
            xticklabels=range(10), yticklabels=range(10))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
print("Saved: confusion_matrix.png")

# ============================================================================
# STEP 8: DECISION BOUNDARY ANALYSIS
# ============================================================================
print("\nAnalyzing decision boundaries...")

# Analyze which classes are most confused with each other
confusion_pairs = []
for i in range(len(actual_labels)):
    if predictions[i] != actual_labels[i]:
        confusion_pairs.append((actual_labels[i], predictions[i]))

if confusion_pairs:
    from collections import Counter
    common_confusions = Counter(confusion_pairs).most_common(5)
    
    print("\nMost common misclassifications:")
    for (true_label, pred_label), count in common_confusions:
        print(f"  {true_label} → {pred_label}: {count} times")

# ============================================================================
# STEP 9: DOCUMENTATION
# ============================================================================
print("\n" + "="*70)
print("Exercise 2 Complete!")
print("="*70)
print("\nWhat you accomplished:")
print("1. ✓ Loaded and queried your trained MNIST model")
print("2. ✓ Analyzed prediction confidence scores")
print("3. ✓ Visualized correct and incorrect classifications")
print("4. ✓ Created confusion matrix")
print("5. ✓ Analyzed decision boundaries")
print("\nKey Insights:")
print(f"  - Overall accuracy on sample: {accuracy:.2f}%")
print(f"  - Average confidence: {confidence_scores.mean():.4f}")
print(f"  - High confidence predictions: {(confidence_scores > 0.9).sum()}/{len(confidence_scores)}")
print("\nRed Team Context:")
print("  Before attacking a model, understanding its behavior is critical.")
print("  This querying/reconnaissance will help you:")
print("  - Identify vulnerable inputs (low confidence predictions)")
print("  - Understand decision boundaries for adversarial crafting")
print("  - Measure attack success (baseline vs adversarial accuracy)")
print("\nNext: Use this knowledge for Week 3 evasion attacks!")
