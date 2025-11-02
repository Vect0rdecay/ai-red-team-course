"""
Week X - Exercise Y: [Exercise Title]

Objective: [What students will learn]

INSTRUCTIONS:
You can use this shell to create a pytorhc model, prep a dataset, train a quick model, wrap with an attack framework, and then do the attack.

"""

# ============================================================================
# IMPORTS
# ============================================================================
# [Add necessary imports with comments explaining their purpose]
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
# from art.    add you specific import needed here
from art.estimators.classification import PyTorchClassifier


# ============================================================================
# STEP 1: Pytorch model creation
# ============================================================================
print("="*70)
print("[Exercise Title]")
print("="*70)

# Create a new type of neural network, inherit from Torch
class SimpleModel(nn.Module):
    def __init__(self):     # class constructor
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)    # fully connected layer, 28*28 input, 128 output
        self.fc2 = nn.Linear(128, 10)         # fully connected layer, 128 input, 10 output
    
    def forward(self, x):                     # forward pass through network
        x = x.view(-1, 784)                    # flatten the input in 784 dimension vector
        x = torch.relu(self.fc1(x))             # apply relu (neg with 0) to first layer
        x = self.fc2(x)                         # second layer
        return x

# ============================================================================
# STEP 2: Dataset preparation
# ============================================================================
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# IMPORTANT: Normalize pixel values from [0, 255] to [0, 1] for consistency with training
# The model was trained on normalized data (transforms.ToTensor() divides by 255), so test data must match
x_test = test_dataset.data[:200].float().unsqueeze(1) / 255.0  # grab 1st 200, conv to float, normalize to [0,1], add 1 dimension for channel
y_test = test_dataset.targets[:200].numpy()  # grab 1st 200 labels, conv torch tensor to numpy array

print(f"\nData preparation:")
print(f"  Test data shape: {x_test.shape}")
print(f"  Test data range: [{x_test.min():.3f}, {x_test.max():.3f}] (should be [0, 1])")
print(f"  Number of test samples: {len(y_test)}")

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# ============================================================================
# STEP 3: Model training
# ============================================================================
model = SimpleModel()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(1):
    for i, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')

model.eval()
with torch.no_grad():
    outputs = model(x_test)
    _, predicted = torch.max(outputs.data, 1)
    print(f'Predicted: {predicted}')
    print(f'Actual: {y_test}')


# ============================================================================
# STEP 3: Wrap with ART framework
# ============================================================================
print("\n" + "="*70)
print("STEP 3: Wrapping model with ART framework")
print("="*70)

classifier = PyTorchClassifier(
    model=model, # our model
    clip_values=(0, 1), # enforce valid data bounds for the attacks
    loss=criterion, # cross entropy loss function for training/gradient computation
    optimizer=optimizer, # updates model weights
    input_shape=(1, 28, 28), # 1 channel, 28x28 pixels, validates input shape
    nb_classes=10 # number of output classes, 0-9
)

# Evaluate original model accuracy before attack
print("\nEvaluating original model on clean test data...")
original_preds = classifier.predict(x_test.numpy())
original_pred_classes = original_preds.argmax(axis=1)
original_acc = (original_pred_classes == y_test).mean()
print(f"Original model accuracy: {original_acc:.4f} ({original_acc*100:.2f}%)")
print(f"Original predictions: {original_pred_classes[:10]}...")
print(f"Actual labels:         {y_test[:10]}...")

# ============================================================================
# STEP 4: Attack Setup
# ============================================================================
print("\n" + "="*70)
print("STEP 4: Setting up adversarial attack")
print("="*70)

from art.attacks.evasion import FastGradientMethod

# Attack configuration
eps = 0.2
print(f"\nAttack type: Fast Gradient Method (FGM)")
print(f"Epsilon (perturbation budget): {eps}")
print(f"Max perturbation per pixel: {eps}")
print(f"Clip values: [0, 1] (pixel values will be clipped to this range)")
print(f"\nNote: Since data is normalized to [0, 1], epsilon={eps} means")
print(f"      we can perturb each pixel by up to {eps*100:.0f}% of its max value.")
print(f"      For MNIST, this is typically visible but not too obvious.")

attack = FastGradientMethod(estimator=classifier, eps=eps)

print(f"\nGenerating adversarial examples for {len(x_test)} samples...")
x_test_np = x_test.numpy()
x_test_adv = attack.generate(x=x_test_np)

# Calculate perturbation statistics
perturbations = x_test_adv - x_test_np
perturbation_magnitude = perturbations.reshape(len(perturbations), -1)
perturbation_l2 = (perturbation_magnitude ** 2).sum(axis=1) ** 0.5
perturbation_linf = np.abs(perturbations).max(axis=(1, 2, 3))

print(f"\nPerturbation statistics:")
print(f"  Mean L2 norm: {perturbation_l2.mean():.6f}")
print(f"  Max L2 norm: {perturbation_l2.max():.6f}")
print(f"  Mean L_inf norm: {perturbation_linf.mean():.6f}")
print(f"  Max L_inf norm: {perturbation_linf.max():.6f}")
print(f"  Mean absolute perturbation: {np.abs(perturbations).mean():.6f}")
print(f"  Max absolute perturbation: {np.abs(perturbations).max():.6f}")

# ============================================================================
# STEP 5: VISUALIZATION AND RESULTS
# ============================================================================
print("\n" + "="*70)
print("STEP 5: Evaluating attack results")
print("="*70)

print("\nEvaluating model on adversarial examples...")
preds_adv = classifier.predict(x_test_adv)
pred_classes_adv = preds_adv.argmax(axis=1)
adv_acc = (pred_classes_adv == y_test).mean()
print(f"\nAdversarial accuracy: {adv_acc:.4f} ({adv_acc*100:.2f}%)")

# Compare before and after
print("\n" + "-"*70)
print("BEFORE vs AFTER COMPARISON")
print("-"*70)
print(f"Original accuracy:  {original_acc:.4f} ({original_acc*100:.2f}%)")
print(f"Adversarial accuracy: {adv_acc:.4f} ({adv_acc*100:.2f}%)")
print(f"Accuracy drop: {original_acc - adv_acc:.4f} ({(original_acc - adv_acc)*100:.2f} percentage points)")

# Identify successful attacks (where prediction changed from correct to incorrect)
correct_before = (original_pred_classes == y_test)
correct_after = (pred_classes_adv == y_test)
successful_attacks = correct_before & ~correct_after
failed_attacks = ~correct_before & correct_after
remained_correct = correct_before & correct_after
remained_incorrect = ~correct_before & ~correct_after

# Verify consistency: all samples should be accounted for
total_accounted = successful_attacks.sum() + failed_attacks.sum() + remained_correct.sum() + remained_incorrect.sum()
assert total_accounted == len(y_test), f"Logic error: {total_accounted} != {len(y_test)}"

print(f"\nAttack success analysis:")
print(f"  Samples correctly classified before attack: {correct_before.sum()}/{len(y_test)}")
print(f"  Samples correctly classified after attack: {correct_after.sum()}/{len(y_test)}")
print(f"  Successful attacks (correct -> incorrect): {successful_attacks.sum()}")
print(f"  Samples where attack failed (incorrect -> correct): {failed_attacks.sum()}")
print(f"  Samples that remained correct: {remained_correct.sum()}")
print(f"  Samples that remained incorrect: {remained_incorrect.sum()}")

# Verify the math relationships
print(f"\nVerification (should match):")
print(f"  Correct before = Successful + Remained correct: {successful_attacks.sum() + remained_correct.sum()} == {correct_before.sum()}")
print(f"  Correct after = Failed + Remained correct: {failed_attacks.sum() + remained_correct.sum()} == {correct_after.sum()}")
print(f"  Total samples = All categories: {total_accounted} == {len(y_test)}")

# Show some examples
print("\n" + "-"*70)
print("EXAMPLE RESULTS (first 15 samples)")
print("-"*70)
print(f"{'Sample':<8} {'Original':<12} {'Adversarial':<12} {'Actual':<10} {'Outcome':<20}")
print("-"*70)
for i in range(min(15, len(y_test))):
    orig_status = "CORRECT" if original_pred_classes[i] == y_test[i] else "WRONG"
    adv_status = "CORRECT" if pred_classes_adv[i] == y_test[i] else "WRONG"
    # Determine outcome type
    if successful_attacks[i]:
        outcome = "SUCCESS"
    elif failed_attacks[i]:
        outcome = "FAILED (helped)"
    elif remained_correct[i]:
        outcome = "NO CHANGE (correct)"
    else:  # remained_incorrect
        outcome = "NO CHANGE (wrong)"
    print(f"{i:<8} {original_pred_classes[i]:<3} ({orig_status:<8}) {pred_classes_adv[i]:<3} ({adv_status:<8}) {y_test[i]:<10} {outcome:<20}")

# Confidence analysis
print("\n" + "-"*70)
print("CONFIDENCE ANALYSIS")
print("-"*70)
original_confidences = original_preds.max(axis=1)
adv_confidences = preds_adv.max(axis=1)
print(f"Original mean confidence: {original_confidences.mean():.4f}")
print(f"Adversarial mean confidence: {adv_confidences.mean():.4f}")
print(f"Confidence drop: {original_confidences.mean() - adv_confidences.mean():.4f}")

# Show confidence for successful attacks
if successful_attacks.sum() > 0:
    print(f"\nConfidence for successfully attacked samples:")
    print(f"  Original mean confidence: {original_confidences[successful_attacks].mean():.4f}")
    print(f"  Adversarial mean confidence: {adv_confidences[successful_attacks].mean():.4f}")
    print(f"  Mean confidence drop: {(original_confidences[successful_attacks] - adv_confidences[successful_attacks]).mean():.4f}")
    
    # Show which samples were successfully attacked
    attacked_indices = np.where(successful_attacks)[0]
    print(f"\n  Successfully attacked sample indices (first 10): {attacked_indices[:10].tolist()}")
    if len(attacked_indices) > 10:
        print(f"  ... and {len(attacked_indices) - 10} more")
    
    # Show prediction changes for successful attacks
    print(f"\n  Prediction changes for successful attacks:")
    print(f"    Original -> Adversarial (Actual label)")
    for idx in attacked_indices[:5]:  # Show first 5 examples
        print(f"    {original_pred_classes[idx]} -> {pred_classes_adv[idx]} ({y_test[idx]})")
else:
    print(f"\nNo successful attacks found (attack did not flip any correct predictions)")

print("\n" + "="*70)
print("Analysis complete")
print("="*70)

