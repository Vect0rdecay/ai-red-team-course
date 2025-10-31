"""
Week 3 - Exercise 1 (Simplified): ART Evasion Attacks on MNIST

Objective: Attack the trained MNIST model using Adversarial Robustness Toolbox (ART).

This demonstrates evasion attacks - making the model misclassify inputs.
FGSM and PGD attacks are performed using the ART library.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent

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

# Load test data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False)

# Get a small sample for attacks
images, labels = next(iter(test_loader))
images_np = images.numpy()
labels_np = labels.numpy()

# Wrap model for ART
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

classifier = PyTorchClassifier(
    model=model,
    loss=criterion,
    optimizer=optimizer,
    input_shape=(1, 28, 28),
    nb_classes=10,
)

# Evaluate baseline accuracy
predictions = classifier.predict(images_np)
baseline_acc = np.sum(np.argmax(predictions, axis=1) == labels_np) / len(labels_np)
print(f"\nBaseline accuracy: {baseline_acc*100:.2f}%")

# Attack 1: Fast Gradient Method (FGM)
print("\nPerforming FGM attack...")
attack_fgm = FastGradientMethod(estimator=classifier, eps=0.3)
adversarial_fgm = attack_fgm.generate(x=images_np)

predictions_fgm = classifier.predict(adversarial_fgm)
fgm_acc = np.sum(np.argmax(predictions_fgm, axis=1) == labels_np) / len(labels_np)
print(f"Accuracy after FGM attack: {fgm_acc*100:.2f}%")
print(f"Attack success rate: {(1-fgm_acc)*100:.2f}%")

# Attack 2: Projected Gradient Descent (PGD)
print("\nPerforming PGD attack...")
attack_pgd = ProjectedGradientDescent(estimator=classifier, eps=0.3, eps_step=0.01, max_iter=40)
adversarial_pgd = attack_pgd.generate(x=images_np)

predictions_pgd = classifier.predict(adversarial_pgd)
pgd_acc = np.sum(np.argmax(predictions_pgd, axis=1) == labels_np) / len(labels_np)
print(f"Accuracy after PGD attack: {pgd_acc*100:.2f}%")
print(f"Attack success rate: {(1-pgd_acc)*100:.2f}%")

# Summary
print("\n" + "="*50)
print("Attack Summary:")
print(f"  Baseline accuracy: {baseline_acc*100:.2f}%")
print(f"  After FGM attack: {fgm_acc*100:.2f}%")
print(f"  After PGD attack: {pgd_acc*100:.2f}%")
print("="*50)

