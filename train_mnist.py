# Simple MNIST training script (PyTorch)
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = "week-01/models"
os.makedirs(SAVE_DIR, exist_ok=True)

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*5*5, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        return self.net(x)

def train(epochs=3, batch_size=64, lr=1e-3):
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = datasets.MNIST(root='data', train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    model = SimpleCNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        total, correct, loss_acc = 0, 0, 0.0
        for xb, yb in trainloader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            opt.step()
            loss_acc += loss.item()
            preds = out.argmax(dim=1)
            correct += (preds==yb).sum().item()
            total += yb.size(0)
        print(f"Epoch {epoch+1}/{epochs} loss={loss_acc/len(trainloader):.4f} acc={correct/total:.4f}")
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, "mnist_cnn.pt"))
    print("Saved model to", os.path.join(SAVE_DIR, "mnist_cnn.pt"))

if __name__ == '__main__':
    train()
