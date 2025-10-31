"""
Week 1 - Exercise 4 (Simplified): Simple Model Deployment

Objective: Deploy model as a basic API endpoint.

This demonstrates how models are deployed in production - understanding this is crucial for red team testing.
"""

import torch
import torch.nn as nn
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import numpy as np
from torchvision import transforms

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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)
model_path = Path(__file__).parent.parent.parent / 'models' / 'mnist_simple.pt'
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

print("Model loaded successfully")

# Create FastAPI app
app = FastAPI(title="MNIST Classifier API", version="1.0")

# Transform for preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

@app.get("/")
def root():
    return {"message": "MNIST Classifier API", "endpoints": ["/predict", "/health"]}

@app.get("/health")
def health():
    return {"status": "healthy", "model": "MNIST SimpleCNN"}

@app.post("/predict")
async def predict(image_data: dict):
    """
    Predict digit from image data.
    
    Expected input: {"image": [784 float values]} - flattened 28x28 image
    Returns: {"prediction": int, "confidence": float, "all_probs": [10 floats]}
    """
    try:
        # Convert input to numpy array then tensor
        image_array = np.array(image_data["image"], dtype=np.float32)
        
        # Reshape to 28x28 if needed
        if len(image_array) == 784:
            image_array = image_array.reshape(28, 28)
        
        # Normalize to [0, 1] range (assuming input is 0-255)
        if image_array.max() > 1.0:
            image_array = image_array / 255.0
        
        # Convert to tensor and add batch/channel dimensions
        image_tensor = torch.tensor(image_array).unsqueeze(0).unsqueeze(0)
        image_tensor = image_tensor.to(device)
        
        # Apply normalization
        mean = torch.tensor([0.1307], device=device)
        std = torch.tensor([0.3081], device=device)
        image_tensor = (image_tensor - mean) / std
        
        # Make prediction
        with torch.no_grad():
            output = model(image_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)
            confidence, predicted = torch.max(probs, 1)
        
        return {
            "prediction": int(predicted.item()),
            "confidence": float(confidence.item()),
            "all_probs": [float(p) for p in probs[0].cpu().numpy()]
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    print("\nStarting FastAPI server on http://localhost:8000")
    print("API endpoints:")
    print("  GET  / - API information")
    print("  GET  /health - Health check")
    print("  POST /predict - Predict digit from image")
    print("\nTo test the API, use:")
    print('  curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d \'{"image": [array of 784 floats]}\'')
    print("\nNote: This is a simplified deployment. Production deployments need:")
    print("  - Input validation")
    print("  - Rate limiting")
    print("  - Authentication")
    print("  - Error handling")
    print("  - Monitoring")
    print("\nStarting server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)

