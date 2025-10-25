# Simple FastAPI server that loads a saved PyTorch MNIST model and serves predictions.
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io

app = FastAPI()

class PredictRequest(BaseModel):
    image_bytes: bytes  # base64 or raw bytes of a 28x28 grayscale image

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

MODEL_PATH = "week-01/models/mnist_cnn.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None

def load_model():
    global model
    model = SimpleCNN().to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
        print("Model loaded.")
    except Exception as e:
        print("Could not load model:", e)

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28,28)),
    transforms.ToTensor()
])

@app.on_event("startup")
def startup_event():
    load_model()

@app.post("/predict")
async def predict(req: PredictRequest):
    if model is None:
        return {"error": "model not loaded"}
    try:
        img = Image.open(io.BytesIO(req.image_bytes)).convert('L')
        x = transform(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out = model(x)
            pred = int(out.argmax(dim=1).cpu().numpy()[0])
        return {"prediction": pred}
    except Exception as e:
        return {"error": str(e)}

if __name__ == '__main__':
    uvicorn.run("serve_model:app", host="0.0.0.0", port=8000, reload=True)
