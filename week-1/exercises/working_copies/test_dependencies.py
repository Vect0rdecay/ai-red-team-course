#!/usr/bin/env python3
"""
Test script to verify all required dependencies are available.
Run this before starting the exercises.
"""

def test_imports():
    """Test if all required packages can be imported."""
    print("Testing dependencies...")
    
    try:
        import torch
        print(f"✓ PyTorch: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
    except ImportError as e:
        print(f"✗ PyTorch: {e}")
        return False
    
    try:
        import torchvision
        print(f"✓ TorchVision: {torchvision.__version__}")
    except ImportError as e:
        print(f"✗ TorchVision: {e}")
        return False
    
    try:
        import matplotlib
        print(f"✓ Matplotlib: {matplotlib.__version__}")
    except ImportError as e:
        print(f"✗ Matplotlib: {e}")
        return False
    
    try:
        import numpy
        print(f"✓ NumPy: {numpy.__version__}")
    except ImportError as e:
        print(f"✗ NumPy: {e}")
        return False
    
    try:
        import seaborn
        print(f"✓ Seaborn: {seaborn.__version__}")
    except ImportError as e:
        print(f"✗ Seaborn: {e}")
        return False
    
    try:
        import sklearn
        print(f"✓ Scikit-learn: {sklearn.__version__}")
    except ImportError as e:
        print(f"✗ Scikit-learn: {e}")
        return False
    
    try:
        import fastapi
        print(f"✓ FastAPI: {fastapi.__version__}")
    except ImportError as e:
        print(f"✗ FastAPI: {e}")
        return False
    
    try:
        import uvicorn
        print(f"✓ Uvicorn: {uvicorn.__version__}")
    except ImportError as e:
        print(f"✗ Uvicorn: {e}")
        return False
    
    try:
        from PIL import Image
        print(f"✓ Pillow: Available")
    except ImportError as e:
        print(f"✗ Pillow: {e}")
        return False
    
    try:
        import requests
        print(f"✓ Requests: {requests.__version__}")
    except ImportError as e:
        print(f"✗ Requests: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("=" * 50)
    print("AI Red Team Course - Week 1 Dependencies Check")
    print("=" * 50)
    
    success = test_imports()
    
    print("\n" + "=" * 50)
    if success:
        print("✓ All dependencies are available!")
        print("You can proceed with the exercises.")
    else:
        print("✗ Some dependencies are missing.")
        print("Please install missing packages:")
        print("pip install -r ../requirements.txt")
    print("=" * 50)
