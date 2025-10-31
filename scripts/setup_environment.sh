#!/bin/bash

# AI Red Team Course - Environment Setup Script
# This script sets up the Python environment and installs all required dependencies

set -e  # Exit on any error

echo "=================================="
echo "AI Red Team Course - Environment Setup"
echo "=================================="
echo ""

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

echo "Python version: $(python3 --version)"
echo ""

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch (CPU version by default, can be modified for CUDA)
echo ""
echo "Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install core ML libraries
echo ""
echo "Installing core ML libraries..."
pip install numpy pandas scikit-learn matplotlib seaborn

# Install AI security tools
echo ""
echo "Installing AI security tools..."
pip install adversarial-robustness-toolbox foolbox

# Install development tools
echo ""
echo "Installing development tools..."
pip install black flake8 pytest

# Install API framework (for Week 1 deployment exercise)
echo ""
echo "Installing FastAPI..."
pip install fastapi uvicorn pydantic

# Create data and models directories
echo ""
echo "Creating data directories..."
mkdir -p data models

# Create .gitkeep files
touch data/.gitkeep models/.gitkeep

echo ""
echo "=================================="
echo "Setup Complete!"
echo "=================================="
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo ""
echo "For CUDA support (if you have a compatible GPU), run:"
echo "  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
echo ""
echo "To deactivate the environment: deactivate"
