# Week 1 Exercises

## Available Formats

All exercises are available in **two formats** for maximum flexibility:

### 1. Python Scripts (.py)
- **Best for**: Automated execution, batch processing, production environments
- **Use when**: You want to run the entire exercise in one go
- **Example**: `python exercise_1_mnist_classifier.py`

### 2. Jupyter Notebooks (.ipynb)
- **Best for**: Interactive learning, experimentation, visualization
- **Use when**: You want to run cells individually, modify and test iteratively
- **Example**: `jupyter notebook exercise_1_mnist_classifier.ipynb`

## Exercises Overview

### Exercise 1: MNIST Classifier
**Files:**
- `exercise_1_mnist_classifier.py` (Python script)
- `exercise_1_mnist_classifier.ipynb` (Jupyter notebook)

**Objective**: Build and train a CNN for handwritten digit classification

**What you'll learn:**
- Loading and preprocessing image data
- Building CNN architectures with PyTorch
- Training neural networks
- Model evaluation and saving

**Red Team Context**: This model becomes your attack target in Week 3

**Estimated Time**: 2-3 hours

---

### Exercise 2: Model Queries
**Files:**
- `exercise_2_model_queries.py` (Python script)
- `exercise_2_model_queries.ipynb` (Jupyter notebook)

**Objective**: Learn to interact with ML models programmatically (reconnaissance)

**What you'll learn:**
- Loading saved models
- Making predictions and extracting confidence scores
- Analyzing model behavior
- Creating confusion matrices

**Red Team Context**: Understanding normal model behavior is critical before attacking

**Estimated Time**: 1-2 hours

---

### Exercise 3: Text Generator
**Files:**
- `exercise_3_text_generator.py` (Python script)
- `exercise_3_text_generator.ipynb` (Jupyter notebook)

**Objective**: Build a simple RNN for character-level text generation

**What you'll learn:**
- Working with text data and character-level modeling
- Building RNNs/LSTMs with PyTorch
- Understanding generative models
- Text generation with temperature sampling

**Red Team Context**: Foundation for understanding LLM mechanics and attacks

**Estimated Time**: 2-3 hours

---

### Exercise 4: Model Deployment
**Files:**
- `exercise_4_fastapi_deployment.md` (Guide only)

**Objective**: Deploy your MNIST model as a web API using FastAPI

**What you'll learn:**
- Creating REST APIs for ML models
- Model serving with FastAPI
- Production deployment considerations
- Testing API endpoints

**Red Team Context**: Most real-world ML attacks target deployed models, not training code

**Estimated Time**: 1 hour

## Getting Started

### Prerequisites
- Python 3.8+
- PyTorch
- Jupyter Notebook (for notebook format)

### Setup
```bash
# Install dependencies
pip install torch torchvision matplotlib numpy seaborn scikit-learn fastapi uvicorn

# For Jupyter notebooks
pip install jupyter notebook
```

### Running Exercises

**Option 1: Python Scripts**
```bash
cd week-1/exercises
python exercise_1_mnist_classifier.py
```

**Option 2: Jupyter Notebooks**
```bash
jupyter notebook
# Then open exercise_1_mnist_classifier.ipynb
```

## Important Notes

1. **Exercise Completion**: All exercises are ~85% complete with TODOs to fill in
2. **Active Learning**: Don't just read - complete the TODOs to build understanding
3. **See EXERCISE_GUIDE.md**: Detailed guide on how to approach the exercises
4. **Error Handling**: Scripts include validation to catch incomplete implementations
5. **Sequential Order**: Complete exercises in order (they build on each other)

## Helpful Resources

- **PyTorch Documentation**: https://pytorch.org/docs/
- **PyTorch Tutorials**: https://pytorch.org/tutorials/
- **Exercise Guide**: See `EXERCISE_GUIDE.md` in this directory

## Troubleshooting

**Import errors**: Make sure all dependencies are installed
**CUDA errors**: Scripts work on CPU; set `device = torch.device('cpu')` if needed
**TODO errors**: Scripts will detect incomplete implementations and provide hints

## Next Steps

After completing all Week 1 exercises:
1. Review your implementations
2. Experiment with hyperparameters
3. Document what you learned
4. Prepare for Week 2: Adversarial ML Foundations
