# Week 1 Exercises

## Simplified Exercises (Recommended)

These simplified exercises contain minimal working code to quickly understand ML foundations and model behavior.

### Exercise 1: Train MNIST Model
**File:** `exercise_1_simple_mnist_train.py`

**Objective**: Train a simple CNN on MNIST dataset

**What you'll learn:**
- Minimal code for loading and preprocessing MNIST
- Simple CNN architecture
- Basic training loop
- Model saving for later exercises

**Time**: ~10 minutes

**Run:** `python exercise_1_simple_mnist_train.py`

---

### Exercise 2: Query Model and Analyze Predictions
**File:** `exercise_2_simple_model_queries.py`

**Objective**: Learn to query ML models and understand baseline behavior

**What you'll learn:**
- Loading saved models
- Making predictions and extracting confidence scores
- Analyzing correct vs incorrect predictions
- Understanding baseline model behavior

**Prerequisites**: Run Exercise 1 first to generate the model

**Time**: ~5 minutes

**Run:** `python exercise_2_simple_model_queries.py`

---

### Exercise 3: Model Sensitivity Analysis
**File:** `exercise_3_model_sensitivity.py`

**Objective**: Understand how models respond to small input changes

**What you'll learn:**
- Testing model behavior with modified inputs
- Observing how confidence scores change
- Understanding model sensitivity
- Foundation for understanding adversarial attacks

**Prerequisites**: Run Exercise 1 first to generate the model

**Time**: ~5 minutes

**Run:** `python exercise_3_model_sensitivity.py`

---

### Exercise 4: Simple Model Deployment (Optional)
**File:** `exercise_4_simple_deployment.py`

**Objective**: Deploy model as a basic API endpoint

**What you'll learn:**
- Creating FastAPI endpoints for ML models
- Basic model serving
- Understanding production deployment basics

**Prerequisites**: Run Exercise 1 first to generate the model

**Time**: ~10 minutes

**Run:** `python exercise_4_simple_deployment.py`

---

## Getting Started (Simplified Exercises)

### Prerequisites
- Python 3.8+
- PyTorch
- FastAPI (for Exercise 4): `pip install fastapi uvicorn`

### Quick Setup
```bash
# Install all dependencies
pip install torch torchvision numpy fastapi uvicorn
```

### Running Exercises
```bash
cd week-1/exercises

# Step 1: Train the model
python exercise_1_simple_mnist_train.py

# Step 2: Query and analyze the model
python exercise_2_simple_model_queries.py

# Step 3: Test model sensitivity
python exercise_3_model_sensitivity.py

# Step 4: (Optional) Deploy model as API
python exercise_4_simple_deployment.py
```

---

## Original Exercises (Still Available)

The original detailed exercises remain in this directory for reference:
- `exercise_1_mnist_classifier.py` - Full CNN training with TODOs
- `exercise_2_model_queries.py` - Model analysis and querying
- `exercise_3_text_generator.py` - Text generation with RNN
- `exercise_4_fastapi_deployment.md` - Model deployment guide

## Exercises Overview (Original)

### Exercise 1: MNIST Classifier
**Files:**
- `exercise_1_mnist_classifier.py` (Python script)

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

### Setup (Original Exercises)
```bash
# Install dependencies
pip install torch torchvision matplotlib numpy seaborn scikit-learn fastapi uvicorn
```

### Running Exercises (Original)

```bash
cd week-1/exercises
python exercise_1_mnist_classifier.py
```

## Important Notes (Original Exercises)

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
