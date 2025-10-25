# Week 2 Exercises

## Overview

Week 2 focuses on **Core AI Adversarial Concepts** with emphasis on membership inference attacks and professional vulnerability reporting. All exercises build toward conducting your first real AI security assessment.

## Exercise Files

### Exercise 1: Membership Inference Attack
**Files**: `exercise_1_membership_inference.py` + `.ipynb`

**Objective**: Execute membership inference attack to detect training data leakage

**Key Concepts**:
- Feature extraction from model predictions
- Training attack models
- Membership detection from confidence scores
- Statistical analysis for attack success

**Expected Results**:
- Attack success rate >60% (random = 50%)
- Confusion matrix visualization
- Understanding of model privacy vulnerabilities

**Red Team Relevance**: This is how you test if a production model leaks training data (HIPAA/GDPR compliance issue)

---

### Exercise 2: Shadow Models
**Files**: `exercise_2_shadow_models.py` + `.ipynb`

**Objective**: Train shadow models to enable sophisticated membership inference attacks

**Key Concepts**:
- Creating shadow models that mimic target behavior
- Training models without access to target training data
- Using shadow models for attack development
- Transferability of attacks from shadow to target

**Expected Results**:
- Trained shadow model with >95% accuracy
- Predictions extracted for attack training
- Understanding of shadow model methodology

**Red Team Relevance**: Shadow models enable attack development without target access (like creating a test environment)

---

### Exercise 3: Vulnerability Reporting
**Files**: `exercise_3_vulnerability_reporting.py` + `.ipynb`

**Objective**: Generate professional AI security vulnerability reports

**Key Concepts**:
- Risk calculation and scoring
- Executive summary generation
- Professional report formatting
- Remediation recommendations
- Compliance impact analysis

**Expected Results**:
- Risk assessment visualizations
- Professional vulnerability report
- Executive summary for stakeholders
- Technical details for engineering teams

**Red Team Relevance**: This is your billable deliverable to clients

---

## Getting Started

### Prerequisites
- Completed Week 1 exercises (need trained MNIST model)
- Python 3.8+
- PyTorch, matplotlib, scikit-learn, seaborn

### Setup
```bash
cd week-2/exercises
pip install torch torchvision matplotlib numpy scikit-learn seaborn
```

### Running Exercises

**Option 1: Python Scripts**
```bash
python exercise_1_membership_inference.py
python exercise_2_shadow_models.py
python exercise_3_vulnerability_reporting.py
```

**Option 2: Jupyter Notebooks**
```bash
jupyter notebook
# Open exercise_1_membership_inference.ipynb
```

---

## Exercise Flow

### Day 1-2: Membership Inference (Exercise 1)
1. Load Week 1's trained model
2. Extract prediction features
3. Train membership inference attack model
4. Evaluate attack success rate
5. Generate confusion matrix

**Deliverable**: Attack achieving >60% success rate

---

### Day 3: Shadow Models (Exercise 2)
1. Train shadow model on similar data
2. Evaluate shadow model performance
3. Extract predictions for attack training
4. Visualize shadow model behavior

**Deliverable**: Trained shadow model + predictions

---

### Day 4-5: Vulnerability Reporting (Exercise 3)
1. Calculate risk metrics from attack results
2. Generate executive summary
3. Create risk visualizations
4. Generate full vulnerability report

**Deliverable**: Professional vulnerability report

---

## Expected Outcomes

After completing all exercises, you will be able to:

1. **Execute membership inference attacks** on production ML models
2. **Train shadow models** for attack development
3. **Calculate risk scores** for AI vulnerabilities
4. **Generate professional reports** documenting findings
5. **Explain compliance impact** of AI security vulnerabilities

---

## Troubleshooting

**Membership Inference Accuracy Too Low**:
- Check feature extraction implementation
- Ensure attack model training is complete
- Verify data preprocessing matches Week 1

**Shadow Model Not Training**:
- Check TODO implementations
- Verify data loading
- Ensure optimizer is configured

**Risk Calculation Errors**:
- Verify attack success rate is between 50-100%
- Check severity thresholds
- Ensure all required parameters are defined

---

## Integration with Week 1

**Week 1 Dependency**: Need trained MNIST model from Exercise 1

**Model Path**: `../models/mnist_cnn.pt`

**If model doesn't exist**:
```bash
cd week-1/exercises
python exercise_1_mnist_classifier.py
# This will train and save the model
```

---

## Red Team Career Connection

**Skills You're Building**:
- AI/ML vulnerability assessment
- Membership inference testing
- Risk assessment and scoring
- Professional report writing
- Compliance impact analysis

**How This Prepares You for AI Red Team Roles**:
- Most engagements involve membership inference testing
- Shadow models are standard attack methodology
- Vulnerability reporting is your primary deliverable
- Understanding risk helps prioritize remediation

**Portfolio Value**:
- Working membership inference attack implementation
- Shadow model training capability
- Professional vulnerability report template
- Risk assessment methodology

---

## Next Steps

After completing Week 2:
1. Review your vulnerability report
2. Update your portfolio
3. Document lessons learned
4. Prepare for Week 3: Evasion attacks (FGSM, PGD)

**Week 3 Preview**: You'll craft adversarial samples that fool your Week 1 model with >90% evasion rate.
