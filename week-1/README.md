# Week 1: ML Foundations for AI Red Teamers

## Overview

This week establishes the foundation for AI red team work by teaching you to build, query, and analyze ML models from a security perspective. Every concept learned here will be directly applied to attacking ML systems in subsequent weeks.

**Estimated Time**: 10-12 hours

---

## Learning Objectives

By the end of Week 1, you will be able to:

1. Build and query ML models to establish baseline behavior for adversarial testing
2. Map OWASP ML Top 10 vulnerabilities to familiar web/cloud attack patterns
3. Document threat models specific to ML systems using familiar pentesting methodology
4. Create attack targets for subsequent weeks' exploitation activities
5. Translate traditional pentest knowledge to AI security assessment

---

## Required Reading

### Primary Textbooks

**Géron, Aurélien.** *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (3rd ed.)

- **Chapter 1 - The Machine Learning Landscape** (pp. 1-42)
  - Focus: Types of ML, supervised vs unsupervised learning
  - Red Team Application: Understand what you'll be attacking (classification models)
  
- **Chapter 10 - Introduction to Artificial Neural Networks with Keras** (pp. 309-350)
  - Focus: Building neural networks, training process
  - Red Team Application: Learn model architecture to identify attack surfaces

- **Chapter 13 - Loading and Preprocessing Data with TensorFlow** (pp. 473-520)
  - Focus: Data pipelines, preprocessing
  - Red Team Application: Understand data flow = understand where to inject adversarial inputs

**Sotiropoulos, John.** *Adversarial AI Attacks, Mitigations and Defense Strategies*

- **Chapter 1 - Introduction to AI Security and Attack Taxonomy** (pp. 1-35)
  - Focus: Attack categories, threat landscape
  - Red Team Application: Framework for understanding all subsequent attacks

### Supplemental Reading

- OWASP ML Security Top 10: https://owasp.org/www-project-machine-learning-security-top-10/
- MITRE ATLAS Framework: https://atlas.mitre.org/

---

## Coding Exercises

### Exercise 1: Build and Train MNIST Classifier (3-4 hours)

**Objective**: Create your first attack target - a CNN that classifies handwritten digits.

**Tasks**:
1. Load MNIST dataset using PyTorch
2. Build CNN architecture (2 conv layers, 2 fully connected layers)
3. Train model for 5 epochs
4. Achieve >95% accuracy on test set
5. Save model to `week-1/models/mnist_cnn.pt`
6. Visualize loss curve and accuracy over epochs

**Deliverable**: `week-1/mnist_classifier.ipynb`

**Red Team Context**: This model becomes your evasion attack target in Week 3. The baseline accuracy you achieve here (e.g., 98%) is what you'll reduce to <5% using adversarial samples.

**Reference**: Géron Ch. 10 - Use PyTorch instead of Keras for consistency with course

### Exercise 2: Model Inference and Querying (1-2 hours)

**Objective**: Learn to interact with ML models programmatically (reconnaissance skill).

**Tasks**:
1. Load your trained MNIST model
2. Query model with 10 test images
3. Log predictions and confidence scores
4. Visualize 5 correctly classified images and 5 misclassified images
5. Identify decision boundary patterns

**Deliverable**: `week-1/model_queries.ipynb`

**Red Team Context**: Before attacking, you must understand normal model behavior. This is the equivalent of service enumeration in traditional pentesting.

**Reference**: See Géron Ch. 13 for data loading patterns

### Exercise 3: Simple Text Generator (2-3 hours)

**Objective**: Understand generative models (prepares for LLM attacks in Week 5).

**Tasks**:
1. Build simple RNN for text generation
2. Train on a small corpus (e.g., Shakespeare sonnets)
3. Generate 10 sample outputs
4. Analyze how model generates text (character-by-character prediction)

**Deliverable**: `week-1/text_generator.ipynb`

**Red Team Context**: LLMs work similarly - understanding this foundation is crucial for prompt injection and jailbreak attacks in Week 5.

**Reference**: Géron Ch. 15 (RNN concepts), though you'll implement from scratch

### Exercise 4: Model Deployment (FastAPI) (1 hour)

**Objective**: Understand how ML models are deployed in production.

**Tasks**:
1. Use provided `serve_model.py` to deploy your MNIST classifier
2. Start FastAPI server on localhost:8000
3. Send POST requests with image data to `/predict` endpoint
4. Document API inputs, outputs, and error handling

**Deliverable**: `week-1/model_deployment_notes.md`

**Red Team Context**: Production ML models are deployed via APIs. Understanding this attack surface is critical - input validation bypasses, rate limiting evasion, etc.

**Reference**: See course `serve_model.py` template

---

## Written Exercises

### Exercise 5: OWASP ML Top 10 Mapping (2-3 hours)

**Objective**: Connect AI security vulnerabilities to familiar web security patterns.

**Tasks**:
1. Create table mapping each OWASP ML Top 10 vulnerability to web/cloud attack patterns
2. For each vulnerability:
   - Web security analogy
   - Attack scenario
   - Business impact
   - How to test for it (pentest workflow)
3. Identify which course weeks cover each vulnerability

**Deliverable**: `week-1/owasp_ml_mapping.md`

**Example**:
```markdown
## M01: Input Manipulation
- **Web Analogy**: SQL Injection
- **AI Attack**: Craft adversarial inputs to fool model
- **Testing Method**: Generate adversarial samples, measure model confidence
- **Covered In**: Week 3 (Evasion Attacks)
```

**Reference**: OWASP ML Top 10 documentation, Sotiropoulos Ch. 1

### Exercise 6: ML Threat Model (2 hours)

**Objective**: Identify attack surfaces in ML systems.

**Tasks**:
1. Diagram complete ML pipeline:
   - Data collection
   - Preprocessing
   - Training
   - Model storage
   - Deployment
   - Inference
2. For each stage, identify:
   - Attack vectors
   - Threat actors
   - Potential vulnerabilities
   - Detection methods
3. Create comparison with traditional application threat model

**Deliverable**: `week-1/ml_threat_model.md`

**Reference**: Sotiropoulos Ch. 1 (threat taxonomy), your pentest methodology experience

### Exercise 7: AI Pentest Methodology (1-2 hours)

**Objective**: Map traditional pentest methodology to AI security.

**Tasks**:
1. Create side-by-side comparison:
   - Traditional Pentest Phase → AI Security Equivalent
   - Tools Used → AI Security Tools
   - Deliverables → AI Security Deliverables
2. Identify which weeks cover each phase

**Deliverable**: `week-1/ai_pentest_methodology.md`

**Example Structure**:
| Traditional Pentest | AI Security Equiv. | Week | Tools |
|-------------------|-------------------|------|-------|
| Reconnaissance | Model fingerprinting | 2 | API queries, architecture inference |
| Vulnerability Scanning | Adversarial testing | 3 | Foolbox, ART |
| Exploitation | Craft adversarial samples | 3-5 | torchattacks, custom exploits |

---

## Deliverables Checklist

- [ ] `mnist_classifier.ipynb` - Trained CNN model (>95% accuracy)
- [ ] `model_queries.ipynb` - Query examples and analysis
- [ ] `text_generator.ipynb` - Working RNN text generator
- [ ] `model_deployment_notes.md` - FastAPI deployment documentation
- [ ] `owasp_ml_mapping.md` - Vulnerability mapping document
- [ ] `ml_threat_model.md` - Complete threat model
- [ ] `ai_pentest_methodology.md` - Methodology comparison
- [ ] Updated `lab-notebook.md` with all commands and findings

---

## Success Criteria

You've successfully completed Week 1 when:

1. ✅ Your MNIST classifier achieves >95% test accuracy
2. ✅ You can query your model and interpret predictions
3. ✅ You've created a working text generator
4. ✅ You've mapped all 10 OWASP ML vulnerabilities with web security analogies
5. ✅ You've created a comprehensive ML threat model
6. ✅ You can explain how pentest methodology maps to AI security

---

## Assessment

**Self-Check Questions**:
1. What makes a neural network "deep"? How does depth relate to attack surface?
2. Explain the difference between training and inference from an attack perspective.
3. What is an adversarial example? How is it similar to a malicious file bypassing antivirus?
4. Which OWASP ML vulnerabilities could be exploited during training vs deployment?
5. How would you scope an AI security assessment using traditional pentest methodology?

**Code Review**:
- Is your code well-commented explaining the security context?
- Are your notebooks reproducible (include dataset loading, model definition)?
- Do your visualizations clearly demonstrate model behavior?

---

## Week 1 → Week 2 Preparation

In Week 2, you will:
- Apply membership inference attacks to your Week 1 MNIST model
- Extract training data information through model queries
- Implement attack techniques from research papers

Ensure your Week 1 models are:
- Saved and accessible (`.pt` files)
- Well-documented (architecture, hyperparameters)
- Baseline metrics recorded (accuracy, loss curves)

---

## Resources and References

**Code Resources**:
- Géron GitHub: https://github.com/ageron/handson-ml3 (reference implementations, don't copy)
- PyTorch Tutorials: https://pytorch.org/tutorials/
- FastAPI Documentation: https://fastapi.tiangolo.com/

**Security Frameworks**:
- OWASP ML Top 10: https://owasp.org/www-project-machine-learning-security-top-10/
- MITRE ATLAS: https://atlas.mitre.org/
- NIST AI Risk Management: https://www.nist.gov/itl/ai-risk-management-framework

**Discussion**:
- Course GitHub Discussions: [Your repo link]
- AI Security Community: Look for Discord/Slack communities

---

## Troubleshooting Common Issues

**Issue**: Model training is slow
- **Solution**: Reduce data size for initial experiments, use CPU if GPU unavailable

**Issue**: Cannot achieve >95% accuracy on MNIST
- **Solution**: Increase model capacity (add layers), train for more epochs, adjust learning rate

**Issue**: FastAPI server not responding
- **Solution**: Check port 8000 isn't in use, verify model path in `serve_model.py`

**Issue**: Need help mapping OWASP vulnerabilities
- **Solution**: Start with M01 (Input Manipulation) and M04 (Evasion) - easiest to map to SQL injection and fuzzing

---

**Next Week**: Core AI Adversarial Concepts - Learn to extract training data through membership inference attacks.
