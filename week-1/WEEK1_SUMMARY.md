# Week 1: ML Foundations Summary - AI Red Team Transition

## Core Concepts

### Machine Learning Model
A mathematical system trained on data to make predictions or decisions. In pentesting terms: like a trained analyst - you query it to understand its decision-making process, then exploit its blind spots.

### Neural Networks / Deep Learning
Layered computational models inspired by brains. **Red Team Perspective**: Each layer creates potential attack surfaces - gradient manipulation at training, input manipulation at inference.

### Training vs Inference
- **Training**: Learning patterns from data (like teaching a human analyst)
- **Inference**: Making predictions on new data (like asking the analyst to evaluate a new case)
- **Attack Vector**: Poison training data, or manipulate inputs during inference

### Model Architecture
The structure of the model (layers, connections). **Pentester Skill**: Understanding architecture tells you where to inject perturbations for evasion attacks (Week 3).

---

## Key AI Security Terminology

### Adversarial Example
Modified input designed to fool ML model. **Analogy**: Phishing email bypassing spam filter.

### Model Querying / Inference
Sending inputs to model to get predictions. **Pentester Skill**: Reconnaissance - understand model behavior before attacking (Week 3).

### Decision Boundary
Line separating different model predictions. **Attack Target**: Cross this boundary with small perturbations.

### Gradient
Direction model would change to improve. **Attack Use**: Gradient descent finds adversarial samples.

### Baseline Performance
Normal model accuracy before attacks. **Benchmark**: Measure attack success against this.

### Model Serving
Deploying models via API (e.g., FastAPI). **Attack Surface**: Like a web API - vulnerable to input manipulation.

---

## OWASP ML Top 10 - Quick Reference

| M# | Vulnerability | Web Security Analogy | Red Team Skill |
|----|--------------|---------------------|----------------|
| M01 | Input Manipulation | SQL Injection | Craft adversarial inputs (Week 3) |
| M02 | Model Theft | API endpoint scraping | Extract architecture via queries (Week 2) |
| M03 | Model Poisoning | Supply chain attack | Poison training data (Week 4) |
| M04 | Evasion Attacks | IDS bypassing | Generate adversarial samples (Week 3) |
| M05 | Data Poisoning | Backdoor in vendor code | Inject triggers in training (Week 4) |
| M06 | Membership Inference | Data breach detection | Infer training data membership (Week 2-3) |
| M07 | Model Tampering | File upload vulnerabilities | Modify deployed models (Week 4) |
| M08 | Model Extraction | IP theft | Clone model via queries (Week 2) |
| M09 | Model Inversion | Data reconstruction | Recover training examples (Week 2) |
| M10 | Prompt Injection | XSS | Inject malicious prompts (Week 5) |

---

## AI Pentest Lifecycle (Week 1 Mapping)

Traditional Pentest → AI Security Equivalent

1. **Reconnaissance** → Model fingerprinting, querying
2. **Scanning** → Architecture extraction, gradient analysis
3. **Enumeration** → Input/output mapping, decision boundary discovery
4. **Vulnerability Analysis** → Adversarial testing, robustness checks
5. **Exploitation** → Craft adversarial samples, trigger backdoors
6. **Post-Exploitation** → Model extraction, membership inference
7. **Reporting** → Document vulnerabilities, recommend defenses

---

## Week 1 Deliverables - Why They Matter

### MNIST Classifier (your attack target)
- Train CNN to recognize handwritten digits
- Establishes baseline: 98% normal accuracy
- **Week 3**: Attack this to achieve <5% adversarial accuracy

### Text Generator (LLM foundation)
- Simple RNN generating text
- Understands generation mechanics
- **Week 5**: Apply this knowledge to jailbreak real LLMs

### OWASP Mapping Document
- Connects familiar web vulnerabilities to AI security
- Translation guide for pentesters
- **All Weeks**: Use this for scoping and reporting

### Threat Model Diagram
- ML pipeline components identified
- Attack surfaces mapped
- **All Weeks**: Guides your testing methodology

---

## Quick Red Team Methodology

### Before Attacking an ML System:
1. **Query the model** - Understand normal behavior
2. **Map inputs/outputs** - Identify attack vectors
3. **Establish baseline** - Measure normal performance
4. **Identify architecture** - Know what you're attacking
5. **Document behavior** - For exploit development

### During Attack:
1. **Choose attack vector** (OWASP ML Top 10)
2. **Craft adversarial inputs** (Week 3-5)
3. **Measure attack success** (compare to baseline)
4. **Document findings** (vulnerability reports)

### After Attack:
1. **Recommend defenses** (Week 7)
2. **Test mitigations** (measure effectiveness)
3. **Write professional report** (Week 7)

---

## Vocabulary Cheat Sheet

| Term | Definition | Why Pentesters Care |
|------|------------|---------------------|
| **Neural Network** | Multi-layer computational model | Each layer = attack surface |
| **Training Data** | Examples model learns from | Poison this = supply chain attack |
| **Overfitting** | Model memorizes training data | Enables membership inference |
| **Gradient** | How to improve predictions | Shows direction for adversarial attacks |
| **Loss Function** | Measures prediction error | Exploited in evasion attacks |
| **Epoch** | One pass through training data | Poison early epochs = harder to detect |
| **Batch** | Group of training examples | Poison entire batch = widespread impact |
| **Model Weights** | Learned parameters | Extract these = model theft |
| **Activation Function** | Adds non-linearity | Creates exploitable decision boundaries |
| **Dropout** | Randomly disable neurons | Can be bypassed in inference attacks |

---

## Week 1 → Future Weeks

**Week 1 builds targets, Week 3 attacks them**
- Your MNIST model becomes your evasion target
- Baseline accuracy: 98% → Adversarial accuracy: <5%

**Week 1 maps vulnerabilities, Week 5 exploits them**
- OWASP M10 (Prompt Injection) → Week 5 jailbreaks
- Text generation knowledge → LLM attack crafting

**Week 1 documents methodology, Week 8 completes it**
- Threat model template → Complete engagement workflow
- Query skills → Systematic model assessment

---

## Bottom Line

**Week 1 teaches you to think like an AI red teamer:**
- ML models are systems to be tested like any other software
- Attack surfaces exist at training, deployment, and inference stages
- Methodology transfers from web pentesting with framework mapping
- Every exercise builds toward your first AI security engagement

**By Week 1's end:** You can query models, map vulnerabilities, and understand where attacks will be effective. This is reconnaissance - the foundation for all subsequent exploitation.
