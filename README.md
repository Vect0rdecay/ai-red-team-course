# 8-Week AI Red Teaming Transition Course  
**From Web/Cloud Pentesting to AI Security — in 2 Months**

> **Goal**: Build upon existing **n+ years of web, API, and cloud pentesting** to become a qualified **AI Red Teamer** (e.g., HiddenLayer).  
> **Outcome**: Portfolio with attack notebooks, reports, tool contributions, and resume-ready evidence of **1+ year equivalent AI security experience**.

**Time Commitment**: 12–18 hrs/week  
**Core Texts**:
- *Adversarial AI Attacks, Mitigations and Defense Strategies* – John Sotiropoulos (Packt)
- *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (3rd ed.) – Aurélien Géron (O'Reilly)
- deeplearning.ai: *Machine Learning Specialization* & *Generative AI with LLMs* (free)

**Hands-On Tools**:
- [Foolbox](https://github.com/bethgelab/foolbox) – Evasion (FGSM, PGD)
- [Adversarial Robustness Toolbox (ART)](https://github.com/Trusted-AI/adversarial-robustness-toolbox) – Poisoning, inference
- [torchattacks](https://github.com/Harry24k/adversarial-attacks-pytorch) – PyTorch attack implementations
- [garak](https://github.com/leondz/garak) – LLM vulnerability scanner
- [Purple Llama](https://github.com/meta-llama/PurpleLlama) – CyberSecEval for LLMs

**Portfolio Repo**: [github.com/yourname/ai-red-team-course](https://github.com/yourname/ai-red-team-course)  

### Course Structure

Each week contains three directories:
- **`exercises/`** - Python scripts (.py files) with 85% complete implementations and TODOs
- **`notebooks/`** - Jupyter notebooks (.ipynb) for interactive learning
- **`notes/`** - Space for your personal notes and additional materials

All exercises include comprehensive educational comments explaining ML/AI concepts for beginners. Both Python scripts and Jupyter notebooks are provided for each exercise.

**Key Files**:
- `PROJECT_STRUCTURE.md` - Detailed repository organization
- `templates/vulnerability_report.md` - Professional AI security report template
- `scripts/setup_environment.sh` - Automated environment setup

---

## Week 1: ML Foundations for Security Pros
**Outcomes**: Identify ML attack surfaces; build & evaluate predictive/generative models.

**Exercises** (85% complete with TODOs):
1. **Exercise 1**: MNIST CNN Classifier (`week-1/exercises/exercise_1_mnist_classifier.py`)
   - Build and train CNN for MNIST classification
   - Implement CNN architecture layers (Conv2d, MaxPool2d, Linear, ReLU)
   - Configure training loop with loss functions and optimizers
   - Apply PyTorch fundamentals to model development
2. **Exercise 2**: Model Queries (`week-1/exercises/exercise_2_model_queries.py`)
   - Query trained models programmatically to extract predictions
   - Apply softmax to convert logits to probabilities
   - Extract confidence scores and predicted classes
   - Practice reconnaissance techniques for AI red teaming
3. **Exercise 3**: Text Generator (`week-1/exercises/exercise_3_text_generator.py`)
   - Build simple RNN/LSTM for character-level text generation
   - Implement embedding layers, LSTM processing, and output layers
   - Apply autoregressive generation with temperature sampling
   - Prepare foundations for LLM attack methods

**Required Reading**: Géron Ch. 1–2, Sotiropoulos Ch. 1
**Deliverables**: Complete TODOs in all three exercises; demonstrate knowledge of core ML concepts through implementation

---

## Week 2: Core AI Adversarial Concepts
**Outcomes**: Implement membership inference attacks; train shadow models; generate vulnerability reports.

**Exercises** (85% complete with TODOs):
1. **Exercise 1**: Membership Inference (`week-2/exercises/exercise_1_membership_inference.py`)
   - Implement membership inference attack algorithm
   - Classify samples as member or non-member of training data
   - Calculate and interpret attack success metrics (accuracy, precision, recall)
2. **Exercise 2**: Shadow Models (`week-2/exercises/exercise_2_shadow_models.py`)
   - Train shadow models to replicate target model behavior
   - Generate attack training data from shadow model outputs
   - Analyze model behavior patterns for attack development
3. **Exercise 3**: Vulnerability Reporting (`week-2/exercises/exercise_3_vulnerability_reporting.py`)
   - Calculate risk scores using attack success rates and business impact
   - Classify vulnerabilities by severity (Low/Medium/High/Critical)
   - Generate professional AI security vulnerability reports for clients

**Required Reading**: Sotiropoulos Ch. 2–3, NIST AI Risk Management Framework
**Deliverables**: Complete all exercises; generate vulnerability report template

---

## Week 3: Evasion & Inference Attacks on Predictive Models
**Outcomes**: Execute gradient-based evasion attacks (FGSM, PGD); compare attack effectiveness; use AI security tools.

**Exercises** (85% complete with TODOs):
1. **Exercise 1**: FGSM Attack (`week-3/exercises/exercise_1_fgsm_attack.py`)
   - Implement Fast Gradient Sign Method (FGSM) algorithm
   - Generate adversarial samples that successfully fool target models
   - Apply gradient computation and sign-based perturbations
2. **Exercise 2**: PGD Attack (`week-3/exercises/exercise_2_pgd_attack.py`)
   - Implement Projected Gradient Descent (PGD) with iterative refinement
   - Generate stronger adversarial samples than single-step FGSM
   - Apply perturbation bounds (epsilon) with random initialization
3. **Exercise 3**: Attack Comparison (`week-3/exercises/exercise_3_attack_comparison.py`)
   - Compare FGSM vs PGD attack effectiveness
   - Measure and contrast evasion rates and computational costs
   - Visualize and interpret attack results in reports
4. **Exercises 4-6**: Visualization, ART attacks, Final report (placeholders)

**Required Reading**: Sotiropoulos Ch. 4, Foolbox documentation, ART tutorial
**Deliverables**: Complete FGSM and PGD implementations; compare attack performance

---

## Week 4: Poisoning & Backdoor Attacks
**Outcomes**: Poison datasets, implant backdoors, test defenses.

| Activity | Details | Time | Output |
|--------|--------|------|--------|
| **Study** | Read ART poisoning tutorial + Sotiropoulos Ch. 5. Analogize to tainted S3 buckets. | 1–2 hrs | `week4/poisoning_notes.md` |
| **Code** | Use ART to poison 10% of MNIST labels. Retrain. Use torchattacks to add BadNets trigger. | 3–4 hrs | `week4/poisoning_backdoor.ipynb` |
| **Exercise** | Apply pruning defense (Sotiropoulos). Compare pre/post accuracy. Plot results. | 2 hrs | `week4/defense_pruning.ipynb` |

---

## Week 5: Generative AI Vulnerabilities
**Outcomes**: Craft jailbreaks, prompt injections; scan LLMs with garak.

| Activity | Details | Time | Output |
|--------|--------|------|--------|
| **Study** | Garak overview. Map prompt injection → XSS. Read Sotiropoulos Ch. 6–7. | 1–2 hrs | `week5/garak_mapping.md` |
| **Code** | Load Llama-2 (Hugging Face). Craft 10 jailbreak prompts (DAN, role-play, suffix). | 3 hrs | `week5/jailbreak_prompts.ipynb` |
| **Exercise** | Run `garak --model_type huggingface --model_name meta-llama/Llama-2-7b-chat-hf` with 10 probes. Log results. | 2–3 hrs | `week5/garak_scan.json`, `week5/garak_findings.md` |

---

## Week 6: Advanced LLM Red Teaming
**Outcomes**: Run Purple Llama evals; chain attacks; simulate purple teaming.

| Activity | Details | Time | Output |
|--------|--------|------|--------|
| **Study** | Purple Llama CyberSecEval guide. Read Sotiropoulos Ch. 8. | 1 hr | `week6/purplellama_notes.md` |
| **Code** | Run 3 CyberSecEval benchmarks (jailbreak, injection, leakage) on Llama. | 3–4 hrs | `week6/cyberseceval_results.ipynb` |
| **Exercise** | Chain garak + Purple Llama. Draft collab email to "data scientist" with mitigation recs. | 2 hrs | `week6/hybrid_attack.ipynb`, `collab/data_scientist_email.md` |

---

## Week 7: Mitigations, Evaluation & Reporting
**Outcomes**: Harden models; write professional AI pentest reports.

| Activity | Details | Time | Output |
|--------|--------|------|--------|
| **Study** | Sotiropoulos Ch. 9–10 + MITRE Adversarial ML Threat Matrix. | 2 hrs | `week7/mitre_matrix_notes.md` |
| **Code** | Apply adversarial training (torchattacks) to Week 3/4 models. Measure robustness gain. | 3 hrs | `week7/adversarial_training.ipynb` |
| **Exercise** | Write full pentest-style report on Week 5/6 attack (exec summary, risk matrix, visuals). | 3–4 hrs | `reports/llm_jailbreak_report.pdf` |

---

## Week 8: Integration, Portfolio & Career Prep
**Outcomes**: Full red team sim; tool contribution; job application ready.

| Activity | Details | Time | Output |
|--------|--------|------|--------|
| **Study** | Scan 5 recent arXiv papers on multimodal attacks. Summarize trends. | 2 hrs | `week8/trends_2025.md` |
| **Code** | Fork garak or ART. Add custom probe (e.g., cloud data poison sim). Test on hybrid model. | 3–4 hrs | `forks/garak-custom-probe/`, `week8/hybrid_attack.ipynb` |
| **Exercise** | Run **end-to-end sim**: attack → mitigate → report. Record 3-min exec explainer video. Update resume. | 4 hrs | `sim/full_red_team_sim.ipynb`, `video/exec_explainer.mp4`, `resume.md` |

---

## Final Portfolio Structure
