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
**Outcomes**: Build and query ML models; understand baseline model behavior before attacking.

**Simplified Exercises** (Minimal working code):
1. **Exercise 1**: Train MNIST Model (`week-1/exercises/exercise_1_simple_mnist_train.py`)
   - Train simple CNN on MNIST dataset
   - Minimal training code (3 epochs)
   - Save model for later exercises
   - **Time**: ~10 minutes
2. **Exercise 2**: Query Model and Analyze Predictions (`week-1/exercises/exercise_2_simple_model_queries.py`)
   - Load trained model and make predictions
   - Extract confidence scores
   - Analyze correct vs incorrect predictions
   - Understand baseline model behavior
   - **Time**: ~5 minutes
3. **Exercise 3**: Model Sensitivity Analysis (`week-1/exercises/exercise_3_model_sensitivity.py`)
   - Test model with modified inputs (noise, brightness)
   - Observe how confidence scores change
   - Understand model sensitivity (foundation for adversarial attacks)
   - **Time**: ~5 minutes
4. **Exercise 4**: Simple Model Deployment (`week-1/exercises/exercise_4_simple_deployment.py`) - Optional
   - Deploy model as FastAPI endpoint
   - Basic model serving demonstration
   - **Time**: ~10 minutes

**Required Reading**: Géron Ch. 1, 10, 13; Sotiropoulos Ch. 1
**Deliverables**: Trained model (>95% accuracy), baseline behavior analysis, sensitivity testing

---

## Week 2: Core AI Adversarial Concepts
**Outcomes**: Perform membership inference attacks; understand attack taxonomy; generate vulnerability reports.

**Exercises**:
1. **Exercise 1**: Membership Inference Attack (`week-2/exercises/exercise_1_membership_inference.py`)
   - Implement membership inference attack algorithm
   - Classify samples as member or non-member of training data
   - Calculate attack success metrics (>60% accuracy)
2. **Exercise 2**: Shadow Models (`week-2/exercises/exercise_2_shadow_models.py`)
   - Train shadow models to replicate target model behavior
   - Generate attack training data from shadow model outputs
   - Improve membership inference attack accuracy
3. **Exercise 3**: Vulnerability Reporting (`week-2/exercises/exercise_3_vulnerability_reporting.py`)
   - Calculate risk scores using attack success rates
   - Generate professional AI security vulnerability reports

**Required Reading**: Sotiropoulos Ch. 2–4, NIST AI Risk Management Framework
**Deliverables**: Membership inference attack (>60% success), shadow models, vulnerability report

---

## Week 3: Evasion & Inference Attacks on Predictive Models
**Outcomes**: Execute evasion attacks using libraries (ART, CleverHans, Foolbox); optionally implement from scratch.

**Simplified Exercises** (Library-first approach):
1. **Exercise 1**: ART Evasion Attacks (`week-3/exercises/exercise_1_art_evasion_attacks.py`)
   - Use ART library for FGM and PGD attacks
   - Achieve 80-90% evasion with FGM, >95% with PGD
   - **Time**: ~5 minutes
2. **Exercise 2**: CleverHans Evasion Attacks (`week-3/exercises/exercise_2_cleverhans_evasion_attacks.py`)
   - Use CleverHans for FGSM and PGD attacks
   - Compare different library implementations
   - **Time**: ~5 minutes
3. **Exercise 3**: Foolbox Evasion Attacks (`week-3/exercises/exercise_3_foolbox_evasion_attacks.py`)
   - Use Foolbox for FGSM, PGD, and L2 iterative attacks
   - Compare multiple attack libraries
   - **Time**: ~5 minutes

**Advanced Exercises** (Optional - From Scratch):
4. **Exercise 4**: FGSM from Scratch (`week-3/exercises/exercise_4_fgsm_attack.py`)
   - Implement FGSM algorithm manually
   - Understand gradient computation and perturbations
5. **Exercise 5**: PGD from Scratch (`week-3/exercises/exercise_5_pgd_attack.py`)
   - Implement PGD with iterative refinement
6. **Exercise 6**: Attack Visualization (`week-3/exercises/exercise_6_attack_comparison.py`)
   - Visualize adversarial samples and compare methods

**Required Reading**: Sotiropoulos Ch. 5–7, Goodfellow et al. (2015), Madry et al. (2018)
**Deliverables**: Library-based evasion attacks (80-90% FGSM, >95% PGD), visualizations, vulnerability report

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
