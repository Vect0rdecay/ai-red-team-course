# 8-Week AI Red Teaming Transition Course  
**From Web/Cloud Pentesting to AI Security — in 2 Months**

> **Goal**: Build upon existing **n+ years of web, API, and cloud pentesting** to become a qualified **AI Red Teamer** (e.g., HiddenLayer).  
> **Outcome**: Portfolio with attack scripts, reports, tool contributions, and resume-ready evidence of **1+ year equivalent AI security experience**.

**Time Commitment**: 12–18 hrs/week  
**Core Texts**:
- *Adversarial AI Attacks, Mitigations and Defense Strategies* – John Sotiropoulos (Packt)
- *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (3rd ed.) – Aurélien Géron (O'Reilly)
- deeplearning.ai: *Machine Learning Specialization* & *Generative AI with LLMs* (free)

**Hands-On Tools**:

**Coding Libraries** (Weeks 1-4, 7-8):
- [Foolbox](https://github.com/bethgelab/foolbox) – Evasion attacks (FGSM, PGD)
- [Adversarial Robustness Toolbox (ART)](https://github.com/Trusted-AI/adversarial-robustness-toolbox) – Poisoning, inference attacks
- [CleverHans](https://github.com/cleverhans-lab/cleverhans) – Evasion attacks (FGSM, PGD)
- [torchattacks](https://github.com/Harry24k/adversarial-attacks-pytorch) – PyTorch attack implementations

**LLM Security Tools** (Weeks 5-6, optional):
- [garak](https://github.com/leondz/garak) – LLM vulnerability scanner
- [Purple Llama](https://github.com/meta-llama/PurpleLlama) – CyberSecEval for LLMs

**CTF Platforms** (Week 5):
- [Lakera Gandalf](https://gandalf.lakera.ai/) – Gamified prompt hacking challenges
- [Security Café's AI Hacking Games](https://securitycafe.ro/2023/05/15/ai-hacking-games-jailbreak-ctfs/) – Prompt injection CTF challenges
- [Hack The Box Academy](https://academy.hackthebox.com/course/preview/prompt-injection-attacks) – Prompt injection attacks course

**Reading Resources** (Week 5):
- ArXiv research papers on jailbreaks and prompt injection (see `week-5/exercises/README.md`)
- [Embrace the Red Blog](https://embracethered.com/blog/) – AI security and red teaming
- [InjectPrompt Blog](https://www.injectprompt.com/) – AI jailbreaks and prompt injections

**Portfolio Repo**: [github.com/yourname/ai-red-team-course](https://github.com/yourname/ai-red-team-course)  

### Course Structure

Each week contains two directories:
- **`exercises/`** - Python scripts (.py files) with 85% complete implementations and TODOs, OR reading materials and CTF resources (Week 5)
- **`notes/`** - Space for your personal notes and additional materials

Most exercises include comprehensive educational comments explaining ML/AI concepts for beginners. Week 5 uses reading materials and CTF challenges instead of coding exercises.

**Key Files**:
- `PROJECT_STRUCTURE.md` - Detailed repository organization
- `templates/vulnerability_report.md` - Professional AI security report template
- `scripts/setup_environment.sh` - Automated environment setup

---

## Week 1: ML Foundations
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

**Simplified Exercises** (Minimal working code):
1. **Exercise 1**: Data Poisoning Attack (`week-4/exercises/exercise_1_data_poisoning.py`)
   - Poison training data by flipping labels (10% of training labels)
   - Retrain model with poisoned data
   - Measure accuracy degradation compared to clean model
   - Understand how poisoned data affects model behavior
   - **Time**: ~15 minutes
2. **Exercise 2**: Backdoor Attack Implementation (`week-4/exercises/exercise_2_backdoor_attack.py`)
   - Implement BadNets-style backdoor attack with trigger pattern
   - Poison training samples with trigger + target label
   - Train model maintaining clean accuracy (>95%)
   - Test backdoor activation rate (>90%)
   - **Time**: ~20 minutes
3. **Exercise 3**: Defense Testing (`week-4/exercises/exercise_3_defense_testing.py`)
   - Apply model pruning defense against backdoor attacks
   - Remove small-weight neurons to reduce backdoor effectiveness
   - Compare pre/post defense accuracy and backdoor activation
   - Evaluate defense effectiveness
   - **Time**: ~15 minutes

**Required Reading**: Sotiropoulos Ch. 8–10 (Data Poisoning, Backdoor Attacks, Defenses); Gu et al. (2017) "BadNets"; Chen et al. (2018) "Targeted Backdoor Attacks"
**Deliverables**: Poisoned model demonstration, backdoor attack with >90% activation, defense evaluation results

---

## Week 5: Generative AI Vulnerabilities
**Outcomes**: Understand jailbreaks and prompt injections through research; practice with CTF challenges; optionally use garak for LLM scanning.

**Reading Exercises** (Research and CTF-based learning):
1. **Reading Exercise 1**: Study ArXiv Research Papers
   - Read at least 3 papers on prompt injection and jailbreak techniques
   - Key papers: "Red Teaming the Mind of the Machine", "Hide Your Malicious Goal Into Benign Narratives", "SequentialBreak"
   - Document key techniques, attack vectors, and defense strategies
   - **Time**: ~4–5 hours
   - **Output**: `week5/reading_notes.md`
2. **Reading Exercise 2**: Review Expert Blogs
   - Study Embrace the Red blog posts on LLM security
   - Review InjectPrompt blog and Lakera blog resources
   - Understand real-world examples and case studies
   - **Time**: ~2 hours
   - **Output**: `week5/blog_analysis.md`
3. **CTF Practice**: Lakera Gandalf and Security Café Challenges
   - Complete Lakera Gandalf prompt hacking challenges (progressive levels)
   - Practice Security Café's AI Hacking Games (context switching, translation, etc.)
   - Document techniques learned and success rates
   - **Time**: ~3–4 hours
   - **Output**: `week5/ctf_notes.md`, `week5/jailbreak_catalog.md`
4. **Research Exercise**: Attack Methodology Analysis
   - Analyze attack methodologies from research papers
   - Create catalog of jailbreak techniques (DAN, role-play, indirect, etc.)
   - Document attack chain flows and dependencies
   - **Time**: ~2–3 hours
   - **Output**: `week5/jailbreak_catalog.md`, `week5/attack_chain.md`
5. **Optional Tool Usage**: garak and Purple Llama Scanning
   - Run `garak` with 10+ probes on LLM model
   - Run Purple Llama CyberSecEval benchmarks
   - Analyze results and create findings report
   - **Time**: ~2–3 hours
   - **Output**: `week5/garak_scan.json`, `week5/garak_findings.md`

**Required Reading**: Sotiropoulos Ch. 11–12 (Generative AI Vulnerabilities, LLM Security and Prompt Attacks); OWASP Top 10 for LLM Applications; ArXiv papers (see `week-5/exercises/README.md`)

**Deliverables**: Reading notes, CTF practice documentation, jailbreak catalog, attack chain analysis, optional garak scan results

**Note**: Week 5 uses reading materials and CTF challenges instead of coding exercises. See `week-5/exercises/README.md` for complete list of resources.

---

## Week 6: Advanced LLM Red Teaming
**Outcomes**: Run Purple Llama evals; chain attacks; simulate purple teaming.

| Activity | Details | Time | Output |
|--------|--------|------|--------|
| **Study** | Purple Llama CyberSecEval guide. Read Sotiropoulos Ch. 8. | 1 hr | `week6/purplellama_notes.md` |
| **Code** | Run 3 CyberSecEval benchmarks (jailbreak, injection, leakage) on Llama. | 3–4 hrs | `week6/cyberseceval_results.py` |
| **Exercise** | Chain garak + Purple Llama. Draft collab email to "data scientist" with mitigation recs. | 2 hrs | `week6/hybrid_attack.py`, `collab/data_scientist_email.md` |

---

## Week 7: Mitigations, Evaluation & Reporting
**Outcomes**: Harden models; write professional AI pentest reports.

| Activity | Details | Time | Output |
|--------|--------|------|--------|
| **Study** | Sotiropoulos Ch. 9–10 + MITRE Adversarial ML Threat Matrix. | 2 hrs | `week7/mitre_matrix_notes.md` |
| **Code** | Apply adversarial training (torchattacks) to Week 3/4 models. Measure robustness gain. | 3 hrs | `week7/adversarial_training.py` |
| **Exercise** | Write full pentest-style report on Week 5/6 attack (exec summary, risk matrix, visuals). | 3–4 hrs | `reports/llm_jailbreak_report.pdf` |

---

## Week 8: Integration, Portfolio & Career Prep
**Outcomes**: Full red team sim; tool contribution; job application ready.

| Activity | Details | Time | Output |
|--------|--------|------|--------|
| **Study** | Scan 5 recent arXiv papers on multimodal attacks. Summarize trends. | 2 hrs | `week8/trends_2025.md` |
| **Code** | Fork garak or ART. Add custom probe (e.g., cloud data poison sim). Test on hybrid model. | 3–4 hrs | `forks/garak-custom-probe/`, `week8/hybrid_attack.py` |
| **Exercise** | Run **end-to-end sim**: attack → mitigate → report. Record 3-min exec explainer video. Update resume. | 4 hrs | `sim/full_red_team_sim.py`, `video/exec_explainer.mp4`, `resume.md` |

---

## Final Portfolio Structure
