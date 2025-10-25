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

**Structure:**
```
ai-red-team-course/
├─ week-01/
├─ attacks/
├─ harness/
├─ reports/
├─ lab-notebook.md
├─ requirements.txt
└─ README.md
```

This repo includes minimal starter scripts like:
- `train_mnist.py` — simple PyTorch MNIST training + model save
- `serve_model.py` — FastAPI example to load and serve a model
- `harness.py` — skeleton CLI for running attack adapters
- `requirements.txt` — pinned list of packages to get started


---

## Week 1: ML Foundations for Security Pros
**Outcomes**: Identify ML attack surfaces; build & evaluate predictive/generative models.

| Activity | Details | Time | Output |
|--------|--------|------|--------|
| **Study** | Map pentest lifecycle to ML stages (recon → data, scanning → model querying). Use Géron Ch. 1–2, Sotiropoulos Ch. 1. | 2–3 hrs | `week1/ml-attack-surfaces.drawio` or Markdown diagram |
| **Code** | Build MNIST CNN classifier + simple text generator (PyTorch). Train, log accuracy, plot loss. | 2–3 hrs | `week1/mnist_classifier.ipynb`, `week1/text_generator.ipynb` |
| **Exercise** | Annotate 5 OWASP ML Top 10 vulns with pentest analogies (e.g., prompt injection = XSS). | 1–2 hrs | `week1/owasp_ml_top10.md` |

---

## Week 2: Core AI Adversarial Concepts
**Outcomes**: Classify attacks; align pentest workflows with AI red teaming.

| Activity | Details | Time | Output |
|--------|--------|------|--------|
| **Study** | Diagram end-to-end AI pen test using your cloud pentest templates. Read Sotiropoulos Ch. 2–3, NIST Taxonomy. | 2–3 hrs | `week2/ai_pentest_lifecycle.md` |
| **Code** | Simulate membership inference: query Week 1 model to extract training data hints. | 2 hrs | `week2/inference_attack.ipynb` |
| **Exercise** | Summarize 3 arXiv papers on poisoning/inference. Focus on real-world risk. | 2–3 hrs | `week2/paper_summaries.md` |

---

## Week 3: Evasion & Inference Attacks on Predictive Models
**Outcomes**: Execute FGSM/PGD; run membership inference; quantify model weakness.

| Activity | Details | Time | Output |
|--------|--------|------|--------|
| **Study** | Read Foolbox quickstart. Compare FGSM/PGD to API fuzzing (Burp Intruder). | 1–2 hrs | `week3/foolbox_notes.md` |
| **Code** | Apply FGSM & PGD to Week 1 MNIST model using Foolbox. Target >90% evasion. Visualize perturbations. | 3–4 hrs | `week3/evasion_fgsm_pgd.ipynb` |
| **Exercise** | Run ART membership inference tutorial. Log success rate. Draft 1-page vuln report. | 2 hrs | `week3/membership_inference.ipynb`, `reports/membership_report.pdf` |

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
