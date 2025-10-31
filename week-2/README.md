# Week 2: Core AI Adversarial Concepts

## Overview

Week 2 bridges your traditional pentesting knowledge with AI security fundamentals. You'll learn to classify attacks, map pentest workflows to AI red teaming, and perform your first real AI security vulnerability assessment.

**Key Transition Skills**: Think of AI models as target systems. Attacks on models are like exploits on web applications - you need reconnaissance, exploitation, and reporting phases.

---

## Learning Objectives

By the end of Week 2, you will be able to:

1. **Classify AI/ML Attacks**: Understand the three main attack categories (evasion, inference, poisoning) and when to use each
2. **Map Pentest Lifecycles**: Translate your traditional pentest methodology to AI/ML targets
3. **Perform Membership Inference**: Execute a real inference attack to detect data leakage
4. **Identify Attack Surfaces**: Map ML pipeline stages to penetration testing checkpoints
5. **Document Vulnerabilities**: Write an AI security finding in pentest report format

---

## Red Team Application

**What You're Actually Learning:**

- **Membership Inference**: Detect if a production model leaks information about its training data (similar to SQL injection leaking database info)
- **Attack Taxonomy**: Know which attacks to run on which model types (mapping to knowing when to use SQLi vs XSS)
- **Methodology Mapping**: Your pentest phases (recon, scanning, exploitation, reporting) apply directly to AI targets
- **Vulnerability Reporting**: Write AI security findings that business stakeholders understand

**Real-World Scenario**: Client has deployed an ML model for customer risk scoring. Your job is to determine if the model leaks sensitive training data and assess overall security posture.

---

## Required Reading

### Primary Texts

**Sotiropoulos (2024)** - "Adversarial AI Attacks, Mitigations and Defense Strategies"
- **Chapter 2**: Attack Taxonomies and Classifications (pp. 23-45)
- **Chapter 3**: Adversarial Attack Lifecycle (pp. 47-70)
- **Chapter 4**: Inference Attacks and Membership Detection (pp. 71-95)

**Géron (2019)** - "Hands-On Machine Learning"
- **Chapter 6**: Decision Trees (pp. 157-185) - for understanding model behavior
- **Chapter 14**: Training with Limited Data (pp. 533-550) - data privacy concepts

### Additional Resources

**NIST AI Risk Management Framework**
- Section 3.1: Adversarial Machine Learning Taxonomy
- Available: https://doi.org/10.6028/NIST.AI.100-1

**OWASP Top 10 for LLM Applications**
- Focus on A1: Prompt Injection, A3: Training Data Poisoning
- Available: https://owasp.org/www-project-top-10-for-large-language-model-applications/

---

## Weekly Structure

### Monday-Tuesday: Attack Taxonomy & Methodology (2-3 hours)

**Activity**: Map Traditional Pentest to AI Pentest Lifecycle

**Tasks**:
1. Read Sotiropoulos Chapters 2-3
2. Create side-by-side comparison: Traditional Pentest Phase → AI Pentest Phase
3. Identify tool mappings (e.g., Nmap → Model fingerprinting, Burp → Adversarial testing)

**Deliverable**: `week-2/ai_pentest_methodology.md`

**Template Sections**:
```markdown
# AI Pentest Methodology - Red Team Mapping

## Phase 1: Reconnaissance
**Traditional**: Port scanning, service enumeration, OS fingerprinting
**AI Equivalent**: Model architecture discovery, API endpoint mapping, input/output analysis
**Tools**: Model fingerprinting scripts, API documentation review, black-box querying

## Phase 2: Vulnerability Discovery
**Traditional**: Automated scanners, manual testing, fuzzing
**AI Equivalent**: Adversarial sample generation, membership inference, model stealing
**Tools**: Foolbox, ART, custom attack scripts

## Phase 3: Exploitation
**Traditional**: Craft exploit payload, establish foothold
**AI Equivalent**: Craft adversarial inputs, bypass model defenses
**Tools**: FGSM/PGD, custom attack frameworks

## Phase 4: Post-Exploitation
**Traditional**: Privilege escalation, lateral movement, data exfiltration
**AI Equivalent**: Model stealing, training data extraction, backdoor implantation
**Tools**: Model extraction attacks, membership inference

## Phase 5: Reporting
**Traditional**: Executive summary, technical details, remediation
**AI Equivalent**: Attack success rate, business impact, defensive recommendations
**Deliverables**: Same format as traditional pentest
```

---

### Wednesday-Thursday: Membership Inference Attack (3-4 hours)

**Activity**: Perform Real Membership Inference on Week 1's MNIST Model

**Learning Objective**: Execute inference attack to detect training data leakage (critical AI security vulnerability)

**Background**:
- **Problem**: ML models can leak information about their training data
- **Attack**: Query model to determine if specific samples were in training set
- **Business Impact**: Privacy violation (HIPAA, GDPR), competitive intelligence leakage
- **Red Team Context**: Like SQL injection leaking database contents, this leaks training data

**Exercises**:
1. **exercise_1_membership_inference.py**: Implement membership inference attack
2. **exercise_2_shadow_model.py**: Train shadow models for attack training
3. **exercise_3_data_leakage_report.py**: Generate vulnerability report with findings

**Expected Attack Success Rate**: 55-65% (random guess = 50%)

---

### Friday: Research Application & Vulnerability Documentation (2-3 hours)

**Activity**: Extract Attack Techniques from Research Papers

**Objective**: Learn to read AI security research and apply techniques to real models

**Tasks**:
1. Read 3 recent arXiv papers on membership inference (provided links)
2. Extract attack methodology from each
3. Implement one technique from a paper
4. Test on Week 1 model
5. Document in vulnerability report format

**Deliverable**: `week-2/paper_techniques_report.md` + working script

**Required Papers** (students will be provided with links):
1. Shokri et al. (2017) "Membership Inference Attacks Against Machine Learning Models"
2. Carlini et al. (2022) "Extracting Training Data from Large Language Models"
3. Choquette-Choo et al. (2021) "Label-Only Membership Inference Attacks"

---

## Creative Challenges (New)

These challenges enhance learning through investigation scenarios, decision-making exercises, optimization challenges, and research implementation.

### Challenge 1: Membership Inference Challenge - Detective Mode (2 hours)

**Objective**: Perform membership inference investigation using incident response methodology.

**Task**: Approach membership inference as a security investigation. Create professional investigation report with reconnaissance, hypothesis formation, attack execution, evidence analysis, and recommendations.

**Deliverable**: `week-2/investigation_report.md`

**Details**: See `week-2/challenges/challenge_1_detective_mode.md`

---

### Challenge 2: Attack Taxonomy Decision Tree (1 hour)

**Objective**: Create visual decision tree for attack selection.

**Task**: Build flowchart to help choose which attack to use based on model access, data access, attack goal, time constraints, and stealth requirements.

**Deliverable**: `week-2/attack_decision_tree.md` or `.png`

**Details**: See `week-2/challenges/challenge_2_attack_decision_tree.md`

---

### Challenge 3: Shadow Model Build-Off (2 hours)

**Objective**: Optimize shadow model efficiency under query constraints.

**Task**: Build most effective shadow model with limited queries (simulating API rate limits). Test optimization strategies: smart sampling, transfer learning, active learning, ensemble approaches.

**Deliverable**: `week-2/shadow_model_optimization.md`

**Details**: See `week-2/challenges/challenge_3_shadow_model_buildoff.md`

---

### Challenge 4: Paper Reading & Implementation Challenge (3 hours)

**Objective**: Read research paper, extract technique, implement it, create presentation.

**Task**: Choose one paper from provided list, read and understand it, implement the key technique, test on your model, create 5-slide presentation.

**Deliverable**: `week-2/paper_presentation.md` + implementation code

**Details**: See `week-2/challenges/challenge_4_paper_challenge.md`

---

## Coding Exercises Overview

### Exercise 1: Membership Inference Attack
**File**: `exercise_1_membership_inference.py`
**Objective**: Detect if specific samples were in training data

**What Students Learn**:
- Black-box model querying techniques
- Confidence score analysis
- Shadow model training for attack training
- Statistical analysis for membership detection

**Red Team Relevance**: This is how you test if a production model leaks training data (HIPAA, GDPR compliance issue)

---

### Exercise 2: Shadow Models and Attack Training
**File**: `exercise_2_shadow_models.py`
**Objective**: Train attack models using shadow models

**What Students Learn**:
- Creating shadow models that mimic target model behavior
- Training neural network-based membership inference attacks
- Feature engineering for attack models
- Model stealing concepts

**Red Team Relevance**: Shadow models are used in real-world attacks to understand target behavior

---

### Exercise 3: AI Vulnerability Reporting
**File**: `exercise_3_vulnerability_report.py`
**Objective**: Generate professional vulnerability findings

**What Students Learn**:
- Documenting AI security findings
- Calculating risk scores
- Creating visualizations for stakeholders
- Writing remediation recommendations

**Red Team Relevance**: This is what you deliver to clients - professional AI pentest reports

---

## Deliverables Checklist

### Core Exercises
- [ ] AI Pentest Methodology document mapping traditional → AI pentest phases
- [ ] Membership inference attack script with 60%+ success rate
- [ ] Shadow model implementation
- [ ] Vulnerability report with executive summary and technical details

### Creative Challenges (New)
- [ ] `investigation_report.md` - Membership inference investigation (Detective Mode)
- [ ] `attack_decision_tree.md` or `.png` - Visual attack selection guide
- [ ] `shadow_model_optimization.md` - Query efficiency optimization results
- [ ] `paper_presentation.md` + implementation - Research paper implementation

### Research & Documentation
- [ ] Research paper technique implementation (if different from Challenge 4)
- [ ] Updated portfolio with Week 2 work

---

## Success Criteria

**You've successfully completed Week 2 when you can**:

1. Explain the three main AI attack categories and when to use each
2. Map your traditional pentest workflow to AI targets
3. Execute membership inference attack achieving >60% accuracy
4. Train shadow models for attack purposes
5. Extract and implement techniques from research papers
6. Write AI security findings in professional vulnerability report format

---

## Self-Assessment Questions

Answer these to confirm understanding:

1. **Attack Taxonomy**: What's the difference between evasion, inference, and poisoning attacks? Which would you use to bypass a fraud detection model?
2. **Membership Inference**: Why is membership inference a privacy concern? What regulations might it violate?
3. **Methodology**: How does your pentest reconnaissance phase translate to AI targets? What tools would you use?
4. **Shadow Models**: Why do we train shadow models for membership inference attacks?
5. **Reporting**: How do you explain AI model vulnerabilities to non-technical stakeholders?

---

## Red Team Career Connection

**Skills You're Building This Week**:
- AI/ML vulnerability assessment
- Membership inference testing
- AI pentest methodology development
- Research-to-exploitation workflow
- Professional vulnerability documentation

**How This Prepares You for AI Red Team Roles**:
- Most engagements involve membership inference testing
- Understanding attack taxonomy helps prioritize testing
- Methodology mapping shows senior leadership your systematic approach
- Vulnerability documentation is your billable deliverable

---

## Troubleshooting Tips

**Membership Inference Attack Accuracy Too Low**:
- Check shadow model training - may need more training data
- Experiment with different features for attack model
- Try different membership inference techniques
- Ensure target model predictions have sufficient variance

**Shadow Model Not Converging**:
- Reduce model complexity
- Increase training epochs
- Check data normalization
- Verify loss function is appropriate

**Cannot Extract Features from Target Model**:
- Start with simpler approach (e.g., confidence scores only)
- Gradually add more sophisticated features
- Consult paper implementations for feature engineering ideas

---

## Next Steps

After completing Week 2:
1. Review membership inference results - did you achieve >60% attack success?
2. Document lessons learned in your lab notebook
3. Update your portfolio with AI pentest methodology document
4. Prepare for Week 3: Evasion attacks (FGSM, PGD) using Week 1's model as target

**Week 3 Preview**: You'll craft adversarial samples that fool your Week 1 MNIST model with >90% success rate.
