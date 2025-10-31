# Week 3: Evasion & Inference Attacks on Predictive Models

## Overview

Week 3 focuses on evasion attacks - making adversarial samples that fool your models. You'll implement industry-standard attacks (FGSM, PGD) and test them on Week 1's MNIST model.

**Key Transition**: Think of evasion attacks like SQL injection - you craft inputs that look normal but bypass security controls (model predictions).

---

## Learning Objectives

By the end of Week 3, you will be able to:

1. **Execute FGSM Attacks**: Fast, simple adversarial sample generation
2. **Implement PGD Attacks**: Iterative, more powerful evasion attacks
3. **Achieve High Evasion Rates**: Fool your model with >90% adversarial success
4. **Visualize Perturbations**: Show how small changes fool the model
5. **Apply Adversarial Defenses**: Test defense mechanisms against attacks

---

## Red Team Application

**What You're Actually Learning**:

- **Evasion Attacks**: Bypass ML-based security controls (like fraud detection, spam filters, malware detection)
- **Adversarial Crafting**: Similar to crafting payloads for XSS/SQLi - inputs that look normal but exploit vulnerabilities
- **Attack Transferability**: If it works on one model, it often works on others
- **Real-World Impact**: Adversarial samples can bypass autonomous vehicle vision, facial recognition, content filtering

**Real-World Scenario**: Client has deployed an ML-based credit card fraud detection system. Your job is to demonstrate that attackers can craft transactions that bypass the fraud detection while appearing legitimate.

---

## Required Reading

**Sotiropoulos (2024)** - "Adversarial AI Attacks, Mitigations and Defense Strategies"
- **Chapter 5**: Evasion Attacks and Adversarial Examples (pp. 125-155)
- **Chapter 6**: Black-Box vs White-Box Attacks (pp. 157-180)
- **Chapter 7**: Defenses Against Evasion Attacks (pp. 181-210)

**Géron (2019)** - "Hands-On Machine Learning"
- **Chapter 14**: Training Models with Limited Data (Sections 14.6-14.7 on adversarial training)

**Research Papers**:
- Goodfellow et al. (2015) "Explaining and Harnessing Adversarial Examples" - FGSM
- Madry et al. (2018) "Towards Deep Learning Models Resistant to Adversarial Attacks" - PGD

---

## Weekly Structure

### Monday-Tuesday: Library-Based Evasion Attacks (2-3 hours)

**Activity**: Use attack libraries (ART, CleverHans, Foolbox) to perform evasion attacks

**Background**:
- Professional attack libraries provide pre-built implementations
- Faster to use than building from scratch
- Demonstrates real-world attack tools
- Establishes baseline attack success rates

**Exercises**:
1. **exercise_1_art_evasion_attacks.py**: Use ART for FGM and PGD attacks
2. **exercise_2_cleverhans_evasion_attacks.py**: Use CleverHans for FGSM and PGD
3. **exercise_3_foolbox_evasion_attacks.py**: Use Foolbox for multiple attack types

**Expected Results**: 
- FGSM/FGM: 80-90% evasion rate
- PGD: >95% evasion rate

---

### Wednesday-Thursday: From-Scratch Implementation (Optional, 3-4 hours)

**Activity**: Implement attacks from scratch for deeper understanding

**Background**:
- Building attacks from scratch provides deeper understanding
- Helps understand attack mechanics
- Useful for custom attack development

**Exercises** (Optional):
1. **exercise_4_fgsm_attack.py**: Implement FGSM from scratch
2. **exercise_5_pgd_attack.py**: Implement PGD from scratch

**Expected Results**: Same evasion rates as library implementations

---

### Friday: Visualization and Reporting (2-3 hours)

**Activity**: Visualize attacks and create vulnerability reports

**Exercises**:
1. **exercise_6_attack_comparison.py**: Visualize and compare attacks
2. **exercise_7_vulnerability_report.py**: Document all findings in pentest format

---

## Creative Challenges (New)

These challenges enhance learning through video creation, tool comparison, real-world scenarios, and professional visualizations.

### Challenge 1: Adversarial Attack Demo Video (1.5 hours)

**Objective**: Create short video demonstrating adversarial attack.

**Task**: Record 2-3 minute video showing: clean image → attack → fooled model. Include narration explaining the attack and business impact.

**Deliverable**: `week-3/attack_demo_video.mp4`

**Details**: See `week-3/challenges/challenge_1_attack_demo_video.md`

---

### Challenge 2: Attack Library Comparison Matrix (1 hour)

**Objective**: Compare attack libraries (ART, CleverHans, Foolbox) for tool selection.

**Task**: Test same attack (FGSM) with all three libraries. Compare: ease of use, attack success rate, speed, documentation, flexibility, community support.

**Deliverable**: `week-3/tool_comparison_matrix.md`

**Details**: See `week-3/challenges/challenge_2_tool_comparison.md`

---

### Challenge 3: Real-World Evasion Scenario - Fraud Detection Bypass (2 hours)

**Objective**: Simulate real-world evasion attack on fraud detection system.

**Task**: Use MNIST as proxy for transaction features. Craft adversarial samples that evade fraud detection while maintaining transaction validity. Create client-ready report.

**Deliverable**: `week-3/fraud_evasion_report.md`

**Details**: See `week-3/challenges/challenge_3_fraud_evasion.md`

---

### Challenge 4: Visualization Gallery (1 hour)

**Objective**: Create presentation-quality attack visualizations.

**Task**: Create 3+ professional visualizations: before/after adversarial samples, perturbation heatmap, attack success vs epsilon. Focus on client-presentation quality.

**Deliverable**: `week-3/visualization_gallery/` directory with images and documentation

**Details**: See `week-3/challenges/challenge_4_visualization_gallery.md`

---

## Simplified Exercises (Recommended)

These simplified exercises use attack libraries first, then build from scratch.

### Exercise 1: ART Evasion Attacks
**File**: `exercise_1_art_evasion_attacks.py`
**Objective**: Perform evasion attacks using Adversarial Robustness Toolbox (ART)

**What You'll Learn**:
- Using ART to generate adversarial examples
- Fast Gradient Method (FGM) attack
- Projected Gradient Descent (PGD) attack
- Measuring attack effectiveness with libraries

**Time**: ~5 minutes
**Run**: `python exercise_1_art_evasion_attacks.py`

---

### Exercise 2: CleverHans Evasion Attacks
**File**: `exercise_2_cleverhans_evasion_attacks.py`
**Objective**: Perform evasion attacks using CleverHans library

**What You'll Learn**:
- CleverHans attack implementations
- Fast Gradient Sign Method (FGSM)
- PGD with CleverHans
- Comparing different library implementations

**Time**: ~5 minutes
**Run**: `python exercise_2_cleverhans_evasion_attacks.py`

---

### Exercise 3: Foolbox Evasion Attacks
**File**: `exercise_3_foolbox_evasion_attacks.py`
**Objective**: Perform evasion attacks using Foolbox library

**What You'll Learn**:
- Using Foolbox for adversarial attacks
- FGSM, PGD, and L2 iterative attacks
- Comparing different attack libraries
- Understanding attack library differences

**Time**: ~5 minutes
**Run**: `python exercise_3_foolbox_evasion_attacks.py`

---

## Advanced Exercises (Optional - From Scratch Implementation)

For deeper understanding, implement attacks from scratch:

### Exercise 4: FGSM Attack Implementation (From Scratch)
**File**: `exercise_4_fgsm_attack.py` + `.ipynb`
**Objective**: Implement Fast Gradient Sign Method from scratch

**What Students Learn**:
- Gradient computation with PyTorch autograd
- Creating adversarial samples manually
- Measuring attack success rate
- Understanding perturbation budgets

**Red Team Relevance**: Understanding attack mechanics helps design better tests

---

### Exercise 5: PGD Attack Implementation (From Scratch)
**File**: `exercise_5_pgd_attack.py` + `.ipynb`
**Objective**: Implement Projected Gradient Descent attacks from scratch

**What Students Learn**:
- Iterative attack optimization
- Projection to bound perturbations
- Multiple gradient steps
- Achieving higher evasion rates

---

### Exercise 6: Attack Visualization and Comparison
**File**: `exercise_6_attack_comparison.py` + `.ipynb`
**Objective**: Visualize and compare different attack methods

**What Students Learn**:
- Visualizing adversarial examples
- Showing imperceptible perturbations
- Comparing library vs from-scratch implementations
- Creating compelling visualizations for reports

**Note**: Visualization functionality may be included in this exercise or separate visualization tools.

---

### Exercise 7: Vulnerability Report
**File**: `exercise_7_vulnerability_report.py` + `.ipynb`
**Objective**: Generate complete vulnerability report

**What Students Learn**:
- Documenting evasion attack findings
- Calculating risk from attack success rates
- Creating executive summaries
- Technical documentation for engineering teams

---

## Deliverables Checklist

### Core Exercises
- [ ] ART evasion attacks (FGM and PGD) achieving expected evasion rates
- [ ] CleverHans evasion attacks (FGSM and PGD) demonstrating library usage
- [ ] Foolbox attacks demonstrating additional attack methods
- [ ] (Optional) From-scratch FGSM implementation
- [ ] (Optional) From-scratch PGD implementation
- [ ] Attack comparison analysis (library vs from-scratch, FGSM vs PGD)

### Creative Challenges (New)
- [ ] `attack_demo_video.mp4` - Video demonstration of attack
- [ ] `tool_comparison_matrix.md` - Library comparison guide
- [ ] `fraud_evasion_report.md` - Real-world scenario report
- [ ] `visualization_gallery/` - Professional attack visualizations

### Reporting & Documentation
- [ ] Visualizations of adversarial samples and perturbations
- [ ] Complete vulnerability report with all findings
- [ ] Updated portfolio with Week 3 work

---

## Success Criteria

**You've successfully completed Week 3 when you can**:

1. Implement FGSM attack achieving >80% evasion
2. Implement PGD attack achieving >95% evasion
3. Visualize adversarial samples showing imperceptible perturbations
4. Explain the difference between FGSM and PGD
5. Use ART library for membership inference attacks
6. Document findings in professional vulnerability report format

---

## Self-Assessment Questions

1. **FGSM vs PGD**: What's the key difference? When would you use each?
2. **Perturbation Budget**: What's ε and why does it matter for attack success?
3. **Transferability**: Why do adversarial samples often transfer between models?
4. **White-box vs Black-box**: Which attack type requires model access?
5. **Defense**: How could you defend against FGSM/PGD attacks?

---

## Red Team Career Connection

**Skills You're Building**:
- Evasion attack implementation
- Adversarial sample crafting
- Model robustness testing
- Attack visualization
- Vulnerability documentation

**How This Prepares You for AI Red Team Roles**:
- Evasion attacks are the most common in real engagements
- Understanding attack mechanics helps design better tests
- Visualization skills essential for client presentations
- FGSM/PGD are industry-standard attack baselines

---

## Troubleshooting Tips

**Attack Success Rate Too Low**:
- Increase perturbation budget (ε)
- Check gradient computation
- Verify model is in evaluation mode
- Ensure correct loss function

**Gradient Errors**:
- Make sure model requires grad on inputs
- Check input tensor has requires_grad=True
- Verify loss.backward() is called properly

**Visualization Issues**:
- Clip image values to [0, 1] after adding perturbations
- Denormalize images before visualizing
- Use appropriate colormaps for perturbations

---

## Next Steps

After completing Week 3:
1. Review your attack success rates - did you achieve >90% evasion?
2. Examine adversarial samples - are perturbations really imperceptible?
3. Document lessons learned in your lab notebook
4. Update your portfolio with attack implementations
5. Prepare for Week 4: Poisoning & Backdoor attacks

**Week 4 Preview**: You'll poison training data and implant backdoors to compromise model behavior.
