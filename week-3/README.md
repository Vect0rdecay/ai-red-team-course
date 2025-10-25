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

### Monday-Tuesday: FGSM Attacks (3-4 hours)

**Activity**: Implement Fast Gradient Sign Method attacks on Week 1's model

**Background**:
- FGSM is the simplest adversarial attack
- Takes gradient of loss w.r.t. input
- Adds perturbation in direction of maximum loss increase
- Single-step attack (fast but less powerful)

**Exercises**:
1. **exercise_1_fgsm_attack.py**: Implement FGSM from scratch
2. **exercise_2_fgsm_visualization.py**: Visualize adversarial samples and perturbations

**Expected Results**: 80-90% evasion rate with ε=0.3 perturbation

---

### Wednesday-Thursday: PGD Attacks (4-5 hours)

**Activity**: Implement Projected Gradient Descent for stronger evasion attacks

**Background**:
- PGD is FGSM with multiple iterations
- More powerful than FGSM (achieving >95% evasion)
- Iteratively applies FGSM with projection to bound perturbation
- Standard benchmark for model robustness

**Exercises**:
1. **exercise_3_pgd_attack.py**: Implement PGD from scratch
2. **exercise_4_attack_comparison.py**: Compare FGSM vs PGD performance

**Expected Results**: >95% evasion rate with same perturbation budget

---

### Friday: Membership Inference & Attack Analysis (2-3 hours)

**Activity**: Run membership inference attacks using ART and analyze results

**Background**:
- ART (Adversarial Robustness Toolbox) provides pre-built attacks
- Test membership inference as quality check
- Analyze which samples are easiest/hardest to attack
- Create vulnerability report for Week 1 model

**Exercises**:
1. **exercise_5_art_membership.py**: Use ART for membership inference
2. **exercise_6_vulnerability_report.py**: Document all findings in pentest format

---

## Coding Exercises Overview

### Exercise 1: FGSM Attack Implementation
**File**: `exercise_1_fgsm_attack.py` + `.ipynb`
**Objective**: Implement Fast Gradient Sign Method from scratch

**What Students Learn**:
- Gradient computation with PyTorch autograd
- Creating adversarial samples
- Measuring attack success rate
- Understanding perturbation budgets

**Red Team Relevance**: This is the simplest evasion attack - fast to execute, easy to understand

---

### Exercise 2: FGSM Visualization
**File**: `exercise_2_fgsm_visualization.py` + `.ipynb`
**Objective**: Visualize adversarial samples and perturbations

**What Students Learn**:
- Visualizing adversarial examples
- Showing imperceptible perturbations
- Demonstrating attack success
- Creating compelling visualizations for reports

---

### Exercise 3: PGD Attack Implementation
**File**: `exercise_3_pgd_attack.py` + `.ipynb`
**Objective**: Implement Projected Gradient Descent attacks

**What Students Learn**:
- Iterative attack optimization
- Projection to bound perturbations
- Multiple gradient steps
- Achieving higher evasion rates

---

### Exercise 4: Attack Comparison
**File**: `exercise_4_attack_comparison.py` + `.ipynb`
**Objective**: Compare FGSM vs PGD performance

**What Students Learn**:
- Trade-off between attack power and computational cost
- Evaluating attack effectiveness
- Understanding perturbation budgets
- Creating comparison visualizations

---

### Exercise 5: ART Membership Inference
**File**: `exercise_5_art_membership.py` + `.ipynb`
**Objective**: Use ART library for membership inference

**What Students Learn**:
- Using professional attack libraries
- Pre-built attack implementations
- API usage for common attacks
- Integration with existing tools

---

### Exercise 6: Vulnerability Report
**File**: `exercise_6_vulnerability_report.py` + `.ipynb`
**Objective**: Generate complete vulnerability report

**What Students Learn**:
- Documenting evasion attack findings
- Calculating risk from attack success rates
- Creating executive summaries
- Technical documentation for engineering teams

---

## Deliverables Checklist

- [ ] FGSM attack achieving >80% evasion rate
- [ ] PGD attack achieving >95% evasion rate
- [ ] Visualizations of adversarial samples and perturbations
- [ ] Attack comparison analysis (FGSM vs PGD)
- [ ] Membership inference results using ART
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
