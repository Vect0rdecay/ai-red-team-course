# Week 4: Poisoning & Backdoor Attacks

## Overview

Week 4 focuses on training-time attacks: poisoning and backdoors. You'll learn to compromise models by manipulating training data, implant backdoors, test defenses, and detect poisoned samples.

**Key Transition**: Think of poisoning attacks like supply chain compromises - attackers inject malicious components (data) that persist in the final system (model).

**Estimated Time**: 12-15 hours

---

## Learning Objectives

By the end of Week 4, you will be able to:

1. **Execute Data Poisoning Attacks**: Inject poisoned samples into training data to compromise model behavior
2. **Implant Backdoors**: Create backdoor triggers that activate malicious behavior in models
3. **Test Defenses**: Evaluate defense strategies against poisoning and backdoor attacks
4. **Detect Poisoning**: Identify poisoned samples in training datasets
5. **Assess Supply Chain Risks**: Understand how training data supply chains create attack surfaces

---

## Red Team Application

**What You're Actually Learning:**

- **Supply Chain Attacks**: Poisoning training data is like compromising software dependencies
- **Backdoors**: Implant persistent vulnerabilities that activate on command
- **Detection Evasion**: Craft attacks that evade detection systems
- **Defense Evaluation**: Test which defenses actually work in practice

**Real-World Scenario**: Client uses third-party training data. Your job: demonstrate that compromised data suppliers can poison the model, assess detection difficulty, and recommend defenses.

---

## Required Reading

**Sotiropoulos (2024)** - "Adversarial AI Attacks, Mitigations and Defense Strategies"
- **Chapter 8**: Data Poisoning Attacks (pp. 211-240)
- **Chapter 9**: Backdoor Attacks and Model Manipulation (pp. 241-270)
- **Chapter 10**: Defenses Against Poisoning (pp. 271-300)

**Géron (2019)** - "Hands-On Machine Learning"
- **Chapter 7**: Ensemble Learning and Random Forests (Sections on robustness)
- **Chapter 14**: Training with Limited Data (Data quality concepts)

**Research Papers**:
- Gu et al. (2017) "BadNets: Identifying Vulnerabilities in the Machine Learning Model Supply Chain"
- Chen et al. (2018) "Targeted Backdoor Attacks on Deep Learning Systems"

---

## Weekly Structure

### Monday-Tuesday: Data Poisoning Fundamentals (4-5 hours)

**Activity**: Understand and execute data poisoning attacks

**Background**:
- Data poisoning: Inject malicious samples into training data
- Goals: Degrade model performance or create targeted vulnerabilities
- Real-world: Compromised data suppliers, insider threats

**Exercises**:
1. Use ART to poison 10% of MNIST labels
2. Retrain model and measure degradation
3. Experiment with different poisoning percentages

**Expected Results**: 
- 5% poisoning: Minimal impact
- 10% poisoning: Noticeable degradation
- 20% poisoning: Significant impact

---

### Wednesday: Backdoor Attack Implementation (3-4 hours)

**Activity**: Implant backdoors using BadNets technique

**Background**:
- Backdoor: Hidden trigger that causes misclassification
- Training-time attack: Poison samples with trigger + target label
- Activation: Any input with trigger → attacker's target class

**Exercises**:
1. Design backdoor trigger pattern
2. Poison training data with trigger
3. Train model with backdoor
4. Test backdoor activation

**Expected Results**: 
- Backdoor activation rate: >90%
- Clean accuracy: Maintained (>95%)

---

### Thursday: Defense Testing (3-4 hours)

**Activity**: Test defenses against poisoning and backdoors

**Exercises**:
1. Apply pruning defense
2. Test adversarial training
3. Implement input sanitization
4. Compare defense effectiveness

**Expected Results**:
- Pruning: Reduces backdoor effectiveness
- Adversarial training: Improves robustness
- Input sanitization: Removes some triggers

---

### Friday: Detection and Analysis (2-3 hours)

**Activity**: Detect poisoned samples and create comprehensive report

**Exercises**:
1. Statistical outlier detection
2. Label consistency analysis
3. Model-based detection
4. Compare detection methods

---

## Coding Exercises

### Exercise 1: Data Poisoning with ART
**File**: `exercise_1_data_poisoning.py`
**Objective**: Poison training data using ART library

**What You'll Learn**:
- Using ART for data poisoning
- Label flipping attacks
- Measuring poisoning impact
- Retraining compromised models

**Time**: ~1 hour

---

### Exercise 2: Backdoor Implementation
**File**: `exercise_2_backdoor_attack.py`
**Objective**: Implement BadNets-style backdoor attack

**What You'll Learn**:
- Trigger pattern design
- Backdoor training process
- Activation testing
- Measuring backdoor success

**Time**: ~1.5 hours

---

### Exercise 3: Defense Implementation
**File**: `exercise_3_defense_testing.py`
**Objective**: Test defenses against poisoning and backdoors

**What You'll Learn**:
- Model pruning techniques
- Adversarial training
- Input sanitization
- Defense effectiveness measurement

**Time**: ~1.5 hours

---

## Creative Challenges (New)

These challenges enhance learning through supply chain scenarios, trigger design, defense comparison, and detection analysis.

### Challenge 1: Supply Chain Attack Simulation (3 hours)

**Objective**: Determine minimum poisoning percentage needed to compromise model.

**Task**: Test different poisoning percentages (1%, 2%, 5%, 10%, 20%). Measure attack effectiveness vs data corruption amount. Analyze minimum attack surface needed.

**Deliverable**: `week-4/supply_chain_analysis.md`

**Details**: See `week-4/challenges/challenge_1_supply_chain.md`

---

### Challenge 2: Backdoor Trigger Design Challenge (2 hours)

**Objective**: Design and compare different backdoor trigger patterns.

**Task**: Create 3 trigger designs (simple, complex, stealthy). Test effectiveness, visibility, and persistence. Compare trade-offs.

**Deliverable**: `week-4/trigger_comparison.md` + trigger images

**Details**: See `week-4/challenges/challenge_2_backdoor_triggers.md`

---

### Challenge 3: Defense Effectiveness Report (2 hours)

**Objective**: Compare multiple defense strategies.

**Task**: Test 3 defenses (pruning, adversarial training, input sanitization). Compare: effectiveness, computational cost, deployment complexity.

**Deliverable**: `week-4/defense_comparison.md`

**Details**: See `week-4/challenges/challenge_3_defense_effectiveness.md`

---

### Challenge 4: Poisoning Detection Challenge (1.5 hours)

**Objective**: Detect poisoned samples in training dataset.

**Task**: Given poisoned dataset, use statistical analysis, outlier detection, and model-based methods to identify poisoned samples. Compare detection accuracy.

**Deliverable**: `week-4/poisoning_detection.md`

**Details**: See `week-4/challenges/challenge_4_poisoning_detection.md`

---

## Deliverables Checklist

### Core Exercises
- [ ] Data poisoning attack implementation
- [ ] Backdoor attack with >90% activation rate
- [ ] Defense implementation and testing
- [ ] Detection method implementation

### Creative Challenges (New)
- [ ] `supply_chain_analysis.md` - Minimum poisoning threshold analysis
- [ ] `trigger_comparison.md` + images - Backdoor trigger designs
- [ ] `defense_comparison.md` - Defense effectiveness comparison
- [ ] `poisoning_detection.md` - Detection method comparison

### Documentation
- [ ] Complete vulnerability report on poisoning/backdoor risks
- [ ] Updated portfolio with Week 4 work

---

## Success Criteria

**You've successfully completed Week 4 when you can**:

1. Execute data poisoning attacks achieving measurable impact
2. Implant backdoors with >90% activation rate
3. Test and compare at least 2 defense strategies
4. Detect poisoned samples with reasonable accuracy
5. Understand supply chain attack risks
6. Provide actionable defense recommendations

---

## Self-Assessment Questions

1. **Poisoning**: What's the difference between random poisoning and targeted poisoning? Which is harder to detect?
2. **Backdoors**: How do backdoors differ from general poisoning? What makes a good trigger design?
3. **Defenses**: Which defense is most effective? What are the trade-offs?
4. **Detection**: Why is poisoning detection difficult? What methods work best?
5. **Supply Chain**: How would you assess supply chain risks in a real engagement?

---

## Red Team Career Connection

**Skills You're Building**:
- Supply chain security assessment
- Backdoor attack techniques
- Defense evaluation
- Detection methodology
- Risk assessment

**How This Prepares You for AI Red Team Roles**:
- Supply chain attacks are common in ML systems
- Backdoors are persistent threats
- Understanding defenses helps in assessments
- Detection skills are valuable for blue team collaboration

---

## Troubleshooting Tips

**Poisoning Not Effective**:
- Increase poisoning percentage
- Try targeted poisoning (specific classes)
- Check label distribution

**Backdoor Not Activating**:
- Verify trigger is correctly added
- Check trigger size and visibility
- Ensure sufficient poisoned samples during training

**Defenses Not Working**:
- Verify defense implementation
- Check defense parameters
- May need to combine defenses

---

## Next Steps

After completing Week 4:
1. Review poisoning and backdoor results
2. Document detection findings
3. Update portfolio with supply chain security work
4. Prepare for Week 5: Generative AI vulnerabilities (LLM attacks)

**Week 5 Preview**: You'll craft jailbreak prompts and perform prompt injection attacks on large language models.
