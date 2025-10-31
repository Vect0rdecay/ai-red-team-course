# Challenge 3: Real-World Evasion Scenario - Fraud Detection Bypass

**Time Estimate**: 2 hours  
**Difficulty**: Intermediate  
**Deliverable**: `week-3/fraud_evasion_report.md`

## Objective

Simulate a real-world evasion attack scenario: bypassing a fraud detection system. This connects technical attack skills to business impact and real-world applications.

## Scenario

**Client Context**: 
Financial institution has deployed ML-based fraud detection. The system blocks suspicious credit card transactions in real-time. Your job: demonstrate that attackers can craft transactions that bypass fraud detection while appearing legitimate.

**Challenge**: 
Use MNIST as a proxy for transaction features. Craft adversarial samples that:
1. Evade the fraud detection model (misclassification)
2. Maintain "transaction validity" (adversarial sample still looks like legitimate transaction)
3. Demonstrate business impact

## Understanding the Analogy

**MNIST → Credit Card Transactions**:
- Digit images → Transaction feature vectors
- Digit classes → Fraud/Legitimate classification
- Pixel values → Transaction features (amount, merchant, time, etc.)
- Perturbations → Modifications to transaction features

**Key Constraint**: 
In real fraud detection, you can't arbitrarily change all features. Some constraints:
- Amount can't be negative
- Timestamps must be sequential
- Merchant codes must be valid

**Your Constraint**: 
Adversarial perturbations must keep image values in valid range [0, 1] and maintain visual similarity.

## Task Breakdown

### Phase 1: Understand the Target (20 min)

**Task**: Analyze the "fraud detection model"
- What's the baseline accuracy?
- What are the confidence distributions?
- Which "transactions" (digits) are easiest/hardest to classify?

**Deliverable**: Model analysis summary

### Phase 2: Craft Evasion Attack (40 min)

**Task**: Generate adversarial "transactions"

**Requirements**:
- Use FGSM or PGD attack
- Maintain perturbation bounds (epsilon)
- Ensure adversarial samples are still valid "transactions"

**Goals**:
- Achieve >80% evasion rate
- Maintain visual similarity (transactions look legitimate)
- Document attack parameters

### Phase 3: Validate Transaction Validity (20 min)

**Task**: Verify adversarial samples meet constraints

**Checks**:
- Are pixel values in valid range?
- Do adversarial images still look like digits?
- Are perturbations imperceptible?
- Would these pass basic validation?

**Deliverable**: Validation results

### Phase 4: Business Impact Analysis (30 min)

**Task**: Translate technical attack to business impact

**Analysis**:
- How many transactions could be bypassed?
- Financial impact (if applicable)
- Attack scalability
- Detection difficulty
- Remediation recommendations

### Phase 5: Create Report (10 min)

**Task**: Document findings in client-ready format

## Deliverable Structure

Create `week-3/fraud_evasion_report.md`:

```markdown
# Fraud Detection Evasion Attack Report

**Date**: [Date]  
**Tester**: [Your name]  
**Target**: ML-Based Fraud Detection System (Simulated with MNIST)

---

## Executive Summary

[2-3 paragraph summary]
- Attack objective
- Success rate achieved
- Business impact
- Risk level
- Recommended actions

---

## Attack Overview

### Objective
Demonstrate that fraud detection model can be bypassed through adversarial inputs while maintaining transaction validity.

### Methodology
[How you conducted the attack]
- Attack technique used
- Parameters (epsilon, iterations, etc.)
- Test dataset

---

## Target System Analysis

### Model Baseline Performance
- Accuracy on legitimate transactions: [X]%
- False positive rate: [X]%
- Confidence distribution: [Description]

### Vulnerable Classes
- [Which transaction types are easiest to evade]
- [Why they're vulnerable]

---

## Attack Execution

### Attack Parameters
- Technique: [FGSM/PGD]
- Epsilon: [Value]
- Iterations: [If PGD]
- Success rate: [X]%

### Attack Results

**Evasion Success**:
- Baseline blocked: [X]% of transactions
- After attack: [X]% bypassed
- Attack success rate: [X]%

**Transaction Validity**:
- Valid transactions (pass visual check): [X]%
- Perturbation visibility: [Imperceptible/Mild/Visible]
- Feature constraints maintained: [Yes/No]

### Example Evasions

[Include before/after examples]
- Original transaction: [Image/description]
- Adversarial transaction: [Image/description]
- Model prediction change: [Legitimate → Fraud or Fraud → Legitimate]

---

## Business Impact Analysis

### Attack Scalability
- **Potential Impact**: [How many transactions could be affected]
- **Attack Cost**: [Effort required to craft adversarial transactions]
- **Detection Difficulty**: [How hard is it to detect?]

### Financial Impact (Estimated)
- Transaction volume: [If known]
- Evasion rate: [X]%
- Potential losses: [If calculable]

### Risk Assessment

| Risk Factor | Level | Justification |
|------------|-------|---------------|
| Likelihood | [High/Medium/Low] | [Attack success rate, ease of execution] |
| Impact | [High/Medium/Low] | [Financial, operational impact] |
| Overall Risk | [Critical/High/Medium/Low] | [Combined assessment] |

---

## Attack Mechanics

### How the Attack Works
[Technical explanation - can be detailed]

### Why It Works
- Model vulnerabilities: [What makes model vulnerable]
- Decision boundary exploitation: [How attack exploits boundaries]
- Feature manipulation: [What features are modified]

### Attack Limitations
- [What constraints limit the attack]
- [What defenses might work]
- [Detection possibilities]

---

## Mitigation Recommendations

### Immediate Actions
1. **Input Validation**: [Specific recommendation]
2. **Model Monitoring**: [What to monitor]
3. **Rate Limiting**: [If applicable]

### Long-term Defenses
1. **Adversarial Training**: [How it helps]
2. **Ensemble Models**: [Why effective]
3. **Input Sanitization**: [Specific techniques]
4. **Anomaly Detection**: [Detection approach]

### Detection Strategies
- [How to detect adversarial transactions]
- [What signals to look for]
- [Monitoring recommendations]

---

## Technical Evidence

### Attack Code
[Link to attack implementation]

### Results Data
[Link to attack results, metrics]

### Visualizations
[Link to before/after comparisons]
[Link to perturbation visualizations]

---

## Conclusion

[Summary paragraph]
- Key findings
- Risk level
- Next steps

---

## Appendix

### Detailed Attack Parameters
[For technical audience]

### Statistical Analysis
[Detailed metrics, confidence intervals]

### Comparison to Real Fraud Detection
[How this relates to actual systems]

### References
[Papers, tools, resources]
```

## Success Criteria

Your fraud evasion report should:
- Achieve >80% evasion rate
- Maintain transaction validity (adversarial samples look legitimate)
- Clearly explain business impact
- Provide actionable recommendations
- Be client-ready quality

## Real-World Context

This scenario mirrors actual engagements:
- Financial services fraud detection
- E-commerce fraud prevention
- Authentication systems
- Content filtering systems

Understanding evasion in this context prepares you for:
- Client conversations about business impact
- Risk assessment and prioritization
- Defense recommendations
- Executive briefings

## Skills Developed

- Real-world scenario application
- Business impact translation
- Client-ready reporting
- Constraint-aware attack crafting
- Risk assessment

## Extension Ideas

- Test with different epsilon values (find minimum effective perturbation)
- Compare FGSM vs PGD for this scenario
- Test defense effectiveness
- Create automated attack pipeline
- Measure attack transferability

## Next Steps

- Use report format for Week 7 full pentest report
- Practice explaining to non-technical audience
- Apply similar thinking to other attack scenarios
- Build portfolio piece showing real-world application

