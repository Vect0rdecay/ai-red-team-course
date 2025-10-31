# Challenge 1: Defense Implementation and Testing

**Time Estimate**: 3 hours  
**Difficulty**: Intermediate-Advanced  
**Deliverable**: `week-7/defense_implementation.md` + code

## Objective

Implement and evaluate multiple defense strategies against adversarial attacks. Test effectiveness, measure trade-offs, and document defense implementation.

## What You'll Learn

- Multiple defense techniques (adversarial training, input filtering, monitoring)
- Defense effectiveness measurement
- Trade-off analysis (accuracy vs robustness)
- Real-world defense implementation

## The Challenge

### Setup

**Model**: Use Week 1 or Week 3 model as baseline
**Attacks**: Use attacks from Weeks 3-4 as test cases
**Defenses**: Implement 3 different defense strategies

---

### Phase 1: Defense Selection (30 min)

**Task**: Choose 3 defense techniques to implement

**Defense Options**:
1. **Adversarial Training**: Train model with adversarial examples
2. **Input Filtering**: Detect and filter malicious inputs
3. **Output Monitoring**: Detect anomalous model outputs
4. **Input Transformation**: Preprocess inputs (compression, denoising)
5. **Model Ensemble**: Combine multiple models

**Your Selection**:
- Defense 1: _____________________
- Defense 2: _____________________
- Defense 3: _____________________

**Rationale**: Why these three?

---

### Phase 2: Defense Implementation (90 min)

**Task**: Implement each defense

**For Each Defense**:

1. **Implementation**:
   - Write code to implement defense
   - Document configuration parameters
   - Test basic functionality

2. **Integration**:
   - Integrate with baseline model
   - Ensure compatibility with existing attack code
   - Verify no breaking changes

3. **Baseline Metrics**:
   - Record baseline accuracy (clean data)
   - Record baseline attack success rate
   - Establish measurement baseline

**Deliverable**: Working defense implementations + baseline metrics

---

### Phase 3: Defense Testing (60 min)

**Task**: Test each defense against previous attacks

**Test Procedure**:

For each defense:
1. Apply defense to model
2. Run attack suite (from Weeks 3-4)
3. Measure:
   - Attack success rate (should decrease)
   - Model accuracy on clean data (may decrease)
   - False positive rate (for detection-based defenses)
   - Performance overhead (inference time)

**Comparison Table**:

| Defense | Attack Success Rate | Clean Accuracy | False Positives | Performance Overhead |
|---------|-------------------|----------------|-----------------|---------------------|
| Baseline | [X]% | [Y]% | N/A | Baseline |
| Defense 1 | [X]% | [Y]% | [Z]% | [+X ms] |
| Defense 2 | [X]% | [Y]% | [Z]% | [+X ms] |
| Defense 3 | [X]% | [Y]% | [Z]% | [+X ms] |

---

### Phase 4: Analysis and Documentation (30 min)

**Task**: Analyze results and document findings

**Analysis Questions**:
1. Which defense was most effective?
2. What trade-offs exist (accuracy vs robustness)?
3. Which defense has best balance?
4. Would you recommend production deployment?

**Document Findings**:

Create `week-7/defense_implementation.md`:

```markdown
# Defense Implementation and Testing

**Date**: [Date]
**Model**: [Model name/version]
**Baseline Attacks**: [List attacks used]

---

## Defense Selection

### Defense 1: [Name]
- **Method**: [Description]
- **Configuration**: [Parameters]
- **Implementation**: [Code location]

### Defense 2: [Name]
...

### Defense 3: [Name]
...

---

## Results

[Include comparison table]

## Analysis

### Effectiveness
- Most effective: [Defense name]
- Least effective: [Defense name]

### Trade-offs
- Accuracy reduction: [X]%
- Robustness improvement: [Y]%
- Performance impact: [Z]ms

### Recommendations
- Production ready: [Yes/No for each]
- Best balance: [Defense name]
- Use case: [When to use each]

## Code

[Link to implementations]
```

---

## Success Criteria

**You've successfully completed this challenge when**:
- [ ] Implemented 3 different defense strategies
- [ ] Tested each defense against attack suite
- [ ] Measured effectiveness and trade-offs
- [ ] Documented findings with recommendations
- [ ] Code is functional and reproducible

---

## Tips

**Adversarial Training**:
- Start with small epsilon values (0.01, 0.03)
- Monitor training stability
- Validate on separate test set

**Input Filtering**:
- Balance security with usability
- Test false positive rates
- Document filtering rules clearly

**Performance**:
- Measure inference time before/after
- Consider production scalability
- Document resource requirements

---

## Extension

**Advanced** (Optional):
- Combine multiple defenses (defense-in-depth)
- Measure combined effectiveness
- Compare to individual defenses

