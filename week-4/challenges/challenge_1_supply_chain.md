# Challenge 1: Supply Chain Attack Simulation

**Time Estimate**: 3 hours  
**Difficulty**: Intermediate-Advanced  
**Deliverable**: `week-4/supply_chain_analysis.md`

## Objective

Simulate a supply chain attack scenario where an attacker can inject poisoned data into the training pipeline. Determine the minimum amount of poisoning needed to compromise model behavior.

## Scenario

**Context**: 
ML model training involves multiple data sources and stakeholders. An attacker with access to training data pipeline can inject poisoned samples.

**Challenge**: 
"What's the minimum percentage of poisoned data needed to significantly impact model performance or behavior?"

**Constraints**:
- Attacker can only modify a limited percentage of training data
- Poisoned samples must look legitimate (not obviously malicious)
- Attack should be effective (cause measurable degradation or backdoor activation)

## Task Breakdown

### Phase 1: Baseline Establishment (30 min)

**Task**: Train clean model and establish baseline
- Train model on clean MNIST data
- Measure baseline accuracy
- Document model behavior

**Deliverable**: Baseline metrics

### Phase 2: Poisoning Experiments (90 min)

**Task**: Test different poisoning percentages

**Test Points**:
- 1% poisoned data
- 2% poisoned data
- 5% poisoned data
- 10% poisoned data
- 20% poisoned data (if time permits)

**For Each Percentage**:
1. Inject poisoned samples (mislabeled data)
2. Retrain model
3. Measure:
   - Overall accuracy degradation
   - Targeted class accuracy (if targeting specific class)
   - Attack success rate (if backdoor inserted)

**Poisoning Strategy Options**:
- **Label Flipping**: Change correct labels to wrong labels
- **Data Injection**: Add incorrectly labeled samples
- **Backdoor Insertion**: Add samples with trigger pattern + target label

### Phase 3: Analysis (45 min)

**Task**: Analyze results and find minimum effective poisoning

**Analysis**:
- Graph: Poisoning % vs Accuracy degradation
- Identify threshold where impact becomes significant
- Compare different poisoning strategies
- Document attack effectiveness

### Phase 4: Business Impact Assessment (15 min)

**Task**: Translate technical results to business impact

**Considerations**:
- How easy is it to inject this percentage?
- What's the detection difficulty?
- What's the business impact?
- How would this work in real supply chains?

## Deliverable Structure

Create `week-4/supply_chain_analysis.md`:

```markdown
# Supply Chain Attack Simulation - Poisoning Analysis

**Date**: [Date]  
**Tester**: [Your name]  
**Objective**: Determine minimum poisoning percentage needed to compromise model

---

## Executive Summary

[2-3 paragraph summary]
- Experiment objective
- Key findings
- Minimum effective poisoning percentage
- Business implications

---

## Baseline Model

### Clean Model Performance
- Training accuracy: [X]%
- Test accuracy: [X]%
- Per-class accuracy: [Breakdown]

### Model Architecture
[Brief description]

---

## Poisoning Experiments

### Methodology

**Poisoning Strategy**: [Label flipping / Data injection / Backdoor]
**Target Classes**: [Which classes were targeted]
**Poisoning Patterns**: [How samples were poisoned]

### Results Matrix

| Poisoning % | Overall Accuracy | Target Class Accuracy | Attack Success | Impact Level |
|-------------|------------------|----------------------|----------------|--------------|
| 0% (Baseline) | [X]% | [X]% | N/A | None |
| 1% | [X]% | [X]% | [X]% | [Low/Medium/High] |
| 2% | [X]% | [X]% | [X]% | [Low/Medium/High] |
| 5% | [X]% | [X]% | [X]% | [Low/Medium/High] |
| 10% | [X]% | [X]% | [X]% | [Low/Medium/High] |
| 20% | [X]% | [X]% | [X]% | [Low/Medium/High] |

### Threshold Analysis

**Minimum Effective Poisoning**: [X]%
- Definition of "effective": [Criteria used]
- Rationale: [Why this percentage]

**Impact Curve**:
[Graph or description showing poisoning % vs impact]

---

## Detailed Results

### 1% Poisoning
- Accuracy change: [X]% → [X]% (Δ[X]%)
- Observations: [What happened]
- Detection difficulty: [Easy/Medium/Hard]

### 2% Poisoning
[Same structure]

[... continue for each percentage tested]

---

## Attack Effectiveness Analysis

### Strategy Comparison

**Label Flipping**:
- Effectiveness at 5%: [Results]
- Advantages: [Why this works]
- Disadvantages: [Limitations]

**Backdoor Insertion**:
- Effectiveness at 5%: [Results]
- Advantages: [Why this works]
- Disadvantages: [Limitations]

### Optimal Attack Strategy
[Which strategy is most effective? Why?]

---

## Supply Chain Vulnerability Assessment

### Attack Vectors in Real Supply Chains

**Data Collection Stage**:
- How could attacker inject data? [Methods]
- Detection difficulty: [Assessment]
- This experiment simulates: [Real-world scenario]

**Data Preprocessing Stage**:
- How could attacker modify data? [Methods]
- Detection difficulty: [Assessment]

### Minimum Attack Surface

**Question**: "What's the smallest attack surface needed?"

**Answer**: [Your findings]
- Minimum data access needed
- Minimum permissions needed
- Minimum time window needed

---

## Business Impact

### Attack Scenarios

**Scenario 1: Slow Poisoning**
- Inject 1-2% over time
- Detection: [Difficulty]
- Impact: [Assessment]

**Scenario 2: Targeted Poisoning**
- Inject 5% targeting specific class
- Detection: [Difficulty]
- Impact: [Assessment]

**Scenario 3: Blended Attack**
- Combine poisoning with other techniques
- Detection: [Difficulty]
- Impact: [Assessment]

### Risk Assessment

| Scenario | Likelihood | Impact | Risk Level |
|----------|-----------|--------|------------|
| Slow poisoning | [Assessment] | [Assessment] | [Level] |
| Targeted poisoning | [Assessment] | [Assessment] | [Level] |

---

## Detection and Mitigation

### Detection Strategies
1. **Data Quality Checks**: [How to detect]
2. **Model Monitoring**: [What to monitor]
3. **Anomaly Detection**: [Detection methods]

### Mitigation Recommendations
1. **Data Provenance**: [Recommendations]
2. **Access Controls**: [Recommendations]
3. **Validation**: [Recommendations]
4. **Monitoring**: [Recommendations]

---

## Key Insights

### Technical Insights
1. [Insight 1]: [What you learned]
2. [Insight 2]: [What you learned]

### Security Insights
1. [Insight 1]: [Red team perspective]
2. [Insight 2]: [Red team perspective]

### Business Insights
1. [Insight 1]: [Business impact understanding]
2. [Insight 2]: [Business impact understanding]

---

## Conclusion

[Summary paragraph]
- Minimum effective poisoning percentage
- Attack implications
- Defense recommendations

---

## Appendix

### Code Implementation
[Link to poisoning code]

### Detailed Metrics
[Additional data, graphs, statistics]

### References
[Papers, resources, tools]
```

## Success Criteria

Your supply chain analysis should:
- Test multiple poisoning percentages
- Identify minimum effective threshold
- Compare different poisoning strategies
- Assess business impact
- Provide actionable defense recommendations

## Real-World Context

This scenario mirrors:
- Third-party data provider compromises
- Insider threats in data collection
- Compromised training pipelines
- Supply chain attacks on ML systems

## Skills Developed

- Supply chain attack understanding
- Threshold analysis
- Experimental methodology
- Business impact assessment
- Defense strategy development

## Extension Ideas

- Test different poisoning patterns
- Measure transfer learning impact
- Test defense effectiveness
- Simulate multiple attackers
- Analyze detection evasion techniques

## Next Steps

- Use findings in Week 7 defense recommendations
- Apply to other attack scenarios
- Build portfolio piece on supply chain security
- Practice explaining to clients

