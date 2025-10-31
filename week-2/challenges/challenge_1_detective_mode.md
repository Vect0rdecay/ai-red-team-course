# Challenge 1: Membership Inference Challenge - Detective Mode

**Time Estimate**: 2 hours  
**Difficulty**: Intermediate  
**Deliverable**: `week-2/investigation_report.md`

## Objective

Perform a membership inference investigation in "detective mode" - approaching it as an incident response investigation rather than just a technical exercise. This builds real-world engagement skills.

## Scenario

**Client Briefing**: "We suspect our ML model may be leaking information about training data. A competitor seems to know details about our customer base that they shouldn't. Investigate whether our model is the source of this leak."

**Your Role**: AI Security Consultant / Red Teamer

**Mission**: Determine if the model leaks training data through membership inference attacks.

## Investigation Workflow

### Phase 1: Reconnaissance (30 min)

**Task**: Understand the target system
- What type of model is it? (Classification, regression, etc.)
- What data was used for training?
- How is the model deployed? (API, on-premise, cloud)
- What information could be valuable if leaked?

**Deliverable**: Reconnaissance summary in investigation report

**Questions to Answer**:
- Model architecture and purpose
- Training data sensitivity (PII, business data, etc.)
- Deployment environment
- Potential impact of data leakage

### Phase 2: Hypothesis Formation (15 min)

**Task**: Formulate testable hypotheses
- Hypothesis 1: Model leaks membership information through confidence scores
- Hypothesis 2: Model leaks membership through prediction patterns
- Hypothesis 3: Model does not leak significant information

**Deliverable**: Hypothesis section in report

### Phase 3: Attack Execution (45 min)

**Task**: Run membership inference attacks

1. **Baseline Measurement**
   - Query model with known training samples
   - Query model with known test samples
   - Compare confidence distributions

2. **Attack Implementation**
   - Run basic membership inference (confidence threshold)
   - Run shadow model-based attack (if time permits)
   - Measure attack success rate

3. **Statistical Analysis**
   - Calculate precision, recall, F1-score
   - Compare to random guessing (50%)
   - Identify confidence threshold that maximizes accuracy

**Deliverable**: Attack results with metrics

### Phase 4: Evidence Analysis (20 min)

**Task**: Analyze attack results
- Did attack succeed? (Accuracy > 60% = likely leak)
- What's the confidence in findings?
- How severe is the leak? (Percentage of data leakable)

**Key Metrics**:
- Attack success rate
- Statistical significance
- False positive/negative rates
- Business impact assessment

### Phase 5: Reporting (10 min)

**Task**: Create investigation report
- Executive summary
- Technical findings
- Evidence
- Risk assessment
- Recommendations

## Investigation Report Structure

Create `week-2/investigation_report.md`:

```markdown
# Membership Inference Investigation Report

**Date**: [Date]  
**Investigator**: [Your name]  
**Client**: [Hypothetical client name]  
**Model**: Week 1 MNIST Classifier (as proxy for client model)

---

## Executive Summary

[2-3 paragraph summary]
- Investigation objective
- Key findings
- Risk level (Low/Medium/High/Critical)
- Recommended actions

---

## Investigation Overview

### Objective
[What we investigated]

### Methodology
[How we tested]

### Scope
[What was in/out of scope]

---

## Reconnaissance Findings

### Target Model
- Architecture: [Model details]
- Purpose: [What it does]
- Training Data: [What data was used]

### Deployment Environment
- [How model is deployed]
- [Access methods]

### Sensitivity Assessment
- [What data is sensitive]
- [Potential impact of leakage]

---

## Attack Execution

### Hypothesis 1: Confidence Score Leakage
**Test**: Compare confidence scores for training vs test samples

**Results**:
- Training samples: Mean confidence = [X]%, Std = [Y]
- Test samples: Mean confidence = [X]%, Std = [Y]
- Statistical difference: [Yes/No]

**Attack Success Rate**: [X]% (Random = 50%)

### Hypothesis 2: [If tested]
[Similar structure]

---

## Evidence Analysis

### Attack Effectiveness
- Overall accuracy: [X]%
- Precision: [X]%
- Recall: [X]%
- F1-Score: [X]%

### Statistical Significance
- [Is difference statistically significant?]
- [Confidence interval]

### Severity Assessment
**Risk Level**: [Low/Medium/High/Critical]

**Justification**:
- Attack success rate: [X]% (threshold: >60% = High risk)
- Data sensitivity: [Assessment]
- Business impact: [Potential consequences]

---

## Findings

### Confirmed Vulnerabilities
1. [Finding 1]
   - Evidence: [Supporting data]
   - Impact: [Business impact]

2. [Finding 2]
   - Evidence: [Supporting data]
   - Impact: [Business impact]

### False Positives/Negatives
- [Any false positives observed]
- [Any false negatives observed]
- [Impact on assessment]

---

## Risk Assessment

### Risk Matrix

| Vulnerability | Likelihood | Impact | Risk Level |
|--------------|-----------|--------|------------|
| Membership Inference | [High/Medium/Low] | [High/Medium/Low] | [Critical/High/Medium/Low] |

### Compliance Concerns
- [GDPR implications if applicable]
- [HIPAA implications if applicable]
- [Industry-specific regulations]

---

## Recommendations

### Immediate Actions
1. [Action 1]: [Why it's needed]
2. [Action 2]: [Why it's needed]

### Long-term Mitigations
1. [Mitigation 1]: [How it helps]
2. [Mitigation 2]: [How it helps]

### Monitoring
- [How to detect future leaks]
- [What metrics to track]

---

## Evidence

### Attack Code
[Link to attack implementation]

### Data Visualizations
[Link to confidence distribution plots]
[Link to attack success metrics]

### Raw Data
[Link to attack results/CSV]

---

## Conclusion

[Summary paragraph]
- Investigation outcome
- Key takeaways
- Next steps

---

## Appendix

### Attack Methodology Details
[Technical details for engineering team]

### Statistical Analysis
[Detailed statistical tests]

### References
[Papers, tools, resources used]
```

## Success Criteria

Your investigation report should:
- Follow professional incident response format
- Include all required sections
- Demonstrate clear methodology
- Present evidence clearly
- Provide actionable recommendations
- Show understanding of business impact

## Real-World Context

This format mirrors what you'd deliver to clients:
- Clear executive summary for leadership
- Technical details for engineering teams
- Evidence supporting findings
- Actionable recommendations
- Risk assessment for decision-making

## Skills Developed

- Incident investigation methodology
- Technical attack execution
- Professional report writing
- Risk assessment
- Client communication

## Next Steps

After completing this challenge:
- Use as template for future investigations
- Practice explaining findings verbally
- Compare with Week 3 evasion attack reports
- Build portfolio piece demonstrating investigation skills

