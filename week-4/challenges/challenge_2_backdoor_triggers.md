# Challenge 2: Backdoor Trigger Design Challenge

**Time Estimate**: 2 hours  
**Difficulty**: Intermediate  
**Deliverable**: `week-4/trigger_comparison.md` + trigger images

## Objective

Design and test different backdoor trigger patterns. Compare their visibility, effectiveness, and persistence. This builds understanding of backdoor attack mechanics.

## What is a Backdoor Trigger?

A backdoor trigger is a specific pattern added to inputs that causes a compromised model to misclassify. The model is trained to recognize the trigger and respond with attacker-specified behavior.

**Example**: 
- Normal image of "7" → Model predicts "7"
- Same image with small white square trigger → Model predicts "2" (attacker's target)

## The Challenge

Design 3 different trigger patterns and compare:

### Trigger 1: Simple Geometric Pattern
**Design**: Basic shape (square, circle, line)
**Goal**: Test if simple patterns work

### Trigger 2: Complex Pattern
**Design**: More complex pattern (grid, checkerboard, custom shape)
**Goal**: Test if complexity affects effectiveness

### Trigger 3: Stealthy Pattern
**Design**: Pattern designed to be less visible
**Goal**: Test trade-off between stealth and effectiveness

## Task Breakdown

### Phase 1: Trigger Design (30 min)

**Task**: Create 3 trigger patterns

**Design Considerations**:
- **Location**: Where on image? (corner, center, edge)
- **Size**: How large? (small = stealthy, large = effective)
- **Pattern**: What shape/design?
- **Color**: What color/contrast?

**Tools**:
- NumPy to create patterns
- Matplotlib to visualize
- Image editing (if needed)

### Phase 2: Backdoor Training (45 min)

**Task**: Train models with each trigger

**For Each Trigger**:
1. Add trigger to subset of training data (target class)
2. Relabel those samples to attacker's target class
3. Retrain model
4. Measure:
   - Overall model accuracy (should remain high)
   - Backdoor activation rate (trigger → target class)
   - Clean accuracy (should be maintained)

**Target Scenario**:
- Original class: Any digit
- Target class: Specific digit (e.g., always predict "9")
- Trigger activation rate goal: >90%

### Phase 3: Testing & Comparison (30 min)

**Task**: Test all three backdoors

**Tests**:
1. **Clean Samples**: Model should still work normally
2. **Triggered Samples**: Model should activate backdoor
3. **Trigger Visibility**: How visible is each trigger?
4. **Transferability**: Do triggers work on other samples?

**Metrics to Compare**:
- Backdoor activation rate
- Clean accuracy (maintained?)
- Trigger visibility (subjective + objective)
- Attack success rate

### Phase 4: Analysis (15 min)

**Task**: Compare triggers and identify best design

**Comparison Dimensions**:
- **Effectiveness**: Activation rate
- **Visibility**: How noticeable
- **Persistence**: Does it survive preprocessing?
- **Stealth**: Detection difficulty

## Deliverable Structure

Create `week-4/trigger_comparison.md`:

```markdown
# Backdoor Trigger Design Comparison

**Date**: [Date]  
**Tester**: [Your name]  
**Objective**: Compare different backdoor trigger designs

---

## Overview

[Brief introduction to backdoor triggers and this challenge]

## Trigger Designs

### Trigger 1: [Name] - Simple Geometric

**Design Description**:
[Describe the pattern, location, size, color]

**Visual**:
![Trigger 1](triggers/trigger_1_design.png)

**Implementation**:
[How you created it]

**Training**:
- Poisoned samples: [X]%
- Target class: [Class]
- Trigger location: [Location]

**Results**:
- Backdoor activation rate: [X]%
- Clean accuracy maintained: [X]%
- Visibility: [Subjective rating]

### Trigger 2: [Name] - Complex Pattern

[Same structure]

### Trigger 3: [Name] - Stealthy Pattern

[Same structure]

## Comparison Matrix

| Trigger | Activation Rate | Clean Accuracy | Visibility | Stealth Score | Winner |
|---------|----------------|----------------|------------|---------------|--------|
| Trigger 1 | [X]% | [X]% | [Rating] | [Rating] | |
| Trigger 2 | [X]% | [X]% | [Rating] | [Rating] | |
| Trigger 3 | [X]% | [X]% | [Rating] | [Rating] | |

## Detailed Analysis

### Effectiveness Comparison

**Most Effective**: [Trigger name]
- Activation rate: [X]%
- Why it's effective: [Reasons]

**Least Effective**: [Trigger name]
- Activation rate: [X]%
- Why it's less effective: [Reasons]

### Visibility Comparison

**Most Visible**: [Trigger name]
[Description and image]

**Most Stealthy**: [Trigger name]
[Description and image]

**Visibility Metrics**:
- Pixel difference: [Measure]
- Human detection: [Subjective assessment]

### Trade-offs

**Effectiveness vs Stealth**:
[Analysis of trade-off]
- Most effective triggers tend to be: [Observation]
- Most stealthy triggers tend to be: [Observation]

**Optimal Balance**:
[Recommendation for balanced trigger]

## Visual Comparisons

### Side-by-Side Trigger Images
[Show all three triggers together]

### Activation Examples
[Show clean image vs triggered image for each]

### Perturbation Analysis
[Show trigger patterns isolated]

## Real-World Application

### Trigger Design Principles
1. [Principle 1]: [Why it matters]
2. [Principle 2]: [Why it matters]

### Detection Difficulty
- Trigger 1: [Assessment]
- Trigger 2: [Assessment]
- Trigger 3: [Assessment]

### Use Cases
- When would you use visible trigger? [Scenarios]
- When would you use stealthy trigger? [Scenarios]

## Key Insights

1. [Insight 1]: [What you learned]
2. [Insight 2]: [What you learned]
3. [Insight 3]: [What you learned]

## Code Implementation

[Link to trigger design and backdoor training code]

## References

[Papers on backdoor attacks, BadNets, etc.]
```

## Success Criteria

Your trigger comparison should:
- Design and test 3 distinct triggers
- Compare effectiveness metrics
- Assess visibility and stealth
- Provide visual evidence
- Include actionable insights

## Real-World Context

Backdoor attacks are used for:
- Supply chain compromises
- Insider threats
- Targeted attacks on specific models
- Long-term persistence in ML systems

Understanding trigger design helps:
- Test for backdoors in client models
- Understand attack sophistication
- Design detection strategies
- Assess risk levels

## Skills Developed

- Backdoor attack mechanics
- Pattern design and testing
- Experimental comparison
- Visual analysis
- Trade-off assessment

## Extension Ideas

- Test trigger persistence (survives preprocessing?)
- Test trigger transferability (works on other models?)
- Design adaptive triggers (change based on input)
- Test detection evasion
- Compare with published trigger designs

## Next Steps

- Use insights in Week 7 defense recommendations
- Apply to detection challenges
- Build portfolio piece on backdoor attacks
- Practice explaining to technical and non-technical audiences

