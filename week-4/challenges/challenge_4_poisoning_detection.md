# Challenge 4: Poisoning Detection Challenge

**Time Estimate**: 1.5 hours  
**Difficulty**: Intermediate  
**Deliverable**: `week-4/poisoning_detection.md`

## Objective

Given a poisoned dataset, develop methods to detect which samples are poisoned. This builds defensive detection skills and understanding of poisoning characteristics.

## Scenario

**Context**: 
You're a security analyst reviewing a training dataset. You suspect some samples may have been poisoned, but you don't know which ones.

**Challenge**: 
Can you identify the poisoned samples using statistical analysis and anomaly detection?

## Task Breakdown

### Phase 1: Create Poisoned Dataset (20 min)

**Task**: Create dataset with known poison samples
- Take clean MNIST training set
- Inject poisoned samples (mislabeled, ~5-10%)
- Mark which samples are poisoned (for validation)
- Save both clean and poisoned datasets

### Phase 2: Detection Methods (60 min)

**Task**: Try multiple detection approaches

**Method 1: Statistical Outlier Detection**
- Measure feature distributions
- Identify samples with unusual features
- Use: Z-score, IQR, clustering

**Method 2: Label Consistency Analysis**
- Compare samples with similar features
- Check if labels are consistent
- Flag inconsistent labels

**Method 3: Model-Based Detection**
- Train clean model on subset
- Check which samples model struggles with
- High loss samples may be poisoned

**Method 4: Ensemble Agreement**
- Train multiple models
- Check samples where models disagree
- Disagreement may indicate poisoning

### Phase 3: Evaluate Detection (20 min)

**Task**: Measure detection accuracy

**Metrics**:
- True positives (correctly identified poisoned)
- False positives (clean samples flagged)
- False negatives (poisoned samples missed)
- Precision, Recall, F1-score

### Phase 4: Analysis (10 min)

**Task**: Compare methods and identify best approach

## Deliverable Structure

Create `week-4/poisoning_detection.md`:

```markdown
# Poisoning Detection Challenge

**Date**: [Date]  
**Tester**: [Your name]  
**Objective**: Detect poisoned samples in training dataset

---

## Dataset Overview

### Clean Dataset
- Samples: [X]
- Classes: [10 for MNIST]
- Distribution: [Per-class counts]

### Poisoned Dataset
- Total samples: [X]
- Poisoned samples: [X] ([X]%)
- Poisoning method: [Description]
- Target classes: [Which classes affected]

## Detection Methods

### Method 1: Statistical Outlier Detection

**Approach**: [Description]
**Implementation**: [How you did it]
**Results**:
- Detected: [X] samples
- True positives: [X]
- False positives: [X]
- Precision: [X]%
- Recall: [X]%

**Analysis**: [What worked, what didn't]

### Method 2: Label Consistency Analysis

[Same structure]

### Method 3: Model-Based Detection

[Same structure]

### Method 4: Ensemble Agreement

[Same structure]

## Comparison Matrix

| Method | Precision | Recall | F1-Score | Best For |
|--------|-----------|--------|----------|----------|
| Statistical Outlier | [X]% | [X]% | [X]% | [Use case] |
| Label Consistency | [X]% | [X]% | [X]% | [Use case] |
| Model-Based | [X]% | [X]% | [X]% | [Use case] |
| Ensemble | [X]% | [X]% | [X]% | [Use case] |

## Best Detection Strategy

**Winner**: [Method name]
**Why**: [Reasons]
**Combined Approach**: [Can methods be combined?]

## Key Insights

1. [Insight 1]: [What you learned]
2. [Insight 2]: [What you learned]

## Code Implementation

[Link to detection code]

## References

[Papers on poisoning detection]
```

## Success Criteria

Your detection analysis should:
- Test multiple detection methods
- Compare quantitative metrics
- Identify most effective approach
- Document limitations
- Provide actionable recommendations

