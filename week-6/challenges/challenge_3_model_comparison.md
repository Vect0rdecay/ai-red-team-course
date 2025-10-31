# Challenge 3: Model Comparison Analysis

**Time Estimate**: 2 hours  
**Difficulty**: Intermediate  
**Deliverable**: `week-6/model_comparison.md`

## Objective

Test the same attack on multiple LLM models (GPT-3.5, Llama-2, Mistral, etc.) and compare vulnerability profiles. Identify model-specific weaknesses.

## Task Breakdown

### Phase 1: Model Selection (15 min)

**Task**: Choose 3 models to compare

**Options**:
- GPT-3.5 / GPT-4
- Llama-2 (7B or 13B)
- Mistral
- Claude (if accessible)
- Local models

### Phase 2: Attack Execution (60 min)

**Task**: Run same attacks on all models

**Attacks to Test**:
- Jailbreak (3-5 techniques)
- Prompt injection (2-3 scenarios)
- Data extraction (if possible)

**For Each Model**:
- Run all attacks
- Document success rates
- Note model-specific behaviors

### Phase 3: Comparison Analysis (30 min)

**Task**: Compare vulnerability profiles

**Analysis**:
- Which models are most vulnerable?
- Which attacks work best on which models?
- Model-specific weaknesses?
- Patterns across models?

### Phase 4: Reporting (15 min)

**Task**: Document findings

## Deliverable Structure

Create `week-6/model_comparison.md`:

```markdown
# LLM Model Security Comparison

**Date**: [Date]  
**Models Tested**: [List]
**Attacks**: [List]

---

## Overview

[Brief introduction]

## Models Tested

### Model 1: [Name]
- Version: [Version]
- Provider: [Provider]
- Size: [Parameters]
- Access: [API/Local]

### Model 2: [Name]
[Same structure]

### Model 3: [Name]
[Same structure]

## Attack Comparison

### Jailbreak Attacks

| Technique | Model 1 | Model 2 | Model 3 | Most Vulnerable |
|-----------|---------|---------|---------|----------------|
| DAN | [X]% | [X]% | [X]% | [Model] |
| Role-Play | [X]% | [X]% | [X]% | [Model] |
| Indirect | [X]% | [X]% | [X]% | [Model] |

### Prompt Injection

[Same comparison table]

### Data Extraction

[Same comparison table]

## Vulnerability Analysis

### Most Vulnerable Model
**Model**: [Name]
**Why**: [Reasons]
**Specific Weaknesses**: [List]

### Most Resilient Model
**Model**: [Name]
**Why**: [Reasons]
**Strength Areas**: [List]

### Model-Specific Findings

**Model 1**: [Unique characteristics, vulnerabilities]
**Model 2**: [Unique characteristics, vulnerabilities]
**Model 3**: [Unique characteristics, vulnerabilities]

## Patterns Across Models

### Common Vulnerabilities
[What works across all models]

### Model-Specific Issues
[What's unique to certain models]

## Recommendations

### For High-Security Use Cases
**Recommended Model**: [Name]
**Why**: [Reasons]

### For Development/Testing
**Recommended Model**: [Name]
**Why**: [Reasons]

## Key Insights

1. [Insight 1]
2. [Insight 2]
3. [Insight 3]
```

## Success Criteria

Your model comparison should:
- Test 3+ models
- Use consistent attack methodology
- Compare quantitative metrics
- Identify model-specific weaknesses
- Provide actionable recommendations

