# Challenge 3: Shadow Model Build-Off

**Time Estimate**: 2 hours  
**Difficulty**: Intermediate  
**Deliverable**: `week-2/shadow_model_optimization.md`

## Objective

Build the most effective shadow model possible with limited queries. This simulates real-world constraints (API rate limits, costs) and teaches efficiency in attack development.

## The Challenge

**Scenario**: You're performing a black-box attack on a production model. You have:
- Limited API queries: 1000 queries maximum
- Cost constraint: Each query costs money
- Goal: Build shadow model that achieves >90% similarity to target model

**Challenge**: Maximize shadow model effectiveness while minimizing query count.

## Strategy Development

### Phase 1: Query Budget Planning (20 min)

**Task**: Plan your query allocation
- How many queries for shadow model training?
- How many for attack model training?
- How many for validation?
- What's your query efficiency target? (e.g., <500 queries for >90% accuracy)

**Deliverable**: Query budget plan

**Questions to Answer**:
- What's the minimum queries needed?
- Can you do better than the baseline?
- What's your optimization strategy?

### Phase 2: Baseline Shadow Model (30 min)

**Task**: Build standard shadow model
- Use Exercise 2 code as baseline
- Train shadow model with standard approach
- Measure: Query count, shadow model accuracy, attack model success rate

**Metrics to Track**:
- Queries used: [X]
- Shadow model accuracy: [X]%
- Attack model success: [X]%
- Time to train: [X] minutes

### Phase 3: Optimization Experiments (60 min)

**Task**: Try optimization strategies

**Strategy 1: Smart Sampling**
- Instead of random samples, use diverse sampling
- Focus queries on edge cases, high-confidence samples
- Measure: Does this improve shadow model quality?

**Strategy 2: Transfer Learning**
- Start with pre-trained model, fine-tune with queries
- Measure: Fewer queries needed?

**Strategy 3: Active Learning**
- Use uncertainty sampling: Query samples model is uncertain about
- Measure: Better shadow model with fewer queries?

**Strategy 4: Ensemble Approach**
- Train multiple small shadow models, ensemble them
- Measure: Better accuracy than single large model?

**For Each Strategy**:
- Document queries used
- Measure shadow model accuracy
- Measure attack model success rate
- Calculate efficiency: Success rate / Queries used

### Phase 4: Analysis & Comparison (10 min)

**Task**: Compare all strategies
- Which used fewest queries?
- Which achieved highest shadow model accuracy?
- Which achieved highest attack success rate?
- Which is most efficient overall?

## Deliverable Structure

Create `week-2/shadow_model_optimization.md`:

```markdown
# Shadow Model Optimization Challenge

## Challenge Overview
[Objective, constraints, goals]

## Query Budget Plan

### Initial Allocation
- Shadow model training: [X] queries
- Attack model training: [X] queries
- Validation: [X] queries
- Total budget: 1000 queries

### Efficiency Target
- Goal: Achieve >90% shadow model accuracy with <500 queries
- Stretch goal: Achieve >95% accuracy with <400 queries

## Baseline Results

### Standard Approach
- Queries used: [X]
- Shadow model accuracy: [X]%
- Attack model success: [X]%
- Time: [X] minutes
- Efficiency score: [Success rate / Queries]

## Optimization Strategies

### Strategy 1: Smart Sampling
**Approach**: [Describe method]
**Results**:
- Queries used: [X]
- Shadow model accuracy: [X]%
- Attack model success: [X]%
- Efficiency: [Score]
**Analysis**: [Did it work? Why/why not?]

### Strategy 2: Transfer Learning
[Same structure]

### Strategy 3: Active Learning
[Same structure]

### Strategy 4: Ensemble Approach
[Same structure]

## Comparison Matrix

| Strategy | Queries | Shadow Acc | Attack Success | Efficiency | Winner |
|----------|---------|------------|----------------|------------|--------|
| Baseline | [X] | [X]% | [X]% | [X] | |
| Smart Sampling | [X] | [X]% | [X]% | [X] | |
| Transfer Learning | [X] | [X]% | [X]% | [X] | |
| Active Learning | [X] | [X]% | [X]% | [X] | |
| Ensemble | [X] | [X]% | [X]% | [X] | |

## Winner Analysis

### Most Efficient Strategy
**Strategy**: [Name]
**Why**: [Reasons]
**Real-world application**: [When would you use this?]

### Key Insights
1. [Insight 1]: [What you learned]
2. [Insight 2]: [What you learned]
3. [Insight 3]: [What you learned]

## Code Implementation

### Optimized Shadow Model Code
[Link to or include optimized code]

### Key Optimizations
1. [Optimization 1]: [Description]
2. [Optimization 2]: [Description]
3. [Optimization 3]: [Description]

## Real-World Application

### When Query Limits Matter
- API rate limits (100 queries/hour)
- Cost constraints ($0.01 per query = $10 per 1000)
- Stealth requirements (fewer queries = less detectable)

### Best Practices
1. [Practice 1]: [Why it helps]
2. [Practice 2]: [Why it helps]
3. [Practice 3]: [Why it helps]

## Future Improvements

### Ideas to Try
- [Idea 1]: [Why it might work]
- [Idea 2]: [Why it might work]

### Research Questions
- [Question 1]: [What you'd like to investigate]
- [Question 2]: [What you'd like to investigate]

## Conclusion

[Summary of challenge, key learnings, real-world application]
```

## Success Criteria

Your optimization report should:
- Document baseline performance
- Test at least 2-3 optimization strategies
- Compare strategies with clear metrics
- Identify most efficient approach
- Explain real-world applicability
- Include working code

## Real-World Context

This challenge simulates:
- Production API constraints
- Cost optimization needs
- Stealth requirements
- Efficiency in attack development

## Skills Developed

- Query efficiency optimization
- Experimental methodology
- Performance measurement
- Cost-benefit analysis
- Real-world constraint handling

## Extension Ideas

- Try additional optimization strategies
- Compare with published research
- Test on different model types
- Measure query efficiency on larger datasets

## Next Steps

- Apply optimization techniques to Week 3 attacks
- Use efficient approaches in future engagements
- Document best practices for your toolkit
- Share findings with team (if collaborative)

