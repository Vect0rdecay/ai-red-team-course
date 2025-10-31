# Challenge 4: Paper Reading & Implementation Challenge

**Time Estimate**: 3 hours  
**Difficulty**: Advanced  
**Deliverable**: `week-2/paper_presentation.md` + implementation code

## Objective

Read an AI security research paper, extract the key technique, implement it, and create a presentation. This builds research-to-practice translation skills essential for staying current in AI security.

## Why This Matters

AI security moves fast. New attack techniques are published monthly. As a red teamer, you need to:
- Read and understand research papers
- Extract implementable techniques
- Apply techniques to real targets
- Stay ahead of defenses

This challenge develops those skills.

## Paper Selection

Choose ONE paper from this list:

### Option 1: Membership Inference
**Paper**: Shokri et al. (2017) "Membership Inference Attacks Against Machine Learning Models"
**Focus**: Basic membership inference techniques
**Difficulty**: Intermediate
**Link**: Search arXiv or Google Scholar

### Option 2: Label-Only Attacks
**Paper**: Choquette-Choo et al. (2021) "Label-Only Membership Inference Attacks"
**Focus**: Attacks with limited model access
**Difficulty**: Intermediate-Advanced
**Link**: Search arXiv

### Option 3: Model Extraction
**Paper**: Tramer et al. (2016) "Stealing Machine Learning Models via Prediction APIs"
**Focus**: Model extraction attacks
**Difficulty**: Advanced
**Link**: Search arXiv or USENIX

### Option 4: Training Data Extraction
**Paper**: Carlini et al. (2022) "Extracting Training Data from Large Language Models"
**Focus**: Data extraction from LLMs (advanced, preview of Week 5)
**Difficulty**: Advanced
**Link**: Search arXiv

## Challenge Phases

### Phase 1: Paper Reading (45 min)

**Task**: Read and understand the paper

**Reading Strategy**:
1. **Abstract & Introduction** (10 min): What problem are they solving?
2. **Methodology** (20 min): How do they solve it? (Focus here)
3. **Experiments & Results** (10 min): How well does it work?
4. **Conclusion** (5 min): Key takeaways

**Take Notes On**:
- Problem statement
- Key technique/methodology
- Implementation details
- Experimental setup
- Results/metrics
- Limitations

**Deliverable**: Reading notes

### Phase 2: Technique Extraction (30 min)

**Task**: Extract the implementable technique

**Questions to Answer**:
- What's the core algorithm?
- What inputs does it need?
- What are the key parameters?
- What's the expected output?
- What are implementation challenges?

**Deliverable**: Technique summary

### Phase 3: Implementation (90 min)

**Task**: Implement the key technique

**Implementation Steps**:
1. Set up test environment (Week 1 model)
2. Implement core algorithm
3. Test on your model
4. Measure results
5. Compare to paper results (if possible)

**Code Requirements**:
- Clean, commented code
- Reproducible (include random seeds)
- Documented (what it does, how to use)
- Tested (works on your model)

**Deliverable**: Working implementation + test results

### Phase 4: Presentation Creation (15 min)

**Task**: Create 5-slide presentation

**Slide Structure**:
1. **Problem**: What problem does this solve?
2. **Method**: How does the technique work? (High-level)
3. **Implementation**: What did you implement?
4. **Results**: What did you achieve?
5. **Application**: How would you use this in red teaming?

**Format**: Markdown slides or simple PDF

## Deliverable Structure

### Presentation (`paper_presentation.md`)

```markdown
# [Paper Title] - Implementation & Analysis

**Paper**: [Full citation]  
**Implementer**: [Your name]  
**Date**: [Date]

---

## Slide 1: Problem Statement

### What Problem Does This Solve?
[2-3 sentences]

### Why Does It Matter for Red Teaming?
[Connection to AI security testing]

---

## Slide 2: Methodology (High-Level)

### Core Technique
[Explain the key idea in 3-4 bullet points]

### How It Works
[Simple explanation, avoid too much math unless critical]

**Key Insight**: [One sentence summary of the clever part]

---

## Slide 3: Implementation

### What I Implemented
- [Feature 1]: [Description]
- [Feature 2]: [Description]
- [Feature 3]: [Description]

### Challenges Faced
- [Challenge 1]: [How you solved it]
- [Challenge 2]: [How you solved it]

### Code Location
[Link to implementation code]

---

## Slide 4: Results

### Attack Performance
- Success rate: [X]%
- Baseline (random): 50%
- Improvement: [X] percentage points

### Comparison to Paper
- Paper reported: [X]%
- My implementation: [X]%
- Difference: [Explanation]

### Visual Evidence
[Link to plots/results]

---

## Slide 5: Red Team Application

### When Would You Use This?
[Specific scenarios]

### What Does It Test For?
[Vulnerability it detects]

### Real-World Example
[How you'd use this in an engagement]

---

## Implementation Details

### Code Structure
[Describe your code organization]

### Key Functions
```python
# Example code snippet showing key function
def membership_inference_attack(model, samples):
    # Your implementation
    pass
```

### Parameters & Configuration
[What parameters matter, how to tune them]

### Usage Example
[How to run your implementation]

---

## Results Analysis

### Quantitative Results
[Tables, metrics, comparisons]

### Qualitative Observations
[What you noticed, edge cases, limitations]

### Limitations
[What didn't work, what's hard about this technique]

---

## Key Learnings

### Technical Insights
1. [Learning 1]
2. [Learning 2]
3. [Learning 3]

### Implementation Insights
1. [Learning 1]
2. [Learning 2]

### Red Team Insights
1. [How this helps in engagements]
2. [When you'd use this]

---

## References

- [Paper citation]
- [Additional resources used]
- [Tools/libraries used]
```

### Code Implementation

Create `week-2/paper_implementation/` with:
- `implemented_technique.py` - Main implementation
- `test_implementation.py` - Tests
- `results/` - Output files, plots
- `README.md` - How to use

## Success Criteria

Your paper challenge should:
- Demonstrate understanding of the paper
- Include working implementation
- Show results on your model
- Explain red team application
- Be presentable to others

## Presentation Tips

**For Slide Creation**:
- Use clear, simple language
- Include visuals (diagrams, plots)
- Focus on practical application
- Avoid excessive technical jargon
- Highlight red team relevance

**For Verbal Presentation** (if presenting):
- 5-7 minutes total
- Practice explaining clearly
- Be ready for questions
- Connect to real engagements

## Real-World Application

This skill is essential because:
- New attacks published constantly
- Defenses evolve quickly
- Clients expect cutting-edge testing
- Research → Practice translation is valuable

## Extension Ideas

- Implement additional techniques from the paper
- Compare with other papers
- Extend the technique (your own variation)
- Write blog post about implementation
- Contribute to open-source tools

## Portfolio Note

This is excellent portfolio material:
- Demonstrates research skills
- Shows implementation capability
- Proves you can stay current
- Shows communication skills (presentation)

## Next Steps

- Present to peers (if collaborative)
- Use technique in future exercises
- Read another paper next week
- Build library of implemented techniques

