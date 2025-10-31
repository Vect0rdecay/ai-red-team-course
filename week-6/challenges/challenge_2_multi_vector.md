# Challenge 2: Multi-Vector Attack Scenario

**Time Estimate**: 3 hours  
**Difficulty**: Advanced  
**Deliverable**: `week-6/multi_vector_attack.md`

## Objective

Combine multiple attack vectors (prompt injection + data extraction + jailbreak) to achieve a complex goal requiring multiple techniques. This demonstrates sophisticated attack methodology.

## Scenario

**Goal**: Extract sensitive training data from LLM, then use that information to craft more effective attacks.

**Attack Chain**:
1. Prompt injection → Extract system prompt
2. Data extraction → Get training data samples
3. Jailbreak → Use extracted info to bypass filters
4. Final objective → Achieve target goal

## Task Breakdown

### Phase 1: Attack Planning (30 min)

**Task**: Design multi-vector attack

**Attack Design**:
- **Vector 1**: [What you'll do first]
- **Vector 2**: [What you build on vector 1]
- **Vector 3**: [Final vector using previous outputs]
- **Dependencies**: [What depends on what]

### Phase 2: Vector 1 - Prompt Injection (45 min)

**Task**: Extract system prompt or instructions

**Objective**: Get information needed for later vectors

### Phase 3: Vector 2 - Data Extraction (60 min)

**Task**: Extract training data samples

**Objective**: Use information from vector 1 if helpful

### Phase 4: Vector 3 - Jailbreak (45 min)

**Task**: Bypass safety filters using extracted information

**Objective**: Use system prompt and training data insights

### Phase 5: Final Objective (30 min)

**Task**: Achieve target goal

**Objective**: Complete the attack chain

## Deliverable Structure

Create `week-6/multi_vector_attack.md`:

```markdown
# Multi-Vector Attack Scenario

**Date**: [Date]  
**Objective**: [Final goal]
**Model**: [Target]

---

## Attack Overview

[Brief description of multi-vector attack]

## Attack Flow

```
Vector 1 (Prompt Injection) 
  → Output 1
  → Vector 2 (Data Extraction)
  → Output 2
  → Vector 3 (Jailbreak)
  → Final Objective
```

## Vector 1: Prompt Injection

**Goal**: Extract system prompt

**Technique**: [Method]
**Prompt**: [What you used]
**Result**: [What you got]
**Output for Next Vector**: [Information passed forward]

## Vector 2: Data Extraction

**Goal**: Extract training data samples

**Technique**: [Method]
**Prompt**: [What you used]
**Information Used from Vector 1**: [How you leveraged previous output]
**Result**: [What you extracted]
**Output for Next Vector**: [Information passed forward]

## Vector 3: Jailbreak

**Goal**: Bypass safety filters

**Technique**: [Method]
**Information Used**: [How you used previous outputs]
**Result**: [Success/failure]
**Final Objective**: [What you achieved]

## Attack Effectiveness

**Vector 1 Success**: [Yes/No]
**Vector 2 Success**: [Yes/No]
**Vector 3 Success**: [Yes/No]
**Overall Success**: [Yes/No]

**Why Chaining Was Necessary**: [Explanation]

## Defense Analysis

### Where Could Chain Be Broken?

**Vector 1 Mitigation**: [How to stop this]
**Vector 2 Mitigation**: [How to stop this]
**Vector 3 Mitigation**: [How to stop this]

### Recommended Defenses
1. [Defense]: [How it breaks chain]
2. [Defense]: [How it breaks chain]

## Key Insights

1. [Insight 1]
2. [Insight 2]
3. [Insight 3]

## Code/Demonstration

[Link to attack code or demonstration]
```

## Success Criteria

Your multi-vector attack should:
- Successfully chain 3+ attack vectors
- Use outputs from previous vectors
- Achieve complex goal requiring chaining
- Document dependencies clearly
- Provide defense recommendations

