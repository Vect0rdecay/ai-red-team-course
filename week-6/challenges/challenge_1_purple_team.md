# Challenge 1: Purple Team Exercise

**Time Estimate**: 3 hours  
**Difficulty**: Intermediate-Advanced  
**Deliverable**: `week-6/purple_team_exercise.md`

## Objective

Simulate collaborative red team (attack) and blue team (defense) exercise. Attack model → Defender implements mitigation → Attack again. Document iterative improvement process.

## What is Purple Teaming?

Purple teaming combines:
- **Red Team**: Attack and find vulnerabilities
- **Blue Team**: Defend and implement mitigations
- **Purple Team**: Collaboration between both

Goal: Improve security through iterative attack → defense → re-attack cycles.

## The Exercise

### Setup

**Red Team (You)**: Attack the LLM model
**Blue Team (You)**: Defend against attacks
**Iterations**: 3 rounds

### Round 1: Initial Attack (45 min)

**Red Team Task**:
- Perform comprehensive attack (jailbreak, injection, extraction)
- Document all successful attacks
- Measure attack success rates
- Create attack report

**Deliverable**: Attack findings

### Round 2: Defense Implementation (60 min)

**Blue Team Task**:
- Review red team findings
- Implement 2-3 defenses
- Test defenses
- Document mitigation approach

**Defense Options**:
- Input filtering/sanitization
- Prompt engineering (system prompts)
- Output filtering
- Rate limiting
- Anomaly detection

**Deliverable**: Defense implementation and testing

### Round 3: Re-Attack (45 min)

**Red Team Task**:
- Attack again, but now with defenses in place
- Find ways around defenses
- Measure: Did defenses work?
- Document: What still works, what doesn't?

**Deliverable**: Re-attack findings

### Analysis (30 min)

**Purple Team Task**:
- Compare Round 1 vs Round 3
- Measure improvement
- Identify remaining vulnerabilities
- Recommend next steps

## Deliverable Structure

Create `week-6/purple_team_exercise.md`:

```markdown
# Purple Team Exercise: LLM Security

**Date**: [Date]  
**Exercise**: Attack → Defend → Re-Attack  
**Model**: [Target model]

---

## Exercise Overview

[Brief description of purple team exercise]

## Round 1: Initial Attack

### Attack Methodology
[What attacks you performed]

### Attack Results

| Attack Type | Success Rate | Examples |
|------------|-------------|----------|
| Jailbreak | [X]% | [Example prompts] |
| Prompt Injection | [X]% | [Example prompts] |
| Data Extraction | [X]% | [What was extracted] |

### Key Vulnerabilities Found
1. [Vulnerability 1]: [Description, impact]
2. [Vulnerability 2]: [Description, impact]
3. [Vulnerability 3]: [Description, impact]

## Round 2: Defense Implementation

### Defense Strategy
[Overall approach]

### Defenses Implemented

**Defense 1: [Name]**
- Method: [How it works]
- Implementation: [What you did]
- Expected protection: [What it should stop]

**Defense 2: [Name]**
[Same structure]

**Defense 3: [Name]**
[Same structure]

### Defense Testing
- Tested against: [What attacks]
- Results: [Did it work?]

## Round 3: Re-Attack

### Attack Strategy Against Defenses
[How you adapted attacks]

### Re-Attack Results

| Attack Type | Round 1 Success | Round 3 Success | Change |
|------------|----------------|-----------------|--------|
| Jailbreak | [X]% | [X]% | [Δ] |
| Prompt Injection | [X]% | [X]% | [Δ] |
| Data Extraction | [X]% | [X]% | [Δ] |

### Vulnerabilities Still Present
1. [Vulnerability]: [Why defense didn't work]
2. [Vulnerability]: [Why defense didn't work]

### New Attack Vectors
[Did you find new ways to attack?]

## Comparison Analysis

### Overall Improvement
- Round 1 vulnerabilities: [X]
- Round 3 vulnerabilities: [X]
- Reduction: [X]%

### Defense Effectiveness
- Most effective defense: [Name] - [Why]
- Least effective defense: [Name] - [Why]

### Remaining Risks
[What still needs work?]

## Iterative Improvement

### What Worked
[Successful defenses and why]

### What Didn't Work
[Unsuccessful defenses and why]

### Next Steps
[Recommendations for further improvement]

## Key Insights

1. [Insight 1]: [What you learned]
2. [Insight 2]: [What you learned]
3. [Insight 3]: [What you learned]

## Conclusion

[Summary of exercise, improvements achieved, remaining work]
```

## Success Criteria

Your purple team exercise should:
- Complete all 3 rounds
- Document attacks and defenses clearly
- Measure improvement quantitatively
- Identify remaining vulnerabilities
- Provide actionable recommendations

## Real-World Application

Purple teaming is used in:
- Production security improvements
- Continuous security testing
- Collaborative security culture
- Risk reduction programs

## Skills Developed

- Attack and defense coordination
- Iterative security improvement
- Defense effectiveness testing
- Collaborative security mindset

## Next Steps

- Apply purple team approach to other systems
- Use methodology in client engagements
- Build portfolio piece on purple teaming
- Practice explaining collaborative security

