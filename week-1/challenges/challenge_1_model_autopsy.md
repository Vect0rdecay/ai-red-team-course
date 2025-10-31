# Challenge 1: Model Autopsy - Real-World ML Security Incident Analysis

**Time Estimate**: 2 hours  
**Difficulty**: Intermediate  
**Deliverable**: `week-1/case_study_analysis.md`

## Objective

Analyze a real-world ML security incident to understand how theoretical attacks manifest in practice. This connects Week 1's foundational concepts to real-world consequences.

## Background

Security incidents involving ML systems occur regularly. By analyzing real cases, you'll:
- See how attacks actually happen
- Understand business impact
- Connect attacks to OWASP ML Top 10
- Prepare for real engagements

## Scenario: Adversarial Patch Attack on Traffic Sign Recognition

**Incident Overview**: Researchers demonstrated that physical adversarial patches (stickers) placed on traffic signs could cause autonomous vehicle vision systems to misclassify stop signs as speed limit signs.

### Incident Timeline

**Phase 1: Research Discovery (2017-2018)**
- Academic researchers publish papers showing physical adversarial attacks
- Proof-of-concept: Printed patches that fool image classifiers
- Minimal real-world impact initially

**Phase 2: Public Demonstration (2019)**
- Security researchers demonstrate attacks on commercial systems
- Tesla, Waymo, and other AV manufacturers acknowledge vulnerabilities
- Media coverage raises public awareness

**Phase 3: Regulatory Response (2020-2021)**
- NHTSA begins investigating adversarial attack risks
- Industry groups develop defense strategies
- Security testing becomes part of AV development lifecycle

## Your Task

1. **Research the Incident** (30 min)
   - Find 2-3 reputable sources about adversarial patch attacks on autonomous vehicles
   - Document key details: attack method, target systems, impact

2. **Create Attack Timeline** (30 min)
   - Use a timeline tool (draw.io, Mermaid, or simple markdown)
   - Include: Discovery → Research → Demonstration → Impact → Response
   - Annotate each phase with key events and dates

3. **Map to OWASP ML Top 10** (30 min)
   - Identify which OWASP ML Top 10 vulnerabilities apply
   - Map specific attack techniques to vulnerability categories
   - Example: Adversarial patch = M01 (Input Manipulation) + M04 (Evasion Attacks)

4. **Analyze Impact** (30 min)
   - Business impact: What risks did this create?
   - Technical impact: How did attacks work?
   - Defense response: What mitigations were developed?

## Deliverable Structure

Create `week-1/case_study_analysis.md` with:

```markdown
# Adversarial Patch Attack - Case Study Analysis

## Incident Summary
[2-3 paragraph overview]

## Attack Timeline
[Visual timeline or structured list]

## OWASP ML Top 10 Mapping
- M01: Input Manipulation - [How it applies]
- M04: Evasion Attacks - [How it applies]
- [Other applicable vulnerabilities]

## Impact Analysis
### Business Impact
- [List key business risks]

### Technical Impact
- [How the attack worked technically]

### Defense Response
- [What mitigations were developed]

## Key Learnings
- [3-5 bullet points of insights]

## Application to Red Teaming
- [How would you test for this vulnerability?]
- [What would you look for in a client engagement?]
```

## Resources

**Recommended Reading:**
- "Robust Physical-World Attacks on Deep Learning Models" (Brown et al., 2017)
- OWASP ML Top 10: M01 and M04
- MITRE ATLAS: Adversarial example patterns

**Search Terms:**
- "adversarial patch traffic sign"
- "physical adversarial attacks autonomous vehicles"
- "robust physical world attacks"

## Success Criteria

Your analysis should demonstrate:
- Understanding of the attack methodology
- Clear connection to OWASP ML Top 10 framework
- Insight into business and technical impacts
- Ability to translate to red team testing scenarios

## Next Steps

After completing this challenge:
- Share insights with peers (if in collaborative setting)
- Use as reference for Week 3 evasion attack exercises
- Build portfolio piece demonstrating real-world security awareness

