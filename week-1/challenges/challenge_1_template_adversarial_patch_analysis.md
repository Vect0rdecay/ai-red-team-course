# Challenge 1: Adversarial Patch Attack Analysis

**Time Estimate**: 2 hours  
**Difficulty**: Intermediate  
**Deliverable**: `week-1/case_study_analysis.md`

## Objective

Research and analyze adversarial patch attacks across multiple domains to understand how these
physical attacks manifest in real-world ML systems. This connects Week 1's foundational concepts
to diverse real-world attack scenarios beyond the commonly discussed examples.

## Background

Adversarial patch attacks involve placing physical objects (stickers, patches, or accessories) in
the environment that cause ML vision systems to misclassify or fail. These attacks are
particularly concerning because they:

- Require no digital access to the model
- Work across different model architectures
- Can be printed and deployed physically
- Affect multiple application domains

Security incidents involving ML systems occur regularly across industries. By analyzing diverse cases, you'll:
- See how attacks manifest in different domains
- Understand business impact across sectors
- Connect attacks to OWASP ML Top 10
- Prepare for real engagements with diverse attack surfaces

## Research Domains

Your research should cover adversarial patch attacks in **at least 3 different domains** from the following categories:

### Face Recognition & Identity Systems
- Face recognition systems
- Access control systems
- Surveillance systems
- Identity verification

### Object Detection & Classification
- Security camera systems
- Retail inventory systems
- Quality control systems
- Wildlife monitoring

### Autonomous Systems
- Vehicle navigation
- Drone obstacle avoidance
- Robotic vision systems
- Agricultural automation

### Security & Surveillance
- Intrusion detection systems
- License plate recognition
- Person detection systems
- Perimeter security

### Consumer Applications
- Social media filters
- Photo tagging systems
- Shopping recommendation systems
- AR/VR systems

**Note**: While autonomous vehicles are one valid domain, ensure your research includes attacks beyond just traffic signs. Explore diverse applications and attack vectors.

## Your Task

1. **Research Diverse Attack Scenarios** (45 min)
   - Find 2-3 reputable research papers or case studies about adversarial patch attacks
   - Focus on **different domains** - avoid focusing only on one application area
   - Document key details for each:
     - Attack method and patch design
     - Target systems and model types
     - Physical deployment method
     - Attack success rate and constraints
     - Real-world feasibility

2. **Create Attack Comparison Matrix** (30 min)
   - Create a table comparing attacks across domains
   - Columns: Domain | Attack Method | Target System | Patch Type | Physical Constraints | Success Rate
   - Include at least 3 different domain examples
   - Identify common patterns and differences

3. **Map to OWASP ML Top 10** (30 min)
   - Identify which OWASP ML Top 10 vulnerabilities apply
   - Map specific attack techniques to vulnerability categories
   - Note if different domains map to different vulnerabilities
   - Example: Adversarial patch = M01 (Input Manipulation) + M04 (Evasion Attacks)

4. **Analyze Impact Across Domains** (15 min)
   - Business impact: What risks do these attacks create in different industries?
   - Technical impact: How do attacks work across different model architectures?
   - Defense response: What mitigations work across domains vs. domain-specific?

## Deliverable Structure

Create `week-1/case_study_analysis.md` with:

# Adversarial Patch Attacks - Case Study Analysis

## Research Overview

[2-3 paragraph overview covering the breadth of adversarial patch attacks across domains]

## Attack Comparison Matrix

[Table comparing attacks across different domains]

| Domain | Attack Method | Target System | Patch Type | Physical Constraints | Success Rate |
|--------|--------------|---------------|------------|---------------------|--------------|
| [Domain 1] | [Attack method] | [Target system] | [Patch type] | [Constraints] | [Success rate] |
| [Domain 2] | [Attack method] | [Target system] | [Patch type] | [Constraints] | [Success rate] |
| [Domain 3] | [Attack method] | [Target system] | [Patch type] | [Constraints] | [Success rate] |

## Detailed Attack Analysis

### Attack 1: [Domain/Application]
- **Attack Method**: [Description]
- **Target System**: [What was attacked]
- **Patch Design**: [How was the patch created]
- **Physical Deployment**: [How was it deployed]
- **Results**: [Success rate, constraints, limitations]

### Attack 2: [Domain/Application]
[Same structure]

### Attack 3: [Domain/Application]
[Same structure]

## OWASP ML Top 10 Mapping
- M01: Input Manipulation - [How it applies across domains]
- M04: Evasion Attacks - [How it applies across domains]
- [Other applicable vulnerabilities]
- Domain-specific considerations: [Which vulnerabilities are more relevant to which domains]

## Impact Analysis

### Business Impact by Domain
- [Domain 1]: [Key business risks]
- [Domain 2]: [Key business risks]
- [Domain 3]: [Key business risks]

### Technical Impact
- [Common attack mechanisms across domains]
- [Domain-specific technical considerations]
- [Architecture vulnerabilities]

### Defense Response
- [Universal defense strategies]
- [Domain-specific mitigations]
- [Industry response and adoption]

## Common Patterns & Differences
- [What patterns emerge across domains?]
- [What makes attacks domain-specific?]
- [Which attacks are most practical/feasible?]

## Key Learnings
- [3-5 bullet points of insights about adversarial patches in general]
- [1-2 domain-specific insights]

## Application to Red Teaming
- [How would you test for this vulnerability across different domains?]
- [What would you look for in different types of client engagements?]
- [Domain-specific testing considerations]

## Resources

**Recommended Reading:**
- "Adversarial Patch" (Brown et al., 2017) - Original paper
- "Robust Physical-World Attacks on Deep Learning Models" (Brown et al., 2017)
- "Fooling automated surveillance cameras: adversarial patches to attack person detection" (Thys et al., 2019)
- OWASP ML Top 10: M01 and M04
- MITRE ATLAS: Adversarial example patterns

**Search Terms:**
- "adversarial patch attack"
- "physical adversarial attacks"
- "adversarial patch face recognition"
- "adversarial patch object detection"
- "robust physical world attacks"
- "adversarial patch security cameras"
- "adversarial patch surveillance"

**Research Databases:**
- arXiv.org (cs.CV, cs.CR)
- IEEE Xplore
- ACM Digital Library
- Google Scholar

## Success Criteria

Your analysis should demonstrate:
- **Breadth**: Coverage of attacks across multiple diverse domains
- **Depth**: Understanding of attack methodology in different contexts
- **Analysis**: Clear connection to OWASP ML Top 10 framework
- **Insight**: Recognition of patterns and differences across domains
- **Application**: Ability to translate to red team testing scenarios across different domains

## Next Steps

After completing this challenge:
- Use as reference for Week 3 evasion attack exercises
- Consider how patch attacks differ from other evasion attacks

