# Challenge 2: MITRE ATLAS Attack Mapping

**Time Estimate**: 2 hours  
**Difficulty**: Intermediate  
**Deliverable**: `week-7/atlas_attack_mapping.md`

## Objective

Map all attacks from Weeks 1-6 to the MITRE ATLAS (Adversarial Threat Landscape for AI Systems) framework. Create comprehensive threat matrix document showing attack classification and mitigation mapping.

## What You'll Learn

- Industry-standard threat framework (MITRE ATLAS)
- Attack classification and taxonomy
- Mitigation mapping
- Professional threat modeling

## Background: MITRE ATLAS

**MITRE ATLAS** = Adversarial Threat Landscape for AI Systems

**Purpose**: Industry-standard framework for classifying AI/ML security threats

**Structure**:
- **Tactics**: High-level attack objectives (like MITRE ATT&CK)
- **Techniques**: Specific attack methods
- **Procedures**: Real-world examples
- **Mitigations**: Defense strategies

**Resource**: https://atlas.mitre.org/

---

## The Challenge

### Phase 1: Framework Review (30 min)

**Task**: Review MITRE ATLAS framework

**Steps**:
1. Visit https://atlas.mitre.org/
2. Review tactic categories
3. Review technique structure
4. Review mitigation mappings
5. Understand framework terminology

**Key Tactics** (examples):
- Initial Access
- Execution
- Persistence
- Evasion
- Discovery
- Collection
- Exfiltration

---

### Phase 2: Attack Inventory (30 min)

**Task**: List all attacks from Weeks 1-6

**Create Attack List**:

| Week | Attack Name | Attack Type | Model Target |
|------|-------------|-------------|--------------|
| 1 | [N/A - Setup] | - | - |
| 2 | Membership Inference | Inference | ML Model |
| 3 | FGSM Evasion | Evasion | ML Model |
| 3 | PGD Evasion | Evasion | ML Model |
| 4 | Data Poisoning | Poisoning | Training Pipeline |
| 4 | Backdoor Attack | Backdoor | ML Model |
| 5 | Jailbreak | Prompt Injection | LLM |
| 5 | Prompt Injection | Prompt Injection | LLM |
| 6 | Multi-Vector Attack | Multiple | LLM |
| 6 | Purple Team Exercise | Comprehensive | LLM |

[Continue listing...]

---

### Phase 3: ATLAS Mapping (45 min)

**Task**: Map each attack to MITRE ATLAS

**For Each Attack**:

1. **Identify Tactic**: Which high-level objective?
2. **Identify Technique**: Which specific technique?
3. **Identify Procedure**: Real-world example
4. **Identify Mitigations**: What defenses work?

**Mapping Template**:

```markdown
## [Attack Name]

**Week**: [X]
**ATLAS Tactic**: [Tactic ID] - [Tactic Name]
**ATLAS Technique**: [Technique ID] - [Technique Name]
**ATLAS Procedure**: [Procedure ID if applicable]

**Attack Description**: [Brief description]

**Mapping Rationale**: [Why this classification?]

**Recommended Mitigations**:
- [Mitigation 1]
- [Mitigation 2]

**Additional Notes**: [Any observations]
```

---

### Phase 4: Threat Matrix Creation (15 min)

**Task**: Create comprehensive threat matrix

**Matrix Format**:

| Attack | ATLAS Tactic | ATLAS Technique | Mitigation | Risk Level |
|--------|--------------|-----------------|------------|------------|
| [Name] | [ID] | [ID] | [Defense] | [High/Med/Low] |
| ... | ... | ... | ... | ... |

---

### Deliverable

Create `week-7/atlas_attack_mapping.md`:

```markdown
# MITRE ATLAS Attack Mapping

**Date**: [Date]
**Course**: AI Red Team Transition
**Weeks Covered**: 1-6

---

## Overview

This document maps all attacks from Weeks 1-6 to the MITRE ATLAS framework.

**Purpose**: Demonstrate industry-standard threat classification and mitigation mapping.

---

## Framework Reference

- **MITRE ATLAS**: https://atlas.mitre.org/
- **Version**: [Date/version]
- **Last Updated**: [Date]

---

## Attack Mappings

[Include mappings for each attack]

---

## Threat Matrix

[Include comprehensive matrix]

---

## Mitigation Summary

**By Tactic**:
- [Tactic 1]: [List mitigations]
- [Tactic 2]: [List mitigations]
- ...

**By Priority**:
- High Risk: [List attacks]
- Medium Risk: [List attacks]
- Low Risk: [List attacks]

---

## Analysis

**Most Common Tactic**: [Tactic name]
**Most Common Technique**: [Technique name]
**Coverage**: [X] tactics, [Y] techniques covered

**Gaps**: [Any attack types not covered in course]
```

---

## Success Criteria

**You've successfully completed this challenge when**:
- [ ] Reviewed MITRE ATLAS framework
- [ ] Listed all attacks from Weeks 1-6
- [ ] Mapped each attack to ATLAS tactic/technique
- [ ] Created comprehensive threat matrix
- [ ] Documented mitigations for each attack
- [ ] Analysis included

---

## Tips

**Mapping Guidance**:
- Some attacks may map to multiple techniques
- Document your rationale clearly
- Use official ATLAS terminology
- Reference ATLAS documentation

**Common Mappings**:
- Evasion attacks → ATLAS.Evasion techniques
- Poisoning → ATLAS.Poisoning techniques
- Prompt injection → ATLAS.LLM techniques
- Data extraction → ATLAS.Collection techniques

---

## Extension

**Advanced** (Optional):
- Create attack chains using ATLAS
- Map purple team exercises to ATLAS
- Create ATLAS-based testing methodology
- Compare with other frameworks (OWASP ML Top 10)

