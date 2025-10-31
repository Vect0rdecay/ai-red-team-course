# Challenge 2: Open-Source Tool Contribution

**Time Estimate**: 4 hours  
**Difficulty**: Advanced  
**Deliverable**: Forked repository + contribution + pull request

## Objective

Contribute to an open-source AI security tool (garak, ART, Purple Llama, etc.). Demonstrate technical depth by implementing custom probe, attack, defense, or enhancement.

## What You'll Learn

- Open-source contribution workflow
- Tool architecture understanding
- Code quality standards
- Professional development practices
- Community engagement

## Background

**Why Contribute?**:
- Demonstrates technical depth
- Shows real-world software skills
- Builds portfolio credibility
- Engages with community
- Shows initiative and expertise

**Popular Tools to Contribute To**:
- **garak**: LLM vulnerability scanner
- **ART**: Adversarial Robustness Toolbox
- **Foolbox**: Adversarial attack library
- **Purple Llama**: Meta's security evaluation framework

---

## The Challenge

### Phase 1: Tool Selection and Fork (30 min)

**Task**: Choose tool and fork repository

**Selection Criteria**:
1. **Tool Choice**:
   - Tool you understand well
   - Active project (recent commits)
   - Clear contribution guidelines
   - Good documentation

2. **Contribution Type**:
   - Custom probe/attack (garak)
   - New attack method (ART/Foolbox)
   - Defense implementation
   - Documentation improvement
   - Bug fix
   - Feature enhancement

**Your Selection**:
- **Tool**: _____________________
- **Contribution Type**: _____________________
- **Rationale**: _____________________

**Fork Repository**:
```bash
# Fork on GitHub/GitLab first, then:
git clone https://github.com/YOUR_USERNAME/tool-name.git
cd tool-name
git remote add upstream https://github.com/ORIGINAL_ORG/tool-name.git
```

---

### Phase 2: Understanding Tool Architecture (60 min)

**Task**: Understand tool structure and contribution process

**Activities**:

1. **Explore Codebase**:
   - Project structure
   - Key modules and classes
   - Extension points
   - Testing framework
   - Documentation style

2. **Review Contribution Guidelines**:
   - CONTRIBUTING.md
   - Code style requirements
   - Testing requirements
   - Pull request process
   - Documentation standards

3. **Study Similar Contributions**:
   - Review existing probes/attacks/defenses
   - Understand patterns and conventions
   - Note code style and structure

4. **Identify Integration Points**:
   - Where your contribution fits
   - Required interfaces
   - Dependencies
   - Configuration options

**Document**: Architecture notes and contribution plan

---

### Phase 3: Contribution Implementation (120 min)

**Task**: Implement your contribution

**Implementation Steps**:

1. **Create Feature Branch**:
   ```bash
   git checkout -b feature/your-contribution-name
   ```

2. **Implement Contribution**:
   - Follow tool's code style
   - Use appropriate interfaces
   - Add proper error handling
   - Include comments and docstrings

3. **Add Tests**:
   - Unit tests for your code
   - Integration tests if applicable
   - Follow tool's testing patterns

4. **Update Documentation**:
   - Add docstrings
   - Update user documentation
   - Add usage examples
   - Update README if needed

5. **Code Quality**:
   - Follow style guidelines
   - Run linters/formatters
   - Ensure all tests pass
   - Review your own code

**Example Contribution Types**:

**For garak (Custom Probe)**:
```python
# Example structure
class YourCustomProbe(garak.Probe):
    """Your custom probe description"""
    
    def __init__(self):
        super().__init__()
        self.name = "your_probe_name"
        # ... initialization
    
    def probe(self, model):
        # ... probe implementation
        return results
```

**For ART (Custom Attack)**:
```python
# Example structure
class YourCustomAttack(Attack):
    """Your custom attack description"""
    
    def __init__(self, estimator, ...):
        super().__init__(estimator=estimator)
        # ... initialization
    
    def generate(self, x, ...):
        # ... attack implementation
        return adversarial_samples
```

---

### Phase 4: Testing and Validation (30 min)

**Task**: Test contribution thoroughly

**Testing**:
1. **Unit Tests**:
   - Test your code in isolation
   - Edge cases
   - Error handling

2. **Integration Tests**:
   - Test with tool's framework
   - Test with example models
   - Verify expected behavior

3. **Manual Testing**:
   - Run your contribution
   - Verify functionality
   - Check output format
   - Validate results

**Validation Checklist**:
- [ ] Code follows style guidelines
- [ ] All tests pass
- [ ] Documentation is complete
- [ ] No errors or warnings
- [ ] Works with example models
- [ ] Follows tool conventions

---

### Phase 5: Pull Request Preparation (30 min)

**Task**: Prepare and submit pull request

**PR Components**:

1. **Commit Message**:
   - Clear, descriptive
   - Follow tool's commit style
   - Reference issues if applicable

2. **Pull Request Description**:
   - What does this contribution do?
   - Why is it useful?
   - How to test it?
   - Screenshots/examples if helpful

3. **PR Checklist** (if tool has one):
   - [ ] Code follows style guidelines
   - [ ] Tests added/updated
   - [ ] Documentation updated
   - [ ] No breaking changes (or documented)
   - [ ] Linked to issue (if applicable)

**Create Pull Request**:
```bash
# Commit changes
git add .
git commit -m "Add: Description of contribution"

# Push to your fork
git push origin feature/your-contribution-name

# Then create PR on GitHub/GitLab
```

**PR Template** (example):
```markdown
## Description
[What your contribution does]

## Motivation
[Why this contribution is useful]

## Changes
[What was added/changed]

## Testing
[How to test your contribution]

## Screenshots/Examples
[If applicable]

## Checklist
- [ ] Code follows style guidelines
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] No breaking changes
```

---

### Phase 6: Contribution Documentation (10 min)

**Task**: Document your contribution

**Create `week-8/tool_contribution.md`**:

```markdown
# Tool Contribution: [Tool Name]

**Date**: [Date]
**Tool**: [Tool Name and Repository]
**Contribution Type**: [Probe/Attack/Defense/etc.]

---

## Contribution Overview

**What**: [Brief description]
**Why**: [Motivation]
**Repository**: [Link to your fork and PR]

---

## Implementation Details

### Contribution Description
[Detailed description of what you implemented]

### Code Structure
[How your code is organized]

### Integration Points
[How it integrates with the tool]

### Configuration
[How to use/configure your contribution]

---

## Testing

### Test Cases
[How you tested]

### Results
[Test results and examples]

---

## Pull Request

**PR Link**: [Link to PR]
**Status**: [Open/Reviewing/Merged]
**Feedback**: [Any feedback received]

---

## Learning Outcomes

- [What you learned]
- [Skills demonstrated]
- [Challenges overcome]
```

---

## Success Criteria

**You've successfully completed this challenge when**:
- [ ] Selected appropriate tool and contribution type
- [ ] Understood tool architecture and contribution process
- [ ] Implemented functional contribution
- [ ] Added tests and documentation
- [ ] Created pull request with proper description
- [ ] Contribution follows tool's standards
- [ ] Documented contribution process

---

## Tips

**Tool Selection**:
- Choose tool you're comfortable with
- Start with small, focused contribution
- Check for "good first issue" labels
- Review contribution guidelines first

**Implementation**:
- Follow existing code patterns
- Write clear, well-documented code
- Add comprehensive tests
- Test thoroughly before submitting

**Pull Request**:
- Write clear description
- Respond to feedback promptly
- Be open to suggestions
- Thank reviewers

**Community Engagement**:
- Engage in discussions
- Help others
- Contribute regularly
- Build reputation

---

## Extension

**Advanced** (Optional):
- Multiple contributions
- Significant feature additions
- Maintainer engagement
- Community involvement
- Blog post about contribution

