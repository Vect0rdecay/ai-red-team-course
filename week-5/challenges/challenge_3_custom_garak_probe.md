# Challenge 3: LLM Vulnerability Scanner Development

**Time Estimate**: 3 hours  
**Difficulty**: Advanced  
**Deliverable**: `week-5/custom_garak_probe/` directory + documentation

## Objective

Extend garak (LLM vulnerability scanner) with a custom probe for a specific vulnerability. This builds tool development skills and deep understanding of LLM security testing.

## What is garak?

Garak is an LLM vulnerability scanner that tests models for various security issues:
- Prompt injection
- Jailbreaks
- Data leakage
- Refusal breaking
- And more

## The Challenge

Create a custom garak probe that tests for a specific vulnerability not already well-covered.

## Probe Ideas

### Option 1: Custom Jailbreak Pattern
Create probe for specific jailbreak technique you discovered

### Option 2: Context Window Exploitation
Test how models handle very long contexts

### Option 3: Multi-Turn Attack
Test vulnerability across conversation turns

### Option 4: Instruction Following Bypass
Test models' ability to follow harmful instructions

### Option 5: Data Extraction Pattern
Test for specific data extraction vulnerabilities

## Task Breakdown

### Phase 1: Understand garak (30 min)

**Task**: Learn garak architecture

**Research**:
- Read garak documentation
- Review existing probe implementations
- Understand probe interface
- Identify probe structure

**Deliverable**: Understanding of garak probe system

### Phase 2: Design Your Probe (30 min)

**Task**: Design custom probe

**Probe Design**:
- **Vulnerability**: What are you testing for?
- **Method**: How does your probe work?
- **Expected Behavior**: What indicates vulnerability?
- **Severity**: How serious if found?

### Phase 3: Implement Probe (90 min)

**Task**: Write probe code

**Implementation Steps**:
1. Create probe class inheriting from garak base
2. Implement probe logic
3. Add test prompts
4. Implement detection logic
5. Add documentation

**Code Structure**:
```python
from garak.probes.base import Probe

class CustomProbe(Probe):
    def __init__(self):
        super().__init__()
        self.name = "custom_probe_name"
        self.description = "What it tests"
        self.bcp47 = "en"  # Language
    
    def _probe(self, model, test_prompts):
        # Your probe logic
        results = []
        for prompt in test_prompts:
            response = model.generate(prompt)
            # Analyze response
            vulnerability_found = self._analyze(response)
            results.append({
                'prompt': prompt,
                'response': response,
                'vulnerable': vulnerability_found
            })
        return results
    
    def _analyze(self, response):
        # Detection logic
        pass
```

### Phase 4: Test Probe (20 min)

**Task**: Test your probe

**Testing**:
- Test on multiple models
- Verify it detects vulnerabilities
- Check for false positives
- Measure effectiveness

### Phase 5: Document (10 min)

**Task**: Document probe and results

## Deliverable Structure

Create `week-5/custom_garak_probe/`:

```
custom_garak_probe/
├── probe_custom.py          # Your probe implementation
├── test_probe.py            # Tests for your probe
├── results/                  # Test results
│   ├── model1_results.json
│   └── model2_results.json
├── README.md                 # Documentation
└── requirements.txt         # Dependencies
```

### README.md

```markdown
# Custom Garak Probe: [Probe Name]

## Overview

[What vulnerability does this probe test for?]

## Vulnerability Description

[Detailed description of the vulnerability]

## Probe Implementation

### How It Works
[Explain your probe's methodology]

### Test Prompts
[What prompts does it use?]

### Detection Logic
[How does it detect the vulnerability?]

## Usage

```bash
# Install garak
pip install garak

# Run your probe
python -m garak --model_type [type] --model_name [name] --probes custom.CustomProbe
```

## Results

### Models Tested
- [Model 1]: [Results]
- [Model 2]: [Results]

### Findings
[Summary of vulnerability findings]

## Code

[Link to or include probe code]

## References

[Papers, resources]
```

## Success Criteria

Your custom probe should:
- Extend garak properly
- Test for specific vulnerability
- Work on multiple models
- Have clear documentation
- Provide useful results

## Real-World Application

Custom probe development helps:
- Extend tool capabilities
- Test specific vulnerabilities
- Contribute to open-source
- Build tool development skills
- Create reusable testing components

## Extension Ideas

- Test probe on more models
- Improve detection logic
- Add more test cases
- Contribute to garak (if appropriate)
- Create probe library

## Next Steps

- Use probe in Week 6 advanced testing
- Share with community (if appropriate)
- Build portfolio piece on tool development
- Apply to client assessments

