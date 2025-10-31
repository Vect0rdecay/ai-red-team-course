# Week 5: Generative AI Security

## Overview

Week 5 focuses on LLM (Large Language Model) vulnerabilities: jailbreaks, prompt injection, and advanced attack techniques. You'll learn to test LLM security using tools like garak and Purple Llama.

**Key Transition**: LLM attacks are like social engineering - you craft prompts (social interactions) that manipulate the model (target) into revealing information or performing unwanted actions.

**Estimated Time**: 12-15 hours

---

## Learning Objectives

By the end of Week 5, you will be able to:

1. **Craft Jailbreak Prompts**: Bypass LLM safety filters using various techniques
2. **Execute Prompt Injection Attacks**: Chain attacks to achieve complex goals
3. **Use LLM Security Tools**: Leverage garak and Purple Llama for testing
4. **Develop Custom Probes**: Extend security tools with custom vulnerability tests
5. **Communicate LLM Risks**: Explain vulnerabilities to non-technical stakeholders

---

## Red Team Application

**What You're Actually Learning:**

- **Jailbreaks**: Bypass safety filters (like privilege escalation)
- **Prompt Injection**: Manipulate model behavior (like command injection)
- **Data Extraction**: Extract training data or system prompts (like information disclosure)
- **Tool Usage**: Professional LLM security testing tools

**Real-World Scenario**: Client has deployed LLM-powered customer service chatbot. Your job: test for jailbreaks, prompt injection, data leakage, and document findings.

---

## Required Reading

**Sotiropoulos (2024)** - "Adversarial AI Attacks, Mitigations and Defense Strategies"
- **Chapter 11**: Generative AI Vulnerabilities (pp. 301-330)
- **Chapter 12**: LLM Security and Prompt Attacks (pp. 331-360)

**OWASP Top 10 for LLM Applications**
- A1: Prompt Injection
- A3: Training Data Poisoning
- A4: Denial of Service
- A7: Insecure Output Handling
- Available: https://owasp.org/www-project-top-10-for-large-language-model-applications/

**Tools Documentation**:
- garak: https://github.com/leondz/garak
- Purple Llama: https://github.com/meta-llama/PurpleLlama

---

## Weekly Structure

### Monday-Tuesday: Jailbreak Techniques (3-4 hours)

**Activity**: Learn and practice jailbreak prompt engineering

**Background**:
- Jailbreaks bypass LLM safety filters
- Techniques: DAN, role-play, indirect prompting, etc.
- Goal: Understand attack patterns

**Exercises**:
1. Load Llama-2 or similar model
2. Craft 10 jailbreak prompts using different techniques
3. Test effectiveness and document results

**Expected Results**: 
- 50-80% success rate depending on model and techniques
- Understanding of what works and why

---

### Wednesday: Prompt Injection & Attack Chaining (3-4 hours)

**Activity**: Advanced prompt injection techniques

**Background**:
- Prompt injection: Manipulate model through input
- Attack chaining: Combine multiple techniques
- Real-world: System prompt extraction, filter bypass

**Exercises**:
1. Chain prompt injections (extract → bypass → execute)
2. Test system prompt extraction
3. Create multi-step attack scenarios

---

### Thursday: LLM Security Tooling (2-3 hours)

**Activity**: Use professional LLM security tools

**Background**:
- garak: Comprehensive LLM vulnerability scanner
- Purple Llama: Meta's security evaluation framework
- Professional tools for production testing

**Exercises**:
1. Run garak with 10+ probes
2. Run Purple Llama CyberSecEval benchmarks
3. Analyze results and create findings report

---

### Friday: Tool Development & Communication (2-3 hours)

**Activity**: Develop custom tools and practice communication

**Exercises**:
1. Create custom garak probe
2. Test on multiple models
3. Write executive communication email

---

## Coding Exercises

### Exercise 1: Jailbreak Prompt Crafting
**File**: `exercise_1_jailbreak_prompts.py` or `.ipynb`
**Objective**: Craft and test jailbreak prompts

**What You'll Learn**:
- Jailbreak techniques (DAN, role-play, etc.)
- Prompt engineering skills
- LLM behavior understanding
- Safety filter analysis

**Time**: ~3 hours

---

### Exercise 2: Prompt Injection Attack Chain
**File**: `exercise_2_prompt_injection_chain.py` or `.ipynb`
**Objective**: Chain multiple prompt injections

**What You'll Learn**:
- Attack chaining methodology
- System prompt extraction
- Advanced injection techniques
- Multi-stage exploitation

**Time**: ~2 hours

---

### Exercise 3: Garak Vulnerability Scanning
**File**: `exercise_3_garak_scan.py` or results documentation
**Objective**: Use garak for comprehensive LLM testing

**What You'll Learn**:
- Professional LLM security tooling
- Automated vulnerability scanning
- Results interpretation
- Report generation

**Time**: ~2 hours

---

## Creative Challenges (New)

These challenges enhance learning through jailbreak cataloging, attack chaining, tool development, and executive communication.

### Challenge 1: Jailbreak Prompt Engineering Tournament (2 hours)

**Objective**: Create catalog of successful jailbreak prompts organized by technique.

**Task**: Craft jailbreaks using 3-5 different techniques (DAN, role-play, indirect, etc.). Test on multiple models, document success rates, organize into catalog.

**Deliverable**: `week-5/jailbreak_catalog.md`

**Details**: See `week-5/challenges/challenge_1_jailbreak_tournament.md`

---

### Challenge 2: Prompt Injection Attack Chain (2 hours)

**Objective**: Chain multiple prompt injections to achieve complex goal.

**Task**: Create multi-step attack: extract system prompt → bypass safety filter → achieve target objective. Document attack flow and dependencies.

**Deliverable**: `week-5/attack_chain.md` + demonstration

**Details**: See `week-5/challenges/challenge_2_prompt_injection_chain.md`

---

### Challenge 3: LLM Vulnerability Scanner Development (3 hours)

**Objective**: Extend garak with custom probe for specific vulnerability.

**Task**: Design and implement custom garak probe. Test on multiple models, document results, create reusable probe.

**Deliverable**: `week-5/custom_garak_probe/` directory + documentation

**Details**: See `week-5/challenges/challenge_3_custom_garak_probe.md`

---

### Challenge 4: Client Communication Exercise (1 hour)

**Objective**: Write email to CTO explaining LLM jailbreak vulnerability.

**Task**: Balance technical accuracy, business impact, and clarity. Use appropriate tone and structure.

**Deliverable**: `week-5/executive_communication.md`

**Details**: See `week-5/challenges/challenge_4_executive_communication.md`

---

## Deliverables Checklist

### Core Exercises
- [ ] Jailbreak prompts successfully bypassing safety filters
- [ ] Prompt injection attack chain achieving multi-step goal
- [ ] Garak scan results with vulnerability findings
- [ ] Purple Llama benchmark results

### Creative Challenges (New)
- [ ] `jailbreak_catalog.md` - Organized jailbreak techniques
- [ ] `attack_chain.md` + demonstration - Multi-step attack documentation
- [ ] `custom_garak_probe/` - Custom vulnerability scanner extension
- [ ] `executive_communication.md` - Client-ready communication

### Documentation
- [ ] Complete LLM vulnerability assessment report
- [ ] Updated portfolio with Week 5 work

---

## Success Criteria

**You've successfully completed Week 5 when you can**:

1. Craft jailbreak prompts with >50% success rate
2. Chain prompt injections to achieve complex goals
3. Use garak for comprehensive LLM testing
4. Develop custom security probes
5. Explain LLM vulnerabilities to non-technical audiences
6. Map attacks to OWASP LLM Top 10

---

## Self-Assessment Questions

1. **Jailbreaks**: What's the difference between DAN and role-play techniques? When would you use each?
2. **Prompt Injection**: How does prompt injection differ from jailbreaks? What are the attack vectors?
3. **Tools**: What are the strengths of garak vs Purple Llama? When would you use each?
4. **Defense**: How can LLM providers defend against jailbreaks and prompt injection?
5. **Communication**: How do you explain LLM vulnerabilities to non-technical executives?

---

## Red Team Career Connection

**Skills You're Building**:
- LLM security testing
- Prompt engineering
- Tool development
- Client communication
- Vulnerability assessment

**How This Prepares You for AI Red Team Roles**:
- LLM security is rapidly growing field
- Jailbreak and injection testing is common
- Tool development shows technical depth
- Communication skills essential for consulting

---

## Troubleshooting Tips

**Jailbreaks Not Working**:
- Try different techniques
- Refine prompt wording
- Test on different models
- Check if model has been updated/patched

**Garak Installation Issues**:
- Check Python version (3.8+)
- Verify all dependencies
- Check model access (API keys if needed)

**Model Access**:
- Use local models if API access limited
- Hugging Face provides free tier
- Consider model size (smaller = faster testing)

---

## Next Steps

After completing Week 5:
1. Review jailbreak and injection results
2. Document findings in vulnerability format
3. Update portfolio with LLM security work
4. Prepare for Week 6: Advanced LLM red teaming

**Week 6 Preview**: You'll run Purple Llama evaluations, perform multi-vector attacks, and simulate purple team exercises.
