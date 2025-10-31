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

**Activity**: Study jailbreak techniques through research papers and expert blogs

**Background**:
- Jailbreaks bypass LLM safety filters
- Techniques: DAN, role-play, indirect prompting, carrier articles, etc.
- Goal: Understand attack patterns and methodologies

**Reading Exercises**:
1. Read ArXiv papers on jailbreak techniques
2. Review Embrace the Red blog posts on LLM security
3. Analyze real-world jailbreak examples and case studies
4. Document key techniques and attack vectors

**Expected Results**: 
- Understanding of jailbreak methodologies
- Knowledge of attack patterns and success factors
- Awareness of defense strategies

---

### Wednesday: Prompt Injection & Attack Chaining (3-4 hours)

**Activity**: Study prompt injection techniques through research

**Background**:
- Prompt injection: Manipulate model through input
- Attack chaining: Combine multiple techniques
- Real-world: System prompt extraction, filter bypass

**Reading Exercises**:
1. Study prompt injection research papers
2. Analyze attack chaining methodologies
3. Review system prompt extraction techniques
4. Document multi-stage exploitation approaches

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

## Reading Exercises

Week 5 uses curated reading materials from research papers and expert blogs instead of coding exercises to demonstrate prompt injection and jailbreak concepts. These resources provide real-world examples and comprehensive analysis of LLM vulnerabilities.

**See**: `exercises/README.md` for complete list of reading resources

### Key Reading Resources

**ArXiv Research Papers**:
- Red Teaming the Mind of the Machine: Systematic Evaluation of Prompt Injection and Jailbreak Vulnerabilities (https://arxiv.org/abs/2505.04806)
- Hide Your Malicious Goal Into Benign Narratives: Jailbreak LLMs through Carrier Articles (https://arxiv.org/abs/2408.11182)
- SequentialBreak: LLMs Can be Fooled by Embedding Jailbreak Prompts into Sequential Prompt Chains (https://arxiv.org/abs/2411.06426)
- "Do Anything Now": Characterizing and Evaluating In-The-Wild Jailbreak Prompts (https://arxiv.org/abs/2308.03825)
- Anyone Can Jailbreak: Prompt-Based Attacks on LLMs and T2Is (https://arxiv.org/abs/2507.21820)

**Blog Resources**:
- Embrace the Red Blog: https://embracethered.com/blog/ (search for "prompt injection" and "LLM security")
- InjectPrompt Blog: https://www.injectprompt.com/
- Lakera Blog - Direct Prompt Injections: https://www.lakera.ai/blog/direct-prompt-injections

**CTF Challenges and Hands-On Practice**:
- Lakera Gandalf: https://gandalf.lakera.ai/ (gamified prompt hacking challenges)
- Security Café's AI Hacking Games: https://securitycafe.ro/2023/05/15/ai-hacking-games-jailbreak-ctfs/
- Hack The Box Academy - Prompt Injection Attacks Course: https://academy.hackthebox.com/course/preview/prompt-injection-attacks
- See `exercises/README.md` for complete list of CTF resources

**Time**: ~6-8 hours reading and analysis, plus optional CTF practice

---

## Creative Challenges (New)

These challenges enhance learning through jailbreak cataloging, attack chaining, tool development, and executive communication.

### Challenge 1: Jailbreak Research Catalog (2 hours)

**Objective**: Create catalog of jailbreak techniques based on research papers and blog analysis.

**Task**: Analyze research papers and blog posts to identify 5-10 different jailbreak techniques. Document each technique, its methodology, success factors, and examples from research. Organize into catalog.

**Deliverable**: `week-5/jailbreak_catalog.md`

**Details**: See `week-5/challenges/challenge_1_jailbreak_tournament.md`

**Note**: This challenge focuses on researching and cataloging techniques rather than implementing them.

---

### Challenge 2: Prompt Injection Attack Analysis (2 hours)

**Objective**: Analyze prompt injection attack chains from research literature.

**Task**: Study multi-step prompt injection attacks from research papers. Document attack flow, dependencies, and methodologies. Analyze how researchers chain techniques to achieve complex goals.

**Deliverable**: `week-5/attack_chain.md` + analysis

**Details**: See `week-5/challenges/challenge_2_prompt_injection_chain.md`

**Note**: This challenge focuses on analyzing research rather than implementing attacks.

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
- [ ] Read and analyze at least 3 ArXiv papers on jailbreaks/prompt injection
- [ ] Review Embrace the Red blog posts on LLM security
- [ ] Document key jailbreak and prompt injection techniques learned
- [ ] Garak scan results with vulnerability findings (if applicable)
- [ ] Purple Llama benchmark results (if applicable)

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

1. Understand jailbreak techniques and methodologies from research
2. Analyze prompt injection attack chains and methodologies
3. Use garak for comprehensive LLM testing (if applicable)
4. Develop custom security probes (if applicable)
5. Explain LLM vulnerabilities to non-technical audiences
6. Map attacks to OWASP LLM Top 10
7. Reference real-world examples from research papers and blogs

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

**Reading Resources**:
- ArXiv papers are freely available and downloadable as PDFs
- Blog posts may require web access
- Some papers may have companion websites or GitHub repositories
- Use paper search tools to find related work

**Garak Installation Issues** (if using):
- Check Python version (3.8+)
- Verify all dependencies
- Check model access (API keys if needed)

**Model Access** (if testing):
- Use local models if API access limited
- Hugging Face provides free tier
- Consider model size (smaller = faster testing)

---

## Next Steps

After completing Week 5:
1. Review jailbreak and injection research papers and blog analyses
2. Document findings and key techniques learned
3. Update portfolio with LLM security knowledge
4. Prepare for Week 6: Advanced LLM red teaming

**Week 6 Preview**: You'll run Purple Llama evaluations, perform multi-vector attacks, and simulate purple team exercises.
