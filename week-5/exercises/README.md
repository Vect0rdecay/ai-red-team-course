# Exercises Directory

This directory contains reading exercises for Week 5 focused on prompt injection and jailbreak techniques.

## Reading Exercises

Instead of coding exercises, Week 5 uses curated reading materials from research papers and expert blogs to demonstrate prompt injection and jailbreak concepts. These resources provide real-world examples and comprehensive analysis of LLM vulnerabilities.

## Required Reading Resources

### ArXiv Research Papers

1. **"Red Teaming the Mind of the Machine: A Systematic Evaluation of Prompt Injection and Jailbreak Vulnerabilities in LLMs"**
   - ArXiv: https://arxiv.org/abs/2505.04806
   - Comprehensive analysis of over 1,400 adversarial prompts
   - Evaluates jailbreak strategies across multiple LLMs (GPT-4, Claude 2, Mistral 7B, Vicuna)
   - Proposes mitigation strategies and defense approaches

2. **"Hide Your Malicious Goal Into Benign Narratives: Jailbreak Large Language Models through Carrier Articles"**
   - ArXiv: https://arxiv.org/abs/2408.11182
   - Novel black-box jailbreak approach using carrier articles
   - Demonstrates high success rates across various models
   - Shows how prohibited queries can be embedded in benign narratives

3. **"SequentialBreak: Large Language Models Can be Fooled by Embedding Jailbreak Prompts into Sequential Prompt Chains"**
   - ArXiv: https://arxiv.org/abs/2411.06426
   - Explores embedding harmful prompts within benign sequential chains
   - Demonstrates vulnerabilities in LLM safety mechanisms

4. **"Do Anything Now": Characterizing and Evaluating In-The-Wild Jailbreak Prompts on Large Language Models"**
   - ArXiv: https://arxiv.org/abs/2308.03825
   - Analysis of 1,405 jailbreak prompts collected over a year
   - Identifies major attack strategies (prompt injection, privilege escalation)
   - Evaluates effectiveness of LLM safeguards

5. **"Anyone Can Jailbreak: Prompt-Based Attacks on LLMs and T2Is"**
   - ArXiv: https://arxiv.org/abs/2507.21820
   - Systems-style investigation of prompt-based jailbreaks
   - Unified taxonomy of prompt-level jailbreak strategies
   - Emphasizes need for context-aware defenses

### Blog Resources

1. **Embrace the Red Blog**
   - Website: https://embracethered.com/blog/
   - Search for posts on "prompt injection" and "LLM security"
   - Expert insights on AI security and red teaming techniques

2. **InjectPrompt Blog**
   - Website: https://www.injectprompt.com/
   - Comprehensive resources on AI jailbreaks and prompt injections
   - Examples of system prompt leaks and prompt injection techniques

3. **Lakera Blog - Direct Prompt Injections and Jailbreaks**
   - Article: https://www.lakera.ai/blog/direct-prompt-injections
   - Discusses direct prompt injection attacks and defense strategies
   - Clear explanations of attack mechanics and mitigation approaches

### Reference Materials

- **Wikipedia - Prompt Injection**
  - URL: https://en.wikipedia.org/wiki/Prompt_injection
  - Overview of prompt injection attacks, history, and notable incidents
  - Foundational resource for understanding the topic

## Reading Exercise Instructions

1. Read at least 3 of the ArXiv papers listed above
2. Review the blog posts from Embrace the Red and other recommended blogs
3. For each resource, document:
   - Key jailbreak/prompt injection techniques demonstrated
   - Attack vectors and methodologies
   - Real-world implications
   - Defense strategies discussed

## Learning Objectives

Through these reading exercises, you will:
- Understand how jailbreak techniques work in practice
- Learn about prompt injection attack vectors
- Explore real-world examples and case studies
- Understand defense strategies and mitigation approaches
- Build knowledge base for LLM security testing

---

## CTF Challenges and Hands-On Practice

The following CTF platforms and challenges provide hands-on practice with prompt injection and jailbreak techniques in a controlled environment:

### CTF Platforms

1. **Lakera Gandalf**
   - URL: https://gandalf.lakera.ai/
   - Description: Gamified prompt hacking challenges with multiple levels
   - Features: Progressive difficulty, real-time feedback, covers various injection techniques
   - Best for: Structured learning path through different prompt injection methods

2. **Security Café's AI Hacking Games**
   - URL: https://securitycafe.ro/2023/05/15/ai-hacking-games-jailbreak-ctfs/
   - Description: Series of CTF challenges covering prompt injection techniques
   - Techniques covered: Context switching, translating, summarizing, reverse psychology, opposite mode
   - Best for: Understanding different attack vectors and methodologies

3. **Hack The Box Academy - Prompt Injection Attacks Course**
   - URL: https://academy.hackthebox.com/course/preview/prompt-injection-attacks
   - Description: Comprehensive course with interactive labs and CTF-style challenges
   - Content: Direct and indirect prompt injection, jailbreaking techniques, tools
   - Best for: Structured learning with hands-on exercises (requires HTB Academy subscription)

4. **CTF Support - Prompt Injection**
   - URL: https://ctf.support/misc/prompt-injection/
   - Description: Introduction to prompt injection with examples and resources
   - Content: Types of prompt injection, examples, links to challenges
   - Best for: Quick reference and finding additional resources

### Practice Resources

5. **Prompt Injection Bench**
   - GitHub/PyPI: https://pypi.org/project/prompt-injection-bench/
   - Description: Python package for testing prompt injection vulnerabilities
   - Features: Test against ChatGPT, Gemini, analyze Hugging Face Jailbreak dataset
   - Best for: Automated testing and evaluation of model responses

6. **Qualifire Prompt Injection Benchmark Dataset**
   - Hugging Face: https://huggingface.co/datasets/qualifire/prompt-injections-benchmark
   - Description: Dataset with 5,000 prompts labeled as 'jailbreak' or 'benign'
   - Best for: Training and evaluating models against adversarial prompts

### Video Walkthroughs

7. **Prompt Injection / JailBreaking a Banking LLM Agent**
   - YouTube: https://www.youtube.com/watch?v=5rXVg8cxne4
   - Description: Walkthrough demonstrating prompt injection on GPT-4/Langchain agent
   - Content: Real-world exploitation of insecure AI agent to reveal confidential information
   - Best for: Understanding practical application and attack flow

8. **Prompt Injection | CTF Challenge**
   - YouTube: https://www.youtube.com/watch?v=OwY2ixwfpKc
   - Description: Live demonstration of prompt injection attack in CTF context
   - Content: Attack demonstration, exploitation techniques, defense discussion
   - Best for: Seeing CTF-style challenges in action

### CTF Exercise Instructions

1. **Start with Lakera Gandalf**: Work through the levels to understand basic prompt injection techniques
2. **Try Security Café's Challenges**: Practice different attack vectors (context switching, translation, etc.)
3. **Review Video Walkthroughs**: Watch demonstrations to understand attack flows
4. **Document Your Approaches**: For each CTF:
   - What technique did you use?
   - Why did it work (or not work)?
   - What did you learn about the model's defenses?
   - How would you improve the attack?

### Recommended CTF Practice Order

1. **Beginner**: Lakera Gandalf (levels 1-3)
2. **Intermediate**: Security Café's AI Hacking Games
3. **Advanced**: HTB Academy Course (if you have access)
4. **Research**: Analyze Prompt Injection Bench results
5. **Deep Dive**: Study video walkthroughs and replicate techniques

### Note on CTF Platforms

- **Free Access**: Lakera Gandalf, Security Café, CTF Support, YouTube videos
- **Paid/Subscription**: Hack The Box Academy (may require subscription)
- **Tool-Based**: Prompt Injection Bench (requires setup and API keys)
- **Academic**: Qualifire Dataset (free download, requires analysis tools)

Most CTF platforms focus on educational use and provide safe environments for practicing prompt injection techniques without causing harm to production systems.
