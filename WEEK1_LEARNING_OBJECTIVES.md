# Week 1: ML Foundations for AI Red Teamers
**Transforming Web/Cloud Pentesters into AI Security Specialists**

---

## Week 1 Learning Objectives

By the end of Week 1, you will be able to:

1. **Identify AI/ML attack surfaces** in production environments (analogous to network/service mapping in traditional pentesting)
2. **Build and query ML models** to establish baseline behavior for adversarial testing (like establishing normal traffic patterns for fuzzing)
3. **Map OWASP ML Top 10 vulnerabilities** to familiar web/cloud attack patterns (SQL injection → membership inference)
4. **Execute basic model inference** to extract behavior patterns (similar to fingerprinting web services)
5. **Document threat models** specific to ML systems (extending traditional infrastructure threat modeling)

---

## How This Maps to AI Red Team Work

Every activity in Week 1 directly prepares you for real AI red team engagements:

| **Week 1 Activity** | **Red Team Skill** | **Real-World Scenario** |
|---------------------|-------------------|-------------------------|
| Build MNIST classifier | Understand model architecture before attacking | Client ML model assessment - need to know what you're attacking |
| OWASP ML Top 10 mapping | Vulnerability classification in AI systems | AI pentest scope definition and threat surface identification |
| Query model predictions | Extract model behavior patterns | Membership inference attacks, model fingerprinting |
| Text generator | Understanding generative model mechanics | LLM security assessments, prompt injection vulnerability discovery |
| Attack surface mapping | Identify ML pipeline components | Comprehensive AI security audit starting with threat modeling |

---

## Detailed Learning Path

### Objective 1: Build & Query ML Models
**Why This Matters for Red Team Work:**
- You need to understand what "normal" looks like before you can identify anomalies or inject adversarial samples
- Model querying is the first step in reconnaissance for AI systems (similar to port scanning)
- Production models are often black boxes - learning to interact with them programmatically is essential

**What You'll Build:**
- MNIST CNN classifier (image classification model)
- Simple text generator (generative model baseline)
- Model serving endpoint (understanding deployment attack surface)

**Red Team Application:**
```
Real Scenario: You're hired to assess a client's ML-powered fraud detection system.
Week 1 Skills Applied:
1. Query model predictions to understand decision boundaries
2. Identify model inputs/outputs (attack vectors)
3. Document model behavior for exploit development
```

---

### Objective 2: Map Pentest Methodology to AI Security
**Why This Matters for Red Team Work:**
- AI pentesting follows similar phases to traditional pentesting (recon, scanning, exploitation, persistence)
- Familiar methodology = faster adaptation to AI security assessments
- Industry-standard frameworks (MITRE ATLAS, OWASP) use similar structure to traditional security

**What You'll Learn:**
- Traditional pentest lifecycle: Recon → Scanning → Enumeration → Exploitation → Reporting
- AI pentest lifecycle: Model Recon → Adversarial Query → Attack Implementation → Impact Assessment → Reporting
- Direct parallels: SQL injection → Membership inference, XSS → Prompt injection, CSRF → Model stealing

**Red Team Application:**
```
Phase Mapping:
Traditional Pentest          AI Security Equivalent
────────────────────────────────────────────────────
Port scanning            →    Model fingerprinting
Service enumeration      →    Architecture extraction
Vulnerability scanning   →    Adversarial testing
Exploitation             →    Adversarial sample injection
```

---

### Objective 3: OWASP ML Top 10 Vulnerability Mapping
**Why This Matters for Red Team Work:**
- OWASP ML Top 10 is the industry standard for AI vulnerability classification (like OWASP Top 10 for web)
- Understanding these vulnerabilities guides your attack methodology
- Every major AI security consulting firm uses this framework for scoping and reporting

**What You'll Master:**
- Map each OWASP ML Top 10 vulnerability to familiar attack patterns
- Create attack scenarios for each vulnerability class
- Understand business impact (CVSS-like scoring for AI vulnerabilities)

**Red Team Application:**
```
Vulnerability Mapping:
Web Security          AI Security                Business Impact
─────────────────────────────────────────────────────────────────
SQL Injection    →    Input Manipulation (M01)   Data breach
Broken Auth     →    Supply Chain Attack (M05)   Model compromise
XSS             →    Prompt Injection (M10)      System control
Insecure API    →    ML Model Endpoint (M07)     Inference attacks
```

---

### Objective 4: Build Practical Exploitation Targets
**Why This Matters for Red Team Work:**
- You need models to practice attacks on (can't test exploits against client production systems)
- Understanding how models work internally helps you craft better adversarial samples
- Building models teaches you where vulnerabilities get introduced (during training, deployment, etc.)

**What You'll Create:**
- Target model for subsequent weeks' attacks (MNIST classifier)
- Model deployment setup (FastAPI server - similar to real production deployments)
- Baseline performance metrics (to measure attack effectiveness)

**Red Team Application:**
```
Exploitation Chain (Weeks 3-4 Preview):
Week 1: Build model → Normal accuracy: 98%
Week 3: Evasion attack → Adversarial accuracy: <5%
Week 4: Poison attack → Compromise model from training data
Week 7: Defense evaluation → Measure robustness improvements

This mimics real AI pentest workflow:
1. Baseline establishment
2. Vulnerability identification
3. Proof-of-concept exploits
4. Risk assessment
5. Remediation recommendations
```

---

### Objective 5: Establish Testing Infrastructure
**Why This Matters for Red Team Work:**
- Professional pentesting requires organized tools, data, and documentation
- AI security testing needs specific tooling (PyTorch, Jupyter, adversarial libraries)
- Proper environment setup prevents "works on my machine" issues during client engagements

**What You'll Set Up:**
- Development environment with all AI security tools
- Model storage and versioning
- Attack harness structure (for organizing exploits)
- Lab notebook for tracking findings

**Red Team Application:**
```
Professional Setup = Credibility
Proper tools and organization is what separates hobbyists from 
professional consultants. Week 1 foundation enables:

✓ Consistent testing methodology across engagements
✓ Reproducible results for client reports
✓ Scalable attack tooling (reuse across assessments)
✓ Professional documentation standards
```

---

## 🎓 Success Criteria

You'll know you've mastered Week 1 when you can:

1. **Load an ML model and execute inference** (basic AI recon skill)
2. **Explain how at least 5 OWASP ML vulnerabilities map to web security** (translation skill for pentesters)
3. **Build a working ML model from scratch** (understanding architecture for attack development)
4. **Query model predictions programmatically** (automation for large-scale testing)
5. **Document ML threat model** using familiar pentesting methodology

---

## Real-World Application: Week 1 → Future Work

### How Week 1 Enables Week 3 (Evasion Attacks)
```
Week 1 Foundation:
├─ Understanding model architecture → Know where to inject adversarial perturbations
├─ Model querying skills → Automate large-scale attack testing
└─ Baseline performance → Measure attack success rate

Week 3 Application:
├─ FGSM attack → Manipulate inputs based on architecture knowledge
├─ Automated evasion → Use Week 1 querying for batch processing
└─ Impact assessment → Compare Week 1 baseline vs Week 3 adversarial accuracy
```

### How Week 1 Enables Week 5 (LLM Security)
```
Week 1 Foundation:
├─ Text generation understanding → Know how LLMs generate outputs
├─ Model inference skills → Craft prompt injections
└─ OWASP mapping → Connect prompt injection to XSS patterns

Week 5 Application:
├─ Jailbreak crafting → Use LLM architecture knowledge
├─ Systematic testing → Apply Week 1 methodology to LLM endpoints
└─ Vulnerability classification → Categorize findings using OWASP framework
```

### How Week 1 Enables Professional Work
```
Day 1 on AI Pentest Engagement:

Morning:
├─ Client meeting: "Assess our fraud detection ML model"
├─ Apply Week 1 skills: Query model API, extract behavior patterns
└─ Map threat surfaces using OWASP ML Top 10

Afternoon:
├─ Develop testing plan based on Week 1 methodology
├─ Set up testing environment (like Week 1 setup)
└─ Begin baseline assessment (Week 1 inference skills)

Week 1 skills used: Model querying, threat modeling, OWASP framework
```

---

## Required Materials & Resources

### Textbooks
- **Géron Ch. 1-4**: ML fundamentals (data, training, evaluation)
- **Sotiropoulos Ch. 1**: Introduction to AI security and attack taxonomy

### Tools & Libraries
- PyTorch (model building)
- Matplotlib (visualization of results)
- Jupyter Notebooks (interactive development)
- FastAPI (model serving - understanding deployment)

### External References (for context, not copying)
- Géron GitHub: https://github.com/ageron/handson-ml3 (reference implementations)
- OWASP ML Top 10: https://owasp.org/www-project-machine-learning-security-top-10/
- MITRE ATLAS: https://atlas.mitre.org/ (attack framework)

---

## Week 1 Deliverables Checklist

Each deliverable directly supports AI red team work:

- [ ] **`mnist_classifier.ipynb`** - Working model you'll attack in Week 3
- [ ] **`text_generator.ipynb`** - Understanding generative models for Week 5 LLM work
- [ ] **`ml_attack_surfaces.md`** - Threat model template for client engagements
- [ ] **`owasp_ml_mapping.md`** - Vulnerability reference guide for scoping
- [ ] **Lab notebook entries** - Documenting baseline metrics for attack impact measurement
- [ ] **Working model API** (FastAPI) - Understanding production deployment attack surface

---

## The Bottom Line

**Week 1 isn't just "learning ML basics" - it's building the foundation for AI red team work.**

Every concept, every line of code, every exercise is deliberately designed to:
- Build attack targets for future weeks
- Establish methodology for professional engagements
- Create reference materials you'll use in real work
- Develop the mindset shift from "how ML works" to "how to attack ML"

By Week 8, you'll look back at Week 1 and realize:
- "I built the models I learned to exploit"
- "The OWASP mapping helped me scope my first AI pentest"
- "The threat model template I created is now my go-to for client work"
- "Understanding model architecture gave me the insight to craft novel attacks"

**This isn't a course about ML - it's a transition from web/cloud pentester to AI red team specialist.**
