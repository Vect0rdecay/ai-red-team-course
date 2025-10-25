# Course Improvement Checklist
**AI Red Teaming Transition Course - Enhancement Tasks**

## Overview
This checklist organizes improvements needed to transform this course from a solid foundation into a production-ready training program for web/cloud pentesters transitioning to AI red teaming.

### Strategic Context
- **Course Materials Strategy**: Create original notebooks and exercises to maintain commercial value
- **Textbook Integration**: Use Sotiropoulos (AI Security) and Géron (ML Fundamentals) as primary references
- **Attribution Approach**: Reference external repos for students' independent study, but build unique implementations
- **Competitive Advantage**: Original code and exercises distinguish this course from other offerings

---

## Phase 1: Infrastructure & Setup (Foundation)

### Environment Setup
- [ ] Create a comprehensive `SETUP.md` with:
  - [ ] Step-by-step environment setup (Python venv, CUDA/CPU)
  - [ ] Platform-specific instructions (Ubuntu 22.04, macOS, Windows WSL2)
  - [ ] Docker Compose setup for consistent environments
  - [ ] Troubleshooting guide for common issues
- [ ] Update `requirements.txt` with:
  - [ ] Pinned versions for reproducibility
  - [ ] Optional dependencies marked clearly
  - [ ] GPU vs CPU variants
  - [ ] Installation order if dependencies conflict
- [ ] Add verification script: `scripts/verify_setup.py`
  - [ ] Check Python version (3.8+)
  - [ ] Verify CUDA/CPU availability
  - [ ] Test imports for all required packages
  - [ ] Check disk space and memory
- [ ] Create `.env.example` for configuration management

### Project Structure
- [x] Add detailed directory structure to README
- [x] Create subdirectories for all 8 weeks:
  - [x] `week-2/` through `week-8/` with placeholder READMEs
  - [x] Consistent structure: `notebooks/`, `exercises/`, `notes/`
- [x] Add `scripts/` directory for helper utilities
- [x] Add `data/` directory with .gitkeep (exclude from git except metadata)
- [x] Create `templates/` directory for reports and notebooks

### Git & Version Control
- [ ] Expand `.gitignore` to exclude:
  - [ ] Jupyter checkpoints (`__pycache__/`, `.ipynb_checkpoints/`)
  - [ ] Large model files (keep `.pt`, `.h5` in .gitignore, document where to get them)
  - [ ] Dataset folders
  - [ ] IDE-specific files (`.vscode/`, `.idea/`)
- [ ] Add CONTRIBUTING.md for future contributors
- [ ] Add LICENSE file
- [ ] Create initial tags/versioning system

---

## Phase 2: Week-by-Week Content Enhancement

### Week 1: ML Foundations
**Reference Texts**: Géron Ch. 1-4 (setup, fundamentals), Sotiropoulos Ch. 1 (introduction)
**External Resources**: 
- Géron GitHub: https://github.com/ageron/handson-ml3
- Use for concepts, create our own implementations

- [ ] Create detailed week README with:
  - [ ] Specific learning objectives (3-5 bullet points)
  - [ ] Prerequisites check
  - [ ] Time estimates per activity
  - [ ] Success criteria for completion
  - [ ] Textbook page references
- [ ] Build **ORIGINAL** starter notebooks (commercial value):
  - [ ] `mnist_classifier.ipynb` - Unique PyTorch implementation with pentester-oriented commentary
  - [ ] `text_generator.ipynb` - Custom simple RNN implementation (not copying Géron's code)
  - [ ] Add cell-level explanations with security context
  - [ ] Include "why this matters for red teaming" callouts
- [ ] Create exercise files:
  - [ ] `owasp_ml_top10_template.md` with template structure
  - [ ] Example completed mapping (week-1/examples/)
  - [ ] Reference diagram: ML attack surfaces (original, not copied)
- [ ] Add assessment:
  - [ ] Quiz on ML fundamentals (10 questions)
  - [ ] Code review checklist for students
- [ ] Create companion resources:
  - [ ] Reference Géron's repo for students' optional practice
  - [ ] Recommended reading (with page numbers from both texts)
  - [ ] Common pitfalls section
- [ ] Attribution note: "Reference Géron's implementation here: [link] for comparison"

### Week 2: Core AI Adversarial Concepts
**Reference Texts**: Sotiropoulos Ch. 2-3 (attack taxonomy), Géron Ch. 13-14 (CNN fundamentals)
**External Resources**: 
- Sotiropoulos GitHub: [research repo, link to be added]
- Create original implementations referencing concepts

- [ ] Detailed lifecycle mapping document (ORIGINAL):
  - [ ] Side-by-side comparison table (PenTest vs AI Red Team)
  - [ ] Mermaid diagram for visual learners (custom design)
- [ ] Paper summary template with required sections:
  - [ ] Problem statement
  - [ ] Methodology
  - [ ] Real-world impact
  - [ ] CVSS-style risk rating
- [ ] **ORIGINAL** membership inference notebook:
  - [ ] Custom implementation (don't copy Sotiropoulos code verbatim)
  - [ ] Data requirements clearly stated
  - [ ] Expected outputs defined
  - [ ] Success metrics (>X% inference rate)
  - [ ] "Inspired by concepts from [paper X], implemented uniquely here"
- [ ] Add "Day in the Life" simulation:
  - [ ] Scenario: "Client asks you to assess their ML pipeline"
  - [ ] Step-by-step workflow
- [ ] Attribution: "For additional exercises, see Sotiropoulos Chapter 2 exercises"

### Week 3: Evasion & Inference Attacks
**Reference Texts**: Sotiropoulos Ch. 4 (evasion attacks), Géron Ch. 11-12 (neural nets)
**External Resources**: Foolbox docs, ART tutorials

- [ ] **ORIGINAL** Foolbox tutorial (custom structured examples):
  - [ ] Installation steps with common errors
  - [ ] Gradual progression: single attack → batch → full pipeline
  - [ ] Visualization requirements (before/after images)
  - [ ] Unique examples: target Week 1's custom MNIST model
- [ ] Specific target metrics:
  - [ ] 90%+ evasion rate for FGSM
  - [ ] Success criteria for PGD (adversarial accuracy <10%)
- [ ] Report template for membership inference findings
- [ ] Add comparative analysis section:
  - [ ] Compare attack effectiveness
  - [ ] Resource requirements (time, compute)
- [ ] MITRE mapping: Map attacks to ATT&CK for ML framework
- [ ] Attribution: "Reference Sotiropoulos Ch. 4 for theory; our implementation provides hands-on practice"

### Week 4: Poisoning & Backdoors
**Reference Texts**: Sotiropoulos Ch. 5 (poisoning attacks)
**External Resources**: ART poisoning tutorials, torchattacks docs

- [ ] **ORIGINAL** poisoning notebook:
  - [ ] Synthetic dataset creation (custom dataset if possible)
  - [ ] Poison injection rate strategies (1%, 5%, 10%, 50%)
  - [ ] Impact measurement methods
  - [ ] Unique scenarios tailored to red team perspective
- [ ] **ORIGINAL** backdoor implementation:
  - [ ] Pixel-level trigger (custom patterns, not copy-paste)
  - [ ] Semantic trigger examples
  - [ ] Activation rate metrics
  - [ ] Test against Week 1 model
- [ ] Defense implementation guide:
  - [ ] Pruning code (original implementation or clear adaptation)
  - [ ] Expected accuracy trade-offs
  - [ ] How to measure robustness improvement
- [ ] Cloud scenario integration:
  - [ ] "Company uses S3 for training data"
  - [ ] Scenario: How would you test data integrity?
- [ ] Attribution: "See Sotiropoulos Ch. 5 for comprehensive theoretical background"

### Week 5: Generative AI Vulnerabilities
**Reference Texts**: Sotiropoulos Ch. 6-7 (LLM attacks)
**External Resources**: Garak docs, Purple Llama guides

- [ ] LLM setup guide (unique to our course):
  - [ ] HuggingFace account creation
  - [ ] Model downloading process
  - [ ] API key management (never commit secrets)
  - [ ] Local vs cloud inference options
  - [ ] Custom helper scripts for model management
- [ ] **ORIGINAL** jailbreak prompt library:
  - [ ] 10+ custom jailbreak templates (not copied from web)
  - [ ] Categorized by technique type
  - [ ] Success rate tracking (systematic testing methodology)
  - [ ] Our own naming/categorization system
- [ ] Garak setup and configuration:
  - [ ] Installation troubleshooting
  - [ ] **ORIGINAL** custom probe creation guide
  - [ ] Output interpretation
  - [ ] Create our own unique probes for course
- [ ] XSS → Prompt Injection analogy exercise (ORIGINAL mapping)
- [ ] Add ethical boundaries section
- [ ] Attribution: "For comprehensive LLM security theory, see Sotiropoulos Ch. 6-7"

### Week 6: Advanced LLM Red Teaming
**Reference Texts**: Sotiropoulos Ch. 8 (advanced LLM attacks)

- [ ] Purple Llama integration (ORIGINAL walkthrough):
  - [ ] Installation walkthrough tailored to our environment
  - [ ] Running benchmarks locally
  - [ ] Interpreting results
  - [ ] Custom benchmarks if possible
- [ ] **ORIGINAL** chained attack example:
  - [ ] Step 1: Jailbreak to extract system prompt
  - [ ] Step 2: Use info for more targeted injection
  - [ ] Step 3: Exfiltrate training data snippets
  - [ ] Unique scenario not found elsewhere
- [ ] **ORIGINAL** collaboration templates:
  - [ ] Email template for "data scientist" (professional pentest style)
  - [ ] Meeting notes template
  - [ ] Remediation tracking sheet
- [ ] Real-world case study analysis (ORIGINAL write-up):
  - [ ] Select recent LLM vulnerability (e.g., from GPT or Claude incident)
  - [ ] Analyze attack vector with our methodology
  - [ ] Our own breakdown/lessons learned
- [ ] Attribution: "Sotiropoulos Ch. 8 covers additional advanced techniques"

### Week 7: Mitigations & Reporting
**Reference Texts**: Sotiropoulos Ch. 9-10 (defenses and mitigations)

- [ ] **ORIGINAL** adversarial training implementation:
  - [ ] Modified training loop code (custom approach, not copy-paste)
  - [ ] Robustness metrics calculation
  - [ ] Trade-off analysis (accuracy vs robustness)
  - [ ] Apply to our Week 1 model
- [ ] **ORIGINAL** professional report template:
  - [ ] Executive summary (1 page)
  - [ ] Technical findings (with screenshots)
  - [ ] Risk scoring matrix
  - [ ] Remediation recommendations
  - [ ] Appendix with raw data
  - [ ] Branded for this course
- [ ] **ORIGINAL** mitigation comparison table:
  - [ ] Technique | Effectiveness | Cost | Implementation Time
  - [ ] Custom evaluation criteria
- [ ] Add MITRE Adversarial ML Threat Matrix exercise
- [ ] Attribution: "Sotiropoulos Ch. 9-10 provides comprehensive defense strategies"

### Week 8: Integration & Portfolio
**Reference Texts**: Sotiropoulos Conclusion, course synthesis

- [ ] **ORIGINAL** end-to-end simulation specification:
  - [ ] Defined threat model (unique to course)
  - [ ] Attack scenarios (3+ variations)
  - [ ] Assessment criteria
  - [ ] Integration of all weeks' concepts
- [ ] **ORIGINAL** portfolio structure guide:
  - [ ] What to include (notebooks, reports, write-ups)
  - [ ] What NOT to include (sensitive customer data)
  - [ ] GitHub README template for portfolio (branded)
- [ ] Tool contribution guide:
  - [ ] Fork workflow
  - [ ] Code quality standards
  - [ ] Pull request process
- [ ] **ORIGINAL** resume/LinkedIn optimization guide:
  - [ ] AI security keywords
  - [ ] Project description templates (for this course)
  - [ ] Quantifiable achievements section
- [ ] Career path resources:
  - [ ] Job boards (HiddenLayer, Adversa, etc.)
  - [ ] Certifications to consider
  - [ ] Interview prep questions
- [ ] Attribution: "Course designed with principles from Sotiropoulos and Géron"

---

## Phase 3: Supporting Materials & Tools

### Documentation
- [ ] Create `GLOSSARY.md` with AI/ML security terms (ORIGINAL definitions)
- [ ] Add `FAQ.md` addressing common questions
- [ ] Build troubleshooting guide (common errors with solutions)
- [ ] Create "Quick Reference" cheat sheet (PDF) - branded
- [ ] Add `CHANGELOG.md` for course updates
- [ ] Add `ATTRIBUTIONS.md` documenting:
  - [ ] Textbook references (Sotiropoulos, Géron)
  - [ ] External repos used for reference
  - [ ] Creative Commons or open source tools used
  - [ ] Clear separation between external resources and original content

### Code Quality
- [ ] Add pre-commit hooks:
  - [ ] Black formatter
  - [ ] Flake8 linting
  - [ ] Pylint for complexity
- [ ] Add docstrings to all Python code
- [ ] Create unit tests for harness.py and critical functions
- [ ] Add type hints to all Python files

### Teaching Aids
- [ ] Create presentation slides (PowerPoint/Google Slides) for each week
- [ ] Develop video script outlines for key concepts
- [ ] Build interactive notebooks with checkpoints
- [ ] Create diagram collection (Mermaid files for all diagrams)
- [ ] Design printable handouts for key concepts

### Assessments
- [ ] Weekly quizzes (5-10 questions each)
- [ ] Mid-course project: Build custom attack tool
- [ ] Final exam: Simulate pentest scenario
- [ ] Peer review process for reports
- [ ] Rubric for grading submissions

---

## Phase 4: Production Readiness

### DevOps & Automation
- [ ] GitHub Actions workflow:
  - [ ] Run tests on PRs
  - [ ] Check code style
  - [ ] Validate notebooks (execute and save outputs)
- [ ] Dockerfile for reproducible environment
- [ ] Docker Compose for multi-service setup (model server + attacker)
- [ ] Script to generate course progress tracking CSV

### Collaboration Features
- [ ] Issue templates for:
  - [ ] Bug reports
  - [ ] Feature requests
  - [ ] Questions
- [ ] Discussion forum setup (GitHub Discussions)
- [ ] Community guidelines (CODE_OF_CONDUCT.md)
- [ ] Maintainer contact information

### Accessibility & Usability
- [ ] Add alt text to all images/diagrams
- [ ] Ensure color-blind friendly plots
- [ ] Test with screen readers
- [ ] Provide keyboard navigation guides
- [ ] Clear font choices in all materials

### Legal & Ethical
- [ ] Add ethical guidelines document
- [ ] Disclaimer about offensive security education
- [ ] Responsible disclosure policy
- [ ] **ATTRIBUTION REQUIREMENTS**:
  - [ ] Document all source code references
  - [ ] Clear attribution for textbook exercises referenced
  - [ ] Separation between original content and adapted content
  - [ ] License compliance check for all dependencies
- [ ] Usage rights clarification:
  - [ ] Course content licensing (for commercial sales)
  - [ ] Student work licensing
  - [ ] Third-party content usage rights
- [ ] Intellectual property protection:
  - [ ] Copyright notices for original content
  - [ ] Trademark considerations
- [ ] Attribution strategy for textbooks:
  - [ ] "This course is designed to complement [Textbook] by [Author]"
  - [ ] "Original implementations created for this course"
  - [ ] "Adapted from concepts in [Textbook] Chapter X"

---

## Phase 5: Advanced Enhancements

### Interactive Learning
- [ ] Jupyter notebook extensions:
  - [ ] Custom prompt injection simulator
  - [ ] Interactive attack builder (GUI)
- [ ] Gamification elements:
  - [ ] Badges for completed modules
  - [ ] Leaderboard for attack success rates
- [ ] Progress tracking dashboard (optional: Streamlit app)

### Real-World Integration
- [ ] Partnerships: Reach out to AI security vendors for guest content
- [ ] Case studies: Document real pentests (anonymized)
- [ ] Interview series: Record conversations with AI red teamers
- [ ] Job board integration: Curated listings

### Ongoing Updates
- [ ] Update schedule: Quarterly review of content
- [ ] News section: Track recent AI security incidents
- [ ] Research tracker: Keep up with latest arXiv papers
- [ ] Tool updates: Track version changes in dependencies

---

## Quick Wins (Do First)
Priority order for maximum impact:

1. Create SETUP.md with environment setup
2. Add verification script (`scripts/verify_setup.py`)
3. Create week-by-week detailed READMEs with objectives
4. Build starter notebooks for Week 1 (with extensive comments) - **ORIGINAL implementations**
5. Expand .gitignore for proper project hygiene
6. Create report templates in `templates/` directory - **ORIGINAL branded templates**
7. Add assessments (quizzes) for first 2 weeks
8. Document common pitfalls and troubleshooting
9. Create `ATTRIBUTIONS.md` file for textbook references
10. Add textbook chapter references to each week's README

---

## Success Metrics
Track these to measure course improvement:

- **Completion Rate**: % of students finishing all 8 weeks
- **Time to First Attack**: How quickly students execute their first evasion
- **Report Quality**: Peer review scores on Week 7 reports
- **Portfolio Outcomes**: Number of students landing AI security roles
- **Community Growth**: Stars, forks, contributions
- **Issue Resolution Time**: Average time to answer student questions

---

**Next Steps**: Start with Quick Wins section, then proceed through phases systematically.
