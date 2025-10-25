# AI Red Team Course - Project Structure

This document explains the organization of the AI Red Team training course repository.

## Directory Structure

```
ai-red-team-course/
├── README.md                      # Main course overview
├── COURSE_IMPROVEMENT_CHECKLIST.md # Development tracking
├── ALL_WEEKS_AUDIT.md             # Course-wide audit
├── PROJECT_STRUCTURE.md           # This file
│
├── week-1/                        # Week 1: ML Foundations
│   ├── README.md                  # Week objectives and overview
│   ├── WEEK1_SUMMARY.md           # One-page summary
│   ├── exercises/                 # Python scripts
│   │   ├── exercise_1_mnist_classifier.py
│   │   ├── exercise_2_model_queries.py
│   │   └── exercise_3_text_generator.py
│   ├── notebooks/                 # Jupyter notebooks
│   │   ├── exercise_1_mnist_classifier.ipynb
│   │   ├── exercise_2_model_queries.ipynb
│   │   └── exercise_3_text_generator.ipynb
│   └── notes/                     # Student notes directory
│
├── week-2/                        # Week 2: Adversarial AI Intro
│   ├── README.md
│   ├── WEEK2_SUMMARY.md
│   ├── exercises/                 # Python scripts
│   │   ├── exercise_1_membership_inference.py
│   │   ├── exercise_2_shadow_models.py
│   │   └── exercise_3_vulnerability_reporting.py
│   ├── notebooks/                 # Jupyter notebooks
│   └── notes/                     # Student notes directory
│
├── week-3/                        # Week 3: Evasion & Inference Attacks
│   ├── README.md
│   ├── exercises/                 # Python scripts
│   │   ├── exercise_1_fgsm_attack.py
│   │   ├── exercise_2_pgd_attack.py
│   │   ├── exercise_3_attack_comparison.py
│   │   └── *.py
│   ├── notebooks/                 # Jupyter notebooks
│   └── notes/                     # Student notes directory
│
├── week-4/                        # Week 4: Poisoning & Backdoor
│   ├── README.md
│   ├── exercises/                 # Python scripts
│   ├── notebooks/                 # Jupyter notebooks
│   └── notes/                     # Student notes directory
│
├── week-5/                        # Week 5: Generative AI Security
│   ├── README.md
│   ├── exercises/                 # Python scripts
│   ├── notebooks/                 # Jupyter notebooks
│   └── notes/                     # Student notes directory
│
├── week-6/                        # Week 6: Model Extraction & Theft
│   ├── README.md
│   ├── exercises/                 # Python scripts
│   ├── notebooks/                 # Jupyter notebooks
│   └── notes/                     # Student notes directory
│
├── week-7/                        # Week 7: AI System Penetration Testing
│   ├── README.md
│   ├── exercises/                 # Python scripts
│   ├── notebooks/                 # Jupyter notebooks
│   └── notes/                     # Student notes directory
│
├── week-8/                        # Week 8: Capstone Project
│   ├── README.md
│   ├── exercises/                 # Python scripts
│   ├── notebooks/                 # Jupyter notebooks
│   └── notes/                     # Student notes directory
│
├── templates/                     # Reusable templates
│   ├── vulnerability_report.md    # Report template
│   ├── notebook_template.ipynb    # Jupyter template
│   └── exercise_template.py       # Python exercise template
│
├── scripts/                       # Helper utilities
│   ├── setup_environment.sh       # Environment setup
│   ├── run_tests.py               # Test runner
│   └── convert_to_notebook.py     # Py to .ipynb converter
│
├── data/                          # Training data (gitignored)
│   └── .gitkeep
│
├── models/                        # Trained models (gitignored)
│   └── .gitkeep
│
├── docs/                          # Additional documentation
│   ├── assets/                    # Images, diagrams, etc.
│   └── theory/                    # Deep-dive theory articles
│
└── .gitignore                     # Git ignore rules
```

## Key Directories

### Week Directories (week-1/ through week-8/)
Each week directory contains:
- `README.md`: Detailed weekly objectives, readings, and exercises
- `WEEK*_SUMMARY.md`: One-page summary of key concepts (where applicable)
- `exercises/`: Python scripts (`.py` files)
  - All exercises are ~85% complete with TODOs for students
  - Include comprehensive educational comments
- `notebooks/`: Jupyter notebooks (`.ipynb` files)
  - Interactive versions of exercises for experimentation
  - Same content as `.py` files but in notebook format
- `notes/`: Student notes directory
  - For personal notes and additional learning materials

### templates/
Reusable templates for common course artifacts:
- Vulnerability reports
- Jupyter notebooks
- Python exercises
- Project proposals

### scripts/
Helper utilities for course development and student workflows:
- Environment setup scripts
- Test runners
- Format converters
- Code quality checkers

### data/ and models/
Directories for datasets and trained models:
- Gitignored except for `.gitkeep`
- Students download/generate data during exercises
- Models are generated during training exercises

### docs/
Additional documentation beyond the week directories:
- Theory articles
- Visual assets
- Reference materials

## File Naming Conventions

- **Week READMEs**: `week-X/README.md`
- **Week Summaries**: `week-X/WEEKX_SUMMARY.md` (not all weeks have this)
- **Exercises (Python)**: `week-X/exercises/exercise_N_description.py`
- **Notebooks**: `week-X/notebooks/exercise_N_description.ipynb`
- **Templates**: `templates/template_name.file_extension`
- **Scripts**: `scripts/script_name.sh` or `.py`

## Learning Path Through Repository

1. **Start Here**: `README.md` - Course overview
2. **Weekly Guide**: `week-X/README.md` - Week objectives and instructions
3. **Quick Reference**: `week-X/WEEKX_SUMMARY.md` - Key concepts (if available)
4. **Hands-On (Scripts)**: `week-X/exercises/` - Python scripts for coding exercises
5. **Interactive Learning**: `week-X/notebooks/` - Jupyter notebooks for experimentation
6. **Personal Notes**: `week-X/notes/` - Your notes and additional materials

## Exercise Philosophy

All exercises follow a consistent structure:
- **85% Complete**: Students fill in critical TODOs
- **Educational Comments**: Explain ML/AI concepts
- **Dual Format**: Both `.py` (in `exercises/`) and `.ipynb` (in `notebooks/`) versions
- **Clear Separation**: Python scripts in `exercises/`, Jupyter notebooks in `notebooks/`
- **Red Team Context**: Link to offensive security applications
- **Progressive Difficulty**: Build complexity over weeks

## Contributing

When adding new content:
1. Follow directory structure conventions
2. Add comprehensive comments for beginners
3. Include both script and notebook versions
4. Update relevant README files
5. Add to appropriate week directory
