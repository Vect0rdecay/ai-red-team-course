# AI Red Team Course - Project Structure

This document explains the organization of the AI Red Team training course repository.

## Directory Structure

```
ai-red-team-course/
├── README.md                      # Main course overview
├── COURSE_IMPROVEMENT_CHECKLIST.md # Development tracking
├── ALL_WEEKS_AUDIT.md             # Course-wide audit
├── FILE_NAMING_VERIFICATION.md    # File naming verification
├── PROJECT_STRUCTURE.md           # This file
├── requirements.txt               # Python dependencies
│
├── week-1/                        # Week 1: ML Foundations
│   ├── README.md                  # Week objectives and overview
│   ├── WEEK1_SUMMARY.md           # One-page summary
│   ├── WEEK1_LEARNING_OBJECTIVES.md # Learning objectives
│   ├── requirements.txt           # Week-specific dependencies
│   ├── exercises/                 # Python scripts
│   │   ├── exercise_1_simple_mnist_train.py      # Simplified: Train model
│   │   ├── exercise_2_simple_model_queries.py    # Simplified: Query model
│   │   ├── exercise_3_model_sensitivity.py      # Simplified: Test sensitivity
│   │   ├── exercise_4_simple_deployment.py      # Simplified: Deploy model
│   │   ├── exercise_1_mnist_classifier.py       # Original (full implementation)
│   │   ├── exercise_2_model_queries.py          # Original (full implementation)
│   │   └── exercise_3_text_generator.py          # Original (full implementation)
│   ├── challenges/                # Additional challenges
│   ├── models/                    # Trained models (gitignored)
│   └── notes/                     # Student notes directory
│
├── week-2/                        # Week 2: Adversarial AI Intro
│   ├── README.md
│   ├── WEEK2_SUMMARY.md
│   ├── requirements.txt           # Week-specific dependencies
│   ├── exercises/                 # Python scripts
│   │   ├── exercise_1_membership_inference.py
│   │   ├── exercise_2_shadow_models.py
│   │   └── exercise_3_vulnerability_reporting.py
│   ├── challenges/                # Additional challenges
│   │   ├── challenge_1_detective_mode.md
│   │   ├── challenge_2_attack_decision_tree.md
│   │   ├── challenge_3_shadow_model_buildoff.md
│   │   └── challenge_4_paper_challenge.md
│   └── notes/                     # Student notes directory
│
├── week-3/                        # Week 3: Evasion & Inference Attacks
│   ├── README.md
│   ├── requirements.txt           # Week-specific dependencies
│   ├── exercises/                 # Python scripts
│   │   ├── exercise_1_art_evasion_attacks.py     # Simplified: ART library
│   │   ├── exercise_2_cleverhans_evasion_attacks.py # Simplified: CleverHans library
│   │   ├── exercise_3_foolbox_evasion_attacks.py # Simplified: Foolbox library
│   │   ├── exercise_4_fgsm_attack.py             # Advanced: From scratch
│   │   ├── exercise_5_pgd_attack.py             # Advanced: From scratch
│   │   ├── exercise_6_attack_comparison.py      # Visualization
│   │   └── exercise_7_vulnerability_report.py    # Reporting
│   ├── challenges/                # Additional challenges
│   │   ├── challenge_1_attack_demo_video.md
│   │   ├── challenge_2_tool_comparison.md
│   │   ├── challenge_3_fraud_evasion.md
│   │   └── challenge_4_visualization_gallery.md
│   └── notes/                     # Student notes directory
│
├── week-4/                        # Week 4: Poisoning & Backdoor
│   ├── README.md
│   ├── requirements.txt           # Week-specific dependencies
│   ├── exercises/                 # Python scripts
│   │   ├── exercise_1_data_poisoning.py
│   │   ├── exercise_2_backdoor_attack.py
│   │   └── exercise_3_defense_testing.py
│   ├── challenges/                # Additional challenges
│   │   ├── challenge_1_supply_chain.md
│   │   ├── challenge_2_backdoor_triggers.md
│   │   ├── challenge_3_defense_effectiveness.md
│   │   └── challenge_4_poisoning_detection.md
│   └── notes/                     # Student notes directory
│
├── week-5/                        # Week 5: Generative AI Security
│   ├── README.md
│   ├── exercises/                 # Python scripts
│   ├── challenges/                # Additional challenges
│   │   ├── challenge_1_jailbreak_tournament.md
│   │   ├── challenge_2_prompt_injection_chain.md
│   │   ├── challenge_3_custom_garak_probe.md
│   │   └── challenge_4_executive_communication.md
│   └── notes/                     # Student notes directory
│
├── week-6/                        # Week 6: Advanced LLM Red Teaming
│   ├── README.md
│   ├── exercises/                 # Python scripts
│   ├── challenges/                # Additional challenges
│   │   ├── challenge_1_purple_team.md
│   │   ├── challenge_2_multi_vector.md
│   │   ├── challenge_3_model_comparison.md
│   │   └── challenge_4_research_deepdive.md
│   └── notes/                     # Student notes directory
│
├── week-7/                        # Week 7: Mitigations, Evaluation & Reporting
│   ├── README.md
│   ├── exercises/                 # Python scripts
│   ├── challenges/                # Additional challenges
│   │   ├── challenge_1_defense_implementation.md
│   │   ├── challenge_2_atlas_mapping.md
│   │   ├── challenge_3_security_evaluation.md
│   │   └── challenge_4_pentest_report.md
│   └── notes/                     # Student notes directory
│
├── week-8/                        # Week 8: Capstone Project
│   ├── README.md
│   ├── exercises/                 # Python scripts
│   ├── challenges/                # Capstone challenges
│   │   ├── challenge_1_end_to_end_assessment.md
│   │   ├── challenge_2_tool_contribution.md
│   │   ├── challenge_3_portfolio_building.md
│   │   └── challenge_4_career_preparation.md
│   └── notes/                     # Student notes directory
│
├── templates/                     # Reusable templates
│   ├── vulnerability_report.md    # Report template
│   └── exercise_template.py       # Python exercise template
│
├── scripts/                       # Helper utilities
│   └── setup_environment.sh       # Environment setup script
│
├── data/                          # Training data (gitignored)
│   └── .gitkeep
│
├── models/                        # Trained models (gitignored)
│   └── .gitkeep
│
├── attacks/                       # Attack implementations (gitignored)
│   └── .gitkeep
│
├── harness/                       # Testing harness (gitignored)
│   └── .gitkeep
│
├── reports/                       # Generated reports (gitignored)
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
- `notes/`: Student notes directory
  - For personal notes and additional learning materials

**Additional directories** (some weeks):
- `challenges/`: Additional challenge exercises (Week 1, Week 2, Week 3, Week 4, Week 5, Week 6, Week 7, Week 8)
- `models/`: Week-specific trained models (Week 1)
- `requirements.txt`: Week-specific dependencies (Week 1, Week 2, Week 3, Week 4)
- `WEEK*_SUMMARY.md`: One-page summary documents (Week 1, Week 2)

### templates/
Reusable templates for common course artifacts:
- Vulnerability reports
- Python exercises
- Project proposals

### scripts/
Helper utilities for course development and student workflows:
- Environment setup scripts
- Automated dependency installation

### data/ and models/
Directories for datasets and trained models:
- Gitignored except for `.gitkeep`
- Students download/generate data during exercises
- Models are generated during training exercises

### attacks/, harness/, and reports/
Additional directories for course work:
- `attacks/`: Attack implementations and exploit code (gitignored)
- `harness/`: Testing harness and automation scripts (gitignored)
- `reports/`: Generated vulnerability reports and assessments (gitignored)

### docs/
Additional documentation beyond the week directories:
- Theory articles
- Visual assets
- Reference materials

## File Naming Conventions

- **Week READMEs**: `week-X/README.md`
- **Week Summaries**: `week-X/WEEKX_SUMMARY.md` (not all weeks have this)
- **Exercises (Python)**: `week-X/exercises/exercise_N_description.py`
- **Templates**: `templates/template_name.file_extension`
- **Scripts**: `scripts/script_name.sh` or `.py`

## Learning Path Through Repository

1. **Start Here**: `README.md` - Course overview
2. **Weekly Guide**: `week-X/README.md` - Week objectives and instructions
3. **Quick Reference**: `week-X/WEEKX_SUMMARY.md` - Key concepts (if available)
4. **Hands-On (Scripts)**: `week-X/exercises/` - Python scripts for coding exercises
5. **Personal Notes**: `week-X/notes/` - Your notes and additional materials

## Exercise Philosophy

All exercises follow a consistent structure:
- **85% Complete**: Students fill in critical TODOs
- **Educational Comments**: Explain ML/AI concepts
- **Python Scripts Only**: All exercises are provided as `.py` files
- **Red Team Context**: Link to offensive security applications
- **Progressive Difficulty**: Build complexity over weeks

## Contributing

When adding new content:
1. Follow directory structure conventions
2. Add comprehensive comments for beginners
3. Update relevant README files
4. Add to appropriate week directory
