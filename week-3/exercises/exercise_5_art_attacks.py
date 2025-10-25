"""
Week 3 - Exercise 5: Using ART Library for Attacks

Objective: Learn to use Adversarial Robustness Toolbox (ART) for attacks

Red Team Context: Using established libraries speeds up testing in engagements.
- Don't reinvent the wheel - use proven tools
- ART provides battle-tested implementations
- Faster to integrate into pentest workflows
- Allows focus on attack strategy rather than implementation details

ART (Adversarial Robustness Toolbox): Industry-standard library
- Maintained by IBM
- Comprehensive attack and defense implementations
- Production-grade code with extensive testing

Note: Requires 'adversarial-robustness-toolbox' package
Install: pip install adversarial-robustness-toolbox

This exercise demonstrates using ART's pre-built attacks:
- Membership inference attacks: Detect training data leakage
- Evasion attacks: Fool models with adversarial samples (FGSM, PGD, etc.)
- Model extraction attacks: Steal model functionality through queries
- Poisoning attacks: Corrupt training data
- Backdoor attacks: Insert hidden triggers

Key concepts:
- Estimator: Wrapper around your model to work with ART
- Attack: Tool that generates adversarial samples
- Attack Success Rate: Percentage of samples that successfully evade the model
"""

print("Week 3 Exercise 5: ART Library Usage")
print("Install: pip install adversarial-robustness-toolbox")
print("See ART documentation for attack implementations")
print("Key attacks to test: EvasionAttack, InferenceAttack")
