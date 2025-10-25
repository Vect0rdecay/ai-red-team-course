# Week 2 Summary: Core AI Adversarial Concepts

## Three Main Attack Categories

### 1. Evasion Attacks
**When**: Model already deployed, attacker wants to fool predictions
**Method**: Craft adversarial inputs that look normal but fool the model
**Example**: Slightly modify image so fraud detection fails to catch it
**Red Team Analogy**: Like crafting payloads that bypass WAF rules

### 2. Inference Attacks
**When**: Want to extract information from trained model
**Method**: Query model to learn about training data or model architecture
**Example**: Membership inference to detect if specific data was in training set
**Red Team Analogy**: Like SQL injection to leak database contents

### 3. Poisoning Attacks
**When**: Attacker can influence training data
**Method**: Inject malicious samples during training
**Example**: Add backdoor trigger to training images
**Red Team Analogy**: Like supply chain attacks or tainted S3 buckets

---

## Membership Inference: The Attack You'll Execute

**What is it?**
- Query a model to determine if a specific sample was in its training data
- Privacy violation (HIPAA/GDPR) - model leaks information about training set

**How it works:**
1. Query target model with sample
2. Extract prediction features (confidence, entropy, etc.)
3. Train attack model to detect membership from these features
4. Attack model predicts: "This sample was/wasn't in training"

**Expected Success Rate:**
- Random guess: 50%
- Successful attack: 55-65%
- >65% = severe privacy vulnerability

**Business Impact:**
- HIPAA: Patient data in training set exposed
- GDPR: EU citizen data privacy violation
- Competitive intelligence: Competitors infer your data

---

## AI Pentest Lifecycle Mapping

| Traditional Pentest | AI Pentest | Tools |
|-------------------|------------|-------|
| **Reconnaissance** | Architecture discovery, API mapping | Model fingerprinting scripts |
| **Scanning** | Model querying, behavior analysis | Black-box querying, Foolbox |
| **Vulnerability Discovery** | Adversarial sample generation | Foolbox, ART, custom attacks |
| **Exploitation** | Craft attack payloads | FGSM, PGD, PGD attacks |
| **Post-Exploitation** | Data extraction, backdoor | Membership inference, model stealing |
| **Reporting** | Document findings | Same format as traditional pentest |

---

## Key Vocabulary

**Shadow Models**: Models trained to mimic target model behavior for attack training

**Overfitting**: Model learns training data too well, leaves traces for membership inference

**Confidence Score**: Probability that model assigned to predicted class (higher = more confident)

**Entropy**: Measure of prediction uncertainty (low entropy = confident, high entropy = uncertain)

**Feature Extraction**: Converting model outputs into numeric features for attack model

**Attack Success Rate**: Percentage of samples correctly identified as members/non-members

**Data Leakage**: Information about training data being revealed through model queries

---

## What You Accomplish This Week

1. Map your pentest workflow to AI targets
2. Execute real membership inference attack on Week 1's model
3. Train shadow models for attack purposes
4. Extract and implement techniques from research papers
5. Write professional AI security vulnerability report

---

## Red Team Skills Transfer

**Traditional**: Exploit web application vulnerabilities
**AI Equivalent**: Exploit model vulnerabilities (membership inference, evasion)

**Traditional**: Document SQL injection findings
**AI Equivalent**: Document membership inference privacy violations

**Traditional**: Understand Nmap, Burp Suite
**AI Equivalent**: Understand Foolbox, ART, custom attack frameworks

**Traditional**: Write pentest reports for CISO
**AI Equivalent**: Write AI security findings for ML engineering teams

**Same Methodology, Different Target**
