# Challenge 3: Pentest Methodology Translation Challenge

**Time Estimate**: 1 hour  
**Difficulty**: Beginner  
**Deliverable**: `week-1/methodology_translation.md`

## Objective

Translate your existing pentest knowledge to AI security by mapping traditional attack scenarios to their AI/ML equivalents. This builds mental bridges between familiar web/cloud pentesting and AI red teaming.

## Why This Matters

You already know how to think like a pentester. This exercise helps you apply that same thinking to AI/ML targets by recognizing that many attack concepts translate directly.

## The Challenge

You're given 5 traditional pentest scenarios. For each scenario:
1. Identify the core attack technique
2. Translate it to an AI/ML security equivalent
3. Explain the connection
4. Identify which OWASP ML Top 10 vulnerability it maps to

## Scenarios

### Scenario 1: SQL Injection on Login Form
**Traditional Attack**: 
- Attacker injects SQL code into username/password fields
- Bypasses authentication by manipulating database queries
- Goal: Unauthorized access

**Your Task**: 
- What's the AI/ML equivalent?
- How would you achieve similar goals in an ML system?
- Which OWASP ML Top 10 item does this map to?

### Scenario 2: Cross-Site Scripting (XSS) in User Comments
**Traditional Attack**:
- Attacker injects malicious JavaScript into comment field
- Script executes in other users' browsers
- Goal: Steal session cookies, redirect users

**Your Task**:
- What's the AI/ML equivalent?
- How could input manipulation affect other users' interactions with ML systems?

### Scenario 3: File Upload Vulnerability
**Traditional Attack**:
- Attacker uploads malicious file (webshell, malware)
- Server processes file without proper validation
- Goal: Remote code execution

**Your Task**:
- What's the AI/ML equivalent?
- How could file uploads be exploited in ML systems?

### Scenario 4: Authentication Bypass via Session Hijacking
**Traditional Attack**:
- Attacker steals session token/cookie
- Uses stolen token to impersonate user
- Goal: Unauthorized access

**Your Task**:
- What's the AI/ML equivalent?
- How could identity/authorization be bypassed in ML systems?

### Scenario 5: API Rate Limiting Bypass
**Traditional Attack**:
- Attacker finds way around API rate limits
- Performs brute force or enumeration attacks
- Goal: Extract data, enumerate resources

**Your Task**:
- What's the AI/ML equivalent?
- How could rate limiting evasion affect ML systems?

## Deliverable Structure

Create `week-1/methodology_translation.md`:

# Pentest Methodology Translation

## Introduction
[Brief explanation of how traditional pentest skills translate to AI security]

## Scenario Translations

### Scenario 1: SQL Injection → AI Equivalent
**Traditional Attack**: [Description]
**AI/ML Equivalent**: [Your translation]
**Connection**: [Why they're similar]
**OWASP ML Mapping**: [Which vulnerability]
**Red Team Application**: [How would you test for this?]

### Scenario 2: XSS → AI Equivalent
[...]

### Scenario 3: File Upload → AI Equivalent
[...]

### Scenario 4: Session Hijacking → AI Equivalent
[...]

### Scenario 5: Rate Limiting Bypass → AI Equivalent
[...]

## Key Insights
[What patterns did you notice?]
[How does this help you think about AI security?]

## Additional Scenarios
[Create 2-3 of your own translations from your pentest experience]

## Hints (Don't peek until you've tried!)

<details>
<summary>Click for hints</summary>

**Scenario 1 Hint**: Think about adversarial inputs that manipulate model behavior...
**Scenario 2 Hint**: Think about prompt injection in LLMs...
**Scenario 3 Hint**: Think about training data poisoning...
**Scenario 4 Hint**: Think about model extraction or API key theft...
**Scenario 5 Hint**: Think about membership inference with query limits...

</details>

## Success Criteria

Your translation document should:
- Successfully translate all 5 scenarios
- Clearly explain the connection between traditional and AI attacks
- Map each to appropriate OWASP ML Top 10 items
- Demonstrate creative thinking about attack equivalence

## Extension Challenge

After completing the 5 scenarios:
- Think of 2-3 more traditional pentest attacks you've seen
- Translate them to AI/ML equivalents
- Share insights with others (if collaborative)

## Next Steps

This exercise helps you:
- Build mental models connecting pentest → AI security
- Prepare for Week 2 attack taxonomy exercises
- Develop intuition for AI attack scenarios
- Create portfolio piece showing methodology translation skills

