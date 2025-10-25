"""
Week X - Exercise Y: [Exercise Title]

Objective: [What students will learn]

Red Team Context: [How this relates to offensive AI security]

INSTRUCTIONS:
This script is ~85% complete. Your task is to fill in the TODO sections.
Read each TODO carefully and implement according to the hints provided.

Key Concepts:
- [Concept 1]: [Brief explanation]
- [Concept 2]: [Brief explanation]
- [Concept 3]: [Brief explanation]
"""

# ============================================================================
# IMPORTS
# ============================================================================
# [Add necessary imports with comments explaining their purpose]
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# STEP 1: SETUP AND CONFIGURATION
# ============================================================================
print("="*70)
print("[Exercise Title]")
print("="*70)

# Configuration
[CONFIG_VARIABLES] = [VALUES]

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================================================
# STEP 2: [Section Title]
# ============================================================================
print("\n[Section description]...")

def function_name(parameters):
    """
    [Function description]
    
    Args:
        parameter: [Description]
    
    Returns:
        return_value: [Description]
    """
    # TODO: [What needs to be implemented]
    # HINT: [Helpful hint for implementation]
    # [Additional context if needed]
    result = None
    
    return result

# ============================================================================
# STEP 3: [Section Title]
# ============================================================================
print("\n[Section description]...")

# TODO: Implement main logic
# HINT: [Guidance for implementation]

# Example structure:
# for item in items:
#     # TODO: [Specific task]
#     # Process item
#     pass

# ============================================================================
# STEP 4: [Section Title]
# ============================================================================
print("\n[Section description]...")

# TODO: [Implementation details]

# ============================================================================
# STEP 5: VISUALIZATION AND RESULTS
# ============================================================================
print("\nGenerating results...")

# TODO: Create visualizations
# HINT: Use matplotlib to plot results

plt.figure(figsize=(10, 6))
# TODO: Add plot code
plt.title("[Plot Title]")
plt.xlabel("[X Label]")
plt.ylabel("[Y Label]")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('[filename].png', dpi=150)
print("Saved: [filename].png")

# ============================================================================
# STEP 6: SUMMARY AND ANALYSIS
# ============================================================================
print("\n" + "="*70)
print("SUMMARY")
print("="*70)

# TODO: Print summary statistics
print(f"[Metric]: {value}")
print(f"[Metric]: {value}")

# Analysis questions for students:
# 1. [Question about understanding]
# 2. [Question about implications]
# 3. [Question about red team application]

# ============================================================================
# STEP 7: SAVE RESULTS
# ============================================================================
print("\nSaving results...")

# TODO: Save any important artifacts
# torch.save(model.state_dict(), 'model.pt')
# np.save('results.npy', results)

print("\nExercise complete!")

# ============================================================================
# NOTES FOR STUDENTS
# ============================================================================
"""
[Optional notes section]

Key Takeaways:
- [Takeaway 1]
- [Takeaway 2]
- [Takeaway 3]

Red Team Application:
[How this exercise applies to actual AI red teaming]

Further Reading:
- [Resource 1]
- [Resource 2]
- [Resource 3]
"""
