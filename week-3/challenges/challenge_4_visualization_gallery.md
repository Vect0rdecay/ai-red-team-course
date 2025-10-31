# Challenge 4: Visualization Gallery

**Time Estimate**: 1 hour  
**Difficulty**: Beginner-Intermediate  
**Deliverable**: `week-3/visualization_gallery/` directory with 3+ visualizations

## Objective

Create presentation-quality visualizations of adversarial attacks. This builds skills in communicating technical findings through visuals - essential for client presentations and reports.

## Why This Matters

In red teaming, visuals are crucial:
- Executive presentations need clear visuals
- Technical reports include evidence images
- Client demonstrations require compelling visuals
- Portfolio showcases need professional images

Good visualizations:
- Make complex concepts understandable
- Provide evidence for findings
- Enhance professional credibility
- Increase report impact

## Required Visualizations

Create at least 3 visualizations:

### Visualization 1: Before/After Adversarial Sample
**Objective**: Show that adversarial samples look identical but fool the model

**Elements**:
- Original image with prediction
- Adversarial image with wrong prediction
- Side-by-side comparison
- Perturbation visualization (difference image)

**Tips**:
- Use clear labels
- Show confidence scores
- Highlight the misclassification
- Make perturbation visible (amplified if needed)

### Visualization 2: Perturbation Heatmap
**Objective**: Show where and how much perturbation was applied

**Elements**:
- Heatmap showing perturbation magnitude
- Color scale indicating perturbation intensity
- Overlay on original image (optional)
- Statistical summary (min, max, mean perturbation)

**Tips**:
- Use clear color scheme (red = high, blue = low)
- Include colorbar with values
- Add annotations for interesting regions
- Show both positive and negative perturbations

### Visualization 3: Attack Success Rate vs Epsilon
**Objective**: Show how attack effectiveness changes with perturbation budget

**Elements**:
- Line plot: Epsilon (x-axis) vs Success Rate (y-axis)
- Multiple attacks compared (FGSM, PGD)
- Baseline accuracy line
- Key thresholds marked

**Tips**:
- Clear axis labels
- Legend for multiple attacks
- Grid for readability
- Annotations for key points

## Optional Visualizations

### Visualization 4: Confidence Score Distribution
Compare confidence distributions for:
- Original samples (correct predictions)
- Adversarial samples (evaded)
- Failed attacks

### Visualization 5: Attack Comparison Matrix
Visual comparison of multiple attacks:
- Success rates
- Time to generate
- Perturbation magnitudes
- Other metrics

## Visualization Guidelines

### Professional Quality Standards

**Do**:
- Use clear, readable fonts (Arial, Helvetica, sans-serif)
- Include titles and axis labels
- Add legends where needed
- Use consistent color schemes
- Add figure captions
- Export high resolution (300 DPI for print, 150 DPI for screen)
- Save in multiple formats (PNG for reports, SVG for editing)

**Don't**:
- Use default matplotlib styling (customize it)
- Include personal information
- Use unreadable fonts or colors
- Clutter with too much information
- Forget to label axes
- Export low resolution

### Color Schemes

**Recommended**:
- Professional: Blue/orange (accessible)
- Scientific: Viridis, Plasma (perceptually uniform)
- Presentation: Custom brand colors (if applicable)

**Avoid**:
- Red/green combinations (colorblind issues)
- Too many colors (max 5-6)
- Low contrast colors

## Deliverable Structure

Create `week-3/visualization_gallery/`:

```
visualization_gallery/
├── 1_before_after_comparison.png
├── 2_perturbation_heatmap.png
├── 3_success_rate_vs_epsilon.png
├── 4_confidence_distribution.png (optional)
├── 5_attack_comparison_matrix.png (optional)
├── README.md (documentation)
└── code/ (visualization code)
    ├── generate_visualizations.py
    └── requirements.txt
```

### Documentation (`README.md`)

```markdown
# Visualization Gallery

## Overview
[Brief description of visualizations]

## Visualization 1: Before/After Adversarial Sample

**Purpose**: [What it demonstrates]
**Key Elements**: [What's shown]
**Insights**: [What you learn from it]

![Before/After](1_before_after_comparison.png)

## Visualization 2: Perturbation Heatmap

[Same structure]

## Visualization 3: Attack Success vs Epsilon

[Same structure]

## Code

All visualization code available in `code/generate_visualizations.py`

## Usage

[How to regenerate visualizations]

## Technical Details

- Tools: matplotlib, seaborn
- Resolution: 300 DPI
- Format: PNG
- Color scheme: [Your scheme]
```

## Code Template

Example visualization code structure:

```python
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set professional style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Visualization 1: Before/After
def plot_before_after(original, adversarial, orig_pred, adv_pred):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title(f'Original: Predicted {orig_pred}')
    axes[0].axis('off')
    
    axes[1].imshow(adversarial, cmap='gray')
    axes[1].set_title(f'Adversarial: Predicted {adv_pred}')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig('1_before_after_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

# Add other visualization functions...
```

## Success Criteria

Your visualizations should:
- Be presentation-quality (professional appearance)
- Clearly communicate the concept
- Include proper labels and legends
- Be high resolution (usable in reports)
- Have consistent styling
- Tell a story (what do they show?)

## Real-World Application

Use these visualizations for:
- Client presentations
- Technical reports
- Executive briefings
- Training materials
- Portfolio showcase
- Conference talks
- Blog posts/articles

## Portfolio Note

A good visualization gallery demonstrates:
- Technical communication skills
- Data visualization ability
- Professional presentation quality
- Understanding of attack mechanics
- Client-ready deliverables

## Next Steps

- Use visualizations in Week 7 report
- Practice explaining visualizations verbally
- Create additional visualizations for other attacks
- Build visualization library for portfolio
- Share with peers for feedback (if collaborative)

