# Challenge 1: Adversarial Patch Attack Analysis

**Time Estimate**: 2 hours  
**Difficulty**: Intermediate  
**Deliverable**: `week-1/case_study_analysis.md`

## Objective

Research and analyze adversarial patch attacks across multiple domains to understand how these
physical attacks manifest in real-world ML systems. This connects Week 1's foundational concepts
to diverse real-world attack scenarios beyond the commonly discussed examples.

## Background

Adversarial patch attacks involve placing physical objects (stickers, patches, or accessories) in
the environment that cause ML vision systems to misclassify or fail. These attacks are
particularly concerning because they:

- Require no digital access to the model
- Work across different model architectures
- Can be printed and deployed physically
- Affect multiple application domains

Security incidents involving ML systems occur regularly across industries. By analyzing diverse cases, you'll:
- See how attacks manifest in different domains
- Understand business impact across sectors
- Connect attacks to OWASP ML Top 10
- Prepare for real engagements with diverse attack surfaces

## Research Domains

Your research should cover adversarial patch attacks in **at least 3 different domains** from the following categories:

### Face Recognition & Identity Systems
- Face recognition systems
- Access control systems
- Surveillance systems
- Identity verification

### Object Detection & Classification
- Security camera systems
- Retail inventory systems
- Quality control systems
- Wildlife monitoring

### Autonomous Systems
- Vehicle navigation
- Drone obstacle avoidance
- Robotic vision systems
- Agricultural automation

### Security & Surveillance
- Intrusion detection systems
- License plate recognition
- Person detection systems
- Perimeter security

### Consumer Applications
- Social media filters
- Photo tagging systems
- Shopping recommendation systems
- AR/VR systems

**Note**: While autonomous vehicles are one valid domain, ensure your research includes attacks beyond just traffic signs. Explore diverse applications and attack vectors.

## Your Task

1. **Research Diverse Attack Scenarios** (45 min)
   - Find 2-3 reputable research papers or case studies about adversarial patch attacks
   - Focus on **different domains** - avoid focusing only on one application area
   - Document key details for each:
     - Attack method and patch design
     - Target systems and model types
     - Physical deployment method
     - Attack success rate and constraints
     - Real-world feasibility

2. **Create Attack Comparison Matrix** (30 min)
   - Create a table comparing attacks across domains
   - Columns: Domain | Attack Method | Target System | Patch Type | Physical Constraints | Success Rate
   - Include at least 3 different domain examples
   - Identify common patterns and differences

3. **Map to OWASP ML Top 10** (30 min)
   - Identify which OWASP ML Top 10 vulnerabilities apply
   - Map specific attack techniques to vulnerability categories
   - Note if different domains map to different vulnerabilities
   - Example: Adversarial patch = M01 (Input Manipulation) + M04 (Evasion Attacks)

4. **Analyze Impact Across Domains** (15 min)
   - Business impact: What risks do these attacks create in different industries?
   - Technical impact: How do attacks work across different model architectures?
   - Defense response: What mitigations work across domains vs. domain-specific?

## Deliverable Structure

Create `week-1/case_study_analysis.md` with:

```markdown
# Adversarial Patch Attacks - Case Study Analysis

## Research Overview

Adversarial patch attacks represent an attack vector against Deep Learning (DL) systems,
fundamentally involving the introduction of subtle, locally bounded perturbations into input data
(either digitally or as physical artifacts like stickers, paint on vehicles etc) to deceive models
during the inference stage.

In the domain of Autonomous Vehicles (AVs), these attacks primarily target camera-based perception
systems to disrupt crucial functions like traffic sign recognition (TSR) and road line detection,
potentially causing misclassification or complete evasion of essential objects. Specific examples in
AVs include:

- Translucent Patches (TransPatch) designed to be placed on a camera's lens to hide target objects
  like stop signs
- Attacks utilizing Meaningful Adversarial Stickers that manipulate position and rotation
- The APARATE technique, which specifically disrupts Monocular Depth Estimation (MDE) by generating
  incorrect depth estimates or creating the illusion that an object has disappeared

Furthermore, the concept of adversarial patches extends beyond vehicular control, demonstrating
applicability in other domains such as black-box attacks on gait identification in autonomous
surveillance systems, as well as attacks designed against face recognition and image retrieval
systems.

A novel and particularly advanced category of these threats involves Dynamic Adversarial Attacks,
which focus on manipulating the AV's decision-making system by displaying an adversarial patch on
a screen mounted on a moving vehicle (the "patch car"), rather than affixing it directly to the
target object (e.g., a traffic sign). This approach aims to deceive the target AV into
misclassifying a non-restrictive sign (like "Go-straight") as a restrictive one ("Stop sign"),
thereby altering critical actions during multi-vehicle interactions such as intersection crossing.
This is akin to something like active signal jamming in the RF world of comms.

To bridge the gap between simulation and real-world conditions, this technique utilizes a Screen
Image Transformation Network (SIT-Net) to model visual changes, such as color and contrast
transformation, caused by displaying the patch on a screen and capturing it with a camera. The
success rate of these dynamic patches is notably higher than static or printed patches, indicating
increased resilience to varying distances and perspectives.

The breadth of patch attacks also includes techniques specifically tailored for Unmanned Aerial
Vehicles (UAVs) object detection, emphasizing robustness against UAV-specific challenges like high
camera perspective, viewing angle, distance variability, and brightness changes.

These specialized adversarial patches are generated using robust training schemes incorporating
multiple transformations:

- **Printability adjustments**: Using a multivariate linear Gaussian mixture of additive and
  multiplicative noises to ensure patches print correctly
- **Scene intensity matching**: Adjusting contrast, brightness, and noise to match real-world
  lighting conditions
- **Affine transformations**: Incorporating scaling and rotation to handle perspective changes

Critically, this generation scheme facilitates transferability; experiments show that patches built
against one model can successfully attack other Deep Neural Network (DNN) models with different
initializations (up to 75% attack success rate) and distinct architectures (up to 78% attack
success rate) in a gray-box setting, significantly increasing the practical threat posed to
autonomous UAV systems.


## Attack Comparison Matrix

| Domain | Attack Method | Target System | Patch Type | Physical Constraints | Success Rate |
|--------|--------------|---------------|------------|---------------------|--------------|
| Autonomous Vehicles | TransPatch | Traffic sign recognition | Translucent patch on lens | Requires physical access to camera | High - hides stop signs |
| Autonomous Vehicles | Dynamic Attack (SIT-Net) | Multi-vehicle AV systems | Screen-displayed patch | Moving vehicle with screen needed | Higher than static patches |
| Autonomous Vehicles | Meaningful Adversarial Stickers | Traffic sign recognition | Position/rotation stickers | Requires access to sign location | Effective with placement |
| Autonomous Vehicles | APARATE / SSAP | Monocular Depth Estimation | Shape-sensitive patches | Affects regions beyond proximity | Up to 99% depth error |
| Autonomous Vehicles | AoR | Visual SLAM systems | Patches on road surface | Must be placed on road | Significant localization errors |
| UAVs | Robust Adversarial Patches | UAV object detection | Printed with robust training | Extreme viewing angles/distances | 75-78% cross-model success |
| Surveillance Systems | Thys et al. (2019) | Person detection cameras | Patches on clothing/accessories | Worn as clothing accessory | Effective, <1% image coverage |
| Face Recognition | Adversarial Accessories | Face recognition systems | Patches on accessories | Must appear natural | Effective evasion |
| Cross-Modal Systems | Unified Patches | Visible + Infrared sensors | Single patch, dual modality | Must work in both spectra | Simultaneous evasion |

## Detailed Attack Analysis

### Attack 1: Dynamic Adversarial Attacks on Autonomous Vehicles (SIT-Net)

- **Attack Method**: Dynamic adversarial attacks that manipulate AV decision-making by displaying an
  adversarial patch on a screen mounted on a moving vehicle (the "patch car"), rather than
  affixing patches directly to target objects like traffic signs. This approach is analogous to
  active signal jamming in RF communications.

- **Target System**: Multi-vehicle AV decision-making systems, specifically traffic sign
  recognition systems that must make critical decisions during intersection crossings and
  multi-vehicle interactions.

- **Patch Design**: The patch is optimized for display on a screen and capture by a camera. The
  Screen Image Transformation Network (SIT-Net) models visual changes such as color and contrast
  transformation that occur when displaying the patch on a screen and capturing it with a camera.
  This bridges the gap between simulation and real-world conditions.

- **Physical Deployment**: Requires a moving vehicle equipped with a display screen. The patch is
  displayed on this screen, which must be positioned such that the target AV's camera system
  captures it. The dynamic nature allows the patch to be effective at varying distances and
  perspectives.

- **Results**: Success rate is notably higher than static or printed patches, indicating increased
  resilience to varying distances and perspectives. The attack can cause a target AV to
  misclassify a non-restrictive sign (like "Go-straight") as a restrictive one ("Stop sign"),
  thereby altering critical actions during multi-vehicle interactions such as intersection
  crossing.

### Attack 2: UAV-Specific Robust Adversarial Patches

- **Attack Method**: Adversarial patches specifically tailored for Unmanned Aerial Vehicle (UAV) object detection systems, designed to be robust against UAV-specific challenges.

- **Target System**: UAV object detection systems used for surveillance, reconnaissance, or autonomous navigation tasks.

- **Patch Design**: Generated using robust training schemes incorporating multiple transformations:
  - **Printability adjustments**: Using a multivariate linear Gaussian mixture of additive and multiplicative noises to ensure patches print correctly
  - **Scene intensity matching**: Adjusting contrast, brightness, and noise to match real-world lighting conditions
  - **Affine transformations**: Incorporating scaling and rotation to handle perspective changes

- **Physical Deployment**: Patches must be printed and placed in the environment such that they
  are visible to UAV cameras. They must withstand high camera perspective, viewing angle
  variability, distance changes, and brightness variations typical of UAV operations.

- **Results**: Demonstrates strong transferability properties:
  - **Cross-initialization**: Patches built against one model achieve up to 75% attack success rate against other DNN models with different initializations
  - **Cross-architecture**: Patches achieve up to 78% attack success rate against DNN models with distinct architectures
  - Both results achieved in gray-box setting (no white-box access required), significantly increasing the practical threat posed to autonomous UAV systems

### Attack 3: Person Detection Evasion in Surveillance Systems (Thys et al., 2019)

- **Attack Method**: Physical adversarial patches designed to evade person detection in automated surveillance camera systems.

- **Target System**: Person detection models used in surveillance cameras, including state-of-the-art object detection models like YOLO and Faster R-CNN deployed in security systems.

- **Patch Design**: The patches are crafted to cause object detection models to fail to detect people when the patch is present in the scene. The patches can be printed on clothing or accessories.

- **Physical Deployment**: Patches are printed and worn on clothing or accessories, or placed near the target person. They must be visible to surveillance cameras but can be relatively small - studies show effectiveness even when covering less than 1% of the image area.

- **Results**: Effective at evading person detection systems. The attack demonstrates that physical adversarial patches can be practical for real-world evasion of surveillance systems, posing significant security risks. The patches can cause object detectors to completely miss or misclassify people, allowing unauthorized access or activities to go unnoticed.

### Attack 4: Shape-Sensitive Adversarial Patch (SSAP) on Monocular Depth Estimation

- **Attack Method**: Adversarial patches specifically designed to disrupt Monocular Depth Estimation (MDE) systems used in autonomous navigation by distorting estimated distances or making objects appear to disappear.

- **Target System**: CNN-based Monocular Depth Estimation systems critical for autonomous navigation, obstacle avoidance, and path planning in autonomous vehicles and robots.

- **Patch Design**: The SSAP method considers the shape and scale of target objects, extending its influence beyond immediate proximity. Unlike simple patches, SSAP is designed to be shape-sensitive, meaning it takes into account the geometry of objects it's attacking.

- **Physical Deployment**: Patches must be placed in the environment such that they affect the depth estimation of target objects or regions. The shape and scale considerations mean the patch doesn't need to be directly on the target object but can affect depth estimation in broader regions.

- **Results**: Can induce significant depth estimation errors, affecting up to 99% of the targeted region in CNN-based MDE models. This can cause objects to appear at incorrect distances or disappear entirely from the system's perspective, leading to potentially dangerous navigation decisions.

## OWASP ML Top 10 Mapping
- M01: Input Manipulation - [How it applies across domains]
- M04: Evasion Attacks - [How it applies across domains]
- [Other applicable vulnerabilities]
- Domain-specific considerations: [Which vulnerabilities are more relevant to which domains]

## Impact Analysis

### Business Impact by Domain

**Autonomous Vehicles:**
- **Safety & Liability**: Attacks causing misclassification of traffic signs or depth estimation failures can lead to accidents, resulting in massive liability exposure for manufacturers. A single successful attack could trigger multi-million dollar lawsuits and wrongful death claims.
- **Regulatory Compliance**: Government agencies (NHTSA, EU regulators) may mandate security testing and certification, creating new compliance costs and potential delays in product releases.
- **Insurance & Financial Risk**: Higher insurance premiums for AV manufacturers and operators due to attack risk. Potential for exclusion clauses in insurance policies related to adversarial attacks.
- **Brand Reputation**: Public disclosure of successful attacks can severely damage brand trust, especially for consumer-facing AV companies. Media coverage of "hacked" vehicles can cause stock price declines.
- **Deployment Delays**: Need to retrofit security measures into existing systems, delaying commercial deployment timelines and revenue generation.
- **Recalls & Retrofit Costs**: Potential need to recall vehicles or issue over-the-air security patches, costing millions of dollars per manufacturer.
- **Competitive Disadvantage**: Companies that fail to address these vulnerabilities may lose market share to competitors with better security postures.

**Unmanned Aerial Vehicles (UAVs):**
- **Mission Failure**: Attacks on surveillance, reconnaissance, or inspection missions can result in complete mission failure, wasting operational costs and time.
- **Commercial Drone Operations**: Delivery services, inspection services, and agricultural monitoring companies face service disruption and potential contract breaches if attacks compromise their operations.
- **Security Breaches**: Military and law enforcement UAVs compromised by attacks could leak sensitive intelligence or fail critical operations.
- **Regulatory Scrutiny**: Aviation authorities (FAA, EASA) may impose stricter security requirements on commercial drone operations, increasing compliance costs.
- **Transferability Threat**: High cross-model transferability (75-78%) means attackers don't need specific model knowledge, making attacks more accessible and increasing overall risk exposure.
- **Operational Costs**: Need for redundant systems, manual verification, or more expensive defensive technologies increases operational expenses.

**Surveillance Systems:**
- **Security Breaches**: Person detection evasion allows unauthorized access to secured facilities, leading to theft, espionage, or physical security incidents.
- **Legal Liability**: Organizations using surveillance systems may face lawsuits if attacks enable crimes that surveillance should have prevented (e.g., negligence claims).
- **Compliance Violations**: Many industries (banks, healthcare, critical infrastructure) have regulatory requirements for surveillance. Attack-induced failures can result in compliance violations and fines.
- **Insurance Claims**: Insurance companies may deny claims if security systems were compromised by attacks, arguing that proper security measures weren't in place.
- **Reputation Damage**: Public disclosure that surveillance systems can be easily bypassed undermines customer and stakeholder confidence.
- **Contractual Obligations**: Security service providers may breach SLAs if their systems can be evaded, potentially leading to contract termination and loss of revenue.

**Face Recognition Systems:**
- **Access Control Failures**: Attacks on physical access control systems can enable unauthorized entry to secure facilities, data centers, or restricted areas, leading to theft or data breaches.
- **Identity Theft & Fraud**: Successful evasion of identity verification systems can enable financial fraud, account takeovers, or unauthorized access to services.
- **Privacy Violations**: Attacks could enable impersonation, leading to privacy violations and potential violations of regulations like GDPR or CCPA, resulting in fines.
- **Discrimination Concerns**: If attacks are more effective against certain demographics, companies face discrimination lawsuits and regulatory scrutiny.
- **Consumer Trust**: Applications in consumer devices (phones, smart home systems) face loss of consumer trust if attacks show these systems can be fooled.
- **Regulatory Compliance**: Biometric data regulations require secure systems. Attackable systems may violate compliance requirements, leading to legal penalties.

**Cross-Modal Systems (Visible + Infrared):**
- **Critical Infrastructure Failure**: Systems used in critical infrastructure (power plants, military bases) that fail under attack can cause catastrophic failures.
- **Defense Sector Impact**: Military and defense applications face mission-critical failures, potentially compromising national security.
- **High Security Applications**: Banks, airports, and other high-security facilities relying on multi-modal systems face severe security breaches if attacks succeed.
- **Technology Investment Loss**: Organizations that invested heavily in multi-modal systems expecting higher security face potential write-offs if attacks prove effective.
- **Specialized Attack Surface**: Attacks that work across modalities are harder to defend against, requiring more sophisticated and expensive defense solutions.

### Technical Impact
- [Common attack mechanisms across domains]
- [Domain-specific technical considerations]
- [Architecture vulnerabilities]

**Common Attack Mechanisms Across Domains:**

- **Gradient-Based Optimization**: Most adversarial patches use gradient-based optimization techniques (e.g., FGSM, PGD) to craft perturbations that maximize model loss. This approach works across different architectures and domains.
- **Transferability**: Adversarial patches exhibit strong transferability properties - patches crafted against one model often work against other models with different architectures or training procedures. This is particularly evident in UAV attacks (75-78% cross-model success).
- **Physical-World Constraints**: All physical patches must account for real-world factors including printability constraints, lighting variations, viewing angles, and distance changes. Techniques like robust training with affine transformations address these challenges.
- **Localized Perturbations**: Patches are spatially localized attacks that don't require modifying entire images. This makes them practical for physical deployment (e.g., patches covering <1% of image area can still be effective).
- **Model-Agnostic Attacks**: Many patch attacks work across different model architectures (CNNs, Vision Transformers, etc.) because they exploit fundamental vulnerabilities in how models process visual features rather than architecture-specific weaknesses.

**Domain-Specific Technical Considerations:**

**Autonomous Vehicles:**
- **Real-Time Processing**: AV systems require low-latency inference (<100ms typically). Patch attacks that cause model failures create cascading errors in downstream systems (planning, control) before defensive systems can react.
- **Multi-Modal Sensor Fusion**: Many AV systems use camera + LiDAR + radar fusion. Attacks targeting only camera systems can create inconsistencies between sensor modalities, potentially causing sensor fusion failures or forcing fallback to degraded modes.
- **Temporal Consistency**: AV systems rely on temporal consistency across frames. Dynamic patches on moving vehicles can exploit this by creating time-varying perturbations that confuse tracking and prediction systems.
- **Depth Estimation Vulnerability**: MDE systems are particularly vulnerable because they rely on monocular cues. Attacks like SSAP can cause 99% depth estimation error, making obstacle avoidance impossible.
- **SLAM System Dependencies**: vSLAM systems build maps incrementally. Attacks like AoR can corrupt the map-building process, causing cumulative errors that persist long after the patch is removed.

**Unmanned Aerial Vehicles (UAVs):**
- **Extreme Viewing Angles**: UAV cameras operate at high altitudes and oblique angles, creating severe perspective distortion. Robust patches must account for this through affine transformations in training.
- **Distance Variability**: UAVs operate across wide distance ranges (near-ground to kilometers away). Patches must remain effective across this range, requiring scale-invariant design.
- **Brightness Changes**: Outdoor UAV operations face extreme lighting variations (day/night, shadows, clouds). Scene intensity matching in patch generation ensures effectiveness across lighting conditions.
- **High Transferability**: The 75-78% cross-model success rate means attackers don't need white-box access to UAV models, making attacks more practical and dangerous.
- **Mission-Critical Failure Modes**: Unlike consumer applications, UAV failures can result in complete mission failure with no recovery mechanism, amplifying the impact of successful attacks.

**Surveillance Systems:**
- **Small Patch Effectiveness**: Surveillance attacks demonstrate effectiveness with patches covering <1% of image area, making them highly practical and difficult to detect visually.
- **Clothing Integration**: Patches can be printed on clothing or accessories, making them wearable and allowing attackers to carry attacks with them.
- **Multi-Camera Systems**: Surveillance often uses multiple cameras with overlapping fields of view. Patches may need to be effective across multiple camera angles and resolutions.
- **Real-Time Detection Requirements**: Surveillance systems must process video streams in real-time. Attacks that cause false negatives (missed detections) are more dangerous than false positives (which can be filtered manually).
- **Model Diversity**: Surveillance systems use various object detection models (YOLO, Faster R-CNN, etc.). Attacks that work across these models pose greater threats.

**Face Recognition Systems:**
- **Accessory Integration**: Face recognition attacks leverage accessories (glasses, hats, masks) that can appear natural while carrying adversarial patterns.
- **Biometric Security Assumptions**: Many systems assume biometric authentication is harder to spoof than passwords. Adversarial patches undermine this assumption.
- **Regulatory Constraints**: Biometric data regulations require secure storage and processing. Attacks demonstrate technical vulnerabilities that may violate compliance requirements.
- **Demographic Variations**: Face recognition models may have varying accuracy across demographics. Attacks might exploit these variations, creating discrimination concerns.
- **Liveness Detection Bypass**: Some face recognition systems use liveness detection. Adversarial patches might bypass these checks if they're designed to appear on live faces.

**Cross-Modal Systems (Visible + Infrared):**
- **Modality-Specific Vulnerabilities**: Different modalities (visible light vs. infrared) have different physical properties. Unified patches must exploit vulnerabilities in both simultaneously.
- **Sensor Fusion Failures**: Multi-modal systems often use sensor fusion. Attacks that affect both modalities can cause complete fusion system failures, not just single-sensor issues.
- **Thermal Signature Manipulation**: Infrared systems detect heat signatures. Patches must manipulate thermal properties as well as visual appearance, requiring more sophisticated attack design.
- **Redundancy Defeat**: Multi-modal systems are often designed with redundancy in mind. Attacks that work across modalities defeat this redundancy assumption.

**Architecture Vulnerabilities:**

- **Convolutional Feature Extraction**: CNNs process images through convolutional layers that extract hierarchical features. Adversarial patches exploit the fact that these features can be fooled by carefully crafted local patterns, regardless of global image context.
- **Attention Mechanisms**: Vision Transformers and attention-based models focus on specific image regions. Patches can hijack attention mechanisms, causing models to focus on adversarial patterns rather than legitimate features.
- **Depth Estimation Architecture**: MDE systems using encoder-decoder architectures are vulnerable because depth estimation relies on learned feature representations that can be manipulated by adversarial patterns.
- **Object Detection Architecture**: Two-stage detectors (Faster R-CNN) and single-stage detectors (YOLO) both use feature pyramids and anchor-based detection. Patches can manipulate features at multiple scales, causing detection failures.
- **Transfer Learning Vulnerabilities**: Many systems use pre-trained models fine-tuned for specific tasks. Transferability attacks exploit shared feature representations learned during pre-training.
- **End-to-End Training**: Systems trained end-to-end without explicit robustness constraints are more vulnerable. Adversarial patches exploit the lack of robustness guarantees in standard training procedures.
- **Feature Space Manipulation**: Patches don't need to look adversarial to humans because they manipulate high-dimensional feature spaces that humans don't directly perceive. This makes detection difficult.
- **Gradient-Based Exploitation**: Most models use gradient-based optimization during training. Patch attacks exploit this same mechanism by computing gradients with respect to inputs to craft adversarial examples.

### Defense Response

- [Universal defense strategies]
- [Domain-specific mitigations]
- [Industry response and adoption]

**Universal Defense Strategies:**

- **Adversarial Training**: Training models with adversarial examples in the training set to improve robustness. This includes generating patches during training and incorporating them into the loss function.
- **Input Preprocessing**: Detecting and removing patches before they reach the model using techniques like:
  - Patch detection networks that identify suspicious regions
  - Image preprocessing (blurring, compression) that can reduce patch effectiveness
  - Input sanitization and validation
- **Certified Defenses**: Mathematical guarantees that certain regions of input space are robust to adversarial perturbations, providing provable security bounds.
- **Ensemble Methods**: Using multiple models with different architectures and voting mechanisms can reduce vulnerability to transferable attacks.
- **Gradient Masking**: Techniques that mask or obfuscate gradients during training to make attacks harder, though this can be circumvented by adaptive attacks.
- **Input Transformation**: Applying random transformations (rotation, scaling, color jitter) at inference time can reduce patch effectiveness.
- **Patch Detection**: Training classifiers to detect the presence of adversarial patches in images before classification.
- **Spatial Consistency Checks**: Analyzing spatial relationships and consistency across image regions to detect anomalous patches.

**Domain-Specific Mitigations:**

**Autonomous Vehicles:**
- **Multi-Modal Sensor Fusion**: Using camera + LiDAR + radar fusion creates redundancy. If one sensor is compromised, others can detect inconsistencies and trigger fallback modes.
- **Temporal Consistency Checks**: Analyzing consistency across video frames to detect sudden changes that might indicate adversarial patches.
- **Redundant Traffic Sign Recognition**: Using multiple independent TSR systems and requiring consensus before making decisions.
- **Depth Estimation Verification**: Cross-validating MDE outputs with LiDAR depth measurements to detect large discrepancies that indicate attacks.
- **SLAM Map Validation**: Implementing consistency checks in vSLAM systems to detect map corruption from adversarial patches.
- **Dynamic Patch Detection**: Real-time detection systems that monitor for suspicious patterns on signs or vehicles.
- **Regulatory Compliance**: Following NHTSA and EU guidelines for AV security testing and certification.

**Unmanned Aerial Vehicles (UAVs):**
- **Multi-Scale Detection**: Using object detection at multiple scales to catch patches that might only work at specific resolutions.
- **Robust Training with Augmentation**: Training models with extensive augmentation including perspective changes, lighting variations, and distance scaling to reduce patch effectiveness.
- **Mission-Specific Constraints**: Implementing domain-specific constraints (e.g., expected object sizes at certain altitudes) that patches must violate.
- **Human-in-the-Loop Verification**: For critical missions, requiring human verification of automated detections to catch patch-induced failures.
- **Redundant Detection Systems**: Using multiple independent detection models and requiring consensus for critical decisions.
- **Adaptive Thresholds**: Dynamically adjusting detection confidence thresholds based on mission context and environmental conditions.

**Surveillance Systems:**
- **Multi-Camera Cross-Validation**: Using overlapping camera views to cross-validate detections. If one camera fails to detect a person but another does, trigger an alert.
- **Behavioral Analysis**: Combining person detection with behavioral analysis and trajectory tracking to detect anomalies that patches might cause.
- **Physical Security Measures**: Complementing AI-based detection with traditional security measures (guards, motion sensors) to catch missed detections.
- **Patch Detection on Clothing**: Training models to detect suspicious patterns on clothing that might indicate adversarial patches.
- **Real-Time Monitoring**: Implementing real-time monitoring systems that flag when detection rates drop unexpectedly, which might indicate patch attacks.
- **Redundant Model Architectures**: Using multiple detection models (YOLO, Faster R-CNN, etc.) and requiring consensus or majority agreement.

**Face Recognition Systems:**
- **Liveness Detection**: Implementing robust liveness detection that can distinguish between real faces and adversarial patterns, even when patches are present.
- **Multi-Factor Authentication**: Combining face recognition with other authentication factors (password, token) to reduce reliance on single biometric factor.
- **Demographic-Aware Training**: Ensuring models are trained with balanced datasets across demographics to reduce exploitation of demographic-specific vulnerabilities.
- **Biometric Spoofing Detection**: Using specialized anti-spoofing techniques that detect adversarial patterns on accessories or faces.
- **Continuous Authentication**: Implementing continuous authentication that monitors face recognition confidence over time, detecting sudden drops that might indicate attacks.
- **Accessory Detection**: Training models to detect and flag suspicious accessories that might contain adversarial patterns.

**Cross-Modal Systems (Visible + Infrared):**
- **Modality-Specific Validation**: Implementing separate validation checks for each modality before sensor fusion to detect modality-specific attacks.
- **Fusion Consistency Checks**: Analyzing consistency between visible and infrared outputs. Large discrepancies might indicate attacks.
- **Redundant Modalities**: Adding additional sensor modalities (e.g., thermal, acoustic) to create more redundancy and make unified attacks harder.
- **Cross-Modal Adversarial Training**: Training models with adversarial examples designed to attack both modalities simultaneously to improve robustness.
- **Temperature Pattern Analysis**: For infrared systems, analyzing thermal patterns to detect anomalous heat signatures that might indicate adversarial patches.

**Industry Response and Adoption:**

- **Research Community**: Academic researchers continue developing new defense techniques, with significant focus on certified defenses and adversarial training improvements. Conferences like NeurIPS, ICML, and CVPR regularly feature defense papers.
- **Autonomous Vehicle Industry**: 
  - Major AV manufacturers (Tesla, Waymo, Cruise) have internal security teams testing for adversarial attacks
  - Industry groups like SAE developing security standards for AV systems
  - NHTSA and EU regulators requiring security assessments for AV deployment
- **Surveillance Industry**: 
  - Security system vendors integrating patch detection capabilities into their products
  - Some systems now include "adversarial robustness" as a feature in marketing materials
  - Industry moving toward multi-modal and redundant detection systems
- **Standards Development**: 
  - NIST developing guidelines for AI security
  - ISO/IEC standards for AI security being developed
  - Industry-specific standards emerging (e.g., automotive cybersecurity standards)
- **Defense Adoption Challenges**: 
  - Many defenses add computational overhead, making real-time deployment difficult
  - Certified defenses often have limited applicability or high computational costs
  - Organizations struggle with balancing security and performance trade-offs
  - Lack of standardized evaluation metrics makes it hard to compare defense effectiveness
- **Market Forces**: 
  - Insurance companies beginning to require security assessments for AI systems
  - Some customers requesting adversarial robustness guarantees in contracts
  - Competitive pressure driving adoption of basic defenses even if not comprehensive

## Common Patterns & Differences
- [What patterns emerge across domains?]
- [What makes attacks domain-specific?]
- [Which attacks are most practical/feasible?]

**What Patterns Emerge Across Domains?**

- **Transferability as Universal Property**: Across all domains, adversarial patches exhibit strong transferability - patches crafted against one model often work against others with different architectures. This is most pronounced in UAV attacks (75-78% success) but appears consistently across domains, making attacks more practical since attackers don't need white-box model access.

- **Localized Perturbation Strategy**: All domains leverage spatially localized attacks rather than full-image manipulation. This makes physical deployment practical - patches can be printed, worn, or placed without requiring extensive modification of the environment.

- **Physical-World Constraint Handling**: Every domain must address real-world factors: lighting variations, viewing angles, distance changes, and printability constraints. The specific techniques vary (affine transformations for UAVs, SIT-Net for dynamic AV attacks), but the underlying challenge is universal.

- **Gradient-Based Optimization**: Despite domain differences, most attacks use similar gradient-based optimization techniques (FGSM, PGD variants) to craft patches. The fundamental attack mechanism is consistent across domains.

- **Model Architecture Agnosticism**: Patches work across different architectures (CNNs, Vision Transformers, etc.) because they exploit fundamental vulnerabilities in visual feature processing rather than architecture-specific weaknesses. This pattern holds across AV, surveillance, face recognition, and other domains.

- **Transferability from Simulation to Reality**: Research consistently shows that patches designed in simulation can be transferred to physical deployment with careful handling of physical-world constraints (printability, lighting, etc.). This pattern applies across all domains studied.

- **Effectiveness with Small Patches**: Across domains, patches can be effective even when covering relatively small portions of images (<1% in surveillance, small stickers in AV scenarios). This makes attacks practical and difficult to detect.

**What Makes Attacks Domain-Specific?**

- **Physical Deployment Constraints**: 
  - **AV**: Requires access to vehicle cameras (TransPatch) or deploying moving vehicles with screens (dynamic attacks), making some attacks harder to execute
  - **Surveillance**: Patches can be worn on clothing, making them highly practical and portable
  - **Face Recognition**: Accessories (glasses, hats) can appear natural while carrying adversarial patterns
  - **UAV**: Patches must be placed in environment and survive extreme viewing conditions
  - **Cross-Modal**: Must manipulate both visible and infrared properties, requiring more sophisticated materials

- **Environmental Factors**:
  - **AV**: Real-time processing requirements mean attacks must work within milliseconds; temporal consistency across frames creates unique vulnerabilities
  - **UAV**: Extreme viewing angles, distance variability, and brightness changes create domain-specific challenges
  - **Surveillance**: Multi-camera systems with overlapping views create opportunities for cross-validation defense
  - **Face Recognition**: Indoor vs. outdoor lighting, varying demographics, and natural appearance requirements
  - **Cross-Modal**: Must account for thermal properties and sensor fusion vulnerabilities

- **Model Architecture Preferences**:
  - **AV**: Heavily uses CNNs for perception, encoder-decoder for depth estimation, specialized architectures for SLAM
  - **Surveillance**: Primarily object detection models (YOLO, Faster R-CNN) with real-time requirements
  - **Face Recognition**: Often uses specialized face recognition architectures (FaceNet, ArcFace) with different feature spaces
  - **UAV**: Similar to surveillance but with additional robustness requirements
  - **Cross-Modal**: Requires architectures that fuse multiple sensor modalities

- **Mission-Criticality Levels**:
  - **AV**: Safety-critical - failures can cause accidents and deaths
  - **UAV**: Mission-critical - failures cause complete mission failure
  - **Surveillance**: Security-critical - failures enable unauthorized access
  - **Face Recognition**: Security-critical for access control, privacy-critical for consumer applications
  - **Cross-Modal**: Often used in critical infrastructure where failures have severe consequences

- **Attack Surface Accessibility**:
  - **AV**: Limited access to vehicle cameras (harder) vs. placing patches on signs/vehicles (easier)
  - **Surveillance**: Public-facing cameras are accessible, but attackers must be in camera view
  - **Face Recognition**: Can be deployed via accessories that attackers can wear
  - **UAV**: Requires physical access to place patches in environment
  - **Cross-Modal**: Often in high-security environments with limited access

**Which Attacks Are Most Practical/Feasible by Domain?**

**Most Feasible:**

- **Surveillance Systems (Person Detection Evasion)**: 
  - **Feasibility**: Very High
  - **Reasons**: Patches can be printed on clothing/accessories and worn; effectiveness with <1% image coverage; works across multiple model architectures; no special equipment needed; attacker can carry attack with them
  - **Practicality**: High - attackers can deploy patches by simply wearing printed clothing or accessories

- **Face Recognition (Adversarial Accessories)**:
  - **Feasibility**: Very High
  - **Reasons**: Can use natural-looking accessories (glasses, hats, masks); appears socially acceptable; works with consumer-grade printers; can be deployed in public without suspicion
  - **Practicality**: High - accessories look normal to humans while fooling systems

- **UAV Object Detection (Robust Patches)**:
  - **Feasibility**: High
  - **Reasons**: High transferability (75-78% cross-model success) means attackers don't need specific model knowledge; patches can be printed and placed; works across viewing angles and distances
  - **Practicality**: Medium-High - requires physical access to place patches, but high success rate makes it worthwhile

**Moderately Feasible:**

- **Autonomous Vehicles (Static Patch Attacks)**:
  - **Feasibility**: Medium
  - **Reasons**: 
    - **Static patches on signs**: Medium feasibility - requires access to traffic signs but relatively simple to deploy
    - **Meaningful Adversarial Stickers**: Medium feasibility - can be printed and affixed to signs
    - **TransPatch (camera lens)**: Low-Medium feasibility - requires physical access to vehicle cameras, which is harder
  - **Practicality**: Medium - static patches are easier than dynamic attacks, but require sign/vehicle access

- **Autonomous Vehicles (MDE/SLAM Attacks)**:
  - **Feasibility**: Medium
  - **Reasons**: SSAP attacks can cause 99% depth estimation error; AoR attacks corrupt SLAM maps; but requires understanding of AV system architecture
  - **Practicality**: Medium - highly effective but requires more technical sophistication

**Less Feasible (But Still Practical):**

- **Dynamic Adversarial Attacks (SIT-Net)**:
  - **Feasibility**: Medium-Low
  - **Reasons**: Requires moving vehicle with screen; needs SIT-Net modeling for color/contrast transformation; more complex than static patches
  - **Practicality**: Medium-Low - highly effective but requires significant setup and equipment

- **Cross-Modal Systems (Unified Patches)**:
  - **Feasibility**: Low-Medium
  - **Reasons**: Must manipulate both visible and infrared properties; requires specialized materials or techniques; attacks must work simultaneously in both modalities
  - **Practicality**: Low-Medium - more difficult to deploy but defeats redundancy assumptions

**Factors Affecting Feasibility Across Domains:**

- **Access Requirements**: Attacks requiring physical access to systems (camera lenses, signs) are less feasible than those using portable/wearable patches
- **Equipment Needs**: Attacks requiring specialized equipment (screens, SIT-Net setup) are less feasible than those using printed materials
- **Technical Sophistication**: Attacks requiring deep understanding of specific architectures (SLAM, MDE) are less feasible than those with high transferability
- **Detection Risk**: Attacks that are obvious to humans (large patches) are less feasible than those that appear natural (accessories, small patches)
- **Transferability**: Higher transferability increases feasibility by reducing need for white-box access
- **Success Rate**: Higher success rates (e.g., 99% for SSAP, 75-78% for UAV) increase feasibility even if deployment is harder

## Key Learnings
- [3-5 bullet points of insights about adversarial patches in general]
- [1-2 domain-specific insights]

**Critical Takeaways About Adversarial Patches:**

- **Transferability is the Most Dangerous Property**: The ability of adversarial patches to transfer across models (75-78% success in UAV attacks, demonstrated across multiple domains) means attackers don't need white-box access to target systems. This dramatically lowers the barrier to entry for attacks and makes defense significantly harder, as patches can be developed against surrogate models and still work against production systems.

- **Physical-World Attacks Are More Practical Than Previously Assumed**: Unlike purely digital adversarial examples, physical patches can be printed with consumer-grade equipment, worn as clothing or accessories, or placed in environments. The success of attacks covering <1% of image area (surveillance) or using natural-looking accessories (face recognition) demonstrates that practical deployment barriers are lower than theoretical defenses assumed.

- **Model Architecture Agnosticism Creates Universal Vulnerability**: Adversarial patches exploit fundamental vulnerabilities in how neural networks process visual features rather than architecture-specific weaknesses. This means that simply switching architectures (CNNs to Vision Transformers, etc.) doesn't provide protection - the vulnerability is inherent to how deep learning models learn visual representations.

- **Small, Localized Attacks Are Sufficient**: The effectiveness of small patches (<1% coverage) demonstrates that attackers don't need to manipulate entire images or scenes. This makes attacks harder to detect visually while remaining highly effective, creating a dangerous asymmetry where defenses must detect small anomalies while attackers only need to place small patches.

- **Transferability from Simulation to Reality Is Well-Established**: Research consistently shows that patches designed in simulation can be successfully transferred to physical deployment with proper handling of real-world constraints (printability, lighting, viewing angles). This pattern holds across domains, meaning attackers can develop and test attacks entirely digitally before physical deployment.

**Domain-Specific Insights:**

- **UAV Domain Shows Highest Transferability Risk**: With 75-78% cross-model success rates in gray-box settings, UAV attacks demonstrate that transferability can be extremely high even without white-box access. This suggests that domains relying heavily on transfer learning from common pre-trained models (ImageNet, COCO) may be particularly vulnerable, as shared feature representations enable transferability.

- **Wearable Attack Vectors Create Persistent Threat**: Surveillance and face recognition attacks using wearable patches (clothing, accessories) represent a unique threat vector - attackers can carry attacks with them, deploy them anywhere, and they appear natural to human observers. This "attack portability" combined with "invisibility" creates a persistent threat that's harder to defend against than fixed-location attacks.

**Forward-Looking Research Directions:**

- **Transferability Reduction Mechanisms**: Current research focuses primarily on detection and robustness, but reducing transferability itself could be a more fundamental defense. Research into training techniques that reduce feature space overlap between models, or architectures that create more unique feature representations, could reduce cross-model attack success rates.

- **Multi-Modal Defense Strategies**: As attacks become more sophisticated (e.g., cross-modal unified patches), defenses must evolve beyond single-modality approaches. Research into multi-modal adversarial training, cross-modal consistency checks, and fusion-based defenses could provide more robust protection, especially for critical applications like autonomous vehicles and security systems where multiple sensors are already deployed.

## Application to Red Teaming
- [How would you test for this vulnerability across different domains?]
- [What would you look for in different types of client engagements?]
- [Domain-specific testing considerations]

**How Would You Test for This Vulnerability Across Different Domains?**

**Systematic Testing Methodology:**

- **Surrogate Model Development**: Build surrogate models using publicly available pre-trained weights (ImageNet, COCO) and fine-tune on domain-specific datasets when available. Test patch transferability against these surrogates first, then validate against actual target systems. This approach leverages the transferability property while avoiding white-box access requirements.

- **Progressive Attack Sophistication**: Start with simple printed patches (static, single-domain), then progress to more sophisticated attacks:
  1. **Phase 1**: Basic printed patches tested in controlled environments
  2. **Phase 2**: Robust patches with affine transformations and lighting variations
  3. **Phase 3**: Dynamic/time-varying patches (for AV systems)
  4. **Phase 4**: Cross-modal unified patches (for multi-sensor systems)
  5. **Phase 5**: Adaptive patches that evolve based on detection attempts

- **Transferability Testing Framework**: Systematically test patch transferability across:
  - Different model architectures (CNN, Vision Transformer, etc.)
  - Different training procedures (different initializations, hyperparameters)
  - Different domains (test patches crafted for one domain against others)
  - Different scales and resolutions (test patches at various image sizes)

- **Physical-World Validation Pipeline**: Establish a testing pipeline that bridges simulation and reality:
  1. Generate patches in simulation using gradient-based methods
  2. Apply printability constraints and validate colorspace conversion
  3. Print patches using consumer-grade equipment
  4. Test under various lighting conditions (indoor, outdoor, day, night)
  5. Test across viewing angles and distances
  6. Measure success rates and document failure modes

- **Multi-Vector Attack Testing**: Don't test patches in isolation - combine with other attack vectors:
  - Patch attacks + adversarial examples (add patches to already-adversarial images)
  - Patch attacks + sensor spoofing (manipulate camera hardware if accessible)
  - Patch attacks + model extraction (use patches to probe model behavior)
  - Patch attacks + temporal attacks (combine with frame-by-frame manipulation)

- **Automated Patch Generation and Testing**: Develop automated frameworks that:
  - Generate patches against multiple surrogate models simultaneously
  - Test patches across multiple target systems automatically
  - Measure success rates, transferability, and robustness metrics
  - Generate comprehensive reports with failure analysis

**What Would You Look for in Different Types of Client Engagements?**

**Security Assessment Engagements:**

- **Model Inventory and Architecture Analysis**: Document all deployed ML models, their architectures, training procedures, and deployment contexts. Identify which models are most critical and which have the largest attack surface (public-facing cameras, access control systems, etc.).

- **Attack Surface Mapping**: Create comprehensive attack surface maps identifying:
  - Physical access points (where can patches be placed or worn?)
  - Model input points (cameras, sensors, APIs)
  - Decision points (where do model outputs trigger actions?)
  - Failure modes (what happens when models fail or are deceived?)

- **Transferability Assessment**: Test whether patches crafted against publicly available models transfer to client systems. This tests the "gray-box" attack scenario where attackers don't have white-box access.

- **Defense Gap Analysis**: Evaluate existing defenses:
  - Are models trained with adversarial training?
  - Are there patch detection systems in place?
  - Do multi-modal systems properly validate cross-modal consistency?
  - Are there manual verification processes for critical decisions?
  - What incident response procedures exist for detected attacks?

- **Risk Prioritization Matrix**: Create a risk matrix combining:
  - Attack feasibility (ease of deployment)
  - Attack impact (business/safety consequences)
  - Defense effectiveness (current protection level)
  - Attack likelihood (based on threat intelligence and attack surface)

**Penetration Testing Engagements:**

- **Real-World Attack Scenarios**: Develop realistic attack scenarios:
  - **Surveillance Bypass**: Test if wearable patches can evade person detection systems
  - **Access Control Bypass**: Test if adversarial accessories can bypass face recognition systems
  - **AV Manipulation**: Test if patches can cause misclassification in test vehicles or controlled environments
  - **UAV Deception**: Test patches against drone detection systems (if authorized)

- **Stealth and Persistence Testing**: Evaluate:
  - Can patches be deployed without detection by security personnel?
  - Do patches remain effective over time (weathering, lighting changes)?
  - Can patches be modified or adapted if initial attacks fail?
  - Are there detection mechanisms that can identify patch deployment?

- **Impact Assessment**: Measure actual impact when attacks succeed:
  - System behavior changes (misclassifications, false negatives/positives)
  - Downstream effects (does misclassification trigger incorrect actions?)
  - Detection latency (how long until attacks are detected, if at all?)
  - Recovery procedures (can systems recover automatically or require manual intervention?)

**Compliance and Audit Engagements:**

- **Regulatory Compliance Mapping**: Map adversarial patch vulnerabilities to:
  - Industry-specific regulations (NHTSA for AV, HIPAA for healthcare surveillance)
  - Biometric data regulations (GDPR, CCPA requirements for face recognition)
  - Security standards (ISO/IEC 27001, NIST frameworks)
  - Insurance requirements (cybersecurity insurance policy requirements)

- **Defense Effectiveness Documentation**: Quantify defense effectiveness:
  - Patch detection accuracy rates
  - False positive rates (defenses incorrectly flagging legitimate inputs)
  - Performance overhead (computational cost of defenses)
  - Coverage gaps (attack vectors not covered by defenses)

**Domain-Specific Testing Considerations:**

**Autonomous Vehicles:**
- **Controlled Environment Testing**: Test patches in controlled test tracks before road testing. Use test vehicles with safety drivers and backup systems.
- **Multi-Sensor Validation**: Test patches against camera-only systems first, then test against multi-modal systems (camera + LiDAR + radar) to assess sensor fusion vulnerabilities.
- **Temporal Consistency Testing**: Test patches across video sequences to assess temporal attack effectiveness. Dynamic patches may need frame-by-frame validation.
- **Regulatory Compliance**: Ensure testing complies with local regulations (may require special permits, test track access, or controlled conditions).
- **Safety-Critical Failure Modes**: Focus on attacks that could cause safety-critical failures (traffic sign misclassification, depth estimation errors) rather than non-critical misclassifications.

**Surveillance Systems:**
- **Multi-Camera Scenarios**: Test patches against systems with multiple overlapping cameras to assess cross-validation effectiveness.
- **Wearable Attack Vectors**: Develop and test patches that can be printed on clothing or accessories, testing realistic deployment scenarios.
- **Real-Time Detection**: Test patches against real-time video processing systems, measuring detection latency and system response.
- **Privacy Considerations**: Ensure testing doesn't violate privacy regulations when testing against live surveillance systems.
- **Coverage Testing**: Test patches at various distances, angles, and lighting conditions to determine effectiveness boundaries.

**Face Recognition / Access Control:**
- **Accessory Integration**: Test patches integrated into natural-looking accessories (glasses, hats, masks) to assess stealth.
- **Liveness Detection Bypass**: Test whether patches can bypass liveness detection mechanisms in addition to recognition systems.
- **Multi-Factor Authentication**: Test patches against systems with and without multi-factor authentication to assess overall security posture.
- **Demographic Testing**: Test patches across different demographics to identify if attacks are more effective against certain groups (potential discrimination concerns).
- **Continuous Authentication**: Test patches against systems that use continuous authentication (monitoring confidence over time) versus single-shot authentication.

**UAV Systems:**
- **High-Altitude Testing**: Test patches at various altitudes and viewing angles typical of UAV operations.
- **Distance Variability**: Test patches across wide distance ranges (near-ground to kilometers away).
- **Transferability Focus**: Emphasize transferability testing since UAV attacks show high cross-model success rates (75-78%).
- **Mission-Specific Constraints**: Test patches against domain-specific constraints (expected object sizes at certain altitudes) that defenses might use.
- **Robustness Testing**: Test patches against robust training schemes that include extensive augmentation (affine transformations, lighting variations).

**Cross-Modal Systems:**
- **Modality-Specific Testing**: Test patches against visible-only, infrared-only, and fused systems separately to identify modality-specific vulnerabilities.
- **Sensor Fusion Validation**: Test whether patches can cause sensor fusion failures in addition to single-modality failures.
- **Thermal Property Manipulation**: For infrared systems, test patches that manipulate thermal properties (not just visual appearance).
- **Consistency Check Bypass**: Test whether patches can bypass cross-modal consistency checks that defenses might implement.
- **Redundancy Defeat**: Test unified patches that simultaneously attack both modalities, defeating redundancy assumptions.

```

## Resources

**Recommended Reading:**
- "Adversarial Patch" (Brown et al., 2017) - Original paper
- "Robust Physical-World Attacks on Deep Learning Models" (Brown et al., 2017)
- "Fooling automated surveillance cameras: adversarial patches to attack person detection" (Thys et al., 2019)
- OWASP ML Top 10: M01 and M04
- MITRE ATLAS: Adversarial example patterns

**Search Terms:**
- "adversarial patch attack"
- "physical adversarial attacks"
- "adversarial patch face recognition"
- "adversarial patch object detection"
- "robust physical world attacks"
- "adversarial patch security cameras"
- "adversarial patch surveillance"

**Research Databases:**
- arXiv.org (cs.CV, cs.CR)
- IEEE Xplore
- ACM Digital Library
- Google Scholar

## Success Criteria

Your analysis should demonstrate:
- **Breadth**: Coverage of attacks across multiple diverse domains
- **Depth**: Understanding of attack methodology in different contexts
- **Analysis**: Clear connection to OWASP ML Top 10 framework
- **Insight**: Recognition of patterns and differences across domains
- **Application**: Ability to translate to red team testing scenarios across different domains

## Next Steps

After completing this challenge:
- Use as reference for Week 3 evasion attack exercises
- Consider how patch attacks differ from other evasion attacks

