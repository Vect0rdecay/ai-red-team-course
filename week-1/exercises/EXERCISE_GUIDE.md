# Exercise Guide: How to Complete the Coding Exercises

## Overview

All coding exercises are intentionally **incomplete** (~85% done). Your job is to fill in the `TODO` sections marked with:

```python
# TODO: Your implementation here
```

## Why This Approach?

**Active Learning > Passive Learning**

- **Reading code** doesn't teach you to write code
- **Completing TODOs** forces you to understand how everything connects
- **Error debugging** teaches you to think through problems
- **This is how real AI security work happens**: you understand existing code and extend it

## How to Approach Each Exercise

### Step 1: Read the Entire File
- Understand the overall structure
- Identify what the exercise is trying to accomplish
- Note what inputs and outputs are expected

### Step 2: Read Each TODO Carefully
- TODOs include **hints** to guide you
- Read the hint before implementing
- Think about what PyTorch function/pattern applies

### Step 3: Fill in TODOs One by One
- Start from the beginning (data loading, model definition)
- Test frequently (add print statements)
- Fix errors as you go
- Don't skip ahead - each TODO builds on previous ones

### Step 4: Run and Debug
- Scripts include error handling to catch incomplete TODOs
- Read error messages carefully
- Common issues: missing variable assignments, incorrect dimensions

### Step 5: Verify Results
- Check that your model trains successfully
- Verify output metrics make sense
- If stuck, compare your implementation with solutions later

## What You ARE Supposed To Do

- Read the hints carefully
- Reference PyTorch documentation
- Experiment with different approaches
- Ask questions when stuck (after trying yourself first)
- Understand WHY each line of code is necessary

## Common Patterns You'll Learn

### Pattern 1: Forward Pass
```python
# TODO: Forward pass in neural network
x = self.layer1(x)  # Apply first layer
x = self.activation(x)  # Apply activation
x = self.layer2(x)  # Apply second layer
return x
```

### Pattern 2: Training Loop
```python
# TODO: Training loop
optimizer.zero_grad()  # 1. Zero gradients
output = model(input)  # 2. Forward pass
loss = criterion(output, target)  # 3. Compute loss
loss.backward()  # 4. Backward pass
optimizer.step()  # 5. Update weights
```

### Pattern 3: Model Evaluation
```python
# TODO: Evaluation mode
with torch.no_grad():  # Disable gradient computation
    output = model(input)  # Forward pass
    # No backward pass during evaluation
```

## Troubleshooting

**Error: "AttributeError: 'NoneType' object has no attribute..."**
- You forgot to assign a value in a TODO
- Check if all TODOs in that section are completed

**Error: "RuntimeError: Expected size [X] but got [Y]"**
- Dimension mismatch in tensor operations
- Check your shapes before and after each operation

**Error: "NameError: name 'loss' is not defined"**
- Variable not assigned in TODO
- Make sure all required variables are defined

**Model doesn't train / accuracy is 0%**
- Check that you implemented forward pass correctly
- Verify loss function is appropriate for task
- Ensure data is properly loaded and normalized

## When You Complete an Exercise

1. **Save your working code** - You'll reference it for future exercises
2. **Document what you learned** - What was confusing? What clicked?
3. **Experiment further** - Can you modify hyperparameters? Architecture?
4. **Check your understanding** - Can you explain each line to someone else?

## Expected Time

- **Exercise 1 (MNIST)**: 2-3 hours
- **Exercise 2 (Queries)**: 1-2 hours  
- **Exercise 3 (Text Gen)**: 2-3 hours
- **Exercise 4 (Deployment)**: 1 hour

**Total**: ~6-9 hours of hands-on coding

## The Bottom Line

This course is designed to teach you to **think like an AI red teamer**.

You're not just following recipes - you're learning to:
- Read and understand ML code
- Identify where security vulnerabilities exist
- Modify code to implement attacks
- Debug issues when things break

Every TODO you complete builds these skills.

**Good luck!**
