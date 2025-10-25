"""
Week 1 - Exercise 3: Simple Text Generator

Objective: Understand generative models (prepares for LLM attacks in Week 5)

INSTRUCTIONS:
This script is ~85% complete. Your task is to fill in the TODO sections.
This exercise teaches the fundamentals of how LLMs generate text.

Red Team Context: LLMs work similarly - understanding this foundation is crucial 
for prompt injection and jailbreak attacks in Week 5.

You'll build a simple RNN that generates text character-by-character.
This teaches you how generative models work, which is essential for attacking LLMs.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# STEP 1: PREPARE TRAINING DATA
# ============================================================================
print("Preparing training data...")

# Simple text corpus (Shakespeare-inspired)
text = """
To be or not to be that is the question
Whether tis nobler in the mind to suffer
The slings and arrows of outrageous fortune
Or to take arms against a sea of troubles
And by opposing end them
"""

# Clean and prepare text
text = text.lower().replace('\n', ' ')
text = ' '.join(text.split())  # Remove extra whitespace

print(f"Training text length: {len(text)} characters")
print(f"First 100 characters: {text[:100]}")

# Create character mappings
chars = sorted(set(text))
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

vocab_size = len(chars)
print(f"\nVocabulary size: {vocab_size} unique characters")
print(f"Characters: {chars}")

# ============================================================================
# STEP 2: CREATE TRAINING SEQUENCES
# ============================================================================
print("\nCreating training sequences...")

def create_sequences(text, seq_length=50):
    """
    Create sequences of characters for training.
    Input: "hello world"
    Output: 
      X: ["hello world", "ello world ", ...]
      y: [next_char for each sequence]
    """
    X, y = [], []
    
    # TODO: Create training sequences
    # HINT: Loop through text, extract sequences of length seq_length
    # For each sequence, the target is the next character
    # Convert characters to indices using char_to_idx dictionary
    
    for i in range(len(text) - seq_length):
        # TODO: Extract sequence and next character
        seq = None  # Extract seq_length characters starting at position i
        next_char = None  # Extract the character immediately after the sequence
        
        # TODO: Convert to indices and append to X and y
        X.append(None)  # Convert seq to list of indices
        y.append(None)  # Convert next_char to index
    
    return torch.tensor(X), torch.tensor(y)

seq_length = 30
X, y = create_sequences(text, seq_length)

print(f"Created {len(X)} training sequences")
print(f"Sequence length: {seq_length}")
print(f"Example sequence: {''.join([idx_to_char[i] for i in X[0]])}")
print(f"Next character: {idx_to_char[y[0].item()]}")

# ============================================================================
# STEP 3: DEFINE RNN MODEL
# ============================================================================
print("\nDefining RNN model...")

class CharRNN(nn.Module):
    """
    Simple character-level RNN for text generation.
    
    Architecture:
    - Embedding layer: convert char indices to vectors
    - LSTM layer: process sequences
    - Linear layer: predict next character
    """
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_layers=2):
        super(CharRNN, self).__init__()
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Embedding: convert char indices to dense vectors
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM: process sequences
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=0.2 if num_layers > 1 else 0)
        
        # Output layer: predict next character
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x, hidden=None):
        # x: (batch, seq_length)
        # TODO: Apply embedding layer
        # HINT: self.embedding(x) converts character indices to dense vectors
        x = None
        
        # TODO: Apply LSTM layer
        # HINT: self.lstm(x, hidden) processes the sequence
        lstm_out, hidden = None, None
        
        # TODO: Get final output and apply fully connected layer
        # HINT: Use lstm_out[:, -1, :] to get last timestep, then apply self.fc
        out = None
        
        return out, hidden
    
    def init_hidden(self, batch_size):
        """Initialize hidden state."""
        weight = next(self.parameters())
        h0 = weight.new_zeros(self.num_layers, batch_size, self.hidden_dim)
        c0 = weight.new_zeros(self.num_layers, batch_size, self.hidden_dim)
        return (h0, c0)

# Initialize model
model = CharRNN(vocab_size)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# ============================================================================
# STEP 4: SET UP TRAINING
# ============================================================================
print("\nSetting up training...")

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training configuration
num_epochs = 200
batch_size = 32

print(f"Training for {num_epochs} epochs...")

# Check if sequences were created properly
if len(X) == 0:
    print("\n❌ ERROR: No training sequences created!")
    print("   Please complete the TODO in create_sequences() function.")
    exit()

# ============================================================================
# STEP 5: TRAINING LOOP
# ============================================================================
print("\nTraining model...")

losses = []
model.train()

# Add try/except to catch TODO-related errors
for epoch in range(num_epochs):
    try:
        # Create random batches
        indices = torch.randperm(len(X))
        epoch_loss = 0.0
        
        for i in range(0, len(X), batch_size):
            batch_indices = indices[i:i + batch_size]
            batch_X = X[batch_indices]
            batch_y = y[batch_indices]
            
            # TODO: Forward pass
            # 1. Zero gradients: optimizer.zero_grad()
            # 2. Initialize hidden state: model.init_hidden(len(batch_X))
            # 3. Get model output: model(batch_X, hidden)
            # 4. Calculate loss: criterion(output, batch_y)
            
            # TODO: Backward pass
            # 1. Compute gradients: loss.backward()
            # 2. Clip gradients: torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            # 3. Update weights: optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / (len(X) // batch_size)
        losses.append(avg_loss)
        
        # Print progress every 20 epochs
        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
    except (AttributeError, RuntimeError, TypeError) as e:
        print(f"\n❌ ERROR during training: {e}")
        print("\nCheck your TODO implementations:")
        print("  - Did you implement create_sequences()?")
        print("  - Did you implement the model forward pass?")
        print("  - Did you implement the training loop TODOs?")
        exit()

print("\nTraining complete!")

# ============================================================================
# STEP 6: VISUALIZE TRAINING PROGRESS
# ============================================================================
plt.figure(figsize=(10, 5))
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.grid(True)
plt.savefig('text_generator_training.png', dpi=150)
print("\nSaved: text_generator_training.png")

# ============================================================================
# STEP 7: GENERATE TEXT
# ============================================================================
print("\nGenerating text...")

def generate_text(model, start_string, length=100, temperature=1.0):
    """
    Generate text starting from a given string.
    
    Args:
        model: Trained RNN model
        start_string: Initial text to start generation
        length: Number of characters to generate
        temperature: Controls randomness (higher = more random)
    """
    model.eval()
    
    # Convert start string to indices
    input_seq = torch.tensor([[char_to_idx[ch] for ch in start_string]])
    
    generated = list(start_string)
    hidden = model.init_hidden(1)
    
    with torch.no_grad():
        # Process start string
        for i in range(len(start_string) - 1):
            output, hidden = model(input_seq[:, i:i+1], hidden)
        
        # Generate new characters
        for _ in range(length):
            # TODO: Get next character prediction
            # 1. Get model output for last character: model(input_seq[:, -1:], hidden)
            
            # TODO: Apply temperature and sample
            # 1. Divide output by temperature
            # 2. Apply softmax to get probabilities
            # 3. Sample from probability distribution using torch.multinomial()
            
            # TODO: Convert to character and add to sequence
            # 1. Convert index to character: idx_to_char[next_char_idx]
            # 2. Add to generated list
            # 3. Append to input_seq for next iteration
            
            generated.append(None)  # Add generated character
            input_seq = None  # Update input sequence
    
    return ''.join(generated)

# Generate 10 samples
start_string = "to be or not"
num_samples = 10

print(f"\nGenerating {num_samples} samples from '{start_string}':\n")
print("="*70)

for i in range(num_samples):
    generated = generate_text(model, start_string, length=80, temperature=0.8)
    print(f"Sample {i+1}:")
    print(f"  {generated}")
    print()

# ============================================================================
# STEP 8: ANALYZE MODEL BEHAVIOR
# ============================================================================
print("\nAnalyzing model behavior...")

# Test different temperatures
print("\nEffect of temperature on generation:")
print("="*70)

temperatures = [0.5, 1.0, 2.0]
for temp in temperatures:
    generated = generate_text(model, "to be", length=50, temperature=temp)
    print(f"\nTemperature {temp}:")
    print(f"  {generated}")

# ============================================================================
# STEP 9: SAVE MODEL
# ============================================================================
model_path = 'char_rnn_model.pt'
torch.save({
    'model_state_dict': model.state_dict(),
    'char_to_idx': char_to_idx,
    'idx_to_char': idx_to_char,
    'vocab_size': vocab_size,
    'model_config': {
        'hidden_dim': 256,
        'num_layers': 2
    }
}, model_path)
print(f"\n✓ Model saved to {model_path}")

# ============================================================================
# DOCUMENTATION
# ============================================================================
print("\n" + "="*70)
print("Exercise 3 Complete!")
print("="*70)
print("\nWhat you accomplished:")
print("1. ✓ Built a character-level RNN for text generation")
print("2. ✓ Trained the model on Shakespeare text")
print("3. ✓ Generated 10 text samples")
print("4. ✓ Analyzed effect of temperature on generation")
print("5. ✓ Saved model for future use")
print("\nKey Concepts Learned:")
print("  - Character-level modeling (input: character, output: next character)")
print("  - LSTM for sequence modeling")
print("  - Text generation via sampling from probability distribution")
print("  - Temperature controls randomness in generation")
print("\nRed Team Context:")
print("  LLMs work similarly to this RNN:")
print("  - They predict the next token (word/character) given context")
print("  - Generation involves sampling from probability distributions")
print("  - Understanding this helps you craft effective prompt injections")
print("\nConnection to Week 5 (LLM Attacks):")
print("  - Prompt injection: manipulating the input context")
print("  - Jailbreaks: exploiting generation patterns")
print("  - Token manipulation: understanding how models process input")
print("\nYour simple RNN foundation → Advanced LLM attacks!")
