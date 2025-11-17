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

# Standard imports for neural networks and data handling
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# STEP 1: PREPARE TRAINING DATA
# ============================================================================
print("Preparing training data...")

# Character-level text generation: Model learns to predict next character
# We'll train on Shakespeare text and the model will generate similar text
# Simple text corpus (Shakespeare-inspired)
text = """
To be or not to be that is the question
Whether tis nobler in the mind to suffer
The slings and arrows of outrageous fortune
Or to take arms against a sea of troubles
And by opposing end them
"""

# Clean and prepare text: Lowercase everything for consistency
text = text.lower().replace('\n', ' ')
text = ' '.join(text.split())  # Remove extra whitespace

print(f"Training text length: {len(text)} characters")
print(f"First 100 characters: {text[:100]}")

# Create character mappings: Convert characters to numbers
# Neural networks work with numbers, not letters directly
# Each unique character gets a number (0, 1, 2, etc.)
chars = sorted(set(text))  # Get all unique characters and sort them
char_to_idx = {ch: i for i, ch in enumerate(chars)}  # 'a' -> 0, 'b' -> 1, etc.
idx_to_char = {i: ch for i, ch in enumerate(chars)}  # Reverse mapping

vocab_size = len(chars)  # How many different characters we have
print(f"\nVocabulary size: {vocab_size} unique characters")
print(f"Characters: {chars}")

# ============================================================================
# STEP 2: CREATE TRAINING SEQUENCES
# ============================================================================
print("\nCreating training sequences...")

def create_sequences(text, seq_length=50):
    """
    Create sequences of characters for training.
    
    Sliding window approach: For text "hello world"
    - Look at "hello wo" -> predict 'r' (next character)
    - Look at "ello wor" -> predict 'l' (next character)
    - And so on...
    
    This is how we teach the model to predict the next character
    """
    X, y = [], []  # X = input sequences, y = target (next character)
    
    # TODO: Create training sequences
    # Process text in sliding windows to create training examples
    # For each position, extract a sequence and its following character
    # HINT: Loop through text, extract sequences of length seq_length
    # For each sequence, the target is the next character
    # Convert characters to indices using char_to_idx dictionary
    
    for i in range(len(text) - seq_length):
        # Extract sequence and next character
        seq = text[i:i + seq_length]  # Extract seq_length characters starting at position i
        next_char = text[i + seq_length]  # Extract the character immediately after the sequence
        
        # Convert to indices and append to X and y
        X.append([char_to_idx[ch] for ch in seq])  # Convert seq to list of indices
        y.append(char_to_idx[next_char])  # Convert next_char to index
    
    return torch.tensor(X), torch.tensor(y)

seq_length = 30
X, y = create_sequences(text, seq_length)

print(f"Created {len(X)} training sequences")
print(f"Sequence length: {seq_length}")
print(f"Example sequence: {''.join([idx_to_char[i.item()] for i in X[0]])}")
print(f"Next character: {idx_to_char[y[0].item()]}")

# ============================================================================
# STEP 3: DEFINE RNN MODEL
# ============================================================================
print("\nDefining RNN model...")

class CharRNN(nn.Module):
    """
    Simple character-level RNN for text generation.
    
    RNNs are perfect for sequences because they remember previous context
    Architecture:
    - Embedding layer: convert char indices to dense vectors (learns character representations)
    - LSTM layer: process sequences (remembers what happened before)
    - Linear layer: predict next character (output probabilities for each character)
    
    LSTM (Long Short-Term Memory): Special RNN that can remember for many steps
    """
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_layers=2):
        super(CharRNN, self).__init__()
        
        self.vocab_size = vocab_size  # Number of unique characters
        self.hidden_dim = hidden_dim  # Size of LSTM's memory
        self.num_layers = num_layers  # Stack multiple LSTM layers
        
        # Embedding: Convert character indices to learnable dense vectors
        # Instead of one-hot encoding, learns meaningful character representations
        # Input: character index (e.g., 5), Output: 128-dimensional vector
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM: Process sequences with memory
        # Takes embedding vectors, outputs hidden states (what it "remembers")
        # batch_first=True: Input format is (batch, seq, features) instead of (seq, batch, features)
        # dropout: Randomly drops 20% of connections to prevent overfitting
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=0.2 if num_layers > 1 else 0)
        
        # Output layer: Convert LSTM's memory to character probabilities
        # Input: hidden_dim (what LSTM remembers)
        # Output: vocab_size (probability for each possible character)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x, hidden=None):
        # x: (batch, seq_length) - batch of character sequences
        # Apply embedding layer: Turn character indices into dense vectors
        # Each character becomes a learnable 128-dim vector
        x = self.embedding(x)
        
        # Apply LSTM layer: Process sequence and build memory
        # LSTM processes each character in sequence, building context
        # Returns: output at each step, and hidden state (memory)
        lstm_out, hidden = self.lstm(x, hidden)
        
        # Get final output and apply fully connected layer
        # Take the last timestep's output (final memory state)
        # Convert to probabilities for each character
        out = self.fc(lstm_out[:, -1, :])
        
        return out, hidden
    
    def init_hidden(self, batch_size):
        """Initialize hidden state (LSTM's initial memory)."""
        # LSTM has two states: h (hidden) and c (cell)
        # Both start as zeros (no prior knowledge)
        weight = next(self.parameters())
        # Create zero tensors with correct shape
        h0 = weight.new_zeros(self.num_layers, batch_size, self.hidden_dim)  # Hidden state
        c0 = weight.new_zeros(self.num_layers, batch_size, self.hidden_dim)  # Cell state
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
    print("\n ERROR: No training sequences created!")
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
            
            # Forward pass
            optimizer.zero_grad()  # 1. Zero gradients
            hidden = model.init_hidden(len(batch_X))  # 2. Initialize hidden state
            output, hidden = model(batch_X, hidden)  # 3. Get model output
            loss = criterion(output, batch_y)  # 4. Calculate loss
            
            # Backward pass
            loss.backward()  # 1. Compute gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)  # 2. Clip gradients
            optimizer.step()  # 3. Update weights
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / (len(X) // batch_size)
        losses.append(avg_loss)
        
        # Print progress every 20 epochs
        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
    except (AttributeError, RuntimeError, TypeError) as e:
        print(f"\n ERROR during training: {e}")
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
    Generate text starting from a given string (autoregressive generation).
    
    Process: Feed model a character, get probabilities for next character, sample one
    Repeat for desired length. Model generates one character at a time.
    
    Args:
        model: Trained RNN model (knows how to predict next character)
        start_string: Initial text to start generation (seed text)
        length: Number of NEW characters to generate
        temperature: Controls randomness
            - Low (0.5): Conservative, picks most likely characters
            - High (2.0): Creative, more diverse but potentially nonsensical
            - 1.0: Balanced
    """
    model.eval()  # Put model in inference mode (no training/learning)
    
    # Convert start string characters to numbers
    input_seq = torch.tensor([[char_to_idx[ch] for ch in start_string]])
    
    generated = list(start_string)  # Start with user's seed text
    hidden = model.init_hidden(1)  # Initialize LSTM memory (no prior context)
    
    with torch.no_grad():  # No gradient needed for inference (faster)
        # First, process the start string to build up context
        # This "warms up" the LSTM's memory with the seed text
        for i in range(len(start_string) - 1):
            output, hidden = model(input_seq[:, i:i+1], hidden)
        
        # Now generate NEW characters one at a time
        for _ in range(length):
            # Get next character prediction
            # Ask model: given current context, what's the next character?
            output, hidden = model(input_seq[:, -1:], hidden)
            
            # Apply temperature and sample
            # Temperature controls randomness in generation
            output = output / temperature  # 1. Divide by temperature
            probs = torch.nn.functional.softmax(output, dim=1)  # 2. Apply softmax
            next_char_idx = torch.multinomial(probs, 1).item()  # 3. Sample from distribution
            
            # Convert to character and add to sequence
            next_char = idx_to_char[next_char_idx]  # 1. Convert index to character
            generated.append(next_char)  # 2. Add to generated list
            input_seq = torch.cat([input_seq, torch.tensor([[next_char_idx]])], dim=1)  # 3. Update input sequence
    
    return ''.join(generated)  # Convert list of chars to string

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
