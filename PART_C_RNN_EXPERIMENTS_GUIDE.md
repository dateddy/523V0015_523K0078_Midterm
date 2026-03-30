# Part C: RNN Architecture Ablation Study

## Executive Summary

Part C extends the sentiment analysis study from MLPs (Part B) to recurrent neural networks (RNNs). Through three controlled experiments, we systematically compare different RNN variants and architectural choices, providing insights into:

1. **Variant Performance**: How do RNN, LSTM, and GRU compare for sequential modeling?
2. **Embedding Expressiveness**: What embedding dimension best captures sentiment semantics?
3. **Network Capacity**: When does adding more layers help or hurt?

---

## Architecture Overview

### RNN vs LSTM vs GRU

#### **Vanilla RNN**
```
Input: x_t
  ↓
h_t = tanh(W_hh * h_{t-1} + W_xh * x_t + b_h)
  ↓
Output based on h_t
```

**Characteristics**:
- Simplest recurrent architecture
- Single hyperbolic tangent activation per step
- Prone to vanishing gradients on long sequences
- **Pros**: Fewest parameters, fastest training
- **Cons**: Struggles with long-term dependencies

#### **LSTM (Long Short-Term Memory)**
```
Input gate:    i_t = σ(W_ii * x_t + W_hi * h_{t-1} + b_i)
Forget gate:   f_t = σ(W_if * x_t + W_hf * h_{t-1} + b_f)
Cell gate:     g_t = tanh(W_ig * x_t + W_hg * h_{t-1} + b_g)
Output gate:   o_t = σ(W_io * x_t + W_ho * h_{t-1} + b_o)

Cell state:    C_t = f_t ⊙ C_{t-1} + i_t ⊙ g_t
Hidden state:  h_t = o_t ⊙ tanh(C_t)
```

**Characteristics**:
- Multiple gating mechanisms (input, forget, output)
- Cell state preserves information across timesteps
- Gradient flow is regulated by gates
- **Pros**: Excellent at capturing long-term dependencies, standard choice
- **Cons**: Most complex, most parameters, slower training

#### **GRU (Gated Recurrent Unit)**
```
Reset gate:    r_t = σ(W_ir * x_t + W_hr * h_{t-1} + b_r)
Update gate:   z_t = σ(W_iz * x_t + W_hz * h_{t-1} + b_z)
New hidden:    h'_t = tanh(W_ih * x_t + W_hh * (r_t ⊙ h_{t-1}) + b_h)
Hidden state:  h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h'_t
```

**Characteristics**:
- Simplified LSTM with 2 gates (instead of 3)
- Combines cell state and hidden state
- Direct gating of hidden state
- **Pros**: Few parameters, simple yet effective, faster than LSTM
- **Cons**: Sometimes loses information faster than LSTM on very long sequences

---

## Experiment 1: RNN Variant Comparison

### Objective
Compare the three RNN variants under identical hyperparameters to isolate architectural differences.

### Design

**Fixed Configuration**:
- `embedding_dim = 128` (moderate semantic richness)
- `hidden_dim = 128` (balanced capacity)
- `n_layers = 1` (single recurrent layer)
- `bidirectional = True` (left-to-right AND right-to-left)
- `dropout = 0.3` (standard regularization)
- Learning rate = 0.001
- Epochs = 10

**Variable**:
- `rnn_type ∈ {'rnn', 'lstm', 'gru'}`

### Expected Outcomes

#### RNN (Vanilla)
- **Fastest training** due to fewest parameters (~200k)
- **May overfit** on IMDb if not carefully regularized
- **Convergence**: Often unstable on longer sequences

#### LSTM
- **Slowest training** but most stable convergence
- **Best long-term dependency capture** (critical for sentiment nuances like negation)
- **Most parameters** (~500k), higher memory usage

#### GRU
- **Middle ground**: Good performance with fewer parameters than LSTM (~350k)
- **Fast convergence**: Often as good as LSTM with less overhead
- **Often preferred** in production due to speed-accuracy tradeoff

### Key Metrics
- **Test Accuracy**: Final performance on unseen test data
- **Test Loss**: BCE loss indicating calibration
- **Training Time**: Wall-clock seconds for 10 epochs
- **Convergence Speed**: How quickly validation accuracy plateaus

### RNN-Specific Considerations

**Why Bidirectional?**
- Reviews contain context clues in both directions
- "The movie was... bad" (context later)
- Bidirectional encoding captures full context

**Why Packed Sequences?**
- Padding tokens waste computation in RNNs
- `torch.nn.utils.pack_padded_sequence` removes padding
- Significant speedup without accuracy loss

---

## Experiment 2: Embedding Dimension

### Objective
Find the optimal word embedding dimension using the best RNN variant from Experiment 1.

### Design

**Fixed Configuration**:
- `rnn_type = [BEST_FROM_EXP1]` (RNN/LSTM/GRU)
- `hidden_dim = 128`
- `n_layers = 1`
- `bidirectional = True`
- `dropout = 0.3`

**Variable**:
- `embedding_dim ∈ {64, 128, 256}`

### Embedding Dimension Trade-off

#### 64-dimensional embeddings
- **Pros**: Compact representation, fewer parameters (~15% less)
- **Cons**: May underfit, limited semantic capacity
- **Best for**: Memory-constrained or simple datasets

#### 128-dimensional embeddings (baseline)
- **Pros**: Well-established for NLP tasks
- **Cons**: May be limiting for very rich vocabularies
- **Best for**: Standard choice, good balance

#### 256-dimensional embeddings
- **Pros**: Rich semantic space, captures nuances
- **Cons**: More parameters, slower computation, overfitting risk
- **Optimal if**: Dataset is large enough (50k samples ✓)

### Expected Learning

1. **Embedding space geometry**:
   - Larger embeddings = broader vector space
   - Can represent "similar" words more distinctly
   - At cost of increased model complexity

2. **Sentiment-specific representatio**:
   - 64-dim may blur sentiment-bearing distinctions
   - 128-dim likely sufficient for binary sentiment
   - 256-dim provides safety margin for edge cases

3. **Diminishing returns**:
   - IMDb is relatively simple (binary sentiment)
   - Expect modest improvements from 128→256
   - Parallel to MLP Part B findings

---

## Experiment 3: Recurrent Layer Depth

### Objective
Analyze the trade-off between model capacity (more layers) and overfitting risk.

### Design

**Fixed Configuration**:
- `rnn_type = LSTM` (best from Exp 1)
- `embedding_dim = 128` (baseline or best from Exp 2)
- `hidden_dim = 128`
- `bidirectional = True`
- `dropout = 0.3`

**Variable**:
- `n_layers ∈ {1, 2}`

### Why Not 3+ Layers?
- Diminishing returns for binary classification
- RNNs harder to train deeply (gradient issues)
- IMDb is relatively simple task

### 1-Layer Architecture

```
Embedding → BiLSTM(128) → Dropout → Linear → Sigmoid
```

**Characteristics**:
- Learns one level of temporal abstraction
- Captures direct word-level + bigram-level patterns
- Sufficient for most sentiment signals

### 2-Layer Architecture

```
Embedding → BiLSTM(128) → Dropout → BiLSTM(128) → Dropout → Linear → Sigmoid
```

**Characteristics**:
- Stacked temporal abstraction
- First layer: word sequences → intermediate features
- Second layer: feature sequences → predictions
- Higher capacity, more prone to overfitting

### Expected Trade-off

**1-Layer wins because**:
- Simpler = easier to train
- Less prone to overfitting
- Sufficient for IMDb's binary classification

**2-Layer might win if**:
- Dataset contains complex sentiment patterns
- Proper regularization (dropout, L2) controls overfitting
- Validation accuracy improves despite more parameters

### Overfitting Indicators to Watch
- **Train loss ↓ but val loss ↑** = overfitting
- **Val accuracy plateaus earlier** = underfitting
- **Train-val gap widening with depth** = capacity > data complexity

---

## Architecture Details: RNNClassifier

### Forward Pass
```python
# 1. Embedding with dropout
x = self.embedding(token_ids)  # [batch, seq_len, emb_dim]
x = self.emb_dropout(x)

# 2. Pack padded sequences (remove padding)
packed = pack_padded_sequence(x, lengths, enforce_sorted=False)

# 3. Apply RNN layers
output, hidden = self.rnn(packed)

# 4. Unpack sequences (restore padding)
output, _ = pad_packed_sequence(output)

# 5. Extract last hidden state
# Handle bidirectional: hidden is (num_layers*num_directions, batch, hidden_dim)
if self.bidirectional:
    h_t = torch.cat((hidden[-2], hidden[-1]), dim=1)  # [batch, 2*hidden_dim]
else:
    h_t = hidden[-1]  # [batch, hidden_dim]

# 6. Classification layer
logits = self.classifier(self.dropout(h_t))  # [batch, 1]
return logits
```

### Key Implementation Choices

1. **Pack/Unpack Sequences**:
   - Only processes actual tokens, skips padding
   - Avoids wasting computation on padding tokens
   - Required for efficiency with variable-length sequences

2. **Sequence Length Tracking**:
   - Computed from: `lengths = (token_ids != 0).sum(dim=1)`
   - Exploits pad token index = 0
   - Essential for correct packing

3. **Bidirectional Hidden State Extraction**:
   ```python
   # After bidirectional LSTM(2 directions, so 2 final hidden states)
   h_t = torch.cat((hidden[-2], hidden[-1]), dim=1)
   #      forward      backward
   # Concatenate gives [batch, 2*hidden_dim]
   ```

4. **Dropout Placement**:
   - After embedding (reduce embedding noise)
   - After RNN (prevent co-adaptation)
   - Before classification (final regularization)

---

## Experimental Protocol

### For Each Configuration

1. **Model Initialization**
   ```python
   model = RNNClassifier(...hyperparams...)
   optimizer = torch.optim.Adam(lr=0.001)
   criterion = nn.BCEWithLogitsLoss()
   ```

2. **Training Loop (10 epochs)**
   ```
   For each epoch:
     ∙ Forward pass
     ∙ Backward pass
     ∙ Gradient clipping (max_norm=1.0) - prevent exploding gradients
     ∙ Optimizer step
     ∙ Validation evaluation
     ∙ Checkpoint best model (based on val loss)
   ```

3. **Gradient Clipping**
   - RNNs prone to exploding gradients
   - Clipping to norm=1.0 prevents instability
   - Standard practice for recurrent models

4. **Evaluation Metrics**
   - **Training Loss**: BCEWithLogitsLoss per batch
   - **Validation Loss**: BCE on full val set
   - **Validation Accuracy**: (predictions == truth).mean()
   - **Test Accuracy**: Final performance
   - **Training Time**: Wall-clock seconds

### Early Stopping (Not Implemented Here)
- Typically: stop if val loss doesn't improve for 3+ epochs
- For comparability, we train all configs for 10 epochs fixed
- Allows fair comparison without convergence bias

---

## Analysis Framework

### Within-Experiment Comparisons

**Experiment 1 Analysis**:
- Which variant has best test accuracy?
- Which converges fastest (fewer epochs to stable val loss)?
- Which uses least wall-clock time?
- Trade-off: accuracy vs speed vs simplicity?

**Experiment 2 Analysis**:
- Is 64-dim clearly insufficient?
- Does 256-dim provide meaningful gains?
- At what embedding size do gains saturate?
- Parameter-accuracy tradeoff?

**Experiment 3 Analysis**:
- Does 2-layer outperform 1-layer?
- If so, by how much? (0.1%? 1%?)
- Does train-val gap widen with depth?
- Evidence of overfitting in deeper model?

### Cross-Experiment Synthesis

- **Best MLP (from Part B)** vs **Best RNN (from Part C)**:
  - Does sequence modeling help?
  - By how much? (accuracy delta)
  - Is it worth the computational cost?

- **Architecture Insights**:
  - Is LSTM overkill for binary sentiment?
  - GRU sufficient and faster?
  - How much does embedding dimension matter?
  - Is RNN depth necessary?

---

## Expected Insights

### Variant Comparison (Exp 1)
**Hypothesis**: LSTM provides best accuracy but GRU offers speed advantage

**Why?**
- LSTM's explicit memory cell excels at long sequences
- Sentiment often depends on distant words (e.g., negation + predicate)
- GRU simplicity may be sufficient for binary task

### Embedding Dimension (Exp 2)
**Hypothesis**: 128 ≈ 256 in performance, both >> 64

**Why?**
- 50k samples support rich embeddings
- Sentiment is relatively dense in embedding space
- Diminishing returns beyond 128-dim

### Network Depth (Exp 3)
**Hypothesis**: 1-layer ≥ 2-layer due to overfitting

**Why?**
- IMDb is a relatively simple binary task
- First RNN layer captures temporal patterns
- Second layer may overfit to training noise
- Regularization (dropout) helps but doesn't fully compensate

---

## Comparison with MLP Baseline

### Key Differences

| Aspect | MLP (Part B) | RNN (Part C) |
|--------|-----------|-----------|
| **Input** | Mean-pooled embeddings | Sequence of embeddings |
| **Word Order** | Ignored (bag-of-words) | Preserved and modeled |
| **Negation** | May miss "not bad" | Captures via sequence |
| **Parameters** | ~2-5k (embedding+layers) | ~200-500k |
| **Training Time** | Fast (~1 sec/epoch) | Moderate (~5 sec/epoch) |
| **Max Sequence** | Fixed (mean over all) | Variable length (packed) |

### When RNN Wins
- Presence of negation/intensifiers
- Sarcasm or irony
- Complex multi-clause reviews
- Where word order determines sentiment

### When MLP Wins
- Speed critical
- Limited compute
- Simple datasets
- When bag-of-words sufficient

---

## Recommendations for Implementation

### Computational Considerations
- **GPU Required**: RNNs benefit significantly from GPU
  - CPU: ~5 min per config
  - GPU: ~30 sec per config
- **Total Runtime**: ~1 hour for all configs (GPU)

### Reproducibility
- Set `torch.manual_seed(42)`
- Fix random state for data splitting
- Document PyTorch version used

### Checkpointing
- Save best model per config
- Format: `rnn_{variant}_emb{dim}_layers{n}_best.pt`
- Allows future loading and ensemble methods

### Monitoring
- Plot training curves (loss over epochs)
- Track gradient norm (diagnose vanishing gradients)
- Monitor validation accuracy plateau

---

## Summary Table

| Experiment | Variable | Controlled | Metrics |
|-----------|----------|-----------|---------|
| 1 | RNN type | emb_dim, hidden_dim, layers | accuracy, loss, time |
| 2 | emb_dim | RNN type, hidden_dim, layers | accuracy, loss, params |
| 3 | n_layers | RNN type, emb_dim, hidden_dim | accuracy, loss, gap |

**Deliverables**:
- ✅ Training curves for each config
- ✅ Test accuracy rankings
- ✅ Convergence analysis
- ✅ Comparative visualizations
- ✅ Best config recommendations
- ✅ MLP vs RNN final comparison

