# Part B: MLP Network Architecture Ablation Study

## Overview

Part B explores how key architectural hyperparameters affect MLP model performance on IMDb sentiment classification. Through three carefully controlled experiments, we systematically analyze:

1. **Experiment 1**: Network depth (1 vs 2 vs 3 hidden layers)
2. **Experiment 2**: Embedding dimension ($d_e \in \{64, 128, 256\}$)
3. **Experiment 3**: Dropout rate ($p \in \{0.2, 0.3, 0.5\}$)

---

## Architecture Overview

### MLP Model Structure

The MLP classifier follows this pipeline:

```
Input: Token IDs
     ↓
[Embedding Layer]
  vocab_size → d_e dimensions
     ↓
[Mean Pooling]
  Aggregate all word embeddings over sequence
  Handles variable-length sequences via masking
     ↓
[Hidden Layers] (variable depth)
  Full-connected layers with ReLU activation
  Dropout applied after each layer
     ↓
[Output Layer]
  Linear: hidden_dim → 1 (binary classification)
  Sigmoid activation: logits → [0, 1] probabilities
```

### Mathematical Formulation

For a batch of sequences with mean-pooled embeddings $\mathbf{x} \in \mathbb{R}^{d_e}$:

$$\mathbf{h}_1 = \sigma(W_1 \mathbf{x} + \mathbf{b}_1)$$

For 2+ layers:
$$\mathbf{h}_{i+1} = \sigma(W_{i+1} \mathbf{h}_i + \mathbf{b}_{i+1}), \quad i = 1, \ldots, L-1$$

Final output:
$$\hat{y} = \sigma(W_L \mathbf{h}_{L-1} + \mathbf{b}_L)$$

Where:
- $\sigma(\cdot)$ is ReLU for hidden layers, sigmoid for output
- Dropout is applied after each hidden layer
- $L$ is the number of hidden layers

---

## Experiment 1: Network Depth

### Objective
Determine the optimal number of hidden layers while keeping total parameter count approximately constant.

### Design

We compare three configurations:
- **1-layer**: `hidden_dims = [512]` → dense representation
- **2-layers**: `hidden_dims = [256, 256]` → moderate hierarchy
- **3-layers**: `hidden_dims = [128, 256, 128]` → deep hierarchy

**Fixed controls:**
- `embed_dim = 128`
- `dropout = 0.3`
- Training epochs = 10
- Learning rate = 0.001

### Expected Outcomes

1. **Depth Benefits:**
   - More layers allow learning of hierarchical feature representations
   - Can capture increasingly abstract sentiment-related patterns

2. **Depth Risks:**
   - Deeper networks are harder to optimize (vanishing gradients)
   - Over-parameterization may lead to overfitting on small datasets
   - Increased computational cost

3. **For Bag-of-Words (BOW) Classification:**
   - Since the model receives aggregated embeddings (not sequential data), depth may offer limited benefit
   - A single hidden layer might be sufficient to separate positive/negative reviews
   - Additional layers risk fitting to noise rather than extracting useful patterns

### Key Metrics
- **Training/Validation Loss**: Monitor convergence and overfitting
- **Test Accuracy**: Final model performance
- **Parameter Count**: Verify comparable model sizes
- **Training Time**: Computational efficiency

---

## Experiment 2: Embedding Dimension

### Objective
Determine the optimal embedding dimension using the best architecture from Experiment 1.

### Design

We test three embedding dimensions:
- **d_e = 64**: Compact representation
- **d_e = 128**: Moderate richness
- **d_e = 256**: Rich semantic space

**Fixed controls:**
- `hidden_dims = [BEST_FROM_EXP1]`
- `dropout = 0.3`
- Training epochs = 10

### Expected Outcomes

1. **Smaller Embeddings (d_e = 64):**
   - Fewer parameters → faster training
   - Lower dimensional space → less expressive, possible underfitting
   - May struggle to capture sentiment nuances

2. **Moderate Embeddings (d_e = 128):**
   - Balance of expressiveness and efficiency
   - Often works well for text classification
   - Enables good generalization

3. **Larger Embeddings (d_e = 256):**
   - Rich semantic representations
   - More parameters → risk of overfitting
   - Higher computational cost
   - May improve accuracy if dataset is large enough

### Key Metrics
- **Test Accuracy**: Does richer embedding help?
- **Parameter Count**: Scaling with embedding dimension
- **Overfitting Gap**: (train_loss - val_loss) to detect overfitting
- **Convergence Speed**: Which dimension trains fastest?

### Relationship to IMDb Dataset

The IMDb dataset (50k reviews) is relatively large, reducing overfitting risk. Larger embeddings can capture:
- Sentiment-specific word relationships
- Negations and intensifiers
- Domain-specific vocabulary

---

## Experiment 3: Dropout Rate

### Objective
Analyze the impact of dropout on overfitting, using the best architecture and embedding dimension from Experiments 1 & 2.

### Design

We test three dropout rates:
- **p = 0.2**: Light regularization, lower overfitting prevention
- **p = 0.3**: Moderate regularization (common baseline)
- **p = 0.5**: Heavy regularization, aggressive noise injection

**Fixed controls:**
- `embed_dim = [BEST_FROM_EXP2]`
- `hidden_dims = [BEST_FROM_EXP1]`
- Training epochs = 10

### Dropout Mechanism

At training time, each neuron is randomly dropped with probability $p$:
$$\mathbf{h}' = \mathbf{h} \odot \mathbf{m}, \quad \mathbf{m}_i \sim \text{Bernoulli}(1-p)$$

This forces the network to learn robust features without relying on specific neurons.

### Expected Outcomes

1. **p = 0.2 (Light Regularization):**
   - Minimal training disruption
   - Model can leverage full capacity
   - Risk: potential overfitting if model capacity >> data complexity
   - Benefit: faster convergence

2. **p = 0.3 (Moderate Regularization):**
   - Standard choice for many applications
   - Balances learning capacity and generalization
   - Effective noise injection without excessive performance penalty

3. **p = 0.5 (Heavy Regularization):**
   - Half the neurons dropped during training → less information flow
   - Robust ensemble-like learning behavior
   - Risk: underfitting if too much capacity is lost
   - Benefit: exceptional generalization on small test data

### Metrics for Overfitting Analysis

- **Train-Val Gap**: Difference between training and validation curves
  - Large gap ($>$ 0.05) → overfitting
  - Small gap ($<$ 0.02) → good regularization
  
- **Test Accuracy**: Generalization performance

- **Learning Curves**: Shape reveals:
  - Smooth convergence → good regularization
  - Noisy curves → high dropout rate disrupting learning
  - Diverging curves → underfitting (dropout too high)

---

## Experimental Protocol

### Training Loop for Each Configuration

1. **Initialize Model**: Create MLPClassifier with specified hyperparameters
2. **Setup Optimizer**: Adam optimizer with lr=0.001, weight_decay=1e-5
3. **Training Phase**:
   - Forward pass through batch
   - Compute BCEWithLogitsLoss
   - Backward pass with gradient clipping (max_norm=1.0)
   - Gradient step
   - Track: loss, accuracy

4. **Validation Phase** (every epoch):
   - Evaluate on validation set
   - No dropout (eval mode)
   - Track: loss, accuracy

5. **Early Stopping**: 
   - Stop if no improvement for 3 epochs
   - Save best model based on validation loss

6. **Test Evaluation**:
   - Load best checkpoint
   - Evaluate on test set
   - Record final metrics

### Metrics Tracked

For each model:
- **Training History**: loss and accuracy per epoch
- **Validation Performance**: best validation accuracy
- **Test Performance**: accuracy, F1-score, precision, recall
- **Parameter Count**: for equivalence verification
- **Training Time**: computational efficiency

---

## Analysis Framework

### Interpreting Results

#### Within-Experiment Analysis
- Compare performance differences between configurations
- Identify which hyperparameter most affects results
- Quantify overfitting (train-val gap) for each setting

#### Cross-Experiment Insights
- Does winner from Exp 1 beat baseline?
- Do Exp 2 and 3 recommendations align?
- Which hyperparameter has largest impact?

#### Statistical Significance
- If differences < 0.5%, consider results inconclusive
- Noise from random initialization/order effects

### Visualization Strategy

For each experiment:

1. **Training Curves**:
   - Plot train/val loss over epochs for all configs
   - Shows convergence speed and overfitting patterns
   - Identify early stopping points

2. **Accuracy Comparison**:
   - Bar charts: test accuracy by configuration
   - Shows which setting performs best

3. **Parameter Count Verification**:
   - Confirm Exp 1 keeps params roughly constant
   - Document scaling for Exp 2

4. **Overfitting Analysis** (Experiment 3):
   - Plot train-validation gap over dropout rates
   - Visualize how dropout controls generalization

---

## Expected Insights

### Depth (Exp 1)
**Hypothesis**: Single hidden layer sufficient for bag-of-words classification
- Mean pooling removes positional information
- No deep hierarchies needed for aggregated features
- Expected: 1-layer ≥ 2-layer ≥ 3-layer due to overfitting

### Embedding Dimension (Exp 2)
**Hypothesis**: Moderate embedding size (128) balances expressiveness and generalization
- 64-dim: May underfit, limited semantic capacity
- 128-dim: Good baseline for text classification
- 256-dim: Marginal gains, risk of overfitting
- Expected: 128 or 256 performs best; 64 noticeably worse

### Dropout Rate (Exp 3)
**Hypothesis**: Higher dropout helps if model is overfitting
- p=0.2: Let model use full capacity → low train-val gap if simple task
- p=0.3: Standard setting, works well
- p=0.5: Aggressive, may underfit
- Expected: p=0.3 or p=0.2 optimal; p=0.5 hurts performance

---

## Recommendations for Running Experiments

### Computational Considerations
- **GPU Recommended**: Each config takes ~5-10 min on CPU, ~30 sec on GPU
- **Total Runtime**: ~30 min (all configs) on GPU, ~3 hours on CPU

### Reproducibility
- Set random seed at notebook start: `torch.manual_seed(42)`
- Document PyTorch and CUDA versions
- Save all training curves for future reference

### Data Handling
- Use same 70/10/20 split from preprocessing
- Ensure same data normalization/tokenization
- Stratified split to maintain class balance

### Model Checkpointing
- Save best model per experiment to `checkpoints/`
- Format: `mlp_depth_{config}.pt`
- Allows resuming analysis without retraining

---

## Summary

This ablation study provides empirical answers to three key design questions:

1. **"How many hidden layers?"** → Measured by Experiment 1
2. **"What embedding size?"** → Measured by Experiment 2
3. **"How much regularization?"** → Measured by Experiment 3

The results form the foundation for fair comparison with RNN models (Part C), ensuring:
- MLPs are properly tuned
- Resources allocated efficiently
- Conclusions about architectures are sound

