# Midterm project 

# Text Sentiment Analysis: Comparing Multi-Layer Perceptrons and Recurrent Networks (RNN/LSTM/GRU)

## Project Overview

This project focuses on **text sentiment analysis**, where the goal is to classify text (e.g., movie reviews) into sentiment categories such as **positive** or **negative**.

The project compares two major neural network approaches:
- **MLP (Multi-Layer Perceptron)** — treats text as bag-of-words (ignores word order)
- **RNN-based models (RNN, LSTM, GRU)** — capture sequential dependencies in text


## Key Objectives

- Develop a complete NLP preprocessing pipeline (text cleaning, vocabulary construction, sequence encoding, and dynamic batch padding).

- Implement and train an MLP for text that takes fixed-length sentence representions as input.

- Understand, implement gating mechanisms and compare three recurrent architectures systematically (RNN, LSTM, or GRU).

- Conduct controlled ablation studies (network depth, embedding dimensions, etc.).

- Perform qualitative and quantitative error analysis.

## Dataset

- IMDb Movie Reviews (50,000 samples, binary sentiment).


## Technical Requirements

### 🔹 Part A: Data Preprocessing
Build a reusable pipeline including:
- Text cleaning (remove HTML, normalize case)
- Tokenization & vocabulary creation (10k–30k tokens)
- Encoding text into sequences
- Padding sequences to fixed length

### 🔹 Part B: MLP Model
- Input: mean-pooled word embeddings

- Architecture:

```
            Embedding → Mean Pool → Linear → ReLU → Dropout → Linear → Output
```

- Train and evaluate performance

- Perform ablation experiments


### 🔹 Part C: RNN Models
Implement at least **2 variants**:
- Vanilla RNN
- LSTM
- GRU

Key requirements:
- Handle variable-length sequences (e.g., packed sequences)
- Run experiments:
1. Compare RNN vs LSTM vs GRU
2. Test embedding sizes (64, 128, 256)
3. Compare 1-layer vs 2-layer networks

### 🔹 Part D: Comparative Analysis
- Compare all models on:
- Accuracy
- Training speed
- Convergence behavior
- Present results in tables and plots (learning curves)

## Experiments
Minimum required:
1. Model comparison (RNN variants)
2. Embedding dimension tuning
3. Network depth analysis

## Project Structure

The project is organized as follows:

```
ID1_ID2_Midterm/
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_mlp.ipynb
│   ├── 03_rnn.ipynb
│   └── 04_analysis.ipynb
├── src/
│   ├── preprocess.py
│   ├── mlp_model.py
│   ├── rnn_model.py
│   ├── train.py
│   └── evaluate.py
├── checkpoints/
│   ├── mlp_best.pt
│   └── rnn_best.pt
├── report/
│   └── report.pdf
├── requirements.txt
└── README.md
```

## Report Requirements
- Max 15 pages
- Include:
  - Abstract (≤200 words)
  - Dataset analysis
  - Model descriptions
  - Experiment results
  - Comparative discussion


## Grading Criteria
- Data preprocessing quality
- Model implementation correctness
- Experiment design and analysis
- Clarity of comparison and reporting