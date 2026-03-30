"""
MLP Model Implementation for IMDb Sentiment Analysis
"""

import torch
import torch.nn as nn


class MLPClassifier(nn.Module):
    """Multi-Layer Perceptron classifier with mean pooling over embeddings."""
    
    def __init__(self, vocab_size: int, embedding_dim: int = 100, 
                 hidden_dims: list = None, dropout: float = 0.3):
        """
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Embedding dimension
            hidden_dims: List of hidden layer dimensions (e.g., [128, 64])
            dropout: Dropout rate
        """
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [128]
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        
        # Build hidden layers dynamically
        layers = []
        prev_dim = embedding_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        self.hidden = nn.Sequential(*layers) if layers else nn.Identity()
        self.classifier = nn.Linear(prev_dim, 1)  # Binary classification
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            token_ids: [batch_size, seq_len]
            
        Returns:
            logits: [batch_size, 1]
        """
        # Embedding: [batch_size, seq_len] -> [batch_size, seq_len, embedding_dim]
        embedded = self.embedding(token_ids)
        embedded = self.dropout(embedded)
        
        # Mean pooling: [batch_size, seq_len, embedding_dim] -> [batch_size, embedding_dim]
        # Create mask to ignore padding tokens
        mask = (token_ids != 0).unsqueeze(-1).float()  # [batch_size, seq_len, 1]
        embedded_masked = embedded * mask
        pooled = embedded_masked.sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        
        # Hidden layers
        hidden_out = self.hidden(pooled)
        
        # Classifier
        logits = self.classifier(hidden_out)
        
        return logits