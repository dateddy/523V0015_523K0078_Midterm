"""
RNN-based Models for Text Sentiment Analysis
Supports RNN, LSTM, and GRU architectures with proper packed sequence handling
"""

import torch
import torch.nn as nn


class RNNClassifier(nn.Module):
    """
    RNN-based classifier for binary text sentiment classification.
    
    Uses packed sequences (pack_padded_sequence) to handle variable-length inputs
    efficiently without wasting computation on padding tokens.
    
    Supports three RNN types:
    - 'rnn': Vanilla RNN
    - 'lstm': LSTM (Long Short-Term Memory)
    - 'gru': GRU (Gated Recurrent Unit)
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 100,
        hidden_dim: int = 128,
        n_layers: int = 1,
        rnn_type: str = 'lstm',
        bidirectional: bool = True,
        dropout: float = 0.3,
        num_classes: int = 1
    ):
        """
        Initialize RNNClassifier.
        
        Args:
            vocab_size (int): Size of vocabulary
            embedding_dim (int): Dimension of word embeddings
            hidden_dim (int): Dimension of RNN hidden state
            n_layers (int): Number of RNN layers (1-based)
            rnn_type (str): Type of RNN - 'rnn', 'lstm', or 'gru'
            bidirectional (bool): Whether to use bidirectional RNN
            dropout (float): Dropout rate (applied between RNN layers)
            num_classes (int): Number of output classes (1 for binary with sigmoid)
        """
        super().__init__()
        
        assert rnn_type.lower() in ['rnn', 'lstm', 'gru'], \
            f"rnn_type must be 'rnn', 'lstm', or 'gru', got {rnn_type}"
        assert n_layers >= 1, f"n_layers must be >= 1, got {n_layers}"
        assert hidden_dim > 0, f"hidden_dim must be > 0, got {hidden_dim}"
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.rnn_type = rnn_type.lower()
        self.bidirectional = bidirectional
        self.num_classes = num_classes
        
        # Embedding layer with padding_idx=0
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=0
        )
        
        # Dropout applied to embeddings
        self.embedding_dropout = nn.Dropout(dropout)
        
        # Select RNN type
        rnn_cls = {'rnn': nn.RNN, 'lstm': nn.LSTM, 'gru': nn.GRU}[self.rnn_type]
        
        # RNN layer(s)
        self.rnn = rnn_cls(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
            bidirectional=bidirectional
        )
        
        # Output layer
        # If bidirectional, RNN output is hidden_dim * 2
        rnn_output_dim = hidden_dim * (2 if bidirectional else 1)
        self.fc = nn.Linear(rnn_output_dim, num_classes)
        
        # Dropout before output layer
        self.fc_dropout = nn.Dropout(dropout)
    
    def forward(self, token_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with packed sequences for efficient variable-length handling.
        
        Args:
            token_ids (torch.Tensor): Shape (batch_size, max_len) - padded token IDs
            lengths (torch.Tensor): Shape (batch_size,) - actual sequence lengths
        
        Returns:
            torch.Tensor: Shape (batch_size, num_classes) - logits
        """
        batch_size = token_ids.size(0)
        device = token_ids.device
        
        # Embedding: (batch_size, max_len) → (batch_size, max_len, embedding_dim)
        embedded = self.embedding(token_ids)
        embedded = self.embedding_dropout(embedded)
        
        # Pack padded sequence for efficient RNN computation
        # lengths must be on CPU for pack_padded_sequence
        lengths_cpu = lengths.cpu()
        
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded,
            lengths_cpu,
            batch_first=True,
            enforce_sorted=False
        )
        
        # RNN forward pass
        if self.rnn_type == 'lstm':
            packed_output, (hidden, cell) = self.rnn(packed_embedded)
            # For LSTM, we use hidden state (not cell state)
        else:
            # RNN or GRU
            packed_output, hidden = self.rnn(packed_embedded)
        
        # Unpack sequence: (batch_size, max_len, hidden_dim*num_directions)
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(
            packed_output,
            batch_first=True
        )
        
        # Extract last valid hidden state
        # hidden shape: (n_layers * num_directions, batch_size, hidden_dim)
        if self.bidirectional:
            # For bidirectional: hidden[-2:] contains last layer's outputs
            # Concatenate forward and backward directions
            last_hidden = hidden[-2:]  # (2, batch_size, hidden_dim)
            last_hidden = last_hidden.permute(1, 0, 2)  # (batch_size, 2, hidden_dim)
            last_hidden = last_hidden.reshape(batch_size, -1)  # (batch_size, hidden_dim*2)
        else:
            # For unidirectional: use last layer's hidden state
            last_hidden = hidden[-1]  # (batch_size, hidden_dim)
        
        # Apply dropout and classify
        last_hidden = self.fc_dropout(last_hidden)
        logits = self.fc(last_hidden)  # (batch_size, num_classes)
        
        return logits
    
    def get_config(self) -> dict:
        """Return model configuration as dictionary."""
        return {
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'hidden_dim': self.hidden_dim,
            'n_layers': self.n_layers,
            'rnn_type': self.rnn_type,
            'bidirectional': self.bidirectional,
            'num_classes': self.num_classes
        }
    
    def __repr__(self) -> str:
        """String representation of the model."""
        config = self.get_config()
        return (
            f"RNNClassifier(\n"
            f"  vocab_size={config['vocab_size']},\n"
            f"  embedding_dim={config['embedding_dim']},\n"
            f"  hidden_dim={config['hidden_dim']},\n"
            f"  n_layers={config['n_layers']},\n"
            f"  rnn_type='{config['rnn_type']}',\n"
            f"  bidirectional={config['bidirectional']}\n"
            f")"
        )


class RNNClassifierMultiConfig(nn.Module):
    """
    Multi-configuration RNN classifier for easy comparison of variants.
    Allows training multiple RNN types with different configurations.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 100,
        hidden_dim: int = 128,
        n_layers: int = 1,
        rnn_type: str = 'lstm',
        bidirectional: bool = True,
        dropout: float = 0.3
    ):
        """Initialize with flexible config."""
        super().__init__()
        self.classifier = RNNClassifier(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            rnn_type=rnn_type,
            bidirectional=bidirectional,
            dropout=dropout
        )
    
    def forward(self, token_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Forward pass delegates to classifier."""
        return self.classifier(token_ids, lengths)
    
    def get_config(self) -> dict:
        """Get configuration."""
        return self.classifier.get_config()
