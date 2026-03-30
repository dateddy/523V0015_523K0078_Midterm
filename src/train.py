"""
Shared training loop with checkpoint saving and early stopping.
"""

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import os


class TrainingConfig:
    """Training configuration."""
    
    def __init__(self, 
                 epochs: int = 10,
                 learning_rate: float = 1e-3,
                 weight_decay: float = 1e-5,
                 batch_size: int = 64,
                 max_grad_norm: float = 1.0,
                 patience: int = 3,
                 checkpoint_dir: str = './checkpoints',
                 model_name: str = 'model'):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        self.patience = patience
        self.checkpoint_dir = checkpoint_dir
        self.model_name = model_name


class Trainer:
    """Model trainer with early stopping and checkpointing."""
    
    def __init__(self, model: nn.Module, config: TrainingConfig, device: str = 'cuda'):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Create checkpoint directory
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        # Optimizer and loss
        self.optimizer = Adam(model.parameters(), 
                             lr=config.learning_rate,
                             weight_decay=config.weight_decay)
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Tracking
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Get batch data
            if len(batch) == 3:  # token_ids, lengths, labels
                token_ids, lengths, labels = batch
                token_ids = token_ids.to(self.device)
                lengths = lengths.to(self.device)
                labels = labels.to(self.device).float()
                
                # Forward pass - RNN model
                logits = self.model(token_ids, lengths)
            else:  # token_ids, labels
                token_ids, labels = batch
                token_ids = token_ids.to(self.device)
                labels = labels.to(self.device).float()
                
                # Forward pass - MLP model
                logits = self.model(token_ids)
            
            # Loss
            loss = self.criterion(logits, labels.unsqueeze(1))
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
                                          self.config.max_grad_norm)
            
            # Optimization step
            self.optimizer.step()
            
            total_loss += loss.item()
            total_batches += 1
            
            # Print progress
            if (batch_idx + 1) % 100 == 0:
                avg_loss = total_loss / total_batches
                print(f"  Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {avg_loss:.4f}")
        
        return total_loss / total_batches
    
    def validate(self, val_loader: DataLoader) -> float:
        """Validate model."""
        self.model.eval()
        total_loss = 0.0
        total_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # Get batch data
                if len(batch) == 3:  # token_ids, lengths, labels
                    token_ids, lengths, labels = batch
                    token_ids = token_ids.to(self.device)
                    lengths = lengths.to(self.device)
                    labels = labels.to(self.device).float()
                    
                    # Forward pass - RNN model
                    logits = self.model(token_ids, lengths)
                else:  # token_ids, labels
                    token_ids, labels = batch
                    token_ids = token_ids.to(self.device)
                    labels = labels.to(self.device).float()
                    
                    # Forward pass - MLP model
                    logits = self.model(token_ids)
                
                # Loss
                loss = self.criterion(logits, labels.unsqueeze(1))
                
                total_loss += loss.item()
                total_batches += 1
        
        return total_loss / total_batches
    
    def save_checkpoint(self, epoch: int, metrics: Dict = None) -> None:
        """Save model checkpoint."""
        checkpoint_path = os.path.join(
            self.config.checkpoint_dir, 
            f'{self.config.model_name}_best.pt'
        )
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics or {}
        }
        
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict:
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Checkpoint loaded: {checkpoint_path}")
        return checkpoint
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Tuple[List, List]:
        """
        Complete training loop with early stopping.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            
        Returns:
            (train_losses, val_losses)
        """
        print(f"\nTraining {self.config.model_name} for {self.config.epochs} epochs")
        print(f"Device: {self.device}, Learning rate: {self.config.learning_rate}")
        print("="*70)
        
        for epoch in range(self.config.epochs):
            # Train
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate(val_loader)
            self.val_losses.append(val_loss)
            
            print(f"Epoch [{epoch + 1}/{self.config.epochs}]")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss:   {val_loss:.4f}")
            
            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint(epoch, {'val_loss': val_loss})
            else:
                self.patience_counter += 1
                print(f"  No improvement. Patience: {self.patience_counter}/{self.config.patience}")
                
                if self.patience_counter >= self.config.patience:
                    print(f"\nEarly stopping triggered after epoch {epoch + 1}")
                    break
            
            print()
        
        print("="*70)
        print(f"Training complete. Best validation loss: {self.best_val_loss:.4f}")
        
        return self.train_losses, self.val_losses


def shared_training_loop(model: nn.Module,
                        train_loader: DataLoader,
                        val_loader: DataLoader,
                        device: str = 'cuda',
                        config: TrainingConfig = None) -> Tuple[List, List, float]:
    """
    Shared training loop for any model.
    
    Args:
        model: PyTorch model (MLPClassifier or RNNClassifier)
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to use ('cuda' or 'cpu')
        config: Training configuration
        
    Returns:
        (train_losses, val_losses, best_val_loss)
    """
    if config is None:
        config = TrainingConfig()
    
    trainer = Trainer(model, config, device)
    train_losses, val_losses = trainer.train(train_loader, val_loader)
    
    return train_losses, val_losses, trainer.best_val_loss
