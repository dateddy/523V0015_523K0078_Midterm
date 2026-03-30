"""
src/mlp_model.py
MLP Classifier for sentiment analysis.
Contains: MLPClassifier, count_params, save_checkpoint, load_checkpoint.
"""

import os
import torch
import torch.nn as nn

PAD = 0 

# ── Model ─────────────────────────────────────────────────────────────────────

class MLPClassifier(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int,
                 hidden_dims: list[int], num_classes: int,
                 dropout: float = 0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD)

        # Build hidden layers dynamically from hidden_dims
        layers = []
        in_dim = embed_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, num_classes))
        self.classifier = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T) — token id sequences
        emb    = self.embedding(x)                        # (B, T, d_e)
        mask   = (x != PAD).unsqueeze(2)                  # (B, T, 1)
        pooled = (emb * mask).sum(1) / mask.sum(1).clamp(min=1)  # (B, d_e)
        return self.classifier(pooled)


# ── Utilities ─────────────────────────────────────────────────────────────────

def count_params(model: nn.Module) -> int:
    """Return total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(model: nn.Module, config: dict,
                    val_acc: float, test_acc: float,
                    path: str) -> None:
    """
    Save model weights + config + metrics to a .pt file.

    Saved keys:
        model_state  : model.state_dict()
        config       : hyperparameter dict used to build the model
        val_acc      : best validation accuracy at checkpoint time
        test_acc     : test accuracy of the best checkpoint

    Usage:
        save_checkpoint(model, cfg, best_val, test_acc,
                        'checkpoints/mlp_depth_1layer.pt')
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'model_state': model.state_dict(),
        'config':      config,
        'val_acc':     val_acc,
        'test_acc':    test_acc,
    }, path)
    print(f'Checkpoint saved → {path}  (val={val_acc:.4f}  test={test_acc:.4f})')


def load_checkpoint(path: str, vocab_size: int,
                    num_classes: int = 2,
                    device: str = 'cpu') -> tuple[nn.Module, dict]:
    """
    Load a saved checkpoint and reconstruct the model.

    Returns:
        model   : MLPClassifier loaded with saved weights, set to eval mode
        metadata: dict with keys config, val_acc, test_acc

    Usage:
        model, meta = load_checkpoint('checkpoints/mlp_depth_1layer.pt',
                                      vocab_size=len(vocab))
        print(meta['val_acc'], meta['test_acc'])
    """
    ckpt = torch.load(path, map_location=device)
    cfg  = ckpt['config']

    model = MLPClassifier(
        vocab_size   = vocab_size,
        embed_dim    = cfg['embed_dim'],
        hidden_dims  = cfg['hidden_dims'],
        num_classes  = num_classes,
        dropout      = cfg['dropout'],
    ).to(device)

    model.load_state_dict(ckpt['model_state'])
    model.eval()

    metadata = {
        'config':   cfg,
        'val_acc':  ckpt['val_acc'],
        'test_acc': ckpt['test_acc'],
    }
    print(f'Checkpoint loaded ← {path}  (val={metadata["val_acc"]:.4f}  test={metadata["test_acc"]:.4f})')
    return model, metadata
