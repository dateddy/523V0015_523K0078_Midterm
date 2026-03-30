"""
IMDb Sentiment Analysis - Data Preprocessing Module (OOP Design)

Object-oriented preprocessing pipeline for maximum reusability across notebooks.
Provides TextPreprocessor, VocabularyBuilder, IMDbDataset, and DataManager classes.
"""

import re
import numpy as np
from pathlib import Path
from collections import Counter
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
from sklearn.model_selection import train_test_split

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS & CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

PAD_IDX, UNK_IDX = 0, 1
MAX_VOCAB_SIZE = 20_000
MAX_LEN = 256
BATCH_SIZE = 64


@dataclass
class DataConfig:
    """Configuration for data preprocessing and loading."""
    max_vocab_size: int = 20_000
    max_len: int = 256
    batch_size: int = 64
    random_state: int = 42
    test_size: float = 0.20
    val_size: float = 0.125


# ──────────────────────────────────────────────────────────────────────────────
# TEXT PREPROCESSING CLASS
# ──────────────────────────────────────────────────────────────────────────────

class TextPreprocessor:
    """Handles text cleaning and tokenization."""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Remove HTML tags, special characters, normalize to lowercase."""
        text = re.sub(r'<[^>]+>', ' ', text)          # Remove HTML tags
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)   # Keep letters/digits/spaces
        text = re.sub(r'\s+', ' ', text)               # Collapse whitespace
        return text.lower().strip()
    
    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Split text into tokens by whitespace."""
        return text.split()


# ──────────────────────────────────────────────────────────────────────────────
# VOCABULARY CLASS
# ──────────────────────────────────────────────────────────────────────────────

class VocabularyBuilder:
    """Builds and manages word-to-index vocabulary."""
    
    def __init__(self):
        self.vocab: Dict[str, int] = {'<pad>': PAD_IDX, '<unk>': UNK_IDX}
        self.counter: Optional[Counter] = None
    
    def build(self, texts: List[str], max_size: int = 20_000) -> None:
        """Build vocabulary from texts."""
        self.counter = Counter(w for text in texts for w in text.split())
        for word, _ in self.counter.most_common(max_size - 2):
            self.vocab[word] = len(self.vocab)
    
    def encode(self, tokens: List[str]) -> List[int]:
        """Convert tokens to integer indices."""
        return [self.vocab.get(w, UNK_IDX) for w in tokens]
    
    def decode(self, indices: List[int]) -> List[str]:
        """Convert integer indices back to tokens."""
        idx_to_word = {v: k for k, v in self.vocab.items()}
        return [idx_to_word.get(i, '<unk>') for i in indices]
    
    def __len__(self) -> int:
        return len(self.vocab)
    
    def __getitem__(self, word: str) -> int:
        return self.vocab.get(word, UNK_IDX)
    
    def get_vocab_dict(self) -> Dict[str, int]:
        """Return underlying vocabulary dictionary."""
        return self.vocab.copy()


# ──────────────────────────────────────────────────────────────────────────────
# DATASET CLASS
# ──────────────────────────────────────────────────────────────────────────────

class IMDbDataset(Dataset):
    """PyTorch Dataset for IMDb sentiment classification."""
    
    def __init__(self, texts: List[str], labels: List[int], 
                 vocab: VocabularyBuilder, max_len: int = 256,
                 preprocessor: Optional[TextPreprocessor] = None):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len
        self.preprocessor = preprocessor or TextPreprocessor()
        
        # Pre-encode all samples
        self.data = []
        for text, label in zip(texts, labels):
            tokens = self.preprocessor.tokenize(text)
            ids = self.vocab.encode(tokens)
            ids = ids[:max_len]  # Truncate if too long
            
            self.data.append((
                torch.tensor(ids, dtype=torch.long),
                torch.tensor(label, dtype=torch.long)
            ))
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx]


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Dynamic batch padding to longest sequence in batch."""
    seqs, labels = zip(*batch)
    seqs_padded = pad_sequence(seqs, batch_first=True, padding_value=PAD_IDX)
    return seqs_padded, torch.stack(labels)


# ──────────────────────────────────────────────────────────────────────────────
# DATA MANAGER CLASS
# ──────────────────────────────────────────────────────────────────────────────

class DataManager:
    """Manages data loading, preprocessing, and splitting."""
    
    def __init__(self, config: DataConfig = None):
        self.config = config or DataConfig()
        self.preprocessor = TextPreprocessor()
        self.vocab = VocabularyBuilder()
        
        self.raw_texts: List[str] = []
        self.raw_labels: List[int] = []
        self.cleaned_texts: List[str] = []
    
    @property
    def total_samples(self) -> int:
        """Total number of samples loaded."""
        return len(self.raw_texts)
    
    def load_data(self) -> None:
        """Load IMDb dataset from Hugging Face."""
        print("Loading IMDb dataset...")
        dataset = load_dataset('imdb')
        self.raw_texts = ([ex['text'] for ex in dataset['train']] + 
                          [ex['text'] for ex in dataset['test']])
        self.raw_labels = ([ex['label'] for ex in dataset['train']] + 
                          [ex['label'] for ex in dataset['test']])
        print(f"  Loaded {len(self.raw_texts):,} samples")
    
    def preprocess(self) -> None:
        """Clean all texts."""
        print("Preprocessing texts...")
        self.cleaned_texts = [self.preprocessor.clean_text(t) for t in self.raw_texts]
        print(f"  Cleaned {len(self.cleaned_texts):,} texts")
    
    def build_vocab(self) -> None:
        """Build vocabulary from cleaned texts."""
        print("Building vocabulary...")
        self.vocab.build(self.cleaned_texts, max_size=self.config.max_vocab_size)
        print(f"  Vocabulary size: {len(self.vocab):,}")
    
    def create_datasets(self) -> Tuple[IMDbDataset, IMDbDataset, IMDbDataset]:
        """Create train, val, test datasets with stratified split."""
        indices = list(range(len(self.cleaned_texts)))
        
        # 80/20 split
        idx_trainval, idx_test = train_test_split(
            indices, test_size=self.config.test_size, 
            stratify=self.raw_labels, random_state=self.config.random_state
        )
        
        # 70/10 split of remaining
        labels_trainval = [self.raw_labels[i] for i in idx_trainval]
        idx_train, idx_val = train_test_split(
            idx_trainval, test_size=self.config.val_size,
            stratify=labels_trainval, random_state=self.config.random_state
        )
        
        # Create datasets
        train_dataset = IMDbDataset(
            [self.cleaned_texts[i] for i in idx_train],
            [self.raw_labels[i] for i in idx_train],
            self.vocab, self.config.max_len, self.preprocessor
        )
        
        val_dataset = IMDbDataset(
            [self.cleaned_texts[i] for i in idx_val],
            [self.raw_labels[i] for i in idx_val],
            self.vocab, self.config.max_len, self.preprocessor
        )
        
        test_dataset = IMDbDataset(
            [self.cleaned_texts[i] for i in idx_test],
            [self.raw_labels[i] for i in idx_test],
            self.vocab, self.config.max_len, self.preprocessor
        )
        
        return train_dataset, val_dataset, test_dataset
    
    def create_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create DataLoaders for train, val, test sets."""
        print("Creating DataLoaders...")
        train_ds, val_ds, test_ds = self.create_datasets()
        
        train_loader = DataLoader(train_ds, batch_size=self.config.batch_size, 
                                 shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_ds, batch_size=self.config.batch_size, 
                               shuffle=False, collate_fn=collate_fn)
        test_loader = DataLoader(test_ds, batch_size=self.config.batch_size, 
                                shuffle=False, collate_fn=collate_fn)
        
        print(f"  Train batches: {len(train_loader):,}")
        print(f"  Val batches: {len(val_loader):,}")
        print(f"  Test batches: {len(test_loader):,}")
        
        return train_loader, val_loader, test_loader
    
    def prepare(self) -> Tuple[DataLoader, DataLoader, DataLoader, VocabularyBuilder]:
        """Complete pipeline: load → clean → build vocab → create loaders."""
        print("="*70)
        print("DATA PREPARATION PIPELINE")
        print("="*70)
        
        self.load_data()
        self.preprocess()
        self.build_vocab()
        train_loader, val_loader, test_loader = self.create_loaders()
        
        print("="*70)
        return train_loader, val_loader, test_loader, self.vocab


# ──────────────────────────────────────────────────────────────────────────────
# CONVENIENCE FUNCTION
# ──────────────────────────────────────────────────────────────────────────────

def prepare_data(config: DataConfig = None) -> Tuple[DataLoader, DataLoader, DataLoader, VocabularyBuilder]:
    """One-call data preparation pipeline."""
    if config is None:
        config = DataConfig()
    
    manager = DataManager(config)
    return manager.prepare()


if __name__ == '__main__':
    # Simple test of the data pipeline
    manager = DataManager()
    train_loader, val_loader, test_loader, vocab = manager.prepare()