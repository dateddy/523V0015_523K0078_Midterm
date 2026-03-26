import re, torch
from collections import Counter
from torch.utils.data import Dataset, DataLoader, Subset
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
from sklearn.model_selection import train_test_split

dataset = load_dataset('imdb')

# Special token indices
PAD, UNK = 0, 1

# ── Hyperparameters to experiment with (spec says N ∈ {10000, 20000, 30000}) ──
MAX_VOCAB_SIZE = 20_000   # try 10_000 / 20_000 / 30_000
MAX_LEN        = 454      # from EDA p90; try 128 / 256 / 512
BATCH_SIZE     = 64
SEED           = 42

def clean_text(text: str) -> str:
    """Remove HTML tags, keep only letters/digits/spaces, lowercase."""
    text = re.sub(r'<[^>]+>', ' ', text)          # strip HTML tags
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)   # keep letters & digits only
    return text.lower().strip()

def build_vocab(texts: list[str], max_size: int = 20_000) -> dict[str, int]:
    """
    Build word→index vocabulary from a list of cleaned texts.
    Reserves index 0 for <pad> and index 1 for <unk>.
    Returns the top (max_size - 2) most frequent tokens.
    """
    counter = Counter(w for text in texts for w in text.split())
    vocab   = {'<pad>': PAD, '<unk>': UNK}
    for word, _ in counter.most_common(max_size - 2):
        vocab[word] = len(vocab)
    return vocab

class SentimentDataset(Dataset):
    """
    PyTorch Dataset for IMDb sentiment.
    Encodes text → integer IDs and truncates to max_len at construction time.
    Padding to a uniform length within each batch is handled by collate_fn.
    """

    def __init__(self, texts: list[str], labels: list[int],
                 vocab: dict[str, int], max_len: int = 256):
        self.data = []
        for text, label in zip(texts, labels):
            ids = [vocab.get(w, UNK) for w in text.split()]  # encode
            ids = ids[:max_len]                               # truncate if too long
            self.data.append((
                torch.tensor(ids,   dtype=torch.long),
                torch.tensor(label, dtype=torch.long)
            ))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


def collate_fn(batch):
    """
    Dynamic batch padding: pads all sequences in the batch to the length
    of the longest sequence in that batch (not the global max_len).
    This avoids wasting compute on excessive padding for short-review batches.
    """
    seqs, labels = zip(*batch)
    seqs_padded  = pad_sequence(seqs, batch_first=True, padding_value=PAD)
    return seqs_padded, torch.stack(labels)

######################################################################################
######################################################################################
######################################################################################

all_texts  = ([clean_text(ex['text']) for ex in dataset['train']] +
              [clean_text(ex['text']) for ex in dataset['test']])
all_labels = ([ex['label'] for ex in dataset['train']] +
              [ex['label'] for ex in dataset['test']])

vocab = build_vocab(all_texts, max_size=MAX_VOCAB_SIZE)

print(f'Total samples after cleaning: {len(all_texts):,}')
print(f'Sample: "{all_texts[0][:120]}..."')

sample_ids = [vocab.get(w, UNK) for w in all_texts[0].split()][:10]

print(f'Vocabulary size : {len(vocab):,}  (including <pad> and <unk>)')
print(f'Top-10 entries  : {list(vocab.items())[2:12]}')

indices = list(range(len(all_texts)))

print('First 10 token IDs of sample[0]:', sample_ids)
print('Decoded back                    :', [list(vocab.keys())[list(vocab.values()).index(i)] for i in sample_ids])

# Step 1: split off 20% test
idx_trainval, idx_test = train_test_split(
    indices, test_size=0.20, stratify=all_labels, random_state=SEED
)

# Step 2: split remaining 80% into 70% train + 10% val
# 10% of total = 10/80 = 12.5% of trainval
labels_trainval = [all_labels[i] for i in idx_trainval]
idx_train, idx_val = train_test_split(
    idx_trainval, test_size=0.125, stratify=labels_trainval, random_state=SEED
)

# Verify split sizes
total = len(all_texts)
print(f'Total  : {total:,}')
print(f'Train  : {len(idx_train):,}  ({len(idx_train)/total*100:.1f}%)')
print(f'Val    : {len(idx_val):,}   ({len(idx_val)/total*100:.1f}%)')
print(f'Test   : {len(idx_test):,}  ({len(idx_test)/total*100:.1f}%)')

# Verify class balance is preserved
for name, idx_split in [('Train', idx_train), ('Val', idx_val), ('Test', idx_test)]:
    labels_split = [all_labels[i] for i in idx_split]
    pos_pct = sum(labels_split) / len(labels_split) * 100
    print(f'  {name} positive ratio: {pos_pct:.1f}%')