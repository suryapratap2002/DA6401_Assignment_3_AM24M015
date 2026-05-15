"""
dataset.py — Multi30k dataset loading and spaCy tokenisation.

Builds source (German) and target (English) vocabularies and saves them
to disk as  src_vocab.pt  and  tgt_vocab.pt  so the Transformer.__init__
can reload them at inference time without needing the datasets library.
"""

import os
import re
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import spacy
from collections import Counter


# ──────────────────────────────────────────────
# Special tokens
# ──────────────────────────────────────────────
PAD_TOKEN = "<pad>"   # 0
UNK_TOKEN = "<unk>"   # 1
SOS_TOKEN = "<sos>"   # 2
EOS_TOKEN = "<eos>"   # 3

SPECIALS = [PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN]
PAD_IDX  = 0
UNK_IDX  = 1
SOS_IDX  = 2
EOS_IDX  = 3


# ──────────────────────────────────────────────
# Vocabulary
# ──────────────────────────────────────────────

class Vocabulary:
    """Simple word-level vocabulary built from a list of tokenised sentences."""

    def __init__(self, min_freq: int = 2):
        self.min_freq = min_freq
        self.stoi: dict = {}   # string → index
        self.itos: dict = {}   # index  → string
        for idx, tok in enumerate(SPECIALS):
            self.stoi[tok] = idx
            self.itos[idx] = tok

    def build(self, tokenised_sentences: list[list[str]]):
        counter: Counter = Counter()
        for tokens in tokenised_sentences:
            counter.update(tokens)
        for token, freq in sorted(counter.items(), key=lambda x: -x[1]):
            if freq >= self.min_freq and token not in self.stoi:
                idx = len(self.stoi)
                self.stoi[token] = idx
                self.itos[idx]   = token

    def __len__(self):
        return len(self.stoi)

    def encode(self, tokens: list[str]) -> list[int]:
        return [self.stoi.get(t, UNK_IDX) for t in tokens]

    def decode(self, ids: list[int]) -> list[str]:
        return [self.itos.get(i, UNK_TOKEN) for i in ids]

    def save(self, path: str):
        torch.save({"stoi": self.stoi, "itos": self.itos}, path)

    @classmethod
    def load(cls, path: str) -> "Vocabulary":
        data = torch.load(path, map_location="cpu")
        obj = cls.__new__(cls)
        obj.stoi = data["stoi"]
        obj.itos = data["itos"]
        obj.min_freq = 2
        return obj


# ──────────────────────────────────────────────
# Tokenisation helpers
# ──────────────────────────────────────────────

def get_tokenizers():
    """Load spaCy German and English models."""
    try:
        spacy_de = spacy.load("de_core_news_sm")
    except OSError:
        import subprocess, sys
        subprocess.run([sys.executable, "-m", "spacy", "download", "de_core_news_sm"], check=True)
        spacy_de = spacy.load("de_core_news_sm")

    try:
        spacy_en = spacy.load("en_core_web_sm")
    except OSError:
        import subprocess, sys
        subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
        spacy_en = spacy.load("en_core_web_sm")

    return spacy_de, spacy_en


def tokenize_de(text: str, nlp) -> list[str]:
    return [tok.text.lower() for tok in nlp(text.strip())]


def tokenize_en(text: str, nlp) -> list[str]:
    return [tok.text.lower() for tok in nlp(text.strip())]


# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────

class Multi30kDataset(Dataset):
    """
    Wraps the HuggingFace Multi30k dataset.
    Each item is a (src_tensor, tgt_tensor) pair with <sos>/<eos> tokens.
    """

    def __init__(
        self,
        split: str,               # "train" | "validation" | "test"
        src_vocab: Vocabulary,
        tgt_vocab: Vocabulary,
        spacy_de,
        spacy_en,
        max_len: int = 128,
    ):
        from datasets import load_dataset
        raw  = load_dataset("bentrevett/multi30k")
        data = raw[split]

        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.spacy_de  = spacy_de
        self.spacy_en  = spacy_en
        self.max_len   = max_len

        self.src_data: list[list[int]] = []
        self.tgt_data: list[list[int]] = []

        for item in data:
            src_tokens = tokenize_de(item["de"], spacy_de)
            tgt_tokens = tokenize_en(item["en"], spacy_en)

            # Truncate to max_len - 2  (leave room for sos/eos)
            src_tokens = src_tokens[: max_len - 2]
            tgt_tokens = tgt_tokens[: max_len - 2]

            src_ids = [SOS_IDX] + src_vocab.encode(src_tokens) + [EOS_IDX]
            tgt_ids = [SOS_IDX] + tgt_vocab.encode(tgt_tokens) + [EOS_IDX]

            self.src_data.append(src_ids)
            self.tgt_data.append(tgt_ids)

    def __len__(self):
        return len(self.src_data)

    def __getitem__(self, idx):
        src = torch.tensor(self.src_data[idx], dtype=torch.long)
        tgt = torch.tensor(self.tgt_data[idx], dtype=torch.long)
        return src, tgt


# ──────────────────────────────────────────────
# Collate + DataLoader factory
# ──────────────────────────────────────────────

def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=PAD_IDX)
    tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=PAD_IDX)
    return src_padded, tgt_padded


def build_vocab_and_loaders(
    batch_size: int = 128,
    max_len: int = 128,
    min_freq: int = 2,
    src_vocab_path: str = "src_vocab.pt",
    tgt_vocab_path: str = "tgt_vocab.pt",
    num_workers: int = 0,          # 0 = safe on Windows; increase on Linux/Mac
):
    """
    Build (or reload) vocabularies and return DataLoaders for
    train / validation / test splits.

    Returns:
        train_loader, val_loader, test_loader, src_vocab, tgt_vocab
    """
    from datasets import load_dataset

    spacy_de, spacy_en = get_tokenizers()

    # ── Build vocabulary from training split ──────────────────────────────
    if os.path.exists(src_vocab_path) and os.path.exists(tgt_vocab_path):
        print("[dataset] Loading cached vocabularies …")
        src_vocab = Vocabulary.load(src_vocab_path)
        tgt_vocab = Vocabulary.load(tgt_vocab_path)
    else:
        print("[dataset] Building vocabularies from training data …")
        raw_train = load_dataset("bentrevett/multi30k")["train"]

        src_sentences, tgt_sentences = [], []
        for item in raw_train:
            src_sentences.append(tokenize_de(item["de"], spacy_de))
            tgt_sentences.append(tokenize_en(item["en"], spacy_en))

        src_vocab = Vocabulary(min_freq=min_freq)
        src_vocab.build(src_sentences)
        tgt_vocab = Vocabulary(min_freq=min_freq)
        tgt_vocab.build(tgt_sentences)

        src_vocab.save(src_vocab_path)
        tgt_vocab.save(tgt_vocab_path)
        print(f"[dataset] src vocab: {len(src_vocab)} | tgt vocab: {len(tgt_vocab)}")

    # ── Build datasets ────────────────────────────────────────────────────
    train_ds = Multi30kDataset("train",      src_vocab, tgt_vocab, spacy_de, spacy_en, max_len)
    val_ds   = Multi30kDataset("validation", src_vocab, tgt_vocab, spacy_de, spacy_en, max_len)
    test_ds  = Multi30kDataset("test",       src_vocab, tgt_vocab, spacy_de, spacy_en, max_len)

    loader_kwargs = dict(
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=(num_workers > 0),   # pin_memory only useful with background workers
        persistent_workers=(num_workers > 0),
    )
    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,  shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader, src_vocab, tgt_vocab


# ──────────────────────────────────────────────
# Quick sanity-check
# ──────────────────────────────────────────────
if __name__ == "__main__":
    train_loader, val_loader, test_loader, sv, tv = build_vocab_and_loaders(batch_size=32)
    src, tgt = next(iter(train_loader))
    print("src shape:", src.shape, "  tgt shape:", tgt.shape)
    print("src vocab size:", len(sv), "  tgt vocab size:", len(tv))