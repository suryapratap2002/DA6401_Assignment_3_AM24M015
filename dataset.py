import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
import spacy

PAD_TOKEN = "<pad>"; PAD_IDX = 0
UNK_TOKEN = "<unk>"; UNK_IDX = 1
SOS_TOKEN = "<sos>"; SOS_IDX = 2
EOS_TOKEN = "<eos>"; EOS_IDX = 3
SPECIALS  = [PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN]


class Vocabulary:
    def __init__(self, min_freq: int = 2):
        self.min_freq = min_freq
        self.stoi: dict = {t: i for i, t in enumerate(SPECIALS)}
        self.itos: dict = {i: t for i, t in enumerate(SPECIALS)}

    def build(self, tokenised_sentences):
        counter = Counter()
        for tokens in tokenised_sentences:
            counter.update(tokens)
        for token, freq in sorted(counter.items(), key=lambda x: -x[1]):
            if freq >= self.min_freq and token not in self.stoi:
                idx = len(self.stoi)
                self.stoi[token] = idx
                self.itos[idx]   = token

    def __len__(self): return len(self.stoi)

    def encode(self, tokens):
        return [self.stoi.get(t, UNK_IDX) for t in tokens]

    def save(self, path):
        torch.save({"stoi": self.stoi, "itos": self.itos}, path)

    @classmethod
    def load(cls, path):
        data = torch.load(path, map_location="cpu", weights_only=True)
        obj = cls.__new__(cls)
        obj.stoi = data["stoi"]
        obj.itos = data["itos"]
        obj.min_freq = 2
        return obj


def get_tokenizers():
    try:    de = spacy.load("de_core_news_sm")
    except OSError:
        import subprocess, sys
        subprocess.run([sys.executable,"-m","spacy","download","de_core_news_sm"],check=True)
        de = spacy.load("de_core_news_sm")
    try:    en = spacy.load("en_core_web_sm")
    except OSError:
        import subprocess, sys
        subprocess.run([sys.executable,"-m","spacy","download","en_core_web_sm"],check=True)
        en = spacy.load("en_core_web_sm")
    return de, en


def tokenize_de(text, nlp): return [t.text.lower() for t in nlp(text.strip())]
def tokenize_en(text, nlp): return [t.text.lower() for t in nlp(text.strip())]


class Multi30kDataset(Dataset):
    def __init__(self, split, src_vocab, tgt_vocab, spacy_de, spacy_en,
                 max_len=128):
        from datasets import load_dataset
        raw  = load_dataset("bentrevett/multi30k")
        data = raw[split]
        self.src_data, self.tgt_data = [], []
        for item in data:
            s = tokenize_de(item["de"], spacy_de)[:max_len-2]
            t = tokenize_en(item["en"], spacy_en)[:max_len-2]
            self.src_data.append([SOS_IDX]+src_vocab.encode(s)+[EOS_IDX])
            self.tgt_data.append([SOS_IDX]+tgt_vocab.encode(t)+[EOS_IDX])

    def __len__(self): return len(self.src_data)
    def __getitem__(self, idx):
        return (torch.tensor(self.src_data[idx], dtype=torch.long),
                torch.tensor(self.tgt_data[idx], dtype=torch.long))


def collate_fn(batch):
    s, t = zip(*batch)
    return (pad_sequence(s, batch_first=True, padding_value=PAD_IDX),
            pad_sequence(t, batch_first=True, padding_value=PAD_IDX))


def build_vocab_and_loaders(batch_size=128, max_len=128, min_freq=2,
                             src_vocab_path="src_vocab.pt",
                             tgt_vocab_path="tgt_vocab.pt",
                             num_workers=0):
    from datasets import load_dataset
    spacy_de, spacy_en = get_tokenizers()

    if os.path.exists(src_vocab_path) and os.path.exists(tgt_vocab_path):
        print("[dataset] Loading cached vocabularies …")
        src_vocab = Vocabulary.load(src_vocab_path)
        tgt_vocab = Vocabulary.load(tgt_vocab_path)
    else:
        print("[dataset] Building vocabularies …")
        raw_train = load_dataset("bentrevett/multi30k")["train"]
        src_sents = [tokenize_de(x["de"], spacy_de) for x in raw_train]
        tgt_sents = [tokenize_en(x["en"], spacy_en) for x in raw_train]
        src_vocab = Vocabulary(min_freq); src_vocab.build(src_sents)
        tgt_vocab = Vocabulary(min_freq); tgt_vocab.build(tgt_sents)
        src_vocab.save(src_vocab_path); tgt_vocab.save(tgt_vocab_path)
        print(f"[dataset] src={len(src_vocab)} tgt={len(tgt_vocab)}")

    kw = dict(batch_size=batch_size, collate_fn=collate_fn,
              num_workers=num_workers, pin_memory=False)
    return (
        DataLoader(Multi30kDataset("train",      src_vocab, tgt_vocab,
                                   spacy_de, spacy_en, max_len), shuffle=True,  **kw),
        DataLoader(Multi30kDataset("validation", src_vocab, tgt_vocab,
                                   spacy_de, spacy_en, max_len), shuffle=False, **kw),
        DataLoader(Multi30kDataset("test",       src_vocab, tgt_vocab,
                                   spacy_de, spacy_en, max_len), shuffle=False, **kw),
        src_vocab, tgt_vocab,
    )
