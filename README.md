# Assignment 3 — Transformer NMT (German → English)

Github link:  https://github.com/suryapratap2002/DA6401_Assignment_3_AM24M015

Wandb Report Link:  https://wandb.ai/spsinghiitian2020-iitmaana/da6401_assignment_3/reports/Assignment-3---VmlldzoxNjk1MjQwMQ?accessToken=lgtjbtu5alk3b69iy4qicokqy05j25zg0tupgwej4careimauvc54t5ce3az5kcb


Implementation of **"Attention Is All You Need"** (Vaswani et al., 2017) for Neural Machine Translation on the Multi30k dataset, built from scratch using PyTorch.

---

## Project Structure

```
assignment3/
├── requirements.txt     # Dependencies
├── README.md            # This file
├── model.py             # Core Transformer architecture (Encoders, Decoders, Multi-Head Attention)
├── utils.py             # Label Smoothing, Noam Scheduler, Masking Utilities
├── dataset.py           # Multi30k dataset loading and spaCy tokenisation
└── train.py             # Training loop and Greedy Decoding inference
```

---

## Setup

### 1. Install PyTorch with CUDA

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### 2. Install remaining dependencies
```bash
pip install -r requirements.txt
```

### 3. Download spaCy language models
```bash
python -m spacy download de_core_news_sm
python -m spacy download en_core_web_sm
```

---

## Training

```bash
python train.py --batch 64 --accum_steps 2 --epochs 30
```

### Key arguments

| Argument | Default | Description |
|---|---|---|
| `--epochs` | 30 | Number of training epochs |
| `--batch` | 64 | Per-step batch size |
| `--accum_steps` | 2 | Gradient accumulation (effective batch = batch × accum_steps) |
| `--d_model` | 512 | Model dimension (base Transformer) |
| `--num_heads` | 8 | Number of attention heads |
| `--num_layers` | 6 | Encoder/Decoder layers |
| `--d_ff` | 2048 | Feed-forward hidden dimension |
| `--dropout` | 0.1 | Dropout rate |
| `--warmup` | 4000 | Noam scheduler warmup steps |
| `--label_smooth` | 0.1 | Label smoothing coefficient ε |
| `--weight_decay` | 1e-4 | AdamW weight decay |
| `--early_stop` | 8 | Early stopping patience (epochs) |
| `--resume` | False | Resume from existing checkpoint |
| `--no_amp` | — | Disable mixed precision (fp16) |

After training, three files are saved:
- `transformer_best.pt` — best model weights (by val loss)
- `src_vocab.pt` — German vocabulary
- `tgt_vocab.pt` — English vocabulary

---

## Architecture

Follows the **base Transformer** configuration from the original paper exactly:

| Component | Details |
|---|---|
| Model dimension | d_model = 512 |
| Attention heads | h = 8, d_k = d_v = 64 |
| Encoder layers | N = 6 |
| Decoder layers | N = 6 |
| FFN dimension | d_ff = 2048 |
| Positional Encoding | Sinusoidal (registered buffer, not trainable) |
| Layer Norm | Post-LayerNorm (after residual addition, as in paper) |
| Attention | Scaled Dot-Product + Multi-Head |
| Masking | Padding mask (encoder) + Causal + Padding mask (decoder) |
| Optimiser | AdamW (β₁=0.9, β₂=0.98, ε=1e-9) |
| LR Schedule | Noam (warmup_steps=4000) |
| Label Smoothing | ε_ls = 0.1 |
| Decoding | Greedy autoregressive |

### Layer Normalisation choice
Post-LayerNorm is used (residual → LayerNorm), matching the original "Attention Is All You Need" paper. Pre-LayerNorm is known to train more stably but was not used here to stay faithful to the paper's base configuration.

---

## Inference (Autograder format)

The autograder evaluates using:

```python
model = Transformer().to(device)
model.eval()
english_sentence = model.infer(german_sentence)
```

`Transformer.__init__` handles everything automatically:
1. Loads spaCy tokenisers (`de_core_news_sm`, `en_core_web_sm`)
2. Downloads `src_vocab.pt` and `tgt_vocab.pt` from Google Drive via `gdown`
3. Downloads `transformer_best.pt` from Google Drive via `gdown`
4. Loads weights into the model

---

## Dataset

**Multi30k** — a multilingual dataset for NMT in resource-constrained environments.

| Split | Sentences |
|---|---|
| Train | 29,000 |
| Validation | 1,014 |
| Test | 1,000 |

Source: `bentrevett/multi30k` on HuggingFace Datasets.  
Tokenisation: spaCy (`de_core_news_sm` for German, `en_core_web_sm` for English), lowercased.  
Vocabulary: min_freq = 2 (words appearing < 2 times mapped to `<unk>`).

