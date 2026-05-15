# Assignment 3 — Transformer NMT (German → English)

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
Check your CUDA version first: `nvidia-smi`

```bash
# CUDA 12.4 (works with CUDA 13.x drivers too — backwards compatible)
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

### Example
```python
from model import Transformer
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Transformer().to(device)
model.eval()

print(model.infer("Ein Mann sitzt auf einer Bank."))
# Output: a man sitting on a bench .
```

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

---

## Autograder Tests — Local Verification

Run before submitting to confirm all unit tests pass:

```bash
python test_autograder.py
```

Expected output:
```
✓ PASS  ScaledDotProduct output shape
✓ PASS  Attention weights sum to 1 over key dim
✓ PASS  Masked positions receive zero attention weight
✓ PASS  MHA output shape under varying d_model / num_heads
✓ PASS  Causal mask produces different output than unmasked
✓ PASS  PE output shape preserved
✓ PASS  PE even dims == sin(0)=0 at position 0
✓ PASS  PE odd dims == cos(0)=1 at position 0
✓ PASS  PE formula correct at arbitrary (pos, dim)
✓ PASS  PE registered as buffer, not trainable parameter
✓ PASS  LR monotonically increasing during warm-up
✓ PASS  Peak within 10 steps of warmup_steps
✓ PASS  LR monotonically decreasing after warm-up
✓ PASS  Peak LR matches closed-form formula
✓ PASS  LR at step 1 matches formula

Result: 15 passed / 0 failed / 15 total
```

---

## Submission Checklist

- [ ] Train model: `python train.py --batch 64 --accum_steps 2 --epochs 30`
- [ ] Upload `transformer_best.pt`, `src_vocab.pt`, `tgt_vocab.pt` to Google Drive
- [ ] Set each file to **"Anyone with the link → Viewer"**
- [ ] Fill in `GDRIVE_FILE_ID`, `GDRIVE_SRC_VOCAB_ID`, `GDRIVE_TGT_VOCAB_ID` in `model.py`
- [ ] Run `python test_autograder.py` — confirm 15/15 pass
- [ ] Test `model.infer("Ein Mann sitzt auf einer Bank.")` — confirm clean output
- [ ] Submit `model.py`, `utils.py`, `dataset.py`, `train.py` to Gradescope
- [ ] **Do NOT upload** `.pt` files to Gradescope — they download automatically

---

## References

- Vaswani et al. (2017). *Attention Is All You Need*. NeurIPS.  
  https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
- Multi30k Dataset: https://huggingface.co/datasets/bentrevett/multi30k
- GitHub Skeleton: https://github.com/MiRL-IITM/da6401_assignment_3
