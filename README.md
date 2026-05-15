# Assignment 3 — Transformer NMT (German → English)

Implementation of **"Attention Is All You Need"** (Vaswani et al., 2017) for Neural Machine Translation on the Multi30k dataset.

## Project Structure

```
assignment3/
├── requirements.txt
├── README.md
├── model.py      # Core Transformer architecture (Encoders, Decoders, Multi-Head Attention)
├── utils.py      # Label Smoothing, Noam Scheduler, Masking Utilities
├── dataset.py    # Multi30k dataset loading and spaCy tokenisation
└── train.py      # Training loops and Greedy Decoding inference
```

## Setup

```bash
pip install -r requirements.txt
python -m spacy download de_core_news_sm
python -m spacy download en_core_web_sm
```

## Training

```bash
python train.py
```

Key hyperparameters (all have defaults):

| Argument | Default | Description |
|---|---|---|
| `--epochs` | 30 | Number of training epochs |
| `--batch` | 128 | Batch size |
| `--d_model` | 512 | Model dimension |
| `--num_heads` | 8 | Attention heads |
| `--num_layers` | 6 | Encoder/Decoder layers |
| `--d_ff` | 2048 | FFN hidden dimension |
| `--warmup` | 4000 | Noam scheduler warmup steps |
| `--label_smooth` | 0.1 | Label smoothing ε |

After training, vocabularies are saved to `src_vocab.pt` and `tgt_vocab.pt`.  
Best checkpoint is saved to `transformer_best.pt`.

## Inference (Autograder format)

```python
model = Transformer().to(device)
model.eval()
english = model.infer("Ein Mann sitzt auf einer Bank.")
```

`Transformer.__init__` automatically:
1. Loads spaCy tokenisers
2. Loads vocabularies from `src_vocab.pt` / `tgt_vocab.pt`
3. Downloads weights from Google Drive via `gdown`
4. Loads the state dict

> **Before submission:** set `Transformer.GDRIVE_FILE_ID` to your Drive file ID and upload `src_vocab.pt` / `tgt_vocab.pt` alongside your code.

## Architecture Details

- **Base Transformer**: d_model=512, h=8, N=6, d_ff=2048
- **Positional Encoding**: sinusoidal (registered buffer, not trainable)
- **Layer Norm**: Post-LayerNorm (original paper style) — applied after residual addition
- **Masking**: padding mask for encoder; causal + padding mask for decoder
- **Optimiser**: Adam (β₁=0.9, β₂=0.98, ε=1e-9) with Noam LR schedule
- **Label Smoothing**: ε_ls = 0.1
- **Decoding**: greedy (token-by-token autoregressive)
