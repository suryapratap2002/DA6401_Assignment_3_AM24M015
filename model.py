"""
model.py — Core Transformer architecture
Implements: Scaled Dot-Product Attention, Multi-Head Attention,
Positional Encoding, Encoder/Decoder stacks, and the full Transformer.
"""

import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# ──────────────────────────────────────────────
# 1.  Scaled Dot-Product Attention
# ──────────────────────────────────────────────

class ScaledDotProductAttention(nn.Module):
    """
    Attention(Q, K, V) = softmax( QK^T / sqrt(d_k) ) V
    """

    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ):
        """
        Args:
            query : (batch, heads, seq_q, d_k)
            key   : (batch, heads, seq_k, d_k)
            value : (batch, heads, seq_k, d_v)
            mask  : broadcastable bool tensor — True where positions should
                    be *ignored* (masked out).
        Returns:
            output  : (batch, heads, seq_q, d_v)
            weights : (batch, heads, seq_q, seq_k)
        """
        d_k = query.size(-1)
        # scores : (batch, heads, seq_q, seq_k)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))

        weights = F.softmax(scores, dim=-1)

        # Replace NaN rows (all -inf → softmax NaN) with 0  [padding rows]
        weights = torch.nan_to_num(weights, nan=0.0)

        weights = self.dropout(weights)
        output = torch.matmul(weights, value)
        return output, weights


# ──────────────────────────────────────────────
# 2.  Multi-Head Attention
# ──────────────────────────────────────────────

class MultiHeadAttention(nn.Module):
    """
    Projects Q, K, V into h heads, runs scaled dot-product attention in
    parallel, then projects the concatenated result back to d_model.
    """

    def __init__(self, d_model: int = 512, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.attention = ScaledDotProductAttention(dropout=dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, seq, d_model) → (batch, heads, seq, d_k)"""
        batch, seq, _ = x.size()
        x = x.view(batch, seq, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, heads, seq, d_k) → (batch, seq, d_model)"""
        batch, _, seq, _ = x.size()
        x = x.transpose(1, 2).contiguous()
        return x.view(batch, seq, self.d_model)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ):
        """
        Args:
            query, key, value : (batch, seq, d_model)
            mask              : (batch, 1, seq_q, seq_k) or compatible — True = mask
        Returns:
            output  : (batch, seq_q, d_model)
            weights : (batch, heads, seq_q, seq_k)
        """
        Q = self._split_heads(self.W_q(query))
        K = self._split_heads(self.W_k(key))
        V = self._split_heads(self.W_v(value))

        attn_out, weights = self.attention(Q, K, V, mask=mask)

        output = self.W_o(self._merge_heads(attn_out))
        return output, weights


# ──────────────────────────────────────────────
# 3.  Positional Encoding
# ──────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding as in the original paper.
    Registered as a buffer (not a trainable parameter).
    """

    def __init__(self, d_model: int = 512, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Build the encoding matrix once
        pe = torch.zeros(max_len, d_model)            # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len,1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd  indices
        pe = pe.unsqueeze(0)                           # (1, max_len, d_model)

        # Register as buffer so it moves with .to(device) but isn't a parameter
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (batch, seq, d_model)
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# ──────────────────────────────────────────────
# 4.  Position-wise Feed-Forward Network
# ──────────────────────────────────────────────

class PositionwiseFeedForward(nn.Module):
    """FFN(x) = max(0, x W_1 + b_1) W_2 + b_2"""

    def __init__(self, d_model: int = 512, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


# ──────────────────────────────────────────────
# 5.  Encoder Layer
# ──────────────────────────────────────────────

class EncoderLayer(nn.Module):
    """
    One encoder layer:
        sublayer-1 : Multi-Head Self-Attention  + Add & Norm
        sublayer-2 : Position-wise FFN          + Add & Norm
    Using Post-LayerNorm (as in the original paper).
    """

    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor | None = None):
        # Self-attention sub-layer
        attn_out, _ = self.self_attn(x, x, x, mask=src_mask)
        x = self.norm1(x + self.dropout(attn_out))
        # FFN sub-layer
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x


# ──────────────────────────────────────────────
# 6.  Decoder Layer
# ──────────────────────────────────────────────

class DecoderLayer(nn.Module):
    """
    One decoder layer:
        sublayer-1 : Masked Multi-Head Self-Attention + Add & Norm
        sublayer-2 : Multi-Head Cross-Attention       + Add & Norm
        sublayer-3 : Position-wise FFN                + Add & Norm
    """

    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        src_mask: torch.Tensor | None = None,
    ):
        # Masked self-attention
        sa_out, _ = self.self_attn(x, x, x, mask=tgt_mask)
        x = self.norm1(x + self.dropout(sa_out))
        # Cross-attention
        ca_out, _ = self.cross_attn(x, memory, memory, mask=src_mask)
        x = self.norm2(x + self.dropout(ca_out))
        # FFN
        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_out))
        return x


# ──────────────────────────────────────────────
# 7.  Encoder Stack
# ──────────────────────────────────────────────

class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 2048,
        max_len: int = 5000,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.scale = math.sqrt(d_model)

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor | None = None):
        x = self.embedding(src) * self.scale
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x, src_mask)
        return x


# ──────────────────────────────────────────────
# 8.  Decoder Stack
# ──────────────────────────────────────────────

class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 2048,
        max_len: int = 5000,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.scale = math.sqrt(d_model)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        src_mask: torch.Tensor | None = None,
    ):
        x = self.embedding(tgt) * self.scale
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x, memory, tgt_mask, src_mask)
        return x


# ──────────────────────────────────────────────
# 9.  Full Transformer
# ──────────────────────────────────────────────

class Transformer(nn.Module):
    """
    Full Transformer for German → English NMT.

    All hyper-parameters have defaults matching the 'base' configuration
    from the original paper.  Vocabulary, tokenizers, and model weights
    are all loaded inside __init__ so the autograder can do:

        model = Transformer().to(device)
        model.eval()
        english = model.infer(german_sentence)
    """

    # ── class-level paths / IDs (edit before submission) ──────────────────
    # Google Drive file-id of your saved checkpoint (weights only, state_dict)
    GDRIVE_FILE_ID:       str = "1KFXlfeR8aL8Br3nmVZjg8oay9_5Isf_i"
    GDRIVE_SRC_VOCAB_ID:  str = "12BOXH_dwIeTWau1BunljHHpIdzUhLyam"
    GDRIVE_TGT_VOCAB_ID:  str = "18WTsnTDU4US-59_r__HpV2mYm-ivp_1J"

    WEIGHTS_FILENAME:  str = "transformer_best.pt"
    SRC_VOCAB_PATH:    str = "src_vocab.pt"
    TGT_VOCAB_PATH:    str = "tgt_vocab.pt"
    # ──────────────────────────────────────────────────────────────────────

    def __init__(
        self,
        src_vocab_size: int = 0,       # auto-loaded from file when 0
        tgt_vocab_size: int = 0,
        d_model: int = 512,            # ← must match what you trained with
        num_heads: int = 8,
        num_encoder_layers: int = 6,   # ← must match what you trained with
        num_decoder_layers: int = 6,
        d_ff: int = 2048,              # ← must match what you trained with
        max_len: int = 128,
        dropout: float = 0.1,
        load_weights: bool = True,
    ):
        super().__init__()

        import spacy   # local import — keeps module-level clean for unit tests
        import gdown   # noqa

        # ── Load spaCy tokenisers ─────────────────────────────────────────
        try:
            self.src_tokenizer = spacy.load("de_core_news_sm")
        except OSError:
            import subprocess, sys
            subprocess.run(
                [sys.executable, "-m", "spacy", "download", "de_core_news_sm"],
                check=True,
            )
            self.src_tokenizer = spacy.load("de_core_news_sm")

        try:
            self.tgt_tokenizer = spacy.load("en_core_web_sm")
        except OSError:
            import subprocess, sys
            subprocess.run(
                [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
                check=True,
            )
            self.tgt_tokenizer = spacy.load("en_core_web_sm")

        # ── Download vocab + weights from Drive if not present ───────────────
        if load_weights:
            if not os.path.exists(self.SRC_VOCAB_PATH):
                print("[Transformer] Downloading src_vocab.pt …")
                gdown.download(
                    f"https://drive.google.com/uc?id={self.GDRIVE_SRC_VOCAB_ID}",
                    self.SRC_VOCAB_PATH, quiet=False,
                )
            if not os.path.exists(self.TGT_VOCAB_PATH):
                print("[Transformer] Downloading tgt_vocab.pt …")
                gdown.download(
                    f"https://drive.google.com/uc?id={self.GDRIVE_TGT_VOCAB_ID}",
                    self.TGT_VOCAB_PATH, quiet=False,
                )

        # ── Load vocabularies ─────────────────────────────────────────────
        src_vocab_data = torch.load(self.SRC_VOCAB_PATH, map_location="cpu",
                                    weights_only=True)
        tgt_vocab_data = torch.load(self.TGT_VOCAB_PATH, map_location="cpu",
                                    weights_only=True)

        self.src_vocab: dict = src_vocab_data["stoi"]  # token → idx
        self.tgt_vocab: dict = tgt_vocab_data["stoi"]
        self.tgt_itos:  dict = tgt_vocab_data["itos"]  # idx → token

        src_vocab_size = len(self.src_vocab)
        tgt_vocab_size = len(self.tgt_vocab)

        # Special token indices
        self.pad_idx   = self.src_vocab.get("<pad>", 0)
        self.sos_idx   = self.tgt_vocab.get("<sos>", 2)
        self.eos_idx   = self.tgt_vocab.get("<eos>", 3)
        self.unk_idx   = self.src_vocab.get("<unk>", 1)

        # ── Build model sub-modules ───────────────────────────────────────
        self.encoder = Encoder(src_vocab_size, d_model, num_heads,
                               num_encoder_layers, d_ff, max_len, dropout)
        self.decoder = Decoder(tgt_vocab_size, d_model, num_heads,
                               num_decoder_layers, d_ff, max_len, dropout)
        self.projection = nn.Linear(d_model, tgt_vocab_size)

        self._d_model = d_model
        self._max_len = max_len

        self._init_parameters()

        # ── Load weights ──────────────────────────────────────────────────
        if load_weights:
            self._load_weights()

    # ── Parameter initialisation ──────────────────────────────────────────

    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # ── Weight loading ────────────────────────────────────────────────────

    def _load_weights(self):
        """Download weights + vocab files from Google Drive if needed, then load."""
        import gdown

        # ── download vocab files if missing ──────────────────────────────
        if not os.path.exists(self.SRC_VOCAB_PATH):
            print("[Transformer] Downloading src_vocab.pt from Google Drive …")
            gdown.download(
                f"https://drive.google.com/uc?id={self.GDRIVE_SRC_VOCAB_ID}",
                self.SRC_VOCAB_PATH, quiet=False,
            )

        if not os.path.exists(self.TGT_VOCAB_PATH):
            print("[Transformer] Downloading tgt_vocab.pt from Google Drive …")
            gdown.download(
                f"https://drive.google.com/uc?id={self.GDRIVE_TGT_VOCAB_ID}",
                self.TGT_VOCAB_PATH, quiet=False,
            )

        # ── download model weights if missing ────────────────────────────
        if not os.path.exists(self.WEIGHTS_FILENAME):
            print("[Transformer] Downloading weights from Google Drive …")
            gdown.download(
                f"https://drive.google.com/uc?id={self.GDRIVE_FILE_ID}",
                self.WEIGHTS_FILENAME, quiet=False,
            )

        device = next(iter(self.parameters()), torch.zeros(1)).device
        state  = torch.load(self.WEIGHTS_FILENAME, map_location=device,
                            weights_only=True)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        self.load_state_dict(state)
        print(f"[Transformer] Weights loaded from '{self.WEIGHTS_FILENAME}'.")

    # ── Mask utilities ────────────────────────────────────────────────────

    def _make_src_mask(self, src: torch.Tensor) -> torch.Tensor:
        """
        Padding mask for encoder.
        Returns: (batch, 1, 1, src_len)  True where token is <pad>
        """
        return (src == self.pad_idx).unsqueeze(1).unsqueeze(2)

    def _make_tgt_mask(self, tgt: torch.Tensor) -> torch.Tensor:
        """
        Combined padding + causal (look-ahead) mask for decoder.
        Returns: (batch, 1, tgt_len, tgt_len)  True where attention is forbidden
        """
        tgt_len = tgt.size(1)
        # Causal mask: upper-triangular (above diagonal) is True
        causal = torch.triu(
            torch.ones(tgt_len, tgt_len, device=tgt.device, dtype=torch.bool),
            diagonal=1,
        )                                                              # (T, T)
        # Padding mask
        pad_mask = (tgt == self.pad_idx).unsqueeze(1).unsqueeze(2)    # (B, 1, 1, T)
        return causal.unsqueeze(0).unsqueeze(0) | pad_mask            # (B, 1, T, T)

    # ── Forward pass ──────────────────────────────────────────────────────

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        tgt_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        src : (batch, src_len)   integer token IDs
        tgt : (batch, tgt_len)   integer token IDs  (teacher-forced)
        Returns logits : (batch, tgt_len, tgt_vocab_size)
        """
        if src_mask is None:
            src_mask = self._make_src_mask(src)
        if tgt_mask is None:
            tgt_mask = self._make_tgt_mask(tgt)

        memory = self.encoder(src, src_mask)
        dec_out = self.decoder(tgt, memory, tgt_mask, src_mask)
        return self.projection(dec_out)

    # ── Tokenisation helpers ──────────────────────────────────────────────

    def _tokenize_src(self, sentence: str) -> list[str]:
        return [tok.text.lower() for tok in self.src_tokenizer(sentence)]

    def _tokenize_tgt(self, sentence: str) -> list[str]:
        return [tok.text.lower() for tok in self.tgt_tokenizer(sentence)]

    def _encode_src(self, tokens: list[str]) -> torch.Tensor:
        ids = [self.src_vocab.get(t, self.unk_idx) for t in tokens]
        return torch.tensor(ids, dtype=torch.long)

    # ── Greedy decoding / infer ───────────────────────────────────────────

    def infer(self, german_sentence: str, max_output_len: int = 100) -> str:
        """
        End-to-end inference:
          1. Tokenise the German sentence with spaCy
          2. Convert to token IDs using the source vocabulary
          3. Run the encoder
          4. Auto-regressively decode until <eos> or max_output_len
          5. Convert predicted IDs back to an English string and return it
        """
        self.eval()
        device = next(self.parameters()).device

        # --- Source encoding ---
        src_tokens = self._tokenize_src(german_sentence)
        src_ids = self._encode_src(src_tokens).unsqueeze(0).to(device)  # (1, src_len)
        src_mask = self._make_src_mask(src_ids)

        with torch.no_grad():
            memory = self.encoder(src_ids, src_mask)

            # Start with <sos>
            tgt_ids = torch.tensor([[self.sos_idx]], dtype=torch.long, device=device)

            for _ in range(max_output_len):
                tgt_mask = self._make_tgt_mask(tgt_ids)
                dec_out = self.decoder(tgt_ids, memory, tgt_mask, src_mask)
                logits = self.projection(dec_out)              # (1, t, vocab)
                next_id = logits[:, -1, :].argmax(dim=-1)      # (1,)

                if next_id.item() == self.eos_idx:
                    break

                tgt_ids = torch.cat([tgt_ids, next_id.unsqueeze(0)], dim=1)

        # Decode to string (skip <sos>)
        predicted_ids = tgt_ids[0, 1:].tolist()
        tokens = [self.tgt_itos.get(i, "<unk>") for i in predicted_ids]
        return " ".join(tokens)