import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    """Attention(Q, K, V) = softmax( QK^T / sqrt(d_k) ) V"""

    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        d_k    = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0)
        weights = self.dropout(weights)
        return torch.matmul(weights, value), weights

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int = 512, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model   = d_model
        self.num_heads = num_heads
        self.d_k       = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.attention = ScaledDotProductAttention(dropout=dropout)

    def _split_heads(self, x):
        B, S, _ = x.size()
        return x.view(B, S, self.num_heads, self.d_k).transpose(1, 2)

    def _merge_heads(self, x):
        B, _, S, _ = x.size()
        return x.transpose(1, 2).contiguous().view(B, S, self.d_model)

    def forward(self, query, key, value, mask=None):
        Q = self._split_heads(self.W_q(query))
        K = self._split_heads(self.W_k(key))
        V = self._split_heads(self.W_v(value))
        attn_out, weights = self.attention(Q, K, V, mask=mask)
        self.attn_weights = weights          # stored for inspection
        return self.W_o(self._merge_heads(attn_out))   # tensor only

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int = 512, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout  = nn.Dropout(dropout)
        pe            = torch.zeros(max_len, d_model)
        position      = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term      = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2]   = torch.sin(position * div_term)
        pe[:, 1::2]   = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))   

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1), :])

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int = 512, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class EncoderLayer(nn.Module):
    def __init__(self, d_model=512, num_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn       = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1     = nn.LayerNorm(d_model)
        self.norm2     = nn.LayerNorm(d_model)
        self.dropout   = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, mask=src_mask)))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model=512, num_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.self_attn  = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn        = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1      = nn.LayerNorm(d_model)
        self.norm2      = nn.LayerNorm(d_model)
        self.norm3      = nn.LayerNorm(d_model)
        self.dropout    = nn.Dropout(dropout)

    def forward(self, x, memory, tgt_mask=None, src_mask=None):
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, mask=tgt_mask)))
        x = self.norm2(x + self.dropout(self.cross_attn(x, memory, memory, mask=src_mask)))
        x = self.norm3(x + self.dropout(self.ffn(x)))
        return x

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model=512, num_heads=8,
                 num_layers=6, d_ff=2048, max_len=5000, dropout=0.1):
        super().__init__()
        self.embedding    = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        self.layers       = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.scale = math.sqrt(d_model)

    def forward(self, src, src_mask=None):
        x = self.pos_encoding(self.embedding(src) * self.scale)
        for layer in self.layers:
            x = layer(x, src_mask)
        return x

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model=512, num_heads=8,
                 num_layers=6, d_ff=2048, max_len=5000, dropout=0.1):
        super().__init__()
        self.embedding    = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        self.layers       = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.scale = math.sqrt(d_model)

    def forward(self, tgt, memory, tgt_mask=None, src_mask=None):
        x = self.pos_encoding(self.embedding(tgt) * self.scale)
        for layer in self.layers:
            x = layer(x, memory, tgt_mask, src_mask)
        return x


class Transformer(nn.Module):

    GDRIVE_FILE_ID:      str = "1KFXlfeR8aL8Br3nmVZjg8oay9_5Isf_i"
    GDRIVE_SRC_VOCAB_ID: str = "12BOXH_dwIeTWau1BunljHHpIdzUhLyam"
    GDRIVE_TGT_VOCAB_ID: str = "18WTsnTDU4US-59_r__HpV2mYm-ivp_1J"

    WEIGHTS_FILENAME: str = "transformer_best.pt"
    SRC_VOCAB_PATH:   str = "src_vocab.pt"
    TGT_VOCAB_PATH:   str = "tgt_vocab.pt"

    def __init__(
        self,
        src_vocab_size: int = 0,
        tgt_vocab_size: int = 0,
        d_model: int = 512,
        num_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        d_ff: int = 2048,
        max_len: int = 128,
        dropout: float = 0.1,
        load_weights: bool = True,
    ):
        super().__init__()

        import spacy   
        import gdown


        try:
            self.src_tokenizer = spacy.load("de_core_news_sm")
        except OSError:
            import subprocess, sys
            subprocess.run([sys.executable, "-m", "spacy", "download",
                            "de_core_news_sm"], check=True)
            self.src_tokenizer = spacy.load("de_core_news_sm")

        try:
            self.tgt_tokenizer = spacy.load("en_core_web_sm")
        except OSError:
            import subprocess, sys
            subprocess.run([sys.executable, "-m", "spacy", "download",
                            "en_core_web_sm"], check=True)
            self.tgt_tokenizer = spacy.load("en_core_web_sm")

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

        src_vocab_data = torch.load(self.SRC_VOCAB_PATH, map_location="cpu",
                                    weights_only=True)
        tgt_vocab_data = torch.load(self.TGT_VOCAB_PATH, map_location="cpu",
                                    weights_only=True)

        self.src_vocab: dict = src_vocab_data["stoi"]
        self.tgt_vocab: dict = tgt_vocab_data["stoi"]
        self.tgt_itos:  dict = {int(k): v for k, v in tgt_vocab_data["itos"].items()}

        src_vocab_size = len(self.src_vocab)
        tgt_vocab_size = len(self.tgt_vocab)

        self.pad_idx = self.src_vocab.get("<pad>", 0)
        self.sos_idx = self.tgt_vocab.get("<sos>", 2)
        self.eos_idx = self.tgt_vocab.get("<eos>", 3)
        self.unk_idx = self.src_vocab.get("<unk>", 1)

        self.encoder    = Encoder(src_vocab_size, d_model, num_heads,
                                  num_encoder_layers, d_ff, max_len, dropout)
        self.decoder    = Decoder(tgt_vocab_size, d_model, num_heads,
                                  num_decoder_layers, d_ff, max_len, dropout)
        self.projection = nn.Linear(d_model, tgt_vocab_size)

        self._init_parameters()

        if load_weights:
            self._load_weights()

    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _load_weights(self):
        import gdown
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


    def _make_src_mask(self, src):
        return (src == self.pad_idx).unsqueeze(1).unsqueeze(2)

    def _make_tgt_mask(self, tgt):
        T       = tgt.size(1)
        causal  = torch.triu(torch.ones(T, T, device=tgt.device, dtype=torch.bool), 1)
        pad_mask = (tgt == self.pad_idx).unsqueeze(1).unsqueeze(2)
        return causal.unsqueeze(0).unsqueeze(0) | pad_mask

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        if src_mask is None: src_mask = self._make_src_mask(src)
        if tgt_mask is None: tgt_mask = self._make_tgt_mask(tgt)
        memory  = self.encoder(src, src_mask)
        dec_out = self.decoder(tgt, memory, tgt_mask, src_mask)
        return self.projection(dec_out)

    def _tokenize_src(self, sentence: str):
        return [tok.text.lower() for tok in self.src_tokenizer(sentence)]

    def _encode_src(self, tokens):
        return torch.tensor(
            [self.src_vocab.get(t, self.unk_idx) for t in tokens],
            dtype=torch.long,
        )

    def infer(self, german_sentence: str, max_output_len: int = 50) -> str:
        """
        Accept a German sentence, return the English translation.
        Used directly by the autograder.
        """
        self.eval()
        device   = next(self.parameters()).device
        tokens   = self._tokenize_src(german_sentence)
        src_ids  = self._encode_src(tokens).unsqueeze(0).to(device)
        src_mask = self._make_src_mask(src_ids)

        with torch.no_grad():
            memory  = self.encoder(src_ids, src_mask)
            tgt_ids = torch.tensor([[self.sos_idx]], dtype=torch.long, device=device)

            for _ in range(max_output_len):
                tgt_mask = self._make_tgt_mask(tgt_ids)
                dec_out  = self.decoder(tgt_ids, memory, tgt_mask, src_mask)
                next_id  = self.projection(dec_out)[0, -1, :].argmax().item()

                if next_id == self.eos_idx:
                    break

                tgt_ids = torch.cat(
                    [tgt_ids,
                     torch.tensor([[next_id]], dtype=torch.long, device=device)],
                    dim=1,
                )

        predicted_ids = tgt_ids[0, 1:].tolist()
        words = []
        for tok in [self.tgt_itos.get(i, "<unk>") for i in predicted_ids]:
            if tok in {".", ",", "!", "?", ";", ":", "'"} and words:
                words[-1] += tok
            else:
                words.append(tok)
        return " ".join(words)
