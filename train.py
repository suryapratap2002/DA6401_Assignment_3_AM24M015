"""
train.py — Training loop, validation, and evaluation for the Transformer NMT model.

Run:
    python train.py                   # full training (memory-safe defaults)
    python train.py --batch 32        # if still crashing, lower batch further
    python train.py --batch 64 --accum_steps 2   # gradient accumulation
"""

import argparse
import gc
import math
import os
import sys
import time

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from model import Encoder, Decoder
from utils import LabelSmoothingLoss, NoamScheduler
from dataset import (
    build_vocab_and_loaders,
    SOS_IDX, EOS_IDX, PAD_IDX,
    get_tokenizers,
    Vocabulary,
)

# Optional: BLEU via sacrebleu (preferred) or nltk
try:
    import sacrebleu
    USE_SACREBLEU = True
except ImportError:
    USE_SACREBLEU = False
    try:
        from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
    except ImportError:
        USE_SACREBLEU = False


# ──────────────────────────────────────────────────────────────────────────────
# Lightweight training-only Transformer wrapper
# ──────────────────────────────────────────────────────────────────────────────

class _TrainTransformer(nn.Module):
    """
    Thin wrapper used ONLY during training.
    Does NOT do weight download / vocab loading — that is Transformer.__init__'s job.
    """
    def __init__(self, src_vocab_size, tgt_vocab_size, args):
        super().__init__()
        self.encoder    = Encoder(src_vocab_size, args.d_model, args.num_heads,
                                  args.num_layers, args.d_ff, args.max_len, args.dropout)
        self.decoder    = Decoder(tgt_vocab_size, args.d_model, args.num_heads,
                                  args.num_layers, args.d_ff, args.max_len, args.dropout)
        self.projection = nn.Linear(args.d_model, tgt_vocab_size)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _src_mask(self, src):
        return (src == PAD_IDX).unsqueeze(1).unsqueeze(2)

    def _tgt_mask(self, tgt):
        T = tgt.size(1)
        causal   = torch.triu(torch.ones(T, T, device=tgt.device, dtype=torch.bool), 1)
        pad_mask = (tgt == PAD_IDX).unsqueeze(1).unsqueeze(2)
        return causal.unsqueeze(0).unsqueeze(0) | pad_mask

    def forward(self, src, tgt):
        sm  = self._src_mask(src)
        tm  = self._tgt_mask(tgt)
        mem = self.encoder(src, sm)
        out = self.decoder(tgt, mem, tm, sm)
        return self.projection(out)


# ──────────────────────────────────────────────────────────────────────────────
# Greedy decoding
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def greedy_decode(model, src, tgt_vocab, device, max_len=100):
    """Greedy decoding for a batch. Returns list of token-lists."""
    model.eval()
    batch_size = src.size(0)
    src        = src.to(device)
    src_mask   = (src == PAD_IDX).unsqueeze(1).unsqueeze(2)

    memory   = model.encoder(src, src_mask)
    tgt_ids  = torch.full((batch_size, 1), SOS_IDX, dtype=torch.long, device=device)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    for _ in range(max_len):
        T        = tgt_ids.size(1)
        causal   = torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), 1)
        tgt_mask = causal.unsqueeze(0).unsqueeze(0)
        dec_out  = model.decoder(tgt_ids, memory, tgt_mask, src_mask)
        next_id  = model.projection(dec_out)[:, -1, :].argmax(dim=-1)

        tgt_ids  = torch.cat([tgt_ids, next_id.unsqueeze(1)], dim=1)
        finished = finished | (next_id == EOS_IDX)
        if finished.all():
            break

    results = []
    for row in tgt_ids[:, 1:].tolist():
        tokens = []
        for idx in row:
            if idx == EOS_IDX:
                break
            tokens.append(tgt_vocab.itos.get(idx, "<unk>"))
        results.append(tokens)
    return results


# ──────────────────────────────────────────────────────────────────────────────
# BLEU evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_bleu(model, loader, tgt_vocab, device, max_len=100):
    hypotheses, references = [], []
    for src, tgt in loader:
        hyps = greedy_decode(model, src, tgt_vocab, device, max_len)
        for i, ref_ids in enumerate(tgt.tolist()):
            ref_tokens = []
            for idx in ref_ids[1:]:
                if idx in (EOS_IDX, PAD_IDX):
                    break
                ref_tokens.append(tgt_vocab.itos.get(idx, "<unk>"))
            hypotheses.append(" ".join(hyps[i]))
            references.append(" ".join(ref_tokens))

    if USE_SACREBLEU:
        return sacrebleu.corpus_bleu(hypotheses, [references]).score
    else:
        refs_tok = [[r.split()] for r in references]
        hyps_tok = [h.split()   for h in hypotheses]
        sf = SmoothingFunction().method1
        return corpus_bleu(refs_tok, hyps_tok, smoothing_function=sf) * 100


# ──────────────────────────────────────────────────────────────────────────────
# Training epoch  (mixed precision + gradient accumulation)
# ──────────────────────────────────────────────────────────────────────────────

def train_epoch(model, loader, criterion, optimizer, scheduler,
                scaler, device, args):
    model.train()
    total_loss, total_tokens = 0.0, 0
    start = time.time()

    optimizer.zero_grad(set_to_none=True)   # set_to_none frees memory faster

    for batch_idx, (src, tgt) in enumerate(loader):
        src     = src.to(device, non_blocking=True)
        tgt     = tgt.to(device, non_blocking=True)
        tgt_in  = tgt[:, :-1]
        tgt_out = tgt[:, 1:]

        # ── forward  (mixed precision) ────────────────────────────────────
        with autocast(enabled=args.amp):
            logits = model(src, tgt_in)                         # (B, T-1, V)
            loss   = criterion(
                logits.reshape(-1, logits.size(-1)),
                tgt_out.reshape(-1),
            )
            # Scale loss for gradient accumulation
            loss = loss / args.accum_steps

        # ── backward ──────────────────────────────────────────────────────
        scaler.scale(loss).backward()

        # ── optimizer step every accum_steps batches ──────────────────────
        if (batch_idx + 1) % args.accum_steps == 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        non_pad       = tgt_out.reshape(-1).ne(PAD_IDX).sum().item()
        total_loss   += loss.item() * args.accum_steps * non_pad
        total_tokens += non_pad

        # ── periodic logging + memory cleanup ────────────────────────────
        if (batch_idx + 1) % args.log_interval == 0:
            avg = total_loss / max(total_tokens, 1)
            mem = torch.cuda.memory_reserved(device) / 1024**3 if args.amp else 0
            print(f"  step {batch_idx+1:5d} | "
                  f"lr {scheduler.get_last_lr():.2e} | "
                  f"loss {avg:.4f} | ppl {math.exp(min(avg,20)):.1f} | "
                  f"VRAM {mem:.1f}GB | {time.time()-start:.1f}s")
            start = time.time()

        # ── free graph every step to avoid VRAM accumulation ─────────────
        del src, tgt, tgt_in, tgt_out, logits, loss
        if (batch_idx + 1) % 50 == 0:
            torch.cuda.empty_cache()

    # handle leftover accumulation steps
    if (len(loader)) % args.accum_steps != 0:
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

    return total_loss / max(total_tokens, 1)


# ──────────────────────────────────────────────────────────────────────────────
# Validation loss
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_loss(model, loader, criterion, device, amp=True):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for src, tgt in loader:
        src, tgt = src.to(device), tgt.to(device)
        tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
        with autocast(enabled=amp):
            logits = model(src, tgt_in)
            loss   = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
        non_pad       = tgt_out.reshape(-1).ne(PAD_IDX).sum().item()
        total_loss   += loss.item() * non_pad
        total_tokens += non_pad
        del src, tgt, tgt_in, tgt_out, logits, loss
    torch.cuda.empty_cache()
    return total_loss / max(total_tokens, 1)


# ──────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train Transformer NMT")

    # ── Model hyper-params ────────────────────────────────────────────────
    p.add_argument("--d_model",      type=int,   default=512,
                   help="Model dimension (256 is safe; 512 needs more VRAM)")
    p.add_argument("--num_heads",    type=int,   default=8)
    p.add_argument("--num_layers",   type=int,   default=6,
                   help="Encoder/Decoder layers (3 is safe; 6 needs more VRAM)")
    p.add_argument("--d_ff",         type=int,   default=2048)
    p.add_argument("--dropout",      type=float, default=0.1)
    p.add_argument("--max_len",      type=int,   default=128)

    # ── Training ──────────────────────────────────────────────────────────
    p.add_argument("--epochs",       type=int,   default=30)
    p.add_argument("--batch",        type=int,   default=64,
                   help="Per-step batch size. Lower if crashing (32 or 16)")
    p.add_argument("--accum_steps",  type=int,   default=2,
                   help="Gradient accumulation steps. "
                        "Effective batch = batch * accum_steps")
    p.add_argument("--warmup",       type=int,   default=4000)
    p.add_argument("--label_smooth", type=float, default=0.1)
    p.add_argument("--clip",         type=float, default=1.0)
    p.add_argument("--amp",          action="store_true", default=True,
                   help="Use mixed-precision (fp16) training")
    p.add_argument("--no_amp",       dest="amp", action="store_false",
                   help="Disable mixed-precision")

    # ── Data / paths ──────────────────────────────────────────────────────
    p.add_argument("--min_freq",     type=int,   default=2)
    p.add_argument("--save_path",    type=str,   default="transformer_best.pt")
    p.add_argument("--src_vocab",    type=str,   default="src_vocab.pt")
    p.add_argument("--tgt_vocab",    type=str,   default="tgt_vocab.pt")
    p.add_argument("--log_interval", type=int,   default=50)
    p.add_argument("--seed",         type=int,   default=42)

    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Device : {device}")
    if device.type == "cuda":
        print(f"[train] GPU    : {torch.cuda.get_device_name(0)}")
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"[train] VRAM   : {total_vram:.1f} GB")

    # ── Limit VRAM fragmentation ──────────────────────────────────────────
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    # ── Data ──────────────────────────────────────────────────────────────
    print("[train] Loading data …")
    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = build_vocab_and_loaders(
        batch_size=args.batch,
        max_len=args.max_len,
        min_freq=args.min_freq,
        src_vocab_path=args.src_vocab,
        tgt_vocab_path=args.tgt_vocab,
        num_workers=0,
    )
    print(f"[train] src_vocab={len(src_vocab)}  tgt_vocab={len(tgt_vocab)}")
    print(f"[train] Effective batch = {args.batch} × {args.accum_steps} = "
          f"{args.batch * args.accum_steps}")

    # ── Model ─────────────────────────────────────────────────────────────
    model = _TrainTransformer(len(src_vocab), len(tgt_vocab), args).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] Parameters: {n_params:,}")

    # ── Optimiser & scheduler ─────────────────────────────────────────────
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9
    )
    scheduler = NoamScheduler(optimizer, d_model=args.d_model,
                              warmup_steps=args.warmup)

    # ── Mixed-precision scaler ────────────────────────────────────────────
    scaler = GradScaler(enabled=args.amp)
    print(f"[train] Mixed precision (AMP): {args.amp}")

    # ── Loss ──────────────────────────────────────────────────────────────
    criterion = LabelSmoothingLoss(
        vocab_size=len(tgt_vocab),
        padding_idx=PAD_IDX,
        smoothing=args.label_smooth,
    )

    # ── Training loop ─────────────────────────────────────────────────────
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        print(f"\n{'─'*60}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'─'*60}")

        train_loss = train_epoch(
            model, train_loader, criterion, optimizer,
            scheduler, scaler, device, args,
        )

        # ── clear cache before validation ─────────────────────────────────
        gc.collect()
        torch.cuda.empty_cache()

        val_loss = evaluate_loss(model, val_loader, criterion, device, args.amp)
        elapsed  = time.time() - t0

        print(
            f"\nEpoch {epoch:3d} summary | "
            f"train_loss {train_loss:.4f} (ppl {math.exp(min(train_loss,20)):.1f}) | "
            f"val_loss {val_loss:.4f} (ppl {math.exp(min(val_loss,20)):.1f}) | "
            f"{elapsed:.0f}s"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), args.save_path)
            print(f"  ✓ Saved best model  (val_loss={val_loss:.4f})")

    # ── Final BLEU on test set ─────────────────────────────────────────────
    print("\n[train] Loading best checkpoint for BLEU evaluation …")
    model.load_state_dict(
        torch.load(args.save_path, map_location=device, weights_only=True)
    )
    bleu = evaluate_bleu(model, test_loader, tgt_vocab, device)
    print(f"[train] Test BLEU: {bleu:.2f}")


if __name__ == "__main__":
    main()