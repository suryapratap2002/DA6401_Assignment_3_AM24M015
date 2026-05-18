"""
train.py — Improved Transformer training for Multi30k BLEU optimization
"""

import argparse
import gc
import math
import os
import time

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

from model import Encoder, Decoder
from utils import LabelSmoothingLoss, NoamScheduler
from dataset import (
    build_vocab_and_loaders,
    SOS_IDX,
    EOS_IDX,
    PAD_IDX,
)

try:
    import sacrebleu
    USE_SACREBLEU = True
except ImportError:
    USE_SACREBLEU = False


# ============================================================
# Training Transformer
# ============================================================

class _TrainTransformer(nn.Module):

    def __init__(self, src_vocab_size, tgt_vocab_size, args):
        super().__init__()

        self.encoder = Encoder(
            src_vocab_size,
            args.d_model,
            args.num_heads,
            args.num_layers,
            args.d_ff,
            args.max_len,
            args.dropout,
        )

        self.decoder = Decoder(
            tgt_vocab_size,
            args.d_model,
            args.num_heads,
            args.num_layers,
            args.d_ff,
            args.max_len,
            args.dropout,
        )

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

        causal = torch.triu(
            torch.ones(T, T, device=tgt.device, dtype=torch.bool),
            diagonal=1
        )

        pad_mask = (tgt == PAD_IDX).unsqueeze(1).unsqueeze(2)

        return causal.unsqueeze(0).unsqueeze(0) | pad_mask

    def forward(self, src, tgt):

        src_mask = self._src_mask(src)
        tgt_mask = self._tgt_mask(tgt)

        memory = self.encoder(src, src_mask)

        out = self.decoder(
            tgt,
            memory,
            tgt_mask,
            src_mask
        )

        return self.projection(out)


# ============================================================
# Beam Search Decoding
# ============================================================

@torch.no_grad()
def beam_search_decode(
    model,
    src,
    tgt_vocab,
    device,
    max_len=100,
    beam_size=4,
):

    model.eval()

    src = src.to(device)

    batch_size = src.size(0)

    src_mask = (src == PAD_IDX).unsqueeze(1).unsqueeze(2)

    memory = model.encoder(src, src_mask)

    results = []

    for b in range(batch_size):

        mem = memory[b:b + 1]
        sm = src_mask[b:b + 1]

        beams = [([SOS_IDX], 0.0)]

        for _ in range(max_len):

            candidates = []

            for seq, score in beams:

                if seq[-1] == EOS_IDX:
                    candidates.append((seq, score))
                    continue

                tgt = torch.tensor(
                    seq,
                    dtype=torch.long,
                    device=device
                ).unsqueeze(0)

                T = tgt.size(1)

                causal = torch.triu(
                    torch.ones(T, T, device=device, dtype=torch.bool),
                    diagonal=1
                )

                tgt_mask = causal.unsqueeze(0).unsqueeze(0)

                out = model.decoder(
                    tgt,
                    mem,
                    tgt_mask,
                    sm
                )

                logits = model.projection(out)[:, -1, :]

                log_probs = torch.log_softmax(logits, dim=-1)

                topk_log_probs, topk_ids = torch.topk(
                    log_probs,
                    beam_size,
                    dim=-1
                )

                for k in range(beam_size):

                    token = topk_ids[0, k].item()
                    token_score = topk_log_probs[0, k].item()

                    candidates.append(
                        (seq + [token], score + token_score)
                    )

            beams = sorted(
                candidates,
                key=lambda x: x[1] / (len(x[0]) ** 0.7),
                reverse=True
            )[:beam_size]

            if all(seq[-1] == EOS_IDX for seq, _ in beams):
                break

        best_seq = beams[0][0]

        tokens = []

        for idx in best_seq[1:]:

            if idx == EOS_IDX:
                break

            tokens.append(
                tgt_vocab.itos.get(idx, "<unk>")
            )

        results.append(tokens)

    return results


# ============================================================
# BLEU
# ============================================================

def detokenize(tokens):

    no_space_before = set(".,!?;:'")

    out = []

    for tok in tokens:

        if tok in no_space_before and out:
            out[-1] += tok
        else:
            out.append(tok)

    return " ".join(out)


@torch.no_grad()
def evaluate_bleu(
    model,
    loader,
    tgt_vocab,
    device,
    max_len=100
):

    hypotheses = []
    references = []

    for src, tgt in loader:

        hyps = beam_search_decode(
            model,
            src,
            tgt_vocab,
            device,
            max_len=max_len,
            beam_size=4
        )

        for i, ref_ids in enumerate(tgt.tolist()):

            ref_tokens = []

            for idx in ref_ids[1:]:

                if idx in (EOS_IDX, PAD_IDX):
                    break

                ref_tokens.append(
                    tgt_vocab.itos.get(idx, "<unk>")
                )

            hypotheses.append(detokenize(hyps[i]))
            references.append(detokenize(ref_tokens))

    if USE_SACREBLEU:
        return sacrebleu.corpus_bleu(
            hypotheses,
            [references],
            force=True
        ).score

    return 0.0


# ============================================================
# Train Epoch
# ============================================================

def train_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scheduler,
    scaler,
    device,
    args,
):

    model.train()

    total_loss = 0.0
    total_tokens = 0

    optimizer.zero_grad(set_to_none=True)

    for batch_idx, (src, tgt) in enumerate(loader):

        src = src.to(device)
        tgt = tgt.to(device)

        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:]

        with autocast(enabled=args.amp):

            logits = model(src, tgt_in)

            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                tgt_out.reshape(-1)
            )

            loss = loss / args.accum_steps

        scaler.scale(loss).backward()

        if (batch_idx + 1) % args.accum_steps == 0:

            scaler.unscale_(optimizer)

            nn.utils.clip_grad_norm_(
                model.parameters(),
                args.clip
            )

            scaler.step(optimizer)
            scaler.update()

            scheduler.step()

            optimizer.zero_grad(set_to_none=True)

        non_pad = tgt_out.reshape(-1).ne(PAD_IDX).sum().item()

        total_loss += loss.item() * args.accum_steps * non_pad
        total_tokens += non_pad

        if (batch_idx + 1) % args.log_interval == 0:

            avg = total_loss / max(total_tokens, 1)

            print(
                f"step {batch_idx+1} | "
                f"loss {avg:.4f} | "
                f"ppl {math.exp(min(avg,20)):.2f}"
            )

        del src, tgt, tgt_in, tgt_out, logits, loss

    return total_loss / max(total_tokens, 1)


# ============================================================
# Validation Loss
# ============================================================

@torch.no_grad()
def evaluate_loss(
    model,
    loader,
    criterion,
    device,
    amp=True
):

    model.eval()

    total_loss = 0.0
    total_tokens = 0

    for src, tgt in loader:

        src = src.to(device)
        tgt = tgt.to(device)

        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:]

        with autocast(enabled=amp):

            logits = model(src, tgt_in)

            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                tgt_out.reshape(-1)
            )

        non_pad = tgt_out.reshape(-1).ne(PAD_IDX).sum().item()

        total_loss += loss.item() * non_pad
        total_tokens += non_pad

    return total_loss / max(total_tokens, 1)


# ============================================================
# Args
# ============================================================

def parse_args():

    p = argparse.ArgumentParser()

    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--num_layers", type=int, default=6)
    p.add_argument("--d_ff", type=int, default=2048)

    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--max_len", type=int, default=128)

    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--accum_steps", type=int, default=2)

    p.add_argument("--warmup", type=int, default=4000)

    p.add_argument("--label_smooth", type=float, default=0.1)

    p.add_argument("--clip", type=float, default=1.0)

    p.add_argument("--early_stop", type=int, default=20)

    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--no_amp", dest="amp", action="store_false")

    p.add_argument("--min_freq", type=int, default=2)

    p.add_argument(
        "--save_path",
        type=str,
        default="transformer_best.pt"
    )

    p.add_argument(
        "--src_vocab",
        type=str,
        default="src_vocab.pt"
    )

    p.add_argument(
        "--tgt_vocab",
        type=str,
        default="tgt_vocab.pt"
    )

    p.add_argument("--log_interval", type=int, default=50)

    return p.parse_args()


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = \
        build_vocab_and_loaders(
            batch_size=args.batch,
            max_len=args.max_len,
            min_freq=args.min_freq,
            src_vocab_path=args.src_vocab,
            tgt_vocab_path=args.tgt_vocab,
            num_workers=0,
        )

    model = _TrainTransformer(
        len(src_vocab),
        len(tgt_vocab),
        args
    ).to(device)

    criterion = LabelSmoothingLoss(
        vocab_size=len(tgt_vocab),
        padding_idx=PAD_IDX,
        smoothing=args.label_smooth,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1.0,
        betas=(0.9, 0.98),
        eps=1e-9
    )

    scheduler = NoamScheduler(
        optimizer,
        d_model=args.d_model,
        warmup_steps=args.warmup,
    )

    scaler = GradScaler(enabled=args.amp)

    best_bleu = 0.0
    no_improve = 0

    for epoch in range(1, args.epochs + 1):

        print("\n" + "=" * 60)
        print(f"Epoch {epoch}/{args.epochs}")
        print("=" * 60)

        train_loss = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scheduler,
            scaler,
            device,
            args,
        )

        gc.collect()
        torch.cuda.empty_cache()

        val_loss = evaluate_loss(
            model,
            val_loader,
            criterion,
            device,
            args.amp
        )

        val_bleu = evaluate_bleu(
            model,
            val_loader,
            tgt_vocab,
            device
        )

        print(
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_bleu={val_bleu:.2f}"
        )

        if val_bleu > best_bleu:

            best_bleu = val_bleu
            no_improve = 0

            torch.save(
                model.state_dict(),
                args.save_path
            )

            print(
                f"Saved best model with BLEU {best_bleu:.2f}"
            )

        else:

            no_improve += 1

            print(
                f"No BLEU improvement "
                f"{no_improve}/{args.early_stop}"
            )

            if no_improve >= args.early_stop:

                print("Early stopping triggered")
                break

    print("\nLoading best model...")

    model.load_state_dict(
        torch.load(
            args.save_path,
            map_location=device,
            weights_only=True
        )
    )

    bleu = evaluate_bleu(
        model,
        test_loader,
        tgt_vocab,
        device
    )

    print(f"\nFinal Test BLEU: {bleu:.2f}")


if __name__ == "__main__":
    main()