"""
experiments.py — W&B experiments for DA6401 Assignment 3 Report (Section 2)

Run:
    python experiments.py --exp 2.1
    python experiments.py --exp 2.2
    python experiments.py --exp 2.3
    python experiments.py --exp 2.4
    python experiments.py --exp 2.5
    python experiments.py --exp all
"""

import multiprocessing
multiprocessing.freeze_support()   # required for Windows + wandb

import argparse
import gc
import math
import os
import sys
import time
import traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

import wandb

# ── local imports ─────────────────────────────────────────────────────────────
from model import (
    PositionalEncoding,
    PositionwiseFeedForward,
    EncoderLayer,
    DecoderLayer,
    MultiHeadAttention,
    ScaledDotProductAttention,
)
from utils import LabelSmoothingLoss, NoamScheduler
from dataset import (
    build_vocab_and_loaders,
    SOS_IDX, EOS_IDX, PAD_IDX,
)

# ─────────────────────────────────────────────────────────────────────────────
WANDB_PROJECT = "da6401-assignment3"

BASE = dict(
    d_model=512, num_heads=8, num_layers=6, d_ff=2048,
    dropout=0.1, max_len=128, batch=64, accum_steps=2,
    epochs=15, warmup=4000, label_smooth=0.1,
    clip=1.0, weight_decay=1e-4,
)


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

class TransformerModel(nn.Module):
    """Full Transformer used in all experiments."""

    def __init__(self, src_vsz, tgt_vsz, cfg, learned_pe=False):
        super().__init__()
        d, h, N, ff = cfg["d_model"], cfg["num_heads"], cfg["num_layers"], cfg["d_ff"]
        dr, ml       = cfg["dropout"], cfg["max_len"]

        self.learned_pe = learned_pe
        self.scale      = math.sqrt(d)

        self.src_emb = nn.Embedding(src_vsz, d, padding_idx=0)
        self.tgt_emb = nn.Embedding(tgt_vsz, d, padding_idx=0)

        if learned_pe:
            self.src_pe = nn.Embedding(ml, d)
            self.tgt_pe = nn.Embedding(ml, d)
            self.drop   = nn.Dropout(dr)
        else:
            self.src_pe = PositionalEncoding(d, ml, dr)
            self.tgt_pe = PositionalEncoding(d, ml, dr)

        self.enc = nn.ModuleList([EncoderLayer(d, h, ff, dr) for _ in range(N)])
        self.dec = nn.ModuleList([DecoderLayer(d, h, ff, dr) for _ in range(N)])
        self.proj = nn.Linear(d, tgt_vsz)

        self._init()

    def _init(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # ── masks ────────────────────────────────────────────────────────────────
    def src_mask(self, src):
        return (src == PAD_IDX).unsqueeze(1).unsqueeze(2)

    def tgt_mask(self, tgt):
        T  = tgt.size(1)
        cm = torch.triu(torch.ones(T, T, device=tgt.device, dtype=torch.bool), 1)
        pm = (tgt == PAD_IDX).unsqueeze(1).unsqueeze(2)
        return cm.unsqueeze(0).unsqueeze(0) | pm

    # ── embedding helpers ─────────────────────────────────────────────────────
    def embed_src(self, src):
        x = self.src_emb(src) * self.scale
        if self.learned_pe:
            p = torch.arange(src.size(1), device=src.device).unsqueeze(0)
            return self.drop(x + self.src_pe(p))
        return self.src_pe(x)

    def embed_tgt(self, tgt):
        x = self.tgt_emb(tgt) * self.scale
        if self.learned_pe:
            p = torch.arange(tgt.size(1), device=tgt.device).unsqueeze(0)
            return self.drop(x + self.tgt_pe(p))
        return self.tgt_pe(x)

    # ── forward ───────────────────────────────────────────────────────────────
    def encode(self, src, sm):
        x = self.embed_src(src)
        for layer in self.enc:
            x = layer(x, sm)
        return x

    def decode(self, tgt, mem, tm, sm):
        x = self.embed_tgt(tgt)
        for layer in self.dec:
            x = layer(x, mem, tm, sm)
        return x

    def forward(self, src, tgt):
        sm  = self.src_mask(src)
        tm  = self.tgt_mask(tgt)
        mem = self.encode(src, sm)
        out = self.decode(tgt, mem, tm, sm)
        return self.proj(out)

    # ── attention weights from last encoder layer ─────────────────────────────
    @torch.no_grad()
    def last_enc_attn(self, src):
        self.eval()
        sm = self.src_mask(src)
        x  = self.embed_src(src)
        weights = None
        for i, layer in enumerate(self.enc):
            attn_out, w = layer.self_attn(x, x, x, mask=sm)
            x = layer.norm1(x + layer.dropout(attn_out))
            x = layer.norm2(x + layer.dropout(layer.ffn(x)))
            if i == len(self.enc) - 1:
                weights = w
        return weights   # (1, heads, T, T)


# ─────────────────────────────────────────────────────────────────────────────
# Unscaled attention (Exp 2.2)
# ─────────────────────────────────────────────────────────────────────────────

class UnscaledAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1))   # no 1/sqrt(dk)
        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))
        w = torch.nan_to_num(F.softmax(scores, dim=-1), nan=0.0)
        return torch.matmul(self.drop(w), V), w


class MHANoScale(nn.Module):
    def __init__(self, d_model=512, num_heads=8, dropout=0.1):
        super().__init__()
        self.h  = num_heads
        self.dk = d_model // num_heads
        self.dm = d_model
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Wo = nn.Linear(d_model, d_model, bias=False)
        self.attn = UnscaledAttention(dropout)

    def forward(self, q, k, v, mask=None):
        B = q.size(0)
        def split(x): return x.view(B,-1,self.h,self.dk).transpose(1,2)
        Q,K,V = split(self.Wq(q)), split(self.Wk(k)), split(self.Wv(v))
        o, w  = self.attn(Q, K, V, mask)
        o = o.transpose(1,2).contiguous().view(B,-1,self.dm)
        return self.Wo(o), w


class EncLayerNoScale(nn.Module):
    def __init__(self, d=512, h=8, ff=2048, dr=0.1):
        super().__init__()
        self.self_attn = MHANoScale(d, h, dr)
        self.ffn   = PositionwiseFeedForward(d, ff, dr)
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.dropout = nn.Dropout(dr)

    def forward(self, x, mask=None):
        a, _ = self.self_attn(x, x, x, mask=mask)
        x = self.norm1(x + self.dropout(a))
        return self.norm2(x + self.dropout(self.ffn(x)))


class DecLayerNoScale(nn.Module):
    def __init__(self, d=512, h=8, ff=2048, dr=0.1):
        super().__init__()
        self.self_attn  = MHANoScale(d, h, dr)
        self.cross_attn = MHANoScale(d, h, dr)
        self.ffn   = PositionwiseFeedForward(d, ff, dr)
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.norm3 = nn.LayerNorm(d)
        self.dropout = nn.Dropout(dr)

    def forward(self, x, mem, tm=None, sm=None):
        a, _ = self.self_attn(x, x, x, mask=tm)
        x = self.norm1(x + self.dropout(a))
        a, _ = self.cross_attn(x, mem, mem, mask=sm)
        x = self.norm2(x + self.dropout(a))
        return self.norm3(x + self.dropout(self.ffn(x)))


class TransformerNoScale(TransformerModel):
    """Same as TransformerModel but uses unscaled attention."""
    def __init__(self, src_vsz, tgt_vsz, cfg):
        nn.Module.__init__(self)
        d, h, N, ff = cfg["d_model"], cfg["num_heads"], cfg["num_layers"], cfg["d_ff"]
        dr, ml       = cfg["dropout"], cfg["max_len"]
        self.learned_pe = False
        self.scale      = math.sqrt(d)
        self.src_emb = nn.Embedding(src_vsz, d, padding_idx=0)
        self.tgt_emb = nn.Embedding(tgt_vsz, d, padding_idx=0)
        self.src_pe  = PositionalEncoding(d, ml, dr)
        self.tgt_pe  = PositionalEncoding(d, ml, dr)
        self.enc  = nn.ModuleList([EncLayerNoScale(d, h, ff, dr) for _ in range(N)])
        self.dec  = nn.ModuleList([DecLayerNoScale(d, h, ff, dr) for _ in range(N)])
        self.proj = nn.Linear(d, tgt_vsz)
        self._init()


# ─────────────────────────────────────────────────────────────────────────────
# Training utilities
# ─────────────────────────────────────────────────────────────────────────────

def make_opt(model, cfg, fixed_lr=None):
    if fixed_lr is not None:
        opt = torch.optim.AdamW(
            model.parameters(), lr=fixed_lr,
            betas=(0.9, 0.98), eps=1e-9,
            weight_decay=cfg["weight_decay"],
        )
        return opt, None
    opt = torch.optim.AdamW(
        model.parameters(), lr=1.0,
        betas=(0.9, 0.98), eps=1e-9,
        weight_decay=cfg["weight_decay"],
    )
    sched = NoamScheduler(opt, d_model=cfg["d_model"],
                          warmup_steps=cfg["warmup"])
    return opt, sched


def train_epoch(model, loader, criterion, opt, sched, scaler,
                device, cfg, step, log_grads=False, prefix=""):
    model.train()
    tot_loss, tot_tok = 0.0, 0
    opt.zero_grad(set_to_none=True)

    for bi, (src, tgt) in enumerate(loader):
        src, tgt    = src.to(device), tgt.to(device)
        ti, to      = tgt[:, :-1], tgt[:, 1:]

        with autocast(enabled=True):
            logits = model(src, ti)
            loss   = criterion(
                logits.reshape(-1, logits.size(-1)), to.reshape(-1)
            ) / cfg["accum_steps"]

        scaler.scale(loss).backward()

        if (bi + 1) % cfg["accum_steps"] == 0:
            scaler.unscale_(opt)

            # log Q/K grad norms for exp 2.2
            if log_grads and step <= 1000:
                qn, kn = [], []
                for layer in model.enc:
                    wq = getattr(layer.self_attn, "Wq",
                                 getattr(layer.self_attn, "W_q", None))
                    wk = getattr(layer.self_attn, "Wk",
                                 getattr(layer.self_attn, "W_k", None))
                    if wq is not None and wq.weight.grad is not None:
                        qn.append(wq.weight.grad.norm().item())
                        kn.append(wk.weight.grad.norm().item())
                if qn:
                    wandb.log({f"{prefix}grad_Wq": np.mean(qn),
                               f"{prefix}grad_Wk": np.mean(kn),
                               "global_step": step})

            nn.utils.clip_grad_norm_(model.parameters(), cfg["clip"])
            scaler.step(opt)
            scaler.update()
            if sched: sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1

        np_ = to.reshape(-1).ne(PAD_IDX).sum().item()
        tot_loss += loss.item() * cfg["accum_steps"] * np_
        tot_tok  += np_
        del src, tgt, ti, to, logits, loss

    return tot_loss / max(tot_tok, 1), step


@torch.no_grad()
def val_loss(model, loader, criterion, device):
    model.eval()
    tl, tt = 0.0, 0
    for src, tgt in loader:
        src, tgt = src.to(device), tgt.to(device)
        ti, to   = tgt[:, :-1], tgt[:, 1:]
        with autocast(enabled=True):
            logits = model(src, ti)
            loss   = criterion(logits.reshape(-1, logits.size(-1)), to.reshape(-1))
        np_ = to.reshape(-1).ne(PAD_IDX).sum().item()
        tl += loss.item() * np_; tt += np_
        del src, tgt, ti, to, logits, loss
    torch.cuda.empty_cache()
    return tl / max(tt, 1)


@torch.no_grad()
def bleu_score(model, loader, tgt_vocab, device):
    import sacrebleu as sb
    model.eval()
    hyps, refs = [], []

    for src, tgt in loader:
        src = src.to(device)
        sm  = model.src_mask(src)
        mem = model.encode(src, sm)
        B   = src.size(0)
        dec = torch.full((B,1), SOS_IDX, dtype=torch.long, device=device)
        done = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(100):
            T    = dec.size(1)
            cm   = torch.triu(torch.ones(T,T,device=device,dtype=torch.bool),1)
            tm   = cm.unsqueeze(0).unsqueeze(0)
            out  = model.decode(dec, mem, tm, sm)
            nxt  = model.proj(out)[:,-1,:].argmax(-1)
            dec  = torch.cat([dec, nxt.unsqueeze(1)], dim=1)
            done = done | (nxt == EOS_IDX)
            if done.all(): break

        for i, ref_ids in enumerate(tgt.tolist()):
            def to_str(ids, skip_first=True):
                toks = []
                for idx in (ids[1:] if skip_first else ids):
                    if idx in (EOS_IDX, PAD_IDX): break
                    toks.append(tgt_vocab.itos.get(int(idx), "<unk>"))
                out = []
                for t in toks:
                    if t in set(".,!?;:'") and out: out[-1] += t
                    else: out.append(t)
                return " ".join(out)
            hyps.append(to_str(dec[i].tolist()))
            refs.append(to_str(ref_ids))

    return sb.corpus_bleu(hyps, [refs], force=True).score


def confidence_on_batch(model, loader, tgt_vocab, device, n_batches=20):
    """Mean softmax probability of the correct token."""
    model.eval()
    confs = []
    with torch.no_grad():
        for bi, (src, tgt) in enumerate(loader):
            if bi >= n_batches: break
            src, tgt = src.to(device), tgt.to(device)
            ti, to   = tgt[:,:-1], tgt[:,1:]
            with autocast(enabled=True):
                logits = model(src, ti)
            probs    = F.softmax(logits.float(), dim=-1)
            flat_p   = probs.reshape(-1, probs.size(-1))
            flat_t   = to.reshape(-1)
            non_pad  = flat_t.ne(PAD_IDX)
            correct  = flat_p[non_pad].gather(1, flat_t[non_pad].unsqueeze(1))
            confs.append(correct.mean().item())
    return float(np.mean(confs))


def run(name, cfg, train_l, val_l, test_l, tgt_vocab, device,
        model_cls=TransformerModel, model_kwargs=None,
        fixed_lr=None, log_grads=False, save=None):
    """
    Generic train + eval loop. Returns (model, bleu).
    All metrics logged to the current wandb run.
    """
    if model_kwargs is None: model_kwargs = {}
    model = model_cls(**model_kwargs).to(device)
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  [{name}] {n:,} parameters")

    crit  = LabelSmoothingLoss(model_kwargs.get("tgt_vsz", 5893),
                               PAD_IDX, cfg["label_smooth"])
    opt, sched = make_opt(model, cfg, fixed_lr)
    scaler     = GradScaler(enabled=True)
    best_val, no_imp, step = float("inf"), 0, 0

    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()
        tl, step = train_epoch(model, train_l, crit, opt, sched,
                               scaler, device, cfg, step,
                               log_grads=log_grads, prefix=f"{name}/")
        gc.collect(); torch.cuda.empty_cache()
        vl   = val_loss(model, val_l, crit, device)
        lr_  = sched.get_last_lr() if sched else fixed_lr
        conf = confidence_on_batch(model, train_l, tgt_vocab, device)

        wandb.log({
            f"{name}/train_loss": tl,
            f"{name}/val_loss":   vl,
            f"{name}/train_ppl":  math.exp(min(tl, 20)),
            f"{name}/val_ppl":    math.exp(min(vl, 20)),
            f"{name}/lr":         lr_,
            f"{name}/confidence": conf,
            "epoch": epoch,
        })
        print(f"  [{name}] ep{epoch:2d} | "
              f"tr {tl:.3f} vl {vl:.3f} ppl {math.exp(min(vl,20)):.1f} "
              f"lr {lr_:.2e} conf {conf:.3f} | {time.time()-t0:.0f}s")

        if vl < best_val:
            best_val, no_imp = vl, 0
            if save: torch.save(model.state_dict(), save)
        else:
            no_imp += 1
            if no_imp >= 6:
                print(f"  [{name}] early stop @ epoch {epoch}")
                break

    if save and os.path.exists(save):
        model.load_state_dict(torch.load(save, map_location=device,
                                         weights_only=True))
    bleu = bleu_score(model, test_l, tgt_vocab, device)
    wandb.log({f"{name}/test_bleu": bleu, f"{name}/best_val": best_val})
    print(f"  [{name}] BLEU = {bleu:.2f}")
    return model, bleu


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 2.1 — Noam Scheduler vs Fixed LR
# ─────────────────────────────────────────────────────────────────────────────

def exp21(train_l, val_l, test_l, src_vocab, tgt_vocab, device):
    print("\n── Exp 2.1: Noam vs Fixed LR ──")
    with wandb.init(project=WANDB_PROJECT, name="2.1-noam-vs-fixed-lr",
                    config={**BASE, "exp": "2.1"}) as wrun:

        mkw = dict(src_vsz=len(src_vocab), tgt_vsz=len(tgt_vocab), cfg=BASE)

        _, b_noam = run("noam", BASE, train_l, val_l, test_l, tgt_vocab, device,
                        model_kwargs=mkw, fixed_lr=None, save="exp21_noam.pt")

        _, b_fix  = run("fixed_lr", BASE, train_l, val_l, test_l, tgt_vocab, device,
                        model_kwargs=mkw, fixed_lr=1e-4, save="exp21_fixed.pt")

        # LR schedule plot
        steps = list(range(1, 8001))
        noam_lrs = [(BASE["d_model"]**-0.5) *
                    min(s**-0.5, s * BASE["warmup"]**-1.5) for s in steps]
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(steps, noam_lrs, label="Noam", color="steelblue")
        ax.axhline(1e-4, color="tomato", ls="--", label="Fixed 1e-4")
        ax.axvline(BASE["warmup"], color="gray", ls=":", alpha=0.6,
                   label=f"warmup={BASE['warmup']}")
        ax.set(xlabel="Step", ylabel="LR",
               title="Noam Schedule vs Fixed LR")
        ax.legend(); ax.grid(alpha=0.3)
        plt.tight_layout()
        wandb.log({"2.1/lr_curve": wandb.Image(fig)})
        plt.close(fig)

        wandb.log({"summary/noam_bleu": b_noam, "summary/fixed_bleu": b_fix})
        print(f"\n[2.1] Noam BLEU={b_noam:.2f}  Fixed BLEU={b_fix:.2f}")


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 2.2 — Scaling Factor 1/√dk Ablation
# ─────────────────────────────────────────────────────────────────────────────

def exp22(train_l, val_l, test_l, src_vocab, tgt_vocab, device):
    print("\n── Exp 2.2: Scaling Factor Ablation ──")
    with wandb.init(project=WANDB_PROJECT, name="2.2-scaling-ablation",
                    config={**BASE, "exp": "2.2"}) as wrun:

        cfg10 = {**BASE, "epochs": 10}
        mkw_s = dict(src_vsz=len(src_vocab), tgt_vsz=len(tgt_vocab), cfg=cfg10)
        mkw_n = dict(src_vsz=len(src_vocab), tgt_vsz=len(tgt_vocab), cfg=cfg10)

        print("  Training WITH scaling …")
        _, b_s = run("scaled", cfg10, train_l, val_l, test_l, tgt_vocab, device,
                     model_cls=TransformerModel, model_kwargs=mkw_s,
                     log_grads=True, save="exp22_scaled.pt")

        print("  Training WITHOUT scaling …")
        _, b_n = run("no_scale", cfg10, train_l, val_l, test_l, tgt_vocab, device,
                     model_cls=TransformerNoScale, model_kwargs=mkw_n,
                     log_grads=True, save="exp22_noscale.pt")

        wandb.log({"summary/scaled_bleu": b_s, "summary/noscale_bleu": b_n})
        print(f"\n[2.2] Scaled BLEU={b_s:.2f}  No-scale BLEU={b_n:.2f}")


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 2.3 — Attention Head Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def exp23(train_l, val_l, test_l, src_vocab, tgt_vocab, device):
    print("\n── Exp 2.3: Attention Head Visualisation ──")
    with wandb.init(project=WANDB_PROJECT, name="2.3-attention-heads",
                    config={**BASE, "exp": "2.3"}) as wrun:

        model = TransformerModel(len(src_vocab), len(tgt_vocab), BASE).to(device)

        if os.path.exists("transformer_best.pt"):
            model.load_state_dict(torch.load("transformer_best.pt",
                                             map_location=device,
                                             weights_only=True))
            print("  Loaded transformer_best.pt")
        else:
            print("  No checkpoint found — training 5 epochs …")
            crit = LabelSmoothingLoss(len(tgt_vocab), PAD_IDX, BASE["label_smooth"])
            opt, sched = make_opt(model, BASE)
            scaler = GradScaler(enabled=True)
            step = 0
            for ep in range(5):
                tl, step = train_epoch(model, train_l, crit, opt, sched,
                                       scaler, device, BASE, step)
                print(f"  ep{ep+1} train_loss={tl:.3f}")

        model.eval()

        # Tokenise a German sentence
        sentence = "Ein Mann sitzt auf einer Bank."
        import spacy
        nlp = spacy.load("de_core_news_sm")
        tokens = ["<sos>"] + [t.text.lower() for t in nlp(sentence)] + ["<eos>"]
        ids = torch.tensor(
            [[src_vocab.stoi.get(t, 1) for t in tokens]],
            dtype=torch.long, device=device,
        )

        # Get attention from last encoder layer
        attn = model.last_enc_attn(ids)   # (1, 8, T, T)
        A    = attn[0].cpu().float().numpy()
        T    = len(tokens)

        # All-heads grid
        fig, axes = plt.subplots(2, 4, figsize=(22, 10))
        fig.suptitle(f'Last Encoder Layer — All 8 Heads\n"{sentence}"', fontsize=13)
        for hi, ax in enumerate(axes.flatten()):
            im = ax.imshow(A[hi, :T, :T], cmap="Blues", aspect="auto")
            ax.set_title(f"Head {hi+1}")
            ax.set_xticks(range(T)); ax.set_xticklabels(tokens, rotation=45,
                                                         ha="right", fontsize=8)
            ax.set_yticks(range(T)); ax.set_yticklabels(tokens, fontsize=8)
            plt.colorbar(im, ax=ax, fraction=0.046)
        plt.tight_layout()
        wandb.log({"2.3/all_heads": wandb.Image(fig)})
        plt.savefig("attention_all_heads.png", dpi=120, bbox_inches="tight")
        plt.close(fig)

        # Individual head + entropy
        entropies = {}
        for hi in range(8):
            fig2, ax2 = plt.subplots(figsize=(6, 5))
            im = ax2.imshow(A[hi, :T, :T], cmap="Blues", aspect="auto")
            ax2.set_title(f"Head {hi+1}")
            ax2.set_xticks(range(T)); ax2.set_xticklabels(tokens, rotation=45,
                                                            ha="right", fontsize=9)
            ax2.set_yticks(range(T)); ax2.set_yticklabels(tokens, fontsize=9)
            plt.colorbar(im, ax=ax2)
            plt.tight_layout()
            wandb.log({f"2.3/head_{hi+1}": wandb.Image(fig2)})
            plt.close(fig2)

            w   = np.clip(A[hi, :T, :T], 1e-9, 1.0)
            ent = float(-np.sum(w * np.log(w), axis=-1).mean())
            entropies[f"2.3/head_{hi+1}_entropy"] = ent
            print(f"  Head {hi+1} entropy: {ent:.4f}")

        wandb.log(entropies)
        print("[2.3] Done — heatmaps logged to W&B")


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 2.4 — Sinusoidal PE vs Learned PE
# ─────────────────────────────────────────────────────────────────────────────

def exp24(train_l, val_l, test_l, src_vocab, tgt_vocab, device):
    print("\n── Exp 2.4: Sinusoidal vs Learned PE ──")
    with wandb.init(project=WANDB_PROJECT, name="2.4-pe-comparison",
                    config={**BASE, "exp": "2.4"}) as wrun:

        mkw = dict(src_vsz=len(src_vocab), tgt_vsz=len(tgt_vocab), cfg=BASE)

        _, b_sin = run("sinusoidal", BASE, train_l, val_l, test_l, tgt_vocab, device,
                       model_kwargs={**mkw, "learned_pe": False},
                       save="exp24_sin.pt")

        _, b_lpe = run("learned_pe", BASE, train_l, val_l, test_l, tgt_vocab, device,
                       model_kwargs={**mkw, "learned_pe": True},
                       save="exp24_learned.pt")

        wandb.log({"summary/sin_bleu": b_sin, "summary/learned_bleu": b_lpe})
        print(f"\n[2.4] Sinusoidal BLEU={b_sin:.2f}  Learned BLEU={b_lpe:.2f}")


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 2.5 — Label Smoothing
# ─────────────────────────────────────────────────────────────────────────────

def exp25(train_l, val_l, test_l, src_vocab, tgt_vocab, device):
    print("\n── Exp 2.5: Label Smoothing ε=0.1 vs ε=0.0 ──")
    with wandb.init(project=WANDB_PROJECT, name="2.5-label-smoothing",
                    config={**BASE, "exp": "2.5"}) as wrun:

        mkw = dict(src_vsz=len(src_vocab), tgt_vsz=len(tgt_vocab), cfg=BASE)

        cfg_s  = {**BASE, "label_smooth": 0.1}
        cfg_ce = {**BASE, "label_smooth": 0.0}

        _, b_s  = run("smooth_0.1", cfg_s,  train_l, val_l, test_l, tgt_vocab, device,
                      model_kwargs={**mkw, "cfg": cfg_s},  save="exp25_smooth.pt")

        _, b_ce = run("smooth_0.0", cfg_ce, train_l, val_l, test_l, tgt_vocab, device,
                      model_kwargs={**mkw, "cfg": cfg_ce}, save="exp25_ce.pt")

        wandb.log({"summary/smooth_bleu": b_s, "summary/ce_bleu": b_ce})
        print(f"\n[2.5] Smoothed BLEU={b_s:.2f}  CE BLEU={b_ce:.2f}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    global WANDB_PROJECT

    p = argparse.ArgumentParser()
    p.add_argument("--exp", default="all",
                   choices=["all","2.1","2.2","2.3","2.4","2.5"])
    p.add_argument("--project", default=WANDB_PROJECT)
    p.add_argument("--epochs",  type=int, default=None)
    args = p.parse_args()

    # Force unbuffered output on Windows
    sys.stdout.reconfigure(line_buffering=True)

    WANDB_PROJECT = args.project
    if args.epochs:
        BASE["epochs"] = args.epochs

    print(f"[main] exp={args.exp}  epochs={BASE['epochs']}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[main] Device : {device}")
    if device.type == "cuda":
        print(f"[main] GPU    : {torch.cuda.get_device_name(0)}")

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    print("[main] Loading data …")
    sys.stdout.flush()
    try:
        train_l, val_l, test_l, src_vocab, tgt_vocab = build_vocab_and_loaders(
            batch_size=BASE["batch"], max_len=BASE["max_len"], num_workers=0,
        )
        print(f"[main] Data OK — src={len(src_vocab)} tgt={len(tgt_vocab)}")
    except Exception as e:
        print(f"[main] ERROR loading data: {e}")
        traceback.print_exc()
        sys.exit(1)

    print("[main] Checking wandb …")
    sys.stdout.flush()
    try:
        import wandb as _wb
        print(f"[main] wandb version: {_wb.__version__}")
    except Exception as e:
        print(f"[main] wandb import error: {e}")
        sys.exit(1)

    dispatch = {
        "2.1": exp21, "2.2": exp22, "2.3": exp23,
        "2.4": exp24, "2.5": exp25,
    }

    to_run = list(dispatch.keys()) if args.exp == "all" else [args.exp]
    for k in to_run:
        print(f"\n[main] ── Running experiment {k} ──")
        sys.stdout.flush()
        try:
            dispatch[k](train_l, val_l, test_l, src_vocab, tgt_vocab, device)
        except Exception as e:
            print(f"[main] ERROR in exp {k}: {e}")
            traceback.print_exc()

    print(f"\n[main] ✓ Done. View at https://wandb.ai — project: {WANDB_PROJECT}")


if __name__ == "__main__":
    main()