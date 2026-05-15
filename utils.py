"""
utils.py — Label Smoothing, Noam LR Scheduler, and Masking Utilities
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────
# 1.  Label-Smoothing Loss
# ──────────────────────────────────────────────

class LabelSmoothingLoss(nn.Module):
    """
    Cross-entropy with label smoothing  (ε_ls = 0.1 by default).

    Instead of putting 100% probability on the correct class,
    the target distribution places (1 - ε) on the correct class
    and ε / (V - 1) on all others (ignoring the padding token).

    Args:
        vocab_size  : size of the target vocabulary
        padding_idx : index of the <pad> token (ignored in loss)
        smoothing   : label-smoothing coefficient ε_ls
    """

    def __init__(
        self,
        vocab_size: int,
        padding_idx: int = 0,
        smoothing: float = 0.1,
    ):
        super().__init__()
        self.vocab_size  = vocab_size
        self.padding_idx = padding_idx
        self.smoothing   = smoothing
        self.confidence  = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits  : (N, vocab_size)  raw (un-softmaxed) scores
            targets : (N,)             ground-truth class indices
        Returns:
            Scalar loss (mean over non-padding tokens).
        """
        log_probs = F.log_softmax(logits, dim=-1)  # (N, V)

        # Build smooth target distribution
        with torch.no_grad():
            smooth_dist = torch.full_like(log_probs, self.smoothing / (self.vocab_size - 2))
            smooth_dist.scatter_(1, targets.unsqueeze(1), self.confidence)
            smooth_dist[:, self.padding_idx] = 0.0

            # Zero out rows where the true target is padding
            mask = targets.eq(self.padding_idx)
            smooth_dist[mask] = 0.0

        loss = -(smooth_dist * log_probs).sum(dim=-1)  # (N,)

        # Average only over non-padding positions
        non_pad = (~mask).sum().clamp(min=1)
        return loss.sum() / non_pad


# ──────────────────────────────────────────────
# 2.  Noam Learning-Rate Scheduler
# ──────────────────────────────────────────────

class NoamScheduler:
    """
    Implements the Noam learning-rate schedule from "Attention Is All
    You Need":

        lrate = d_model^{-0.5} · min(step^{-0.5}, step · warmup^{-1.5})

    Usage:
        optimizer = torch.optim.Adam(model.parameters(), lr=1, betas=(0.9, 0.98), eps=1e-9)
        scheduler = NoamScheduler(optimizer, d_model=512, warmup_steps=4000)

        for step in training_loop:
            ...
            optimizer.step()
            scheduler.step()
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        d_model: int = 512,
        warmup_steps: int = 4000,
        factor: float = 1.0,
    ):
        self.optimizer     = optimizer
        self.d_model       = d_model
        self.warmup_steps  = warmup_steps
        self.factor        = factor
        self._step_num     = 0
        # Initialise optimizer lr to something reasonable
        self._update_lr()

    def step(self):
        self._step_num += 1
        self._update_lr()

    def _compute_lr(self, step: int) -> float:
        if step == 0:
            step = 1  # avoid division by zero
        return (
            self.factor
            * (self.d_model ** -0.5)
            * min(step ** -0.5, step * (self.warmup_steps ** -1.5))
        )

    def _update_lr(self):
        lr = self._compute_lr(self._step_num)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def get_last_lr(self) -> float:
        return self._compute_lr(self._step_num)

    @property
    def current_step(self) -> int:
        return self._step_num


# ──────────────────────────────────────────────
# 3.  Masking utilities (stand-alone, reusable)
# ──────────────────────────────────────────────

def make_src_mask(src: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    """
    Padding mask for the encoder.

    Returns:
        (batch, 1, 1, src_len)  Bool tensor, True where token == pad_idx.
    """
    return (src == pad_idx).unsqueeze(1).unsqueeze(2)


def make_tgt_mask(tgt: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    """
    Combined causal (look-ahead) + padding mask for the decoder.

    Returns:
        (batch, 1, tgt_len, tgt_len)  Bool tensor, True where attention
        should be blocked.
    """
    tgt_len = tgt.size(1)

    # Upper-triangular causal mask (True = future position)
    causal = torch.triu(
        torch.ones(tgt_len, tgt_len, device=tgt.device, dtype=torch.bool),
        diagonal=1,
    )                                                         # (T, T)

    # Padding mask
    pad_mask = (tgt == pad_idx).unsqueeze(1).unsqueeze(2)    # (B, 1, 1, T)

    # Expand causal to batch and combine
    return causal.unsqueeze(0).unsqueeze(0) | pad_mask        # (B, 1, T, T)


def make_causal_mask(size: int, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """
    Pure causal mask (no padding).

    Returns:
        (1, 1, size, size)  Bool tensor.
    """
    return torch.triu(
        torch.ones(1, 1, size, size, device=device, dtype=torch.bool),
        diagonal=1,
    )
