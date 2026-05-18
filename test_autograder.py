"""
test_autograder.py
==================
Mirrors every criterion listed in the automated evaluation pipeline so you
can verify locally before submitting to Gradescope.

Run:
    python test_autograder.py
"""

import math
import sys
import traceback
import torch
import torch.nn as nn

# ── colour helpers ────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
RESET  = "\033[0m"

pass_count = 0
fail_count = 0


def ok(name: str):
    global pass_count
    pass_count += 1
    print(f"  {GREEN}✓ PASS{RESET}  {name}")


def fail(name: str, reason: str = ""):
    global fail_count
    fail_count += 1
    print(f"  {RED}✗ FAIL{RESET}  {name}" + (f"  →  {reason}" if reason else ""))


def section(title: str):
    print(f"\n{YELLOW}{'─'*60}{RESET}")
    print(f"{YELLOW}  {title}{RESET}")
    print(f"{YELLOW}{'─'*60}{RESET}")


# ─────────────────────────────────────────────────────────────────────────────
# Import modules under test
# ─────────────────────────────────────────────────────────────────────────────
try:
    from model import (
        ScaledDotProductAttention,
        MultiHeadAttention,
        PositionalEncoding,
        Encoder,
        Decoder,
    )
    from utils import NoamScheduler, LabelSmoothingLoss
except Exception as e:
    print(f"{RED}Import error: {e}{RESET}")
    traceback.print_exc()
    sys.exit(1)


# =============================================================================
# SECTION 1 — Multi-Head Attention  [10M]
# =============================================================================
section("1 · Multi-Head Attention")

B, H, T_q, T_k, D_K = 2, 8, 10, 12, 64
D_MODEL = H * D_K  # 512

# ── 1-a  Scaled dot-product output shape ─────────────────────────────────────
try:
    sdpa = ScaledDotProductAttention(dropout=0.0)
    Q = torch.randn(B, H, T_q, D_K)
    K = torch.randn(B, H, T_k, D_K)
    V = torch.randn(B, H, T_k, D_K)
    out, w = sdpa(Q, K, V)
    assert out.shape == (B, H, T_q, D_K), f"got {out.shape}"
    assert w.shape   == (B, H, T_q, T_k), f"got {w.shape}"
    ok("ScaledDotProduct output shape")
except Exception as e:
    fail("ScaledDotProduct output shape", str(e))

# ── 1-b  Attention weights sum to 1 over key dimension ───────────────────────
try:
    sdpa = ScaledDotProductAttention(dropout=0.0)
    Q = torch.randn(B, H, T_q, D_K)
    K = torch.randn(B, H, T_k, D_K)
    V = torch.randn(B, H, T_k, D_K)
    _, w = sdpa(Q, K, V)
    row_sums = w.sum(dim=-1)                          # (B, H, T_q)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), \
        f"max deviation {(row_sums - 1).abs().max().item():.2e}"
    ok("Attention weights sum to 1 over key dim")
except Exception as e:
    fail("Attention weights sum to 1 over key dim", str(e))

# ── 1-c  Masked positions receive zero attention weight ──────────────────────
try:
    sdpa = ScaledDotProductAttention(dropout=0.0)
    Q = torch.randn(1, 1, 4, D_K)
    K = torch.randn(1, 1, 4, D_K)
    V = torch.randn(1, 1, 4, D_K)
    # mask the last two key positions
    mask = torch.zeros(1, 1, 4, 4, dtype=torch.bool)
    mask[:, :, :, 2:] = True
    _, w = sdpa(Q, K, V, mask=mask)
    assert w[:, :, :, 2:].abs().max().item() < 1e-6, \
        f"masked weights non-zero: {w[:, :, :, 2:].abs().max().item()}"
    ok("Masked positions receive zero attention weight")
except Exception as e:
    fail("Masked positions receive zero attention weight", str(e))

# ── 1-d  MHA output shape under varying d_model / num_heads ──────────────────
try:
    for dm, nh in [(256, 4), (512, 8), (512, 16)]:
        mha = MultiHeadAttention(d_model=dm, num_heads=nh, dropout=0.0)
        x   = torch.randn(B, T_q, dm)
        out = mha(x, x, x)
        assert out.shape == (B, T_q, dm), f"dm={dm} nh={nh}: got {out.shape}"
    ok("MHA output shape under varying d_model / num_heads")
except Exception as e:
    fail("MHA output shape under varying d_model / num_heads", str(e))

# ── 1-e  Causal mask produces different output than unmasked ─────────────────
try:
    torch.manual_seed(0)
    mha = MultiHeadAttention(d_model=D_MODEL, num_heads=H, dropout=0.0)
    x   = torch.randn(1, 8, D_MODEL)
    T   = 8
    causal = torch.triu(torch.ones(1, 1, T, T, dtype=torch.bool), diagonal=1)

    out_unmasked = mha(x, x, x, mask=None)
    out_masked   = mha(x, x, x, mask=causal)
    diff = (out_masked - out_unmasked).abs().max().item()
    assert diff > 1e-4, f"outputs too similar (diff={diff:.2e})"
    ok("Causal mask produces different output than unmasked")
except Exception as e:
    fail("Causal mask produces different output than unmasked", str(e))


# =============================================================================
# SECTION 2 — Positional Encoding  [10M]
# =============================================================================
section("2 · Positional Encoding")

D_MODEL_PE = 512
MAX_LEN    = 100

pe_module = PositionalEncoding(d_model=D_MODEL_PE, max_len=MAX_LEN, dropout=0.0)
pe_buf    = pe_module.pe  # (1, max_len, d_model)

# ── 2-a  Output shape preservation ───────────────────────────────────────────
try:
    x_in  = torch.randn(3, 20, D_MODEL_PE)
    x_out = pe_module(x_in)
    assert x_out.shape == x_in.shape, f"got {x_out.shape}"
    ok("PE output shape preserved")
except Exception as e:
    fail("PE output shape preserved", str(e))

# ── 2-b  Even-indexed dims == sin(0) = 0 at position 0 ───────────────────────
try:
    even_vals = pe_buf[0, 0, 0::2]   # position 0, even dims
    assert torch.allclose(even_vals, torch.zeros_like(even_vals), atol=1e-6), \
        f"max={even_vals.abs().max().item():.2e}"
    ok("PE even dims == sin(0)=0 at position 0")
except Exception as e:
    fail("PE even dims == sin(0)=0 at position 0", str(e))

# ── 2-c  Odd-indexed dims == cos(0) = 1 at position 0 ────────────────────────
try:
    odd_vals = pe_buf[0, 0, 1::2]    # position 0, odd dims
    assert torch.allclose(odd_vals, torch.ones_like(odd_vals), atol=1e-6), \
        f"max dev={(odd_vals - 1).abs().max().item():.2e}"
    ok("PE odd dims == cos(0)=1 at position 0")
except Exception as e:
    fail("PE odd dims == cos(0)=1 at position 0", str(e))

# ── 2-d  Formula correctness at an arbitrary (pos, dim) pair ─────────────────
try:
    # Paper: PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    # arange(0,d,2) gives values [0,2,4,6,...] which ARE the "2i" indices directly
    # For even dim=6: 2i=6, so sin(pos / 10000^(6/d_model))
    pos, dim = 17, 6                  # even dimension
    expected = math.sin(pos / (10000 ** (dim / D_MODEL_PE)))
    actual   = pe_buf[0, pos, dim].item()
    assert abs(actual - expected) < 1e-5, f"even dim={dim}: expected {expected:.6f}, got {actual:.6f}"

    # For odd dim=7: 2i+1=7 → 2i=6, so cos(pos / 10000^(6/d_model))
    pos2, dim2 = 23, 7               # odd dimension
    expected2  = math.cos(pos2 / (10000 ** ((dim2 - 1) / D_MODEL_PE)))
    actual2    = pe_buf[0, pos2, dim2].item()
    assert abs(actual2 - expected2) < 1e-5, f"odd dim={dim2}: expected {expected2:.6f}, got {actual2:.6f}"
    ok("PE formula correct at arbitrary (pos, dim)")
except Exception as e:
    fail("PE formula correct at arbitrary (pos, dim)", str(e))

# ── 2-e  Buffer registered (not a trainable parameter) ───────────────────────
try:
    param_names   = {n for n, _ in pe_module.named_parameters()}
    buffer_names  = {n for n, _ in pe_module.named_buffers()}
    assert "pe" not in param_names,  "'pe' should not be a Parameter"
    assert "pe" in buffer_names,     "'pe' must be registered as a buffer"
    ok("PE registered as buffer, not trainable parameter")
except Exception as e:
    fail("PE registered as buffer, not trainable parameter", str(e))


# =============================================================================
# SECTION 3 — Noam LR Scheduler  [10M]
# =============================================================================
section("3 · Noam LR Scheduler")

D_MODEL_SCHED  = 512
WARMUP_STEPS   = 4000

def _make_scheduler(d_model=D_MODEL_SCHED, warmup=WARMUP_STEPS):
    dummy_model = nn.Linear(10, 10)
    opt = torch.optim.Adam(dummy_model.parameters(), lr=1.0,
                           betas=(0.9, 0.98), eps=1e-9)
    sched = NoamScheduler(opt, d_model=d_model, warmup_steps=warmup)
    return sched

def _lrs_for_steps(steps):
    sched = _make_scheduler()
    lrs = []
    for _ in range(steps):
        sched.step()
        lrs.append(sched.get_last_lr())
    return lrs

# ── 3-a  Monotonically increasing during warm-up ─────────────────────────────
try:
    lrs = _lrs_for_steps(WARMUP_STEPS)
    warmup_lrs = lrs[:WARMUP_STEPS - 1]
    assert all(warmup_lrs[i] < warmup_lrs[i+1] for i in range(len(warmup_lrs)-1)), \
        "LR not strictly increasing during warm-up"
    ok("LR monotonically increasing during warm-up")
except Exception as e:
    fail("LR monotonically increasing during warm-up", str(e))

# ── 3-b  Peak occurs within 10 steps of warmup_steps ────────────────────────
try:
    lrs = _lrs_for_steps(WARMUP_STEPS + 500)
    peak_step = int(torch.tensor(lrs).argmax().item()) + 1  # 1-indexed
    assert abs(peak_step - WARMUP_STEPS) <= 10, \
        f"peak at step {peak_step}, expected near {WARMUP_STEPS}"
    ok(f"Peak within 10 steps of warmup_steps (peak@{peak_step})")
except Exception as e:
    fail("Peak within 10 steps of warmup_steps", str(e))

# ── 3-c  Monotonically decreasing after warm-up ──────────────────────────────
try:
    lrs = _lrs_for_steps(WARMUP_STEPS + 2000)
    post = lrs[WARMUP_STEPS + 10 :]     # give a few steps of grace
    assert all(post[i] >= post[i+1] for i in range(len(post)-1)), \
        "LR not monotonically decreasing after warm-up"
    ok("LR monotonically decreasing after warm-up")
except Exception as e:
    fail("LR monotonically decreasing after warm-up", str(e))

# ── 3-d  Peak value matches closed-form formula ──────────────────────────────
try:
    lrs   = _lrs_for_steps(WARMUP_STEPS + 100)
    peak_actual = max(lrs)
    # closed-form peak ≈ d_model^{-0.5} * warmup^{-0.5}
    peak_formula = (D_MODEL_SCHED ** -0.5) * (WARMUP_STEPS ** -0.5)
    assert abs(peak_actual - peak_formula) / peak_formula < 0.02, \
        f"actual {peak_actual:.6f} vs formula {peak_formula:.6f}"
    ok("Peak LR matches closed-form formula")
except Exception as e:
    fail("Peak LR matches closed-form formula", str(e))

# ── 3-e  LR at step 1 matches formula ────────────────────────────────────────
try:
    sched = _make_scheduler()
    sched.step()
    lr_step1 = sched.get_last_lr()
    formula_step1 = (D_MODEL_SCHED ** -0.5) * min(1 ** -0.5, 1 * WARMUP_STEPS ** -1.5)
    assert abs(lr_step1 - formula_step1) / formula_step1 < 1e-5, \
        f"got {lr_step1:.8f}, expected {formula_step1:.8f}"
    ok("LR at step 1 matches formula")
except Exception as e:
    fail("LR at step 1 matches formula", str(e))


# =============================================================================
# SECTION 4 — Label Smoothing (bonus checks)
# =============================================================================
section("4 · Label Smoothing (bonus)")

try:
    V = 100
    ls = LabelSmoothingLoss(vocab_size=V, padding_idx=0, smoothing=0.1)
    logits  = torch.randn(8, V)
    targets = torch.randint(1, V, (8,))
    loss = ls(logits, targets)
    assert loss.item() > 0, "loss must be positive"
    assert loss.ndim == 0,  "loss must be scalar"
    ok("LabelSmoothingLoss returns positive scalar")
except Exception as e:
    fail("LabelSmoothingLoss returns positive scalar", str(e))

try:
    V = 100
    ls_smooth = LabelSmoothingLoss(vocab_size=V, padding_idx=0, smoothing=0.1)
    ls_hard   = LabelSmoothingLoss(vocab_size=V, padding_idx=0, smoothing=0.0)
    logits  = torch.randn(8, V)
    targets = torch.randint(1, V, (8,))
    loss_s = ls_smooth(logits, targets)
    loss_h = ls_hard(logits, targets)
    # smoothed loss should generally differ from hard loss
    assert abs(loss_s.item() - loss_h.item()) > 1e-6, "smoothing has no effect"
    ok("Label smoothing changes loss value vs hard targets")
except Exception as e:
    fail("Label smoothing changes loss value vs hard targets", str(e))

try:
    V = 50
    ls = LabelSmoothingLoss(vocab_size=V, padding_idx=0, smoothing=0.1)
    logits  = torch.randn(4, V)
    targets = torch.zeros(4, dtype=torch.long)   # all padding
    loss = ls(logits, targets)
    assert loss.item() == 0.0 or math.isnan(loss.item()) == False, \
        "unexpected behaviour on all-padding batch"
    ok("LabelSmoothingLoss handles all-padding batch gracefully")
except Exception as e:
    fail("LabelSmoothingLoss handles all-padding batch gracefully", str(e))


# =============================================================================
# Summary
# =============================================================================
total = pass_count + fail_count
print(f"\n{'═'*60}")
print(f"  Result: {GREEN}{pass_count} passed{RESET}  /  {RED}{fail_count} failed{RESET}  /  {total} total")
print(f"{'═'*60}\n")

if fail_count > 0:
    sys.exit(1)