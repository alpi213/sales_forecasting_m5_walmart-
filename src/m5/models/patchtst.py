"""PatchTST (Nie et al., ICLR 2023) written from scratch on top of `torch.nn`.

Architecture, for one series and a lookback window of length L:
  1. Instance normalisation: subtract the window mean and divide by the window std (RevIN
     without the affine part). Sales series differ by orders of magnitude; without this the
     model would spend its capacity on scale rather than shape. Statistics are re-applied at
     the output.
  2. Patching: the window is cut into overlapping patches of `patch_len` with stride `stride`,
     giving N = (L - patch_len) // stride + 1 tokens. Tokens are local sub-series, which is
     both cheaper (attention over N instead of L positions) and more informative than single
     time-steps.
  3. Each patch is projected linearly to d_model, a learnable positional embedding is added and,
     optionally, a learned per-series embedding (a light-weight way to give a channel-independent
     model some memory of *which* item it is looking at).
  4. A standard pre-norm Transformer encoder (`nn.TransformerEncoderLayer`, batch_first).
  5. A flatten + linear head maps the N x d_model tokens to the H forecast values.
  6. De-normalisation and a (sharpened) softplus so the forecast is strictly positive (needed
     for Tweedie); a window with no sales at all is mapped to ~0.

The model is channel-independent: every series is a separate sample with shared weights, which
is exactly the "global model" setting of the LightGBM baseline.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PatchTSTConfig:
    lookback: int = 224
    horizon: int = 28
    patch_len: int = 16
    stride: int = 8
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 128
    dropout: float = 0.2
    n_series: int = 0          # > 0 enables the per-series embedding
    eps: float = 1e-5

    @property
    def n_patches(self) -> int:
        return (self.lookback - self.patch_len) // self.stride + 1


class PatchTST(nn.Module):
    def __init__(self, cfg: PatchTSTConfig):
        super().__init__()
        if cfg.patch_len > cfg.lookback:
            raise ValueError("patch_len must be <= lookback")
        self.cfg = cfg
        self.patch_proj = nn.Linear(cfg.patch_len, cfg.d_model)
        self.pos_emb = nn.Parameter(torch.empty(1, cfg.n_patches, cfg.d_model))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        self.series_emb = nn.Embedding(cfg.n_series, cfg.d_model) if cfg.n_series > 0 else None
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)
        self.final_norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.n_patches * cfg.d_model, cfg.horizon),
        )

    def forward(self, x: torch.Tensor, series_idx: torch.Tensor | None = None) -> torch.Tensor:
        """x: (B, L) raw sales. Returns positive forecasts (B, H)."""
        if x.dim() != 2 or x.shape[1] != self.cfg.lookback:
            raise ValueError(f"expected input (B, {self.cfg.lookback}), got {tuple(x.shape)}")
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True, unbiased=False) + self.cfg.eps
        xn = (x - mean) / std                                          # (B, L)
        patches = xn.unfold(dimension=1, size=self.cfg.patch_len, step=self.cfg.stride)  # (B, N, P)
        z = self.patch_proj(patches) + self.pos_emb                   # (B, N, D)
        if self.series_emb is not None and series_idx is not None:
            z = z + self.series_emb(series_idx)[:, None, :]
        z = self.final_norm(self.encoder(z))
        out = self.head(z)                                             # (B, H), normalised scale
        out = out * std + mean
        # softplus with beta=4 keeps the forecast strictly positive (required by the Tweedie
        # loss) while behaving like identity above ~1 unit and reaching ~0 for negative inputs;
        # windows with no sales at all are forced to (almost) zero rather than the softplus floor.
        mu = F.softplus(out, beta=4.0) + 1e-6
        dormant = (x.abs().sum(dim=1, keepdim=True) == 0)
        return torch.where(dormant, torch.full_like(mu, 1e-6), mu)


def tweedie_deviance(y: torch.Tensor, mu: torch.Tensor, p: float = 1.1) -> torch.Tensor:
    """Mean Tweedie unit deviance for 1 < p < 2 (the LightGBM objective, so both models optimise
    the same quantity). mu must be strictly positive."""
    if not 1.0 < p < 2.0:
        raise ValueError("power must be in (1, 2)")
    term1 = torch.pow(y.clamp_min(0), 2 - p) / ((1 - p) * (2 - p))
    term2 = y * torch.pow(mu, 1 - p) / (1 - p)
    term3 = torch.pow(mu, 2 - p) / (2 - p)
    return 2.0 * (term1 - term2 + term3).mean()


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def receptive_field_note(cfg: PatchTSTConfig) -> str:
    return (
        f"lookback={cfg.lookback} patches={cfg.n_patches} (patch_len={cfg.patch_len}, stride={cfg.stride}) "
        f"d_model={cfg.d_model} heads={cfg.n_heads} layers={cfg.n_layers} "
        f"attention cost ~ {cfg.n_patches ** 2} vs {cfg.lookback ** 2} for point tokens "
        f"(x{cfg.lookback ** 2 / max(1, cfg.n_patches ** 2):.0f} cheaper); log2 = {math.log2(cfg.n_patches):.1f}"
    )
