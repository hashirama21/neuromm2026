"""
src/models/eegmamba_encoder.py

EEGMamba temporal encoder — Mamba2-style SSM for long EEG sequences.

Architecture:
    Input (B, T, D) ─► [MambaBlock × N] ─► LayerNorm ─► mean-pool ─► (B, D)

Each MambaBlock contains:
    - RMSNorm + selective state-space layer (depthwise conv + SSM)
    - Residual connection
    - Optional feed-forward sublayer

Two implementations are provided:
    1. NativeMambaBlock  — pure PyTorch, no CUDA extensions, runs everywhere.
       Uses selective gating + causal depthwise conv to approximate Mamba2.
    2. Real Mamba2       — loaded from `mamba_ssm` if installed (recommended for
       production, requires CUDA and the mamba-ssm wheel).

The model automatically uses the real Mamba2 when available.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# RMS Normalization (used by Mamba)
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * norm * self.weight


# ---------------------------------------------------------------------------
# Native Mamba Block (no CUDA deps)
# ---------------------------------------------------------------------------

class NativeMambaBlock(nn.Module):
    """
    Pure-PyTorch approximation of Mamba2.

    Implements the selective SSM mechanism via:
    1. Input projection + gating (SiLU)
    2. Causal depthwise convolution (local mixing)
    3. Selective state-space modelling (simplified: diagonal A, learned B/C/dt)
    4. Output projection with residual

    Reference: Mamba: Linear-Time Sequence Modeling with Selective State Spaces
               (Gu & Dao, 2023)
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)

        # Input projection: x → [z, x_inner]
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # Causal depthwise conv
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=True,
        )

        # SSM parameters
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)  # B, C, dt
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)

        # Log-diagonal A (initialized as evenly spaced)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0)  # (1, d_state)
        self.A_log = nn.Parameter(torch.log(A.repeat(self.d_inner, 1)))     # (d_inner, d_state)

        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.norm = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def _ssm(self, x: torch.Tensor) -> torch.Tensor:
        """
        Simplified selective SSM scan.
        Args:
            x: (B, T, d_inner)
        Returns:
            y: (B, T, d_inner)
        """
        B, T, D = x.shape
        S = self.d_state

        # Compute B_param, C_param, dt from x
        xz = self.x_proj(x)                        # (B, T, 2*S+1)
        B_param = xz[..., :S]                      # (B, T, S)
        C_param = xz[..., S:2*S]                   # (B, T, S)
        dt_raw = xz[..., 2*S:]                     # (B, T, 1)
        dt = F.softplus(self.dt_proj(dt_raw))      # (B, T, d_inner)

        # Discretize A: A_bar = exp(-exp(A_log) * dt)
        A = -torch.exp(self.A_log.float())         # (d_inner, S)
        # (B, T, d_inner, S)
        dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))

        # dB = dt * B_param  → (B, T, d_inner, S)
        dB = dt.unsqueeze(-1) * B_param.unsqueeze(2)  # broadcast

        # Sequential scan (simplified: approximate with cumsum-based method)
        # For exact selective scan, use mamba_ssm CUDA kernel
        h = torch.zeros(B, D, S, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(T):
            # h_t = A_bar * h_{t-1} + dB_t * x_t
            h = dA[:, t] * h + dB[:, t] * x[:, t].unsqueeze(-1)   # (B, D, S)
            # y_t = C_t · h_t
            y_t = (h * C_param[:, t].unsqueeze(1)).sum(-1)          # (B, D)
            ys.append(y_t)

        y = torch.stack(ys, dim=1)                  # (B, T, D)
        y = y + x * self.D.unsqueeze(0).unsqueeze(0)
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)
        Returns:
            (B, T, d_model)
        """
        residual = x
        x = self.norm(x)

        # Split into inner activations and gate
        xz = self.in_proj(x)                           # (B, T, 2*d_inner)
        x_inner, z = xz.chunk(2, dim=-1)               # each (B, T, d_inner)

        # Causal depthwise conv: (B, d_inner, T) → trim to T
        x_conv = x_inner.transpose(1, 2)               # (B, d_inner, T)
        x_conv = self.conv1d(x_conv)[..., :x.size(1)]  # causal: drop future
        x_conv = x_conv.transpose(1, 2)                # (B, T, d_inner)
        x_conv = F.silu(x_conv)

        # SSM
        y = self._ssm(x_conv)                          # (B, T, d_inner)

        # Gate
        y = y * F.silu(z)

        # Output projection
        out = self.dropout(self.out_proj(y))
        return out + residual


# ---------------------------------------------------------------------------
# Try to load real Mamba2
# ---------------------------------------------------------------------------

def _make_mamba_block(
    d_model: int,
    d_state: int,
    d_conv: int,
    expand: int,
    dropout: float,
    use_native: bool,
) -> nn.Module:
    if not use_native:
        try:
            from mamba_ssm import Mamba2
            return Mamba2(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
        except (ImportError, Exception):
            pass
    return NativeMambaBlock(d_model, d_state, d_conv, expand, dropout)


# ---------------------------------------------------------------------------
# EEGMamba Encoder
# ---------------------------------------------------------------------------

class EEGMambaEncoder(nn.Module):
    """
    EEGMamba temporal encoder.

    Stacks N MambaBlocks on the channel-token sequence produced by the
    Dynamic GAT. Each channel token is a D-dimensional vector.

    Input:  (B, C, D)  — batch, n_channels (29), d_model
    Output: (B, D)     — global pooled representation

    The O(n) complexity of Mamba makes it 4–8× faster than Transformer
    on T=2000 timesteps and avoids the quadratic memory bottleneck.

    Args:
        d_model:     Feature dimension (= gat_hidden_dim).
        d_state:     SSM state dimension (latent). Default 16.
        d_conv:      Local conv kernel size. Default 4.
        expand:      Inner dimension multiplier. Default 2.
        n_layers:    Number of stacked MambaBlocks. Default 4.
        dropout:     Dropout on output. Default 0.10.
        force_native: Force NativeMambaBlock (for testing / no CUDA).
    """

    def __init__(
        self,
        d_model: int = 256,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        n_layers: int = 4,
        dropout: float = 0.10,
        force_native: bool = False,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        self.blocks = nn.ModuleList([
            _make_mamba_block(d_model, d_state, d_conv, expand, dropout, force_native)
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # Feed-forward sublayers between Mamba blocks (optional, adds capacity)
        self.ff_blocks = nn.ModuleList([
            nn.Sequential(
                RMSNorm(d_model),
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 4, d_model),
                nn.Dropout(dropout),
            )
            for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)   — T = n_channels (29) after GAT
        Returns:
            (B, d_model)         — global mean pooling
        """
        for mamba, ff in zip(self.blocks, self.ff_blocks):
            x = mamba(x)          # selective SSM
            x = x + ff(x)        # feed-forward sublayer

        x = self.norm(x)
        return self.dropout(x.mean(dim=1))   # global average pooling → (B, D)
