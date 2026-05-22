"""
src/models/dynamic_gat.py

Dynamic Graph Attention Network for EEG inter-channel interactions.

Key design decision: the adjacency matrix is NOT fixed by the 10-20
electrode topology. Instead it is computed dynamically from channel
embeddings via scaled dot-product attention:

    A_ij^(t) = softmax( Q(X_i) · K(X_j)^T / sqrt(d) )

This allows the model to learn which channel pairs are relevant for
each specific EEG pattern, rather than relying on physical proximity.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicGATLayer(nn.Module):
    """
    Single dynamic GAT layer with multi-head self-attention over channels.

    Args:
        in_dim:   Input feature dimension per channel.
        out_dim:  Output feature dimension per channel.
        n_heads:  Number of attention heads.
        dropout:  Dropout on attention weights.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        n_heads: int = 8,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        assert out_dim % n_heads == 0, f"out_dim ({out_dim}) must be divisible by n_heads ({n_heads})"

        self.n_heads = n_heads
        self.head_dim = out_dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(in_dim, out_dim, bias=False)
        self.k_proj = nn.Linear(in_dim, out_dim, bias=False)
        self.v_proj = nn.Linear(in_dim, out_dim, bias=False)
        self.out_proj = nn.Linear(out_dim, out_dim)

        self.attn_drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_dim)

        self.residual = (
            nn.Linear(in_dim, out_dim, bias=False)
            if in_dim != out_dim else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:  x: (B, C, in_dim)
        Returns: (B, C, out_dim)
        """
        B, C, _ = x.shape
        H, Dh = self.n_heads, self.head_dim

        Q = self.q_proj(x).view(B, C, H, Dh).transpose(1, 2)   # (B, H, C, Dh)
        K = self.k_proj(x).view(B, C, H, Dh).transpose(1, 2)
        V = self.v_proj(x).view(B, C, H, Dh).transpose(1, 2)

        # Dynamic adjacency: A = softmax(Q K^T / sqrt(d))  →  (B, H, C, C)
        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, V)                              # (B, H, C, Dh)
        out = out.transpose(1, 2).contiguous().view(B, C, -1)   # (B, C, out_dim)
        out = self.out_proj(out)

        return self.norm(out + self.residual(x))


class DynamicGAT(nn.Module):
    """
    Stack of DynamicGATLayers with feed-forward sublayers.

    Processes (B, C, in_dim) → (B, C, hidden_dim) enriching each
    channel embedding with inter-channel context.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        n_layers: int = 2,
        n_heads: int = 8,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()

        dims = [in_dim] + [hidden_dim] * n_layers
        self.gat_layers = nn.ModuleList([
            DynamicGATLayer(dims[i], dims[i + 1], n_heads, dropout)
            for i in range(n_layers)
        ])
        self.ff_layers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.Dropout(dropout),
            )
            for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:  x: (B, C, in_dim)
        Returns: (B, C, hidden_dim)
        """
        for gat, ff in zip(self.gat_layers, self.ff_layers):
            x = gat(x)
            x = x + ff(x)
        return x
