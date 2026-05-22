"""
src/models/heads.py

All three track heads:

Track1Head  — MLP binary classifier (EEG only)
Track2Head  — Uncertainty-aware Gating + Cross-Attention (EEG + Video)
Track3Head  — Temporal Attention over channel features + Video late fusion
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


###############################################################################
# Track 1 Head
###############################################################################

class Track1Head(nn.Module):
    """
    Binary MLP head for IED detection (EEG only).

    Input:  (B, embed_dim)
    Output: (B,)  logit
    """

    def __init__(
        self,
        embed_dim: int = 256,
        hidden_dims: Optional[list[int]] = None,
        dropout: float = 0.40,
    ) -> None:
        super().__init__()
        hidden_dims = hidden_dims or [256, 128]

        layers: list[nn.Module] = []
        in_d = embed_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(in_d, h),
                nn.LayerNorm(h),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            in_d = h
        layers.append(nn.Linear(in_d, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, embed: torch.Tensor) -> torch.Tensor:
        return self.mlp(embed).squeeze(-1)   # (B,)


###############################################################################
# Track 2 Head — Uncertainty-aware Gating + Cross-Attention
###############################################################################

class VideoProjLayer(nn.Module):
    """Project one video backbone's features to a common target_dim."""

    def __init__(self, in_dim: int, target_dim: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, target_dim),
            nn.LayerNorm(target_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_frames, in_dim) or (B, 1, in_dim)
        return self.proj(x)


class UncertaintyGating(nn.Module):
    """
    MC-Dropout based uncertainty estimator.

    Runs the EEG embedding through a lightweight scorer K times with
    dropout active to estimate predictive entropy. The entropy is
    converted to a soft gate value in [0,1]:
        - gate → 0 : EEG is confident  → video is suppressed
        - gate → 1 : EEG is uncertain  → video cross-attention is activated

    This mirrors clinical practice: use the camera when EEG is ambiguous
    (e.g. movement artefacts), trust EEG when signal is clean.
    """

    def __init__(
        self,
        embed_dim: int,
        mc_passes: int = 5,
        dropout_rate: float = 0.30,
        entropy_threshold: float = 0.50,
        learnable_scale: bool = True,
    ) -> None:
        super().__init__()
        self.mc_passes = mc_passes
        self.mc_dropout = nn.Dropout(dropout_rate)
        self.scorer = nn.Sequential(
            nn.Linear(embed_dim, 64), nn.GELU(), nn.Linear(64, 1)
        )
        self.entropy_threshold = entropy_threshold
        self.scale = nn.Parameter(torch.ones(1)) if learnable_scale else None
        self.bias = nn.Parameter(torch.zeros(1)) if learnable_scale else None

    def forward(self, embed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            gate:    (B,) in [0,1]
            entropy: (B,) predictive entropy estimate
        """
        probs = []
        for _ in range(self.mc_passes):
            noisy = self.mc_dropout(embed)
            p = torch.sigmoid(self.scorer(noisy).squeeze(-1))
            probs.append(p)

        p_stack = torch.stack(probs, dim=0).mean(dim=0)   # (B,)

        eps = 1e-7
        entropy = -(
            p_stack * (p_stack + eps).log()
            + (1 - p_stack) * (1 - p_stack + eps).log()
        )   # (B,) ∈ [0, log(2) ≈ 0.693]

        # Normalise to [0,1] and apply threshold
        h_norm = entropy / 0.693
        if self.scale is not None:
            gate = torch.sigmoid(self.scale * (h_norm - self.entropy_threshold) + self.bias)
        else:
            gate = torch.sigmoid(10.0 * (h_norm - self.entropy_threshold))

        return gate, entropy


class CrossAttentionFusion(nn.Module):
    """
    Asymmetric cross-attention: EEG queries video keys/values.

    Q = EEG embed token  (B, 1, D)
    K = V = video tokens (B, n_frames, D)
    """

    def __init__(self, embed_dim: int, n_heads: int = 8, n_layers: int = 2, dropout: float = 0.10) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "norm_q":  nn.LayerNorm(embed_dim),
                "norm_kv": nn.LayerNorm(embed_dim),
                "attn":    nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, batch_first=True),
                "ff":      nn.Sequential(
                    nn.LayerNorm(embed_dim),
                    nn.Linear(embed_dim, embed_dim * 2), nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(embed_dim * 2, embed_dim),
                ),
            })
            for _ in range(n_layers)
        ])

    def forward(self, eeg_q: torch.Tensor, video_kv: torch.Tensor) -> torch.Tensor:
        """
        Args:
            eeg_q:    (B, 1, D)
            video_kv: (B, F, D)
        Returns:
            (B, 1, D)
        """
        x = eeg_q
        for layer in self.layers:
            q  = layer["norm_q"](x)
            kv = layer["norm_kv"](video_kv)
            attended, _ = layer["attn"](q, kv, kv)
            x = x + attended
            x = x + layer["ff"](x)
        return x


class Track2Head(nn.Module):
    """
    Track 2: Uncertainty-aware Gating + Cross-Attention + Binary Classifier.

    Pipeline:
        EEG embed → MC-Dropout → entropy → gate ∈ [0,1]
        Video features → project to embed_dim
        Cross-attention: EEG queries video (activated by gate)
        Fused embed → MLP classifier → logit
    """

    def __init__(
        self,
        embed_dim: int = 256,
        video_configs: Optional[dict] = None,
        target_dim: int = 256,
        n_ca_heads: int = 8,
        n_ca_layers: int = 2,
        mc_passes: int = 5,
        mc_dropout: float = 0.30,
        entropy_threshold: float = 0.50,
        learnable_gate_scale: bool = True,
        hidden_dims: Optional[list[int]] = None,
        dropout: float = 0.40,
    ) -> None:
        super().__init__()

        # Default: VideoMAE-Large + DINOv2-Large
        if video_configs is None:
            video_configs = {
                "videomae-large": {"n_frames": 1, "feat_dim": 1024},
                "dinov2-large":   {"n_frames": 8, "feat_dim": 1024},
            }

        # Video projectors: one per backbone
        self.video_projs = nn.ModuleDict({
            name: VideoProjLayer(vcfg["feat_dim"], target_dim)
            for name, vcfg in video_configs.items()
        })

        # Uncertainty gate
        self.gating = UncertaintyGating(
            embed_dim=embed_dim,
            mc_passes=mc_passes,
            dropout_rate=mc_dropout,
            entropy_threshold=entropy_threshold,
            learnable_scale=learnable_gate_scale,
        )

        # Cross-attention (EEG queries concatenated video tokens)
        self.cross_attn = CrossAttentionFusion(target_dim, n_ca_heads, n_ca_layers)

        # EEG embed dim may differ from target_dim → project if needed
        self.eeg_proj = (
            nn.Linear(embed_dim, target_dim, bias=False)
            if embed_dim != target_dim else nn.Identity()
        )

        # Fusion MLP: concat(raw EEG, attended)  → embed
        fusion_in = target_dim * 2
        self.fusion = nn.Sequential(
            nn.LayerNorm(fusion_in),
            nn.Linear(fusion_in, target_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Final classifier
        hidden_dims = hidden_dims or [256, 128]
        layers: list[nn.Module] = []
        in_d = target_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_d, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)]
            in_d = h
        layers.append(nn.Linear(in_d, 1))
        self.classifier = nn.Sequential(*layers)

    def forward(
        self,
        embed: torch.Tensor,
        video_features: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            logit: (B,)
            gate:  (B,) for diagnostics
        """
        gate, _ = self.gating(embed)   # (B,)

        # Project video backbones → shared dim, concat frames
        video_tokens = []
        for name, feat in video_features.items():
            if name in self.video_projs:
                video_tokens.append(self.video_projs[name](feat))  # (B, F, D)
        if not video_tokens:
            return self.classifier(self.eeg_proj(embed)).squeeze(-1), gate

        video_kv = torch.cat(video_tokens, dim=1)  # (B, sum_F, D)

        # Cross-attention
        eeg_q = self.eeg_proj(embed).unsqueeze(1)  # (B, 1, D)
        attended = self.cross_attn(eeg_q, video_kv).squeeze(1)  # (B, D)

        # Soft gating: blend raw EEG with attended
        g = gate.unsqueeze(-1)
        fused_embed = (1 - g) * self.eeg_proj(embed) + g * attended  # (B, D)

        # Additional fusion path
        cat = torch.cat([self.eeg_proj(embed), fused_embed], dim=-1)
        out = self.fusion(cat)

        return self.classifier(out).squeeze(-1), gate


###############################################################################
# Track 3 Head — 5-class localization
###############################################################################

class ChannelTemporalAttn(nn.Module):
    """
    Multi-head self-attention over 29 channel GAT tokens.
    Captures the spatial propagation pattern of IEDs across channels,
    which distinguishes the 5 epileptogenic zone classes.
    """

    def __init__(self, in_dim: int, hidden_dim: int, n_heads: int, n_layers: int, dropout: float) -> None:
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads,
            dim_feedforward=hidden_dim * 2, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, ch_feats: torch.Tensor) -> torch.Tensor:
        # ch_feats: (B, C, in_dim) → (B, hidden_dim)
        x = self.proj(ch_feats)
        x = self.encoder(x)
        return self.norm(x).mean(dim=1)


class Track3Head(nn.Module):
    """
    Track 3: 5-class epileptogenic zone localization.

    Uses:
        - Global EEG embed from backbone
        - Channel-level GAT features (spatial propagation)
        - Optional DINOv2 late fusion for visual context
    """

    def __init__(
        self,
        embed_dim: int = 256,
        channel_feat_dim: int = 128,
        video_config: Optional[dict] = None,  # {"n_frames":8, "feat_dim":1024}
        hidden_dim: int = 256,
        n_classes: int = 5,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.30,
    ) -> None:
        super().__init__()

        self.ch_attn = ChannelTemporalAttn(channel_feat_dim, hidden_dim, n_heads, n_layers, dropout)

        fusion_dim = embed_dim + hidden_dim

        self.use_video = video_config is not None
        if self.use_video:
            n_frames = video_config["n_frames"]
            feat_dim = video_config["feat_dim"]
            self.video_proj = nn.Sequential(
                nn.Linear(feat_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()
            )
            self.frame_pool = nn.Linear(n_frames, 1) if n_frames > 1 else None
            fusion_dim += hidden_dim

        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(
        self,
        embed: torch.Tensor,
        channel_feats: torch.Tensor,
        video_features: Optional[dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Returns: logits (B, n_classes)
        """
        spatial = self.ch_attn(channel_feats)  # (B, hidden_dim)
        parts = [embed, spatial]

        if self.use_video and video_features:
            v = next(iter(video_features.values()))   # (B, F, feat_dim)
            v = self.video_proj(v)                    # (B, F, hidden_dim)
            if self.frame_pool is not None:
                v = self.frame_pool(v.transpose(1, 2)).squeeze(-1)  # (B, hidden_dim)
            else:
                v = v.squeeze(1)
            parts.append(v)

        return self.classifier(torch.cat(parts, dim=-1))
