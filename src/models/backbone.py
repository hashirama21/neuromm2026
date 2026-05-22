"""
src/models/backbone.py

Shared EEG Backbone: LocalCNN → DynamicGAT → EEGMamba

Architecture flow for input (B, 29, 2000):

    1. LocalCNN     : Per-channel 1D CNN captures spike morphology (20–70 ms).
                      Output: (B, 29, cnn_out_channels)

    2. DynamicGAT   : Multi-head self-attention over 29 channel tokens.
                      Learns inter-channel propagation patterns dynamically.
                      Output: (B, 29, gat_hidden_dim)

    3. EEGMamba     : Stacked Mamba2 SSM blocks over the 29 channel tokens.
                      O(n) complexity vs O(n²) Transformer.
                      Output: (B, embed_dim)  — global mean-pooled

Supports:
    - Foundation model loading (EEGMamba checkpoint) + LoRA fine-tuning
    - return_channel_feats=True for Track 3 localization head
    - freeze / unfreeze backbone
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from .dynamic_gat import DynamicGAT
from .eegmamba_encoder import EEGMambaEncoder



@dataclass
class BackboneConfig:
    n_channels: int = 29
    n_timesteps: int = 2000

    # LocalCNN
    cnn_out_channels: int = 64
    cnn_kernel_size: int = 7
    cnn_n_layers: int = 2
    cnn_dropout: float = 0.10

    # DynamicGAT
    gat_hidden_dim: int = 128
    gat_n_layers: int = 2
    gat_n_heads: int = 8
    gat_dropout: float = 0.10

    # EEGMamba
    mamba_d_model: int = 256
    mamba_d_state: int = 16
    mamba_d_conv: int = 4
    mamba_expand: int = 2
    mamba_n_layers: int = 4
    mamba_dropout: float = 0.10

    # Output
    embed_dim: int = 256

    @classmethod
    def from_dict(cls, d: dict) -> "BackboneConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Local temporal CNN
# ---------------------------------------------------------------------------

class LocalCNN(nn.Module):
    """
    Per-channel 1D CNN capturing spike morphology.

    Each of the 29 channels is processed independently with the same
    convolutional filters (shared weights — a form of channel-wise
    depthwise processing). This captures local temporal patterns like
    the sharp rising/falling edge of an IED (20-70 ms = 10-35 pts at 500 Hz).
    """

    def __init__(
        self,
        out_channels: int = 64,
        kernel_size: int = 7,
        n_layers: int = 2,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        in_ch = 1
        for i in range(n_layers):
            out_ch = out_channels if i == n_layers - 1 else out_channels // 2
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2, bias=False),
                nn.BatchNorm1d(out_ch),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            in_ch = out_ch

        self.cnn = nn.Sequential(*layers)
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:  x: (B, C, T)
        Returns: (B, C, out_channels)  — mean-pooled over time
        """
        B, C, T = x.shape
        x = x.view(B * C, 1, T)          # treat each channel independently
        x = self.cnn(x)                  # (B*C, out_channels, T')
        x = x.mean(dim=-1)               # global avg pool over time
        x = x.view(B, C, self.out_channels)
        return x


# ---------------------------------------------------------------------------
# Full Backbone
# ---------------------------------------------------------------------------

class EEGBackbone(nn.Module):
    """
    Full EEGMamba Backbone.

    forward() returns:
        embed: (B, embed_dim)

    forward(return_channel_feats=True) returns:
        embed: (B, embed_dim)
        channel_feats: (B, C, gat_hidden_dim)   ← used by Track 3 head
    """

    def __init__(self, cfg: BackboneConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # Stage 1: per-channel local CNN
        self.local_cnn = LocalCNN(
            out_channels=cfg.cnn_out_channels,
            kernel_size=cfg.cnn_kernel_size,
            n_layers=cfg.cnn_n_layers,
            dropout=cfg.cnn_dropout,
        )

        # Stage 2: Dynamic GAT over 29 channel tokens
        self.dynamic_gat = DynamicGAT(
            in_dim=cfg.cnn_out_channels,
            hidden_dim=cfg.gat_hidden_dim,
            n_layers=cfg.gat_n_layers,
            n_heads=cfg.gat_n_heads,
            dropout=cfg.gat_dropout,
        )

        # Projection: gat_hidden_dim → mamba_d_model
        self.gat_to_mamba = nn.Sequential(
            nn.LayerNorm(cfg.gat_hidden_dim),
            nn.Linear(cfg.gat_hidden_dim, cfg.mamba_d_model),
        )

        # Stage 3: EEGMamba encoder over channel tokens as sequence
        self.mamba = EEGMambaEncoder(
            d_model=cfg.mamba_d_model,
            d_state=cfg.mamba_d_state,
            d_conv=cfg.mamba_d_conv,
            expand=cfg.mamba_expand,
            n_layers=cfg.mamba_n_layers,
            dropout=cfg.mamba_dropout,
        )

        # Final projection to embed_dim
        self.head = nn.Sequential(
            nn.LayerNorm(cfg.mamba_d_model),
            nn.Linear(cfg.mamba_d_model, cfg.embed_dim),
            nn.GELU(),
        )

    def forward(
        self,
        eeg: torch.Tensor,
        return_channel_feats: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            eeg: (B, C, T)  normalised EEG signal
            return_channel_feats: also return (B, C, gat_hidden_dim)
        Returns:
            embed: (B, embed_dim)
            channel_feats (optional): (B, C, gat_hidden_dim)
        """
        # 1. Local CNN: spatial+temporal → per-channel features
        x = self.local_cnn(eeg)             # (B, C, cnn_out_channels)

        # 2. Dynamic GAT: inter-channel attention
        channel_feats = self.dynamic_gat(x) # (B, C, gat_hidden_dim)

        # 3. Project to Mamba dim, run SSM over channel sequence
        x_mamba = self.gat_to_mamba(channel_feats)  # (B, C, mamba_d_model)
        mamba_out = self.mamba(x_mamba)     # (B, mamba_d_model)

        # 4. Final projection
        embed = self.head(mamba_out)        # (B, embed_dim)

        if return_channel_feats:
            return embed, channel_feats
        return embed


    def freeze(self) -> None:
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze(self) -> None:
        for p in self.parameters():
            p.requires_grad = True

    def freeze_cnn(self) -> None:
        for p in self.local_cnn.parameters():
            p.requires_grad = False

    def get_embed_dim(self) -> int:
        return self.cfg.embed_dim

    def get_channel_feat_dim(self) -> int:
        return self.cfg.gat_hidden_dim

    def load_foundation_weights(
        self,
        ckpt_path: str | Path,
        strict: bool = False,
    ) -> None:
        """
        Load weights from an EEGMamba foundation model checkpoint.
        Uses strict=False to allow partial loading (e.g. different n_channels).
        """
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state = ckpt.get("model_state", ckpt.get("backbone_state", ckpt))
        missing, unexpected = self.load_state_dict(state, strict=strict)
        print(f"[Backbone] Loaded foundation weights from {ckpt_path}")
        if missing:
            print(f"  Missing keys ({len(missing)}): {missing[:5]}...")
        if unexpected:
            print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")

    def apply_lora(self, r: int = 16, alpha: int = 32, dropout: float = 0.05) -> None:
        """
        Apply LoRA adapters to all Linear layers in the Mamba encoder.
        Requires the `peft` library: pip install peft
        """
        try:
            from peft import get_peft_model, LoraConfig, TaskType
            lora_cfg = LoraConfig(
                r=r, lora_alpha=alpha, lora_dropout=dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "out_proj",
                                 "in_proj", "out_proj", "x_proj", "dt_proj"],
            )
            # LoRA on the mamba encoder only
            self.mamba = get_peft_model(self.mamba, lora_cfg)
            print(f"[Backbone] LoRA applied: r={r}, alpha={alpha}")
        except ImportError:
            print("[Backbone] peft not installed — skipping LoRA. pip install peft")
