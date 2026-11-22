from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from .model import TSMixerBlock


class BaselineWrapper(nn.Module):
    def __init__(self, name: str, module: nn.Module):
        super().__init__()
        self.name = name
        self.module = module

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {"prediction": self.module(batch["flat_hist"], batch["target_calendar"])}


class MixerBackbone(nn.Module):
    def __init__(self, seq_len: int, feat_dim: int, hidden_dim: int = 96, depth: int = 2):
        super().__init__()
        self.input_proj = nn.Linear(feat_dim, hidden_dim)
        self.blocks = nn.Sequential(*[TSMixerBlock(seq_len, hidden_dim, 0.1) for _ in range(depth)])
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + 7, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 4),
        )

    def forward(self, x: torch.Tensor, calendar: torch.Tensor) -> torch.Tensor:
        h = self.blocks(self.input_proj(x))
        pooled = h[:, -1] + h.mean(dim=1)
        return self.head(torch.cat([pooled, calendar], dim=-1))


class PatchTransformerBackbone(nn.Module):
    def __init__(self, seq_len: int, feat_dim: int, patch_size: int = 4, hidden_dim: int = 96):
        super().__init__()
        self.patch_size = patch_size
        self.patch_proj = nn.Linear(feat_dim * patch_size, hidden_dim)
        encoder = nn.TransformerEncoderLayer(hidden_dim, 4, hidden_dim * 2, 0.1)
        self.encoder = nn.TransformerEncoder(encoder, 2)
        self.head = nn.Sequential(nn.Linear(hidden_dim + 7, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 4))

    def forward(self, x: torch.Tensor, calendar: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, feat_dim = x.shape
        usable = (seq_len // self.patch_size) * self.patch_size
        x = x[:, -usable:]
        patches = x.reshape(bsz, usable // self.patch_size, feat_dim * self.patch_size)
        h = self.encoder(self.patch_proj(patches).transpose(0, 1)).transpose(0, 1)
        pooled = h[:, -1] + h.mean(dim=1)
        return self.head(torch.cat([pooled, calendar], dim=-1))


class InvertedTransformerBackbone(nn.Module):
    def __init__(self, seq_len: int, feat_dim: int, hidden_dim: int = 96):
        super().__init__()
        self.feature_proj = nn.Linear(seq_len, hidden_dim)
        encoder = nn.TransformerEncoderLayer(hidden_dim, 4, hidden_dim * 2, 0.1)
        self.encoder = nn.TransformerEncoder(encoder, 2)
        self.head = nn.Sequential(nn.Linear(hidden_dim + 7, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 4))

    def forward(self, x: torch.Tensor, calendar: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        h = self.encoder(self.feature_proj(x).transpose(0, 1)).transpose(0, 1)
        pooled = h.mean(dim=1)
        return self.head(torch.cat([pooled, calendar], dim=-1))


class ConvInceptionBackbone(nn.Module):
    def __init__(self, seq_len: int, feat_dim: int, hidden_dim: int = 96):
        super().__init__()
        self.input_proj = nn.Linear(feat_dim, hidden_dim)
        self.conv3 = nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1)
        self.conv5 = nn.Conv1d(hidden_dim, hidden_dim, 5, padding=2)
        self.conv7 = nn.Conv1d(hidden_dim, hidden_dim, 7, padding=3)
        self.norm = nn.LayerNorm(hidden_dim * 3)
        self.head = nn.Sequential(nn.Linear(hidden_dim * 3 + 7, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 4))

    def forward(self, x: torch.Tensor, calendar: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x).transpose(1, 2)
        mixed = torch.cat([self.conv3(h), self.conv5(h), self.conv7(h)], dim=1).transpose(1, 2)
        pooled = self.norm(mixed).mean(dim=1)
        return self.head(torch.cat([pooled, calendar], dim=-1))


def create_generic_baselines(seq_len: int, feat_dim: int) -> Dict[str, nn.Module]:
    return {
        "PatchTST": BaselineWrapper("PatchTST", PatchTransformerBackbone(seq_len, feat_dim, patch_size=4)),
        "TSMixer": BaselineWrapper("TSMixer", MixerBackbone(seq_len, feat_dim, hidden_dim=96, depth=2)),
        "TimesNet": BaselineWrapper("TimesNet", ConvInceptionBackbone(seq_len, feat_dim, hidden_dim=64)),
        "iTransformer": BaselineWrapper("iTransformer", InvertedTransformerBackbone(seq_len, feat_dim, hidden_dim=96)),
        "TimeMixer": BaselineWrapper("TimeMixer", MixerBackbone(seq_len, feat_dim, hidden_dim=128, depth=3)),
        "TimeXer": BaselineWrapper("TimeXer", PatchTransformerBackbone(seq_len, feat_dim, patch_size=7, hidden_dim=96)),
    }


def create_load_specific_baselines(seq_len: int, feat_dim: int) -> Dict[str, nn.Module]:
    return {
        "MG-ASTL": BaselineWrapper("MG-ASTL", ConvInceptionBackbone(seq_len, feat_dim, hidden_dim=80)),
        "TL-STLF": BaselineWrapper("TL-STLF", MixerBackbone(seq_len, feat_dim, hidden_dim=80, depth=2)),
        "LoadFormer": BaselineWrapper("LoadFormer", PatchTransformerBackbone(seq_len, feat_dim, patch_size=7, hidden_dim=80)),
        "STGNN-Load": BaselineWrapper("STGNN-Load", InvertedTransformerBackbone(seq_len, feat_dim, hidden_dim=80)),
        "ASTGCN-Load": BaselineWrapper("ASTGCN-Load", ConvInceptionBackbone(seq_len, feat_dim, hidden_dim=96)),
        "GAT-LoadMixer": BaselineWrapper("GAT-LoadMixer", MixerBackbone(seq_len, feat_dim, hidden_dim=72, depth=3)),
    }
