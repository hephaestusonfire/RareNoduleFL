"""
app_core/model_registry.py
Manages model loading, caching, and availability.

Architecture definitions are duplicated minimally here (not imported from
training/ scripts) to ensure the app layer is independent of training-time
imports like google.colab, flwr, or monai training utilities.

The exact architecture (UNet channels, strides, res_units, norm) must match
the checkpoints saved during training. Any mismatch causes a state_dict error.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Optional, Tuple
import warnings
warnings.filterwarnings("ignore")

from app_core.config import (
    CHECKPOINT_PATHS, CHECKPOINT_SD_KEY,
    IMAGE_SIZE, UNET_CHANNELS, UNET_STRIDES, UNET_RES_UNITS
)

# ── Architecture definition ───────────────────────────────────────────────────
# Exact copy of Model1_UNet from training — channels/strides/norm must match.
# We avoid importing from training/ to keep the app layer clean.

def _build_unet() -> nn.Module:
    """
    Build the U-Net architecture that matches all three checkpoints.
    Uses MONAI UNet. Falls back to a pure-PyTorch UNet if MONAI unavailable.
    """
    try:
        from monai.networks.nets   import UNet
        from monai.networks.layers import Norm
        model = UNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            channels=UNET_CHANNELS,
            strides=UNET_STRIDES,
            num_res_units=UNET_RES_UNITS,
            norm=Norm.BATCH,
            dropout=0.1,
        )
        return model
    except ImportError:
        # Fallback: minimal pure-PyTorch U-Net for architecture display
        # This will load weights but inference quality may differ slightly
        return _build_fallback_unet()


def _build_fallback_unet() -> nn.Module:
    """Minimal pure-PyTorch fallback if MONAI is not installed."""

    class ConvBlock(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            )
        def forward(self, x): return self.block(x)

    class FallbackUNet(nn.Module):
        """Simplified U-Net for display purposes only."""
        def __init__(self):
            super().__init__()
            self.note = "fallback_architecture"
            chs = UNET_CHANNELS
            self.enc1 = ConvBlock(1,     chs[0])
            self.enc2 = ConvBlock(chs[0],chs[1])
            self.enc3 = ConvBlock(chs[1],chs[2])
            self.enc4 = ConvBlock(chs[2],chs[3])
            self.bot  = ConvBlock(chs[3],chs[4])
            self.dec4 = ConvBlock(chs[4]+chs[3],chs[3])
            self.dec3 = ConvBlock(chs[3]+chs[2],chs[2])
            self.dec2 = ConvBlock(chs[2]+chs[1],chs[1])
            self.dec1 = ConvBlock(chs[1]+chs[0],chs[0])
            self.out  = nn.Conv2d(chs[0], 1, 1)
            self.pool = nn.MaxPool2d(2)
            self.up   = nn.Upsample(scale_factor=2, mode="bilinear",
                                     align_corners=True)

        def forward(self, x):
            e1=self.enc1(x);  e2=self.enc2(self.pool(e1))
            e3=self.enc3(self.pool(e2)); e4=self.enc4(self.pool(e3))
            b =self.bot(self.pool(e4))
            d4=self.dec4(torch.cat([self.up(b),  e4],1))
            d3=self.dec3(torch.cat([self.up(d4), e3],1))
            d2=self.dec2(torch.cat([self.up(d3), e2],1))
            d1=self.dec1(torch.cat([self.up(d2), e1],1))
            return self.out(d1)

    return FallbackUNet()


# ── Model cache ───────────────────────────────────────────────────────────────
_model_cache: Dict[str, Optional[nn.Module]] = {}
_availability: Dict[str, bool] = {}


def _normalize_state_dict_keys(sd: dict) -> dict:
    """
    Normalize common training wrappers in checkpoint keys.

    Handles keys saved as:
    - 'unet.model....' -> 'model....'
    - 'module.model....' (DataParallel) -> 'model....'
    """
    if not sd:
        return sd

    sample_key = next(iter(sd.keys()))
    if sample_key.startswith("unet."):
        return {k[len("unet."):]: v for k, v in sd.items()}
    if sample_key.startswith("module."):
        stripped = {k[len("module."):]: v for k, v in sd.items()}
        # Some checkpoints may include both wrappers: module.unet.*
        sample2 = next(iter(stripped.keys()))
        if sample2.startswith("unet."):
            return {k[len("unet."):]: v for k, v in stripped.items()}
        return stripped
    return sd


def check_availability() -> Dict[str, bool]:
    """Return which model checkpoints exist on disk."""
    result = {}
    for key, path in CHECKPOINT_PATHS.items():
        result[key] = Path(path).exists()
    _availability.update(result)
    return result


def load_model(model_key: str) -> Tuple[Optional[nn.Module], str]:
    """
    Load a model by key. Returns (model, status_message).
    Uses cache to avoid repeated disk reads.
    Runs on CPU — safe for HF Spaces free tier.
    """
    if model_key in _model_cache and _model_cache[model_key] is not None:
        return _model_cache[model_key], "loaded_from_cache"

    ckpt_path = Path(CHECKPOINT_PATHS[model_key])
    if not ckpt_path.exists():
        _model_cache[model_key] = None
        return None, f"checkpoint_not_found: {ckpt_path.name}"

    try:
        model = _build_unet()
        ckpt  = torch.load(ckpt_path, map_location="cpu",
                           weights_only=False)
        sd_key = CHECKPOINT_SD_KEY[model_key]

        # Try primary key, then fallback to alternative
        if sd_key in ckpt:
            sd = ckpt[sd_key]
        elif "model_state_dict" in ckpt:
            sd = ckpt["model_state_dict"]
        elif "global_model_state_dict" in ckpt:
            sd = ckpt["global_model_state_dict"]
        else:
            return None, f"no_state_dict_key_in_checkpoint: {list(ckpt.keys())}"

        sd = _normalize_state_dict_keys(sd)
        model.load_state_dict(sd, strict=True)
        model.eval()
        _model_cache[model_key] = model
        return model, "ok"

    except Exception as e:
        _model_cache[model_key] = None
        return None, f"load_error: {e}"


def get_model_info(model_key: str) -> Dict:
    """Return metadata about a model without loading it."""
    from app_core.config import KNOWN_METRICS
    ckpt_path = Path(CHECKPOINT_PATHS[model_key])
    metrics   = KNOWN_METRICS.get(model_key, {})
    return {
        "key":            model_key,
        "checkpoint":     ckpt_path.name,
        "available":      ckpt_path.exists(),
        "dice":           metrics.get("dice", "N/A"),
        "sensitivity":    metrics.get("sensitivity", "N/A"),
        "froc_1fp":       metrics.get("froc_1fp", "N/A"),
        "ece":            metrics.get("ece", "N/A"),
        "description":    metrics.get("description", ""),
    }


def preload_all() -> Dict[str, str]:
    """
    Preload all available models at app startup.
    Returns dict of {model_key: status}.
    Skips missing checkpoints gracefully.
    """
    statuses = {}
    for key in CHECKPOINT_PATHS:
        _, status = load_model(key)
        statuses[key] = status
        print(f"  [{key}]: {status}")
    return statuses
