"""
app_core/preprocessing.py
Lightweight, deterministic preprocessing for app inference.

Deliberately does NOT import from training/ scripts.
This is a self-contained preprocessing path for uploaded images.
"""

from pathlib import Path
from typing import Tuple, Union
import numpy as np
import torch
from PIL import Image

from app_core.config import IMAGE_SIZE


def load_image(source: Union[str, Path, np.ndarray]) -> np.ndarray:
    """
    Load an image from a file path or numpy array.
    Returns a float32 numpy array of shape [H, W] normalised to [0, 1].
    """
    if isinstance(source, np.ndarray):
        img = source.copy().astype(np.float32)
        if img.ndim == 3:
            # Take luminance: if RGB/RGBA, convert to grayscale
            if img.shape[2] == 4:
                img = img[:, :, :3]
            img = 0.299*img[:,:,0] + 0.587*img[:,:,1] + 0.114*img[:,:,2]
    else:
        pil = Image.open(str(source)).convert("L")   # grayscale
        img = np.array(pil, dtype=np.float32)

    # Normalise to [0, 1]
    mn, mx = img.min(), img.max()
    if mx > mn:
        img = (img - mn) / (mx - mn)
    else:
        img = np.zeros_like(img)

    return img


def resize_image(img: np.ndarray, size: int = IMAGE_SIZE) -> np.ndarray:
    """Resize a 2D float array to [size, size] using bilinear interpolation."""
    pil = Image.fromarray((img * 255).astype(np.uint8))
    pil = pil.resize((size, size), Image.BILINEAR)
    return np.array(pil, dtype=np.float32) / 255.0


def to_display(img: np.ndarray) -> np.ndarray:
    """Convert normalised float [H,W] to uint8 [H,W,3] for Gradio display."""
    img8 = (img * 255).clip(0, 255).astype(np.uint8)
    return np.stack([img8, img8, img8], axis=2)


def to_tensor(img: np.ndarray) -> torch.Tensor:
    """
    Convert normalised float [H,W] or [H,W,C] numpy image to
    model-ready tensor [1, 1, H, W] on CPU.
    """
    if img.ndim == 3:
        img = 0.299*img[:,:,0] + 0.587*img[:,:,1] + 0.114*img[:,:,2]
    return torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)


def prepare(
    source: Union[str, Path, np.ndarray]
) -> Tuple[np.ndarray, np.ndarray, torch.Tensor]:
    """
    Full preprocessing pipeline.

    Args:
        source: file path or numpy array

    Returns:
        (display_img, model_img, tensor)
        display_img : uint8 [H, W, 3] — for Gradio Image component
        model_img   : float32 [H, W]  — 256×256 normalised
        tensor      : float32 [1,1,H,W] — model-ready
    """
    raw      = load_image(source)
    resized  = resize_image(raw, IMAGE_SIZE)
    display  = to_display(resized)
    tensor   = to_tensor(resized)
    return display, resized, tensor
