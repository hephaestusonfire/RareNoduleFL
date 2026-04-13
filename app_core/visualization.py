"""
app_core/visualization.py
Convert model outputs to display-ready images for Gradio.
All functions return uint8 numpy arrays [H, W, 3] or PIL Images.
Gradio Image accepts both.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import Optional


def mask_to_rgb(
    binary_mask: np.ndarray,
    color: tuple = (45, 180, 150),   # teal accent
) -> np.ndarray:
    """
    Convert binary float mask [H,W] to coloured uint8 [H,W,3].
    Positive pixels are teal, background is near-black.
    """
    h, w   = binary_mask.shape
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[binary_mask > 0.5] = color
    canvas[binary_mask <= 0.5] = (15, 20, 25)
    return canvas


def prob_to_heatmap(prob_map: np.ndarray) -> np.ndarray:
    """
    Convert probability map [H,W] to coloured heatmap uint8 [H,W,3].
    Uses 'viridis' colormap for a clean academic look.
    """
    normed  = np.clip(prob_map, 0, 1)
    colored = (cm.viridis(normed)[:, :, :3] * 255).astype(np.uint8)
    return colored


def make_overlay(
    img_np: np.ndarray,
    prob_map: np.ndarray,
    alpha: float = 0.45,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Overlay probability map on CT image with a teal/hot colourmap.

    Args:
        img_np   : float32 [H,W] normalised CT image
        prob_map : float32 [H,W] sigmoid probabilities
        alpha    : overlay opacity
        threshold: draw contour at this level

    Returns:
        uint8 [H,W,3]
    """
    # Base CT as RGB
    ct_rgb = np.stack([img_np, img_np, img_np], axis=2)
    ct_rgb = (ct_rgb * 255).clip(0, 255).astype(np.uint8)

    # Coloured probability overlay
    heat   = (cm.plasma(np.clip(prob_map, 0, 1))[:, :, :3] * 255).astype(np.uint8)

    # Blend only where probability is meaningful
    mask_a = (prob_map > 0.2).astype(np.float32)
    blend  = (ct_rgb.astype(np.float32) * (1 - alpha * mask_a[:,:,None]) +
              heat.astype(np.float32)   * (alpha * mask_a[:,:,None]))
    result = blend.clip(0, 255).astype(np.uint8)

    # Draw contour at threshold using matplotlib
    try:
        fig, ax = plt.subplots(1, 1, figsize=(256/96, 256/96), dpi=96)
        ax.imshow(result)
        ax.contour(prob_map, levels=[threshold],
                   colors=["#00e5cc"], linewidths=1.5)
        ax.axis("off")
        fig.tight_layout(pad=0)

        # Render to numpy array
        fig.canvas.draw()
        w_px, h_px = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        buf = buf.reshape(h_px, w_px, 3)
        plt.close(fig)

        # Resize back to 256×256 if needed
        if buf.shape[:2] != (256, 256):
            pil = Image.fromarray(buf).resize((256, 256), Image.BILINEAR)
            buf = np.array(pil)
        return buf

    except Exception:
        # Fallback if matplotlib rendering fails
        plt.close("all")
        return result


def add_caption(img: np.ndarray, caption: str,
                font_size: int = 14) -> np.ndarray:
    """Add a text caption band at the bottom of an image."""
    pil     = Image.fromarray(img)
    band_h  = font_size + 10
    canvas  = Image.new("RGB", (pil.width, pil.height + band_h), (18, 24, 32))
    canvas.paste(pil, (0, 0))
    draw    = ImageDraw.Draw(canvas)
    draw.text((6, pil.height + 4), caption, fill=(180, 200, 210))
    return np.array(canvas)


def build_comparison_row(
    display_img: np.ndarray,
    results: dict,
) -> list:
    """
    Build a list of (image, caption) tuples for Gradio Gallery.
    One entry per model: original, mask, overlay.
    """
    gallery = []
    gallery.append((display_img, "Input CT Slice"))

    for key, result in results.items():
        if result["status"] == "ok":
            mask_img    = result["binary_mask"]
            overlay_img = result["overlay"]
        else:
            h, w = display_img.shape[:2]
            mask_img    = np.zeros((h, w, 3), dtype=np.uint8)
            overlay_img = display_img.copy()

        gallery.append((mask_img,    f"{key} — Mask"))
        gallery.append((overlay_img, f"{key} — Overlay"))

    return gallery
