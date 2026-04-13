"""
app_core/inference.py
Single-model and multi-model inference for the Gradio app.
CPU-safe. No training loops, no FL orchestration.
"""

from typing import Dict, Optional, Tuple, Union
from pathlib import Path
import torch
import numpy as np

from app_core.config   import DEFAULT_THRESHOLD, KNOWN_METRICS, MODEL_KEYS
from app_core.model_registry import load_model


def predict_mask(
    model: torch.nn.Module,
    tensor: torch.Tensor,
    threshold: float = DEFAULT_THRESHOLD,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run segmentation inference on a preprocessed tensor.

    Args:
        model   : loaded nn.Module in eval mode
        tensor  : [1, 1, H, W] float32 CPU tensor
        threshold: sigmoid threshold for binary mask

    Returns:
        (prob_map, binary_mask)  both float32 [H, W] in [0,1]
    """
    model.eval()
    with torch.no_grad():
        logits   = model(tensor)
        prob_map = torch.sigmoid(logits).squeeze().numpy()
    binary_mask = (prob_map > threshold).astype(np.float32)
    return prob_map, binary_mask


def run_single_inference(
    source: Union[str, Path, np.ndarray],
    model_key: str,
    threshold: float = DEFAULT_THRESHOLD,
) -> Dict:
    """
    Full inference pipeline for a single model.

    Returns a dict with:
        display_img   : uint8 [H,W,3]
        prob_map      : float32 [H,W]
        binary_mask   : float32 [H,W]
        overlay       : uint8 [H,W,3]
        summary       : str
        model_key     : str
        status        : "ok" | error message
    """
    from app_core.preprocessing import prepare
    from app_core.visualization import make_overlay, mask_to_rgb

    # Preprocess
    display_img, model_img, tensor = prepare(source)

    # Load model
    model, load_status = load_model(model_key)
    if model is None:
        metrics = KNOWN_METRICS.get(model_key, {})
        return {
            "display_img":  display_img,
            "prob_map":     np.zeros((256, 256), dtype=np.float32),
            "binary_mask":  np.zeros((256, 256), dtype=np.float32),
            "overlay":      display_img,
            "summary":      _no_checkpoint_summary(model_key, metrics),
            "model_key":    model_key,
            "status":       f"checkpoint_unavailable: {load_status}",
        }

    # Inference
    prob_map, binary_mask = predict_mask(model, tensor, threshold)
    overlay  = make_overlay(model_img, prob_map, alpha=0.45)
    mask_rgb = mask_to_rgb(binary_mask)
    summary  = _format_summary(model_key, prob_map, binary_mask, threshold)

    return {
        "display_img":  display_img,
        "prob_map":     prob_map,
        "binary_mask":  mask_rgb,
        "overlay":      overlay,
        "summary":      summary,
        "model_key":    model_key,
        "status":       "ok",
    }


def run_comparison(
    source: Union[str, Path, np.ndarray],
    threshold: float = DEFAULT_THRESHOLD,
) -> Dict[str, Dict]:
    """
    Run inference with all three models on the same input.
    Returns a dict keyed by model_key, each containing the same
    structure as run_single_inference().
    """
    from app_core.preprocessing import prepare
    display_img, model_img, tensor = prepare(source)

    results = {}
    for key in MODEL_KEYS:
        result = run_single_inference(source, key, threshold)
        result["display_img"] = display_img   # share same display image
        results[key] = result

    return results


def _format_summary(
    model_key: str,
    prob_map: np.ndarray,
    binary_mask: np.ndarray,
    threshold: float,
) -> str:
    """Build a short plain-text summary of the prediction."""
    n_detected  = int(binary_mask.sum())
    max_conf    = float(prob_map.max())
    mean_conf   = float(prob_map[prob_map > 0.1].mean()) if (prob_map > 0.1).any() else 0.0
    small_nodule = n_detected > 0 and n_detected < 200

    metrics = KNOWN_METRICS.get(model_key, {})
    lines = [
        f"Model          : {model_key}",
        f"Threshold      : {threshold:.2f}",
        f"Detected pixels: {n_detected}",
        f"Max confidence : {max_conf:.4f}",
        f"Mean confidence: {mean_conf:.4f}",
        f"Rare nodule?   : {'Likely (<200px)' if small_nodule and n_detected > 0 else 'Standard or none'}",
        "",
        "Model performance (test set):",
        f"  Dice score   : {metrics.get('dice', 'N/A')}",
        f"  Sensitivity  : {metrics.get('sensitivity', 'N/A')}",
        f"  FROC @1FP    : {metrics.get('froc_1fp', 'N/A')}",
        f"  ECE          : {metrics.get('ece', 'N/A')}",
        "",
        metrics.get("description", ""),
        "",
        "⚕️  Research demo only — not a clinical device.",
    ]
    return "\n".join(lines)


def _no_checkpoint_summary(model_key: str, metrics: Dict) -> str:
    """Summary shown when checkpoint is not available."""
    lines = [
        f"Model          : {model_key}",
        f"Status         : Checkpoint not found in models/",
        "",
        "Known performance (from training logs):",
        f"  Dice score   : {metrics.get('dice', 'N/A')}",
        f"  Sensitivity  : {metrics.get('sensitivity', 'N/A')}",
        f"  FROC @1FP    : {metrics.get('froc_1fp', 'N/A')}",
        f"  ECE          : {metrics.get('ece', 'N/A')}",
        "",
        "To enable live inference, add the following file to models/:",
        f"  {model_key.lower().replace(' ','_').replace('(','').replace(')','').replace('+','plus')}_best.pth",
        "",
        "See README.md for checkpoint download instructions.",
        "",
        "⚕️  Research demo only — not a clinical device.",
    ]
    return "\n".join(lines)
