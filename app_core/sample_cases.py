"""
app_core/sample_cases.py
Sample discovery and fallback generation for the demo.

Behavior:
- Any image file placed under assets/samples is auto-discovered.
- If no sample images are present, a small synthetic fallback set is generated.
- A sample can optionally map to a precomputed XAI visual from results/.
"""

from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
from PIL import Image

from app_core.config import SAMPLES_DIR, FIGURES


# ── Sample registry (dynamic) ────────────────────────────────────────────────
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

_FALLBACK_SPECS = [
    # (filename, nodule_center, radius, intensity, bg_texture, seed)
    ("sample_01.png", (90, 170), 8, 0.72, 0.28, 42),    # small pleural nodule
    ("sample_02.png", (128, 128), 18, 0.75, 0.25, 43),  # central standard nodule
    ("sample_03.png", (110, 150), 14, 0.55, 0.30, 44),  # ground-glass
]


# ── Sample generation ─────────────────────────────────────────────────────────

def _make_ct_like_sample(
    nodule_center: tuple,
    nodule_radius: int,
    nodule_intensity: float,
    bg_texture: float,
    size: int = 256,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate a synthetic CT-like grayscale image with a simulated nodule.
    Returns uint8 [H, W, 3].
    """
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size), dtype=np.float32)

    # Lung-like dark background
    cy, cx = size // 2, size // 2
    for r in range(size):
        for c in range(size):
            d = np.sqrt((r-cy)**2 + (c-cx)**2)
            if d < size * 0.42:
                img[r, c] = bg_texture + rng.normal(0, 0.04)

    # Rib-like bright structures (subtle)
    for i in range(4):
        rib_y = int(size * (0.25 + i * 0.15))
        img[rib_y:rib_y+3, cx-80:cx+80] += 0.3

    # Nodule (slightly brighter disc)
    ny, nx = nodule_center
    yy, xx = np.ogrid[:size, :size]
    dist = np.sqrt((yy-ny)**2 + (xx-nx)**2)
    nodule_mask = dist < nodule_radius
    img[nodule_mask] = nodule_intensity + rng.normal(0, 0.03, nodule_mask.sum())

    # Clip and convert
    img = np.clip(img, 0, 1)
    img8 = (img * 255).astype(np.uint8)
    return np.stack([img8, img8, img8], axis=2)


def _discover_sample_files() -> List[Path]:
    """Return all supported sample images in assets/samples."""
    if not SAMPLES_DIR.exists():
        return []
    files = [p for p in SAMPLES_DIR.iterdir() if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS]
    return sorted(files, key=lambda p: p.name.lower())


def _display_name_from_file(path: Path, idx: int) -> str:
    """Create a readable dropdown label from file name."""
    pretty = path.stem.replace("_", " ").replace("-", " ").strip().title()
    return f"Sample {idx:02d} — {pretty}"


def _xai_figure_for_file(path: Path):
    """Map known sample names to an XAI figure; default to Grad-CAM panels."""
    stem = path.stem.lower()
    if "sample_02" in stem:
        return FIGURES.get("entropy_maps")
    if "sample_03" in stem:
        return FIGURES.get("calibration")
    return FIGURES.get("gradcam_panels")


def _build_samples_from_files(files: List[Path]) -> List[Dict]:
    samples: List[Dict] = []
    for idx, path in enumerate(files, start=1):
        samples.append({
            "id": f"sample_{idx:02d}",
            "name": _display_name_from_file(path, idx),
            "file": path,
            "caption": f"User-provided demo sample: {path.name}",
            "xai_figure": _xai_figure_for_file(path),
            "xai_caption": "Precomputed explainability panel for demonstration.",
        })
    return samples


def _get_samples() -> List[Dict]:
    """Load current sample registry from disk."""
    return _build_samples_from_files(_discover_sample_files())


def ensure_samples_exist() -> None:
    """
    Generate synthetic sample images if they don't exist.
    Called once at app startup.
    """
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    if _discover_sample_files():
        return

    for filename, *spec in _FALLBACK_SPECS:
        path = SAMPLES_DIR / filename
        if not path.exists():
            img = _make_ct_like_sample(*spec)
            Image.fromarray(img).save(str(path))
            print(f"  Generated sample: {path.name}")


def get_sample_names() -> List[str]:
    """Return list of sample display names for Gradio Dropdown."""
    return [s["name"] for s in _get_samples()]


def get_sample_by_name(name: str) -> Optional[Dict]:
    """Return sample dict by display name."""
    for s in _get_samples():
        if s["name"] == name:
            return s
    return None


def get_sample_path(name: str) -> Optional[Path]:
    """Return file path for a sample by display name."""
    s = get_sample_by_name(name)
    return Path(s["file"]) if s else None


def get_xai_figure(name: str) -> Optional[np.ndarray]:
    """
    Return the precomputed XAI figure associated with a sample.
    Returns None if figure file is missing.
    """
    s = get_sample_by_name(name)
    if not s:
        return None
    fig_path = s.get("xai_figure")
    if not fig_path:
        return None
    fig_path = Path(fig_path)
    if not fig_path.exists():
        # Try the gradient panels as universal fallback
        fallback = FIGURES.get("gradcam_panels")
        if fallback and Path(fallback).exists():
            from PIL import Image as PILImage
            return np.array(PILImage.open(fallback).convert("RGB"))
        return None
    from PIL import Image as PILImage
    return np.array(PILImage.open(fig_path).convert("RGB"))
