# docs/app_notes.md
# RareNoduleFL — App Deployment Technical Notes

---

## What Was Added

The following files were added on top of the existing research pipeline
without modifying any research code:

```
app.py                     Gradio Blocks entrypoint — 6 tabs
requirements.txt           9 CPU-safe pip dependencies
README.md                  Updated: Gradio HF Spaces YAML + full docs
.gitignore                 Excludes checkpoints, NPY data, caches

app_core/
  __init__.py
  config.py                All paths, constants, known metrics, stat highlights
  model_registry.py        Lazy model loading, caching, graceful fallback
  preprocessing.py         Upload → resize → normalise → tensor
  inference.py             Single + multi-model inference + summary formatting
  visualization.py         Mask, overlay, heatmap rendering
  results_loader.py        Load figures/CSVs/stats from results/
  sample_cases.py          Auto-discovers assets/samples + synthetic fallback generation

assets/
  css/app.css              Academic medical dashboard CSS
  samples/                 User-managed demo images for Inference/Compare dropdowns
                          (synthetic fallback set generated only if folder is empty)

docs/
  app_notes.md             This file
```

---

## How the App Layer Maps to the Research Layer

```
Research Layer                        App Layer
──────────────────────────────────────────────────────────────────
training/lidc_step2_step3_step4.py    ← architecture in model_registry.py
                                        (minimal duplication — exact
                                         channel/stride/norm to match checkpoints)

evaluation/complete_project_finish.py ← figures/CSVs consumed by results_loader.py
                                        (app reads from results/, does not re-run)

models/*.pth (from Google Drive)      ← loaded lazily by model_registry.py
                                        (app works without them — shows known metrics)

results/*.png, *.csv, *.txt           ← loaded at runtime by results_loader.py
                                        (with embedded fallbacks if files missing)
```

The app deliberately **does not import** from `preprocessing/`, `training/`,
`federated/`, or `evaluation/`. This keeps the deployment layer clean and
avoids pulling in training-time dependencies (Flower, Google Colab, etc.).

---

## Live Inference vs Precomputed Display

| Tab | Mode | Notes |
|-----|------|-------|
| Overview | Precomputed | Pipeline figure from `results/pipeline_overview.png` |
| Inference | Live (if checkpoint) | Falls back to known metrics if `.pth` missing |
| Compare | Live (if checkpoint) | Same fallback per model |
| Explainability | Precomputed | Loads from `results/xai_gradcam_panels.png` etc. |
| Results | Precomputed | Figures gallery + ablation CSV + stat tests text |
| About | Static | Hardcoded text |

**Live inference** = uploads image → preprocessing → model forward pass → sigmoid → mask/overlay.

**Precomputed display** = reads PNG/CSV/TXT from `results/` and renders directly.

The app is designed so that **all tabs provide meaningful content** even on a
Hugging Face Space that has no checkpoint files uploaded — because research
result figures and embedded known metrics cover every tab.

---

## App Assumptions

1. **`results/` folder exists and is populated** by running
   `evaluation/complete_project_finish.py` on Google Colab.
   If figures are missing, tabs display placeholder messages.

2. **`models/` folder** may be empty. The app checks each checkpoint path
   at startup via `check_availability()` and adjusts UI labels accordingly.

3. **CPU execution only.** No CUDA tensors. All `torch.load` calls use
   `map_location="cpu"`. Model `eval()` mode is enforced before inference.

4. **No Google Drive paths.** All paths are resolved relative to the repo root
   via `Path(__file__).parent.parent` in `config.py`.

5. **Sample source** is the `assets/samples/` folder. Any image placed there
   appears in Inference/Compare dropdowns. If the folder is empty, the app
   auto-generates a synthetic fallback set (CT-like grayscale nodules) so the
   inference pipeline remains demo-ready.

---

## Architecture Duplication Note

`app_core/model_registry.py` contains a minimal re-definition of the U-Net
architecture (`_build_unet()`) using MONAI. This is intentional — importing
from `training/lidc_step2_step3_step4.py` would pull in:
- `from google.colab import drive`
- `import flwr`
- MONAI training transforms not needed at inference time

The re-definition uses exactly the same hyperparameters as the training checkpoints:
```python
UNet(spatial_dims=2, in_channels=1, out_channels=1,
     channels=(32,64,128,256,512), strides=(2,2,2,2),
     num_res_units=2, norm=Norm.BATCH, dropout=0.1)
```
Any change to this in `training/` must be reflected here.

A `FallbackUNet` pure-PyTorch implementation is also provided in case MONAI
is unavailable, though the `requirements.txt` includes `monai>=1.3.0`.

---

## How to Extend Later

### Add a new model
1. Add the checkpoint path to `CHECKPOINT_PATHS` in `app_core/config.py`
2. Add the state dict key to `CHECKPOINT_SD_KEY`
3. Add known metrics to `KNOWN_METRICS`
4. Add the model key to `MODEL_KEYS`
5. If the architecture differs, add a branch in `model_registry._build_unet()`

### Add a new result figure
1. Add the figure path to `FIGURES` in `app_core/config.py`
2. Add a `(key, caption)` entry in `results_loader.load_all_figures()`
3. It will appear automatically in the Results tab gallery

### Enable Grad-CAM live in the app
Current: Explainability tab shows precomputed figures.
To enable live: add `torchcam` to `requirements.txt`, import `GradCAMpp`
in `inference.py`, and add a `run_gradcam()` function following the pattern
in `evaluation/complete_project_finish.py` (lines 570-610).
Note: GradCAMpp requires `inp.requires_grad_(True)` and a forward pass
outside `torch.no_grad()`.

### Deploy with GPU on HF Spaces
Change the Space hardware to T4 in the HF Spaces settings.
No code changes required — the model_registry already runs on CPU,
which is correct for T4 (`map_location="cpu"` → then `.to(device)` if needed).
Add `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`
in `app_core/config.py` and update `model_registry.load_model()` to use it.

---

## Local Run Checklist

```bash
# 1. Clone repo
git clone https://huggingface.co/spaces/YOUR_HF_USERNAME/RareNoduleFL
cd RareNoduleFL

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Add model checkpoints
mkdir -p models
# Copy step4.1_unet_best.pth, fl_iid_global_best.pth,
#       fl_noniid_rus_global_best.pth from Google Drive → models/

# 4. (Optional) Add result figures
# Copy results/ folder from Google Drive → results/

# 5. Run
python app.py
# Open http://localhost:7860
```

---

## Hugging Face Spaces Deployment Checklist

```
✅ README.md has valid YAML front matter:
     sdk: gradio
     app_file: app.py

✅ app.py ends with demo.launch() (no share=True needed on Spaces)

✅ requirements.txt lists all dependencies
   (Spaces installs from this file automatically)

✅ All paths in app_core/config.py are repo-relative
   (REPO_ROOT = Path(__file__).parent.parent)

✅ No Colab / Google Drive imports in app layer

✅ No training-time imports (flwr, Colab, etc.) in app layer

✅ App works without model checkpoints
   (precomputed results cover all tabs)

✅ App works without results/ files
   (embedded fallback values in results_loader.py)

✅ Sample images auto-generated at startup
   (ensure_samples_exist() called in app.py)
```

---

## Missing Assets Checklist

Items that must be added to the repo manually (not auto-generated):

| Item | Source | Required for |
|------|--------|-------------|
| `models/step4.1_unet_best.pth` | Google Drive (training output) | Live inference |
| `models/fl_iid_global_best.pth` | Google Drive (training output) | Live inference |
| `models/fl_noniid_rus_global_best.pth` | Google Drive (training output) | Live inference |
| `results/pipeline_overview.png` | Google Drive (evaluation output) | Overview tab figure |
| `results/fl_learning_curves.png` | Google Drive (evaluation output) | Results tab |
| `results/fl_client_drift.png` | Google Drive (evaluation output) | Results tab |
| `results/fl_ece_comparison.png` | Google Drive (evaluation output) | Results tab |
| `results/xai_entropy_heatmaps.png` | Google Drive (evaluation output) | Explainability tab |
| `results/xai_gradcam_panels.png` | Google Drive (evaluation output) | Explainability tab |
| `results/fl_heterogeneity_bars.png` | Google Drive (evaluation output) | Results tab |
| `results/froc_curve_fixed.png` | Google Drive (evaluation output) | Results tab |
| `results/calibration_analysis.png` | Google Drive (evaluation output) | Results tab |
| `results/fl_ablation_table.csv` | Google Drive (evaluation output) | Results tab table |
| `results/fl_statistical_tests.txt` | Google Drive (evaluation output) | Results tab stats |

All evaluation outputs are generated by:
```bash
# On Google Colab T4 (~1 compute unit)
python evaluation/complete_project_finish.py
```
