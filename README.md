---

## title: RareNoduleFL Demo
emoji: 🫁
colorFrom: teal
colorTo: blue
sdk: gradio
app_file: app.py
pinned: false
license: mit
short_description: Federated learning and explainable AI for rare lung nodule segmentation
tags:
  - federated-learning
  - medical-imaging
  - lung-nodule
  - pytorch
  - monai
  - gradio
  - xai
  - lidc-idri

# RareNoduleFL 🫁

**Privacy-Preserving Federated Learning for Rare Lung Nodule Detection**

> End-to-end implementation of federated learning across simulated hospital silos
> for detecting rare (<10mm) lung nodules in LIDC-IDRI CT scans, with
> Rare Uncertainty Sensitivity (RUS) loss, Grad-CAM++ explainability,
> and FROC evaluation.

---

## Live Demo

The Gradio app is hosted on this Hugging Face Space.
Upload any grayscale CT slice (PNG/JPG) or choose demo samples from `assets/samples`
to run segmentation inference, compare all three models, and browse
all project figures and statistical results.

---

## Key Results


| Experiment               | Dice      | Sensitivity | FROC @1FP/scan | ECE         |
| ------------------------ | --------- | ----------- | -------------- | ----------- |
| Central U-Net (baseline) | **0.844** | **90.9%**   | 0.355          | 0.0103      |
| FL IID — FedAvg          | 0.767     | 82.1%       | **0.959**      | 0.00063     |
| **FL Non-IID + RUS**     | 0.766     | 81.3%       | **0.962**      | **0.00065** |
| FL Non-IID (no RUS)      | 0.767     | 81.1%       | 0.959          | 0.00065     |


- **96.2% detection sensitivity** at 1 FP/scan — exceeds radiologist baseline (68%)
and project target (85%)
- **16× ECE improvement** under FL training (0.0103 → 0.00063)
- **McNemar p = 0.021** — RUS loss significantly changes rare-nodule prediction patterns
- **Paired t-test p = 0.708** — RUS does not degrade overall Dice

---

## Repository Structure

```
RareNoduleFL/
│
├── app.py                              ← Gradio app entrypoint (HF Spaces)
├── requirements.txt                    ← App dependencies
├── README.md
├── gitignore
│
├── app_core/                           ← App modules (deployment layer)
│   ├── __init__.py
│   ├── config.py                       ← Paths, labels, known metrics
│   ├── model_registry.py               ← Lazy model loading + caching
│   ├── preprocessing.py                ← Upload → tensor pipeline
│   ├── inference.py                    ← Single + multi-model inference
│   ├── visualization.py                ← Mask/overlay/heatmap rendering
│   ├── results_loader.py               ← Load figures, CSVs, stats text
│   └── sample_cases.py                 ← Auto-discover `assets/samples` + fallback generation
│
├── assets/
│   ├── css/app.css                     ← Custom academic dashboard CSS
│   └── samples/                        ← User-provided demo images (auto-discovered)
│
├── docs/
│   └── app_notes.md                    ← Deployment technical notes
│
├── preprocessing/
│   └── lidc_preprocessing.py           ← Step 1: LIDC-IDRI → NPY slices
│
├── training/
│   └── training.py
│
├── federated_learning/
│   └── federated_learning.py
│
├── evaluation/
│   └── evaluation.py
│
├── postprocessing/
│   └── postproccessing_and_initial_training.py
│
├── demo/
│   └── colab_demo.py
│
├── models/                             ← Model checkpoints (.pth, optional)
│
└── results/                            ← Precomputed figures/CSVs used by app
```

---

## Model Checkpoints

The app works without checkpoints — it displays precomputed results from `results/`
and shows known metrics from training logs.

To enable **live inference**, add these files to `models/`:


| File                            | Source                     | Dice  |
| ------------------------------- | -------------------------- | ----- |
| `step4.1_unet_best.pth`         | Central training (Step 6)  | 0.844 |
| `fl_iid_global_best.pth`        | FL IID experiment (Step 8) | 0.767 |
| `fl_noniid_rus_global_best.pth` | FL Non-IID+RUS (Step 9)    | 0.766 |


These are saved to Google Drive during training.
Download from your Drive and place in the `models/` folder.

All three use the **same architecture**:

```python
monai.networks.nets.UNet(
    spatial_dims=2, in_channels=1, out_channels=1,
    channels=(32,64,128,256,512), strides=(2,2,2,2),
    num_res_units=2, norm="batch", dropout=0.1
)
```

---

## Run Locally

```bash
git clone https://huggingface.co/spaces/YOUR_HF_USERNAME/RareNoduleFL
cd RareNoduleFL
pip install -r requirements.txt
python app.py
# Open http://localhost:7860
```

---

## Deploy to Hugging Face Spaces

1. Create a new Space at huggingface.co → New Space
2. Select **Gradio** SDK
3. Push this repo (or upload files via the Files tab)
4. Spaces auto-detects `app_file: app.py` from the README YAML
5. The app starts automatically — no GPU required (CPU Space is sufficient)

The app gracefully handles missing model checkpoints by displaying precomputed
metrics and figures from `results/`.

---

## Reproduce the Research Pipeline

The research pipeline runs entirely on Google Colab Pro.
See `evaluation/complete_project_finish.py` for the complete evaluation script.

### Run Order

```bash
# Step 1: Preprocessing (T4, ~1 unit)
preprocessing/lidc_preprocessing.py

# Steps 2-6: Central baselines (A100, ~18 units)
training/lidc_step2_step3_step4.py
training/step6_optimise_m1_target90_m3.py

# Steps 7-10: Federated Learning (L4, ~28 units)
federated/steps7_to_10_fl_experiments.py

# Steps 11-15 + all figures + fixes (T4, ~1 unit)
evaluation/complete_project_finish.py

# Step 18: Live demo (T4, ~0.2 units, optional)
demo/colab_demo.py
```

**Total compute: ~48 Colab Pro units.**
All random seeds fixed at 42. All dependency versions pinned.

---

## Dataset

LIDC-IDRI is publicly available from
[The Cancer Imaging Archive (TCIA)](https://www.cancerimagingarchive.net/collection/lidc-idri/).
This repository contains no raw CT data.

---

## Citation

```bibtex
@misc{rarenodulefl2025,
  title     = {RareNoduleFL: Federated Learning for Rare Lung Nodule Detection
               with Uncertainty-Aware Calibration and Grad-CAM++ Explainability},
  author    = {[Nandika Raj Varma]},
  year      = {2025},
  institution = {RV University}
}
```

---

## License

MIT — see [LICENSE](LICENSE)
