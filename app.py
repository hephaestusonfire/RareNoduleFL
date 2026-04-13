"""
app.py — RareNoduleFL Gradio Spaces entrypoint
Run locally: python app.py
Hugging Face Spaces: auto-detected via app_file in README YAML

Tabs: Overview | Inference | Compare | Explainability | Results | About
"""

from pathlib import Path
import numpy as np
import gradio as gr
from PIL import Image
from html import escape

from app_core.config import (
    APP_TITLE, APP_SUBTITLE, APP_DESCRIPTION,
    MODEL_KEYS, DEFAULT_MODEL, DEFAULT_THRESHOLD,
    KNOWN_METRICS, STAT_HIGHLIGHTS, FIGURES,
)
from app_core.sample_cases import (
    ensure_samples_exist, get_sample_names,
    get_sample_path, get_xai_figure,
)
from app_core.model_registry import check_availability, preload_all
from app_core.inference      import run_single_inference, run_comparison
from app_core.results_loader import (
    load_all_figures, load_ablation_table,
    load_statistical_tests, build_metrics_summary_html,
    load_result_image,
)

# ── Startup ────────────────────────────────────────────────────────────────────
print("Starting RareNoduleFL ...")
ensure_samples_exist()
availability = check_availability()
print("Checkpoints:", {k: ("✅" if v else "❌") for k, v in availability.items()})
preload_all()

# ── CSS ────────────────────────────────────────────────────────────────────────
_css_path = Path(__file__).parent / "assets" / "css" / "app.css"
CUSTOM_CSS = _css_path.read_text() if _css_path.exists() else ""

# ── Helpers ────────────────────────────────────────────────────────────────────

def _avail_html(key: str) -> str:
    ok = availability.get(key, False)
    icon = "●" if ok else "○"
    text = "Live inference" if ok else "Precomputed metrics only"
    cls = "eg-avail eg-avail-live" if ok else "eg-avail eg-avail-off"
    return f'<span class="{cls}">{icon} {text}</span>'


def _stat_card(value: str, label: str) -> str:
    return (f'<div class="stat-card">'
            f'<div class="stat-value">{value}</div>'
            f'<div class="stat-label">{label}</div>'
            f'</div>')


def _model_status_md() -> str:
    lines = ["### Model Checkpoint Status\n"]
    for k in MODEL_KEYS:
        ok = availability.get(k, False)
        m  = KNOWN_METRICS.get(k, {})
        icon = "✅" if ok else "⚠️"
        lines.append(
            f"{icon} **{k}** — "
            f"Dice `{m.get('dice','—')}` | "
            f"FROC@1FP `{m.get('froc_1fp','—')}` | "
            f"ECE `{m.get('ece','—')}`"
        )
        if not ok:
            lines.append(
                "   *Checkpoint missing. "
                "See README → Model Checkpoints section.*\n"
            )
    return "\n".join(lines)


# ── Callbacks ─────────────────────────────────────────────────────────────────

def _resolve_source(image_input, sample_name):
    if image_input is not None:
        return image_input
    if sample_name and sample_name != "— Upload your own —":
        p = get_sample_path(sample_name)
        if p and Path(p).exists():
            return str(p)
    return None


def cb_infer(image_input, sample_name, model_key, threshold):
    source = _resolve_source(image_input, sample_name)
    empty  = np.zeros((256, 256, 3), dtype=np.uint8)
    if source is None:
        return empty, empty, empty, "No input. Upload an image or select a sample."
    r = run_single_inference(source, model_key, float(threshold))
    return r["display_img"], r["binary_mask"], r["overlay"], r["summary"]


def cb_clear():
    empty = np.zeros((256, 256, 3), dtype=np.uint8)
    return None, "— Upload your own —", empty, empty, empty, ""


def cb_compare(image_input, sample_name, threshold):
    source = _resolve_source(image_input, sample_name)
    empty  = np.zeros((256, 256, 3), dtype=np.uint8)
    if source is None:
        return (*([empty] * 6), "No input provided.")
    results  = run_comparison(source, float(threshold))
    out_imgs = []
    summaries = []
    for key in MODEL_KEYS:
        r = results.get(key, {})
        out_imgs.append(r.get("binary_mask", empty))
        out_imgs.append(r.get("overlay",     empty))
        summaries.append(_comparison_summary_card(key, r.get("summary", "")))
    return (*out_imgs, "".join(summaries))


def cb_xai_sample(name):
    img = get_xai_figure(name)
    if img is None:
        img = load_result_image("gradcam_panels")
    if img is None:
        img = np.zeros((400, 800, 3), dtype=np.uint8)
    return img


def _comparison_summary_card(model_key: str, summary_text: str) -> str:
    """Render structured comparison report card from plain summary text."""
    fields = {}
    perf = {}
    in_perf = False
    notes = []
    for raw in summary_text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.lower().startswith("model performance"):
            in_perf = True
            continue
        if ":" in line:
            k, v = [x.strip() for x in line.split(":", 1)]
            if in_perf:
                perf[k] = v
            else:
                fields[k] = v
        elif "research demo only" not in line.lower():
            notes.append(line)

    note = escape(notes[0]) if notes else "Research comparison summary."
    return f"""
    <div class="eg-panel eg-panel-strong eg-compare-card">
      <div class="eg-section-title">{escape(model_key)}</div>
      <div class="eg-kv-grid">
        <div><span class="eg-k">Threshold</span><span class="eg-v">{escape(fields.get('Threshold', 'N/A'))}</span></div>
        <div><span class="eg-k">Detected Pixels</span><span class="eg-v">{escape(fields.get('Detected pixels', 'N/A'))}</span></div>
        <div><span class="eg-k">Max Confidence</span><span class="eg-v">{escape(fields.get('Max confidence', 'N/A'))}</span></div>
        <div><span class="eg-k">Mean Confidence</span><span class="eg-v">{escape(fields.get('Mean confidence', 'N/A'))}</span></div>
      </div>
      <div class="eg-subtitle">Performance Block</div>
      <div class="eg-kv-grid">
        <div><span class="eg-k">Dice</span><span class="eg-v">{escape(perf.get('Dice score', 'N/A'))}</span></div>
        <div><span class="eg-k">Sensitivity</span><span class="eg-v">{escape(perf.get('Sensitivity', 'N/A'))}</span></div>
        <div><span class="eg-k">FROC @1FP</span><span class="eg-v">{escape(perf.get('FROC @1FP', 'N/A'))}</span></div>
        <div><span class="eg-k">ECE</span><span class="eg-v">{escape(perf.get('ECE', 'N/A'))}</span></div>
      </div>
      <div class="eg-callout">{note}</div>
      <div class="eg-disclaimer-inline">Research demo only — not a medical device.</div>
    </div>
    """


def cb_sample_preview(sample_name):
    """Preview the currently selected demo sample from assets/samples."""
    if not sample_name or sample_name == "— Upload your own —":
        return None
    p = get_sample_path(sample_name)
    if p and Path(p).exists():
        return np.array(Image.open(p).convert("RGB"))
    return None


# ── UI ─────────────────────────────────────────────────────────────────────────

sample_choices = ["— Upload your own —"] + get_sample_names()

with gr.Blocks(
    title=f"{APP_TITLE}",
    css=CUSTOM_CSS,
    theme=gr.themes.Base(
        primary_hue="teal",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Inter"),
    ),
) as demo:

    # Hero
    gr.HTML(f"""
    <div class="hero-header">
      <div class="hero-title">🫁 {APP_TITLE}</div>
      <div class="hero-subtitle">{APP_SUBTITLE}</div>
    </div>
    """)

    with gr.Tabs():

        # ── Tab 1: Overview ────────────────────────────────────────────────────
        with gr.Tab("📋 Overview"):
            with gr.Row():
                for v, l in STAT_HIGHLIGHTS[:3]:
                    gr.HTML(_stat_card(v, l))
            with gr.Row():
                for v, l in STAT_HIGHLIGHTS[3:]:
                    gr.HTML(_stat_card(v, l))

            gr.Markdown(f"""
## What This App Demonstrates

{APP_DESCRIPTION}

**Explore via the tabs above:**
- **Inference** — upload any CT slice, run segmentation with one model
- **Compare** — run all 3 models side-by-side on the same input
- **Explainability** — precomputed Grad-CAM++ and entropy heatmaps
- **Results** — full figure gallery, ablation table, statistical tests
""")
            _pipeline = load_result_image("pipeline")
            if _pipeline is not None:
                gr.Image(value=_pipeline, label="Figure 1 — System Pipeline",
                         interactive=False)
            else:
                gr.Markdown(
                    "*Pipeline figure appears here after running "
                    "`evaluation/complete_project_finish.py`.*"
                )
            gr.Markdown(_model_status_md())

        # ── Tab 2: Inference ───────────────────────────────────────────────────
        with gr.Tab("🔬 Inference"):
            gr.Markdown("### Single-Model Segmentation")
            with gr.Row():
                with gr.Column(scale=1):
                    inf_upload  = gr.Image(label="Upload CT Slice",
                                           type="numpy", image_mode="L",
                                           sources=["upload"])
                    inf_sample  = gr.Dropdown(choices=sample_choices,
                                              value=sample_choices[0],
                                              label="Or choose from assets/samples")
                    inf_sample_preview = gr.Image(
                        label="Selected sample preview (assets/samples)",
                        interactive=False,
                    )
                    inf_model   = gr.Dropdown(choices=MODEL_KEYS,
                                              value=DEFAULT_MODEL,
                                              label="Model")
                    inf_thr     = gr.Slider(0.1, 0.9, DEFAULT_THRESHOLD, step=0.05,
                                            label="Detection Threshold")
                    with gr.Row():
                        inf_run   = gr.Button("▶  Run", variant="primary")
                        inf_clear = gr.Button("✕  Clear")

                    for k in MODEL_KEYS:
                        gr.HTML(
                            _avail_html(k)
                            + f' &nbsp;<span class="eg-model-name">— {k}</span>'
                        )

                with gr.Column(scale=2):
                    with gr.Row():
                        inf_orig = gr.Image(label="Input",   interactive=False)
                        inf_mask = gr.Image(label="Mask",    interactive=False)
                        inf_over = gr.Image(label="Overlay", interactive=False)
                    inf_summary = gr.Textbox(label="Summary", lines=14,
                                             interactive=False)

            gr.HTML('<div class="disclaimer">⚕️ <strong>Research demo only</strong> — not a medical device.</div>')

            inf_run.click(cb_infer,
                [inf_upload, inf_sample, inf_model, inf_thr],
                [inf_orig, inf_mask, inf_over, inf_summary])
            inf_sample.change(cb_sample_preview, [inf_sample], [inf_sample_preview])
            inf_clear.click(cb_clear, [],
                [inf_upload, inf_sample, inf_orig, inf_mask, inf_over, inf_summary])

        # ── Tab 3: Compare ─────────────────────────────────────────────────────
        with gr.Tab("⚖️ Compare"):
            gr.Markdown("### All Three Models — Side-by-Side")
            with gr.Row():
                cmp_upload = gr.Image(label="Upload CT Slice", type="numpy",
                                      image_mode="L", sources=["upload"], scale=1)
                cmp_sample = gr.Dropdown(choices=sample_choices,
                                         value=sample_choices[1],
                                         label="Or select sample", scale=1)
                cmp_thr    = gr.Slider(0.1, 0.9, DEFAULT_THRESHOLD, step=0.05,
                                       label="Threshold", scale=1)
                cmp_btn    = gr.Button("▶  Compare All", variant="primary", scale=1)

            gr.Markdown("---")

            cmp_outputs = []
            with gr.Row():
                for key in MODEL_KEYS:
                    with gr.Column():
                        gr.Markdown(f"#### {key}")
                        gr.HTML(_avail_html(key))
                        mask_img = gr.Image(label="Mask",    interactive=False)
                        over_img = gr.Image(label="Overlay", interactive=False)
                        cmp_outputs += [mask_img, over_img]

            cmp_summary = gr.HTML(label="Comparison Summary")
            gr.HTML('<div class="disclaimer">⚕️ Research demo only.</div>')

            cmp_btn.click(cb_compare,
                [cmp_upload, cmp_sample, cmp_thr],
                cmp_outputs + [cmp_summary])

        # ── Tab 4: Explainability ──────────────────────────────────────────────
        with gr.Tab("🧠 Explainability"):
            gr.HTML("""
            <div class="eg-panel eg-panel-soft">
              <div class="eg-section-title">Grad-CAM++ and Uncertainty Maps</div>
              <p class="eg-body">Precomputed from the LIDC test set using the FL Non-IID+RUS model.</p>
              <p class="eg-body"><strong>Grad-CAM++ (Figure 5):</strong> activation overlays highlight where the model attends while segmenting candidate nodules.</p>
              <p class="eg-body"><strong>Entropy maps (Figure 4b):</strong> uncertainty rises near ambiguous boundaries and clinically sensitive regions.</p>
            </div>
            """)
            xai_sel = gr.Dropdown(choices=get_sample_names(),
                                   value=get_sample_names()[0],
                                   label="Sample → linked XAI figure")

            with gr.Row():
                with gr.Column():
                    gc_img = gr.Image(label="Grad-CAM++ Panels",
                                      interactive=False,
                                      value=load_result_image("gradcam_panels"))
                with gr.Column():
                    ent_img = gr.Image(label="Entropy — FL+RUS vs FL-IID",
                                       interactive=False,
                                       value=load_result_image("entropy_maps"))

            gr.HTML("""
            <div class="eg-panel eg-panel-soft">
              <div class="eg-section-title">How to Read Figure 5</div>
              <ul class="eg-list">
                <li><strong>Green</strong>: Hit — model correctly localises rare nodule.</li>
                <li><strong>Red</strong>: Miss — nodule is not detected.</li>
                <li><strong>Amber</strong>: Boundary — partial detection.</li>
              </ul>
              <div class="eg-caption">White contour: ground truth. Yellow dashed contour: model prediction.</div>
            </div>
            """)
            xai_sel.change(cb_xai_sample, [xai_sel], [gc_img])

        # ── Tab 5: Results ─────────────────────────────────────────────────────
        with gr.Tab("📊 Results"):
            gr.HTML(build_metrics_summary_html())
            gr.Markdown("---\n### Ablation Table — Paper Table 2")
            gr.Dataframe(value=load_ablation_table(),
                         label="", interactive=False, wrap=True)
            gr.Markdown("""
> FL models trade ~7.7% Dice for complete data privacy.
> FROC @1FP (96.2%) is the detection headline: can the model find a nodule when it fires?
> ECE 16× improvement reflects more calibrated confidence under federated aggregation.
""")
            gr.Markdown("---\n### Statistical Validation")
            gr.Textbox(value=load_statistical_tests(),
                       label="fl_statistical_tests.txt",
                       lines=22, interactive=False)

            gr.Markdown("---\n### All Project Figures")
            _all_figs = load_all_figures()
            if _all_figs:
                gr.Gallery(value=_all_figs,
                           label="Figures 1–7 + Calibration (scroll / click to enlarge)",
                           columns=3, rows=3, height=620,
                           object_fit="contain", allow_preview=True)
            else:
                gr.Markdown(
                    "*Run `evaluation/complete_project_finish.py` on Colab "
                    "to generate figures, then add them to `results/`.*"
                )

            gr.Markdown("""
---
| Figure | Description |
|--------|-------------|
| Figure 1 | System pipeline overview |
| Figure 3 | FL learning curves — global Dice, sensitivity, ECE per round |
| Figure 3b | Per-client drift across rounds |
| Figure 4 | ECE comparison: central vs FL IID vs FL+RUS |
| Figure 4b | Entropy heatmaps — rare nodule confidence |
| Figure 5 | Grad-CAM++ panels — hits, misses, boundary cases |
| Figure 6 | Per-client heterogeneity bars |
| Figure 7 | FROC curve — IoU threshold sweep |
| Calibration | Detection ECE decomposition: overall vs rare |
""")

        # ── Tab 6: About ───────────────────────────────────────────────────────
        with gr.Tab("ℹ️ About"):
            gr.HTML("""
            <div class="eg-panel eg-panel-strong">
              <div class="eg-section-title">About RareNoduleFL</div>
              <p class="eg-body"><strong>Problem Statement.</strong> Rare nodules (&lt;10mm) are frequently missed while cross-site data sharing remains constrained by privacy regulations.</p>
              <p class="eg-body"><strong>Solution.</strong> Federated learning (FedAvg across five simulated hospital clients) with uncertainty-aware RUS loss and explainability workflows.</p>
              <p class="eg-body"><strong>Dataset.</strong> LIDC-IDRI, 700 patients, 15,116 slices (256×256), evaluated with IID and non-IID client splits.</p>
              <div class="eg-subtitle">RUS Equation</div>
              <pre class="eg-code">RUS = mean(is_small_mask × entropy(p)) + ECE(p, y) × 0.1</pre>
              <div class="eg-subtitle">Key Findings</div>
              <ul class="eg-list">
                <li>FROC@1FP reaches 96.2% with FL Non-IID + RUS.</li>
                <li>ECE improves 16× under federated aggregation.</li>
                <li>McNemar p=0.021 confirms meaningful RUS effect.</li>
                <li>Paired t-test p=0.708 shows no Dice degradation.</li>
              </ul>
              <div class="eg-subtitle">Repository Structure</div>
              <pre class="eg-code">RareNoduleFL/
├── app.py
├── requirements.txt
├── README.md
├── gitignore
├── app_core/
├── assets/
├── docs/
├── training/
├── federated_learning/
├── evaluation/
├── preprocessing/
├── postprocessing/
├── demo/
├── models/
└── results/</pre>
              <div class="eg-caption">RV University · Final Year Project · Nandika Raj Varma</div>
            </div>
            """)

    # Footer
    gr.HTML("""
    <div class="app-footer">
      RareNoduleFL &nbsp;·&nbsp; LIDC-IDRI &nbsp;·&nbsp;
      PyTorch + MONAI + Flower + Gradio &nbsp;·&nbsp;
      <strong>Research demo only — not a medical device</strong>
    </div>
    """)


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=True)
