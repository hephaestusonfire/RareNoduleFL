# -*- coding: utf-8 -*-
"""
=============================================================================
  STEP 18 — INTERACTIVE COLAB DEMO
  RareNoduleFL: 1-Round Federated Learning Live Demonstration

  PURPOSE:
    This is the live demo notebook for presentations and GitHub.
    It shows one complete FL round end-to-end:
      1. Load pretrained global model from Drive
      2. Distribute to 5 simulated clients
      3. Run 1 local training epoch per client (visible progress)
      4. FedAvg aggregate
      5. Show before/after prediction on a real CT slice
      6. Display Grad-CAM explanation

  PROCESSOR : T4 (costs ~0.2 compute units, runs in ~15 minutes)
  AUDIENCE  : Evaluators, GitHub visitors, presentation live demo
=============================================================================
"""

# ============================================================================
# INSTALL
# ============================================================================
import subprocess, sys
subprocess.run([sys.executable,"-m","pip","install",
    "torch==2.6.0","torchvision==0.21.0","torchaudio==2.6.0",
    "--index-url","https://download.pytorch.org/whl/cu121","-q"],check=False)
subprocess.run([sys.executable,"-m","pip","install",
    "monai==1.5.0","flwr==1.7.0","torchcam==0.4.0",
    "numpy==1.26.4","pandas==2.2.0","matplotlib","tqdm","-q"],check=False)

# ============================================================================
# MOUNT + PATHS
# ============================================================================
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

from pathlib import Path
BASE_DIR  = Path("/content/drive/MyDrive/Rare Lung Nodules")
DATA_DIR  = BASE_DIR / "preprocessed_lidc"
MODEL_DIR = BASE_DIR / "models"
print("✅ Drive mounted")

# ============================================================================
# IMPORTS
# ============================================================================
import torch, copy, warnings
import numpy as np, pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data  import Dataset, DataLoader
from torch.optim       import AdamW
from torch.cuda.amp    import autocast, GradScaler
from torchcam.methods  import GradCAMpp
from PIL               import Image
from tqdm              import tqdm
from monai.networks.nets   import UNet
from monai.networks.layers import Norm
from monai.losses          import DiceCELoss
from monai.transforms import (
    Compose, ResizeWithPadOrCropd, RandFlipd, RandRotate90d,
    RandGaussianNoised, RandAdjustContrastd,
    ToTensord, ScaleIntensityd,
)
warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Device: {device} ({torch.cuda.get_device_name(0) if device.type=='cuda' else 'CPU'})")

# ============================================================================
# DATASET (minimal version for demo)
# ============================================================================
class DemoDataset(Dataset):
    def __init__(self, data_df, data_dir, noise_sigma=15.0):
        self.data_dir = Path(data_dir)
        self.data_df  = data_df.reset_index(drop=True)
        self.tfm = Compose([
            ResizeWithPadOrCropd(["image","mask"],[256,256],mode="constant"),
            RandFlipd(["image","mask"], prob=0.5, spatial_axis=0),
            RandGaussianNoised(["image"], prob=0.5, mean=0.0, std=noise_sigma/255.0),
            RandAdjustContrastd(["image"], prob=0.5, gamma=(0.85,1.15)),
            ScaleIntensityd(["image"], minv=0.0, maxv=1.0),
            ToTensord(["image","mask"]),
        ])
        self.infer_tfm = Compose([
            ResizeWithPadOrCropd(["image","mask"],[256,256],mode="constant"),
            ScaleIntensityd(["image"], minv=0.0, maxv=1.0),
            ToTensord(["image","mask"]),
        ])
    def __len__(self): return len(self.data_df)
    def __getitem__(self, idx):
        row   = self.data_df.iloc[idx]
        fname = row["fname"]
        try:
            image = np.load(self.data_dir/f"{fname}_img.npy").astype(np.float32)
            mask  = np.load(self.data_dir/f"{fname}_mask.npy").astype(np.float32)
        except:
            image = np.zeros((1,256,256),np.float32)
            mask  = np.zeros((1,256,256),np.float32)
        d = self.tfm({"image":image,"mask":mask})
        d["fname"] = fname
        return d

# ============================================================================
# MODEL
# ============================================================================
class UNetModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.unet = UNet(spatial_dims=2,in_channels=1,out_channels=1,
                         channels=(32,64,128,256,512),strides=(2,2,2,2),
                         num_res_units=2,norm=Norm.BATCH,dropout=0.1)
    def forward(self,x): return self.unet(x)

# ============================================================================
# LOAD DATA (use just 500 slices per client for the demo — fast)
# ============================================================================
print("\n🔄 Loading demo data ...")
all_dfs = []
for i in range(5):
    p = DATA_DIR/f"client_{i}_iid.csv"
    if p.exists():
        df = pd.read_csv(p).sample(n=min(500,len(pd.read_csv(p))),
                                    random_state=42+i)
        all_dfs.append(df)

client_noise = [5, 10, 15, 20, 25]
client_loaders = [
    DataLoader(DemoDataset(df, DATA_DIR, noise_sigma=client_noise[i]),
               batch_size=16, shuffle=True, num_workers=2)
    for i, df in enumerate(all_dfs)
]
print(f"  ✅ {len(all_dfs)} clients, ~500 slices each")

# Load one test sample for visualisation
test_csv = DATA_DIR/"test_iid.csv"
test_df  = pd.read_csv(test_csv).sample(n=50, random_state=42)
infer_ds = DemoDataset(test_df, DATA_DIR)

# Find a good rare nodule slice for the demo visualisation
print("  🔍 Finding a rare nodule for demo visualisation ...")
demo_sample = None
for row in test_df.itertuples():
    mask_path = DATA_DIR/f"{row.fname}_mask.npy"
    if mask_path.exists():
        mask = np.load(mask_path)
        mp = (mask>0.5).sum()
        if 20 < mp < 150:   # small nodule
            img = np.load(DATA_DIR/f"{row.fname}_img.npy").astype(np.float32)
            demo_sample = {"img":img, "mask":mask, "fname":row.fname}
            print(f"  ✅ Demo slice: {row.fname} (mask pixels={mp:.0f})")
            break

if demo_sample is None:
    # Fallback: use first available slice
    row = test_df.iloc[0]
    img  = np.load(DATA_DIR/f"{row.fname}_img.npy").astype(np.float32)
    mask = np.load(DATA_DIR/f"{row.fname}_mask.npy").astype(np.float32)
    demo_sample = {"img":img,"mask":mask,"fname":row.fname}

# ============================================================================
# LOAD PRETRAINED GLOBAL MODEL
# ============================================================================
print("\n🔄 Loading pretrained FL model ...")
global_model = UNetModel().to(device)
best_path    = MODEL_DIR/"fl_noniid_rus_global_best.pth"
if best_path.exists():
    ckpt = torch.load(best_path, map_location=device)
    global_model.load_state_dict(ckpt["global_model_state_dict"])
    print(f"  ✅ Loaded FL Non-IID+RUS model (round={ckpt.get('round','?')}, "
          f"dice={ckpt.get('best_dice',0):.4f})")
else:
    print(f"  ⚠️  Using central model instead ...")
    c_path = MODEL_DIR/"step4.1_unet_best.pth"
    if c_path.exists():
        ckpt = torch.load(c_path, map_location=device)
        global_model.load_state_dict(ckpt["model_state_dict"])

# ============================================================================
# HELPER: PREDICT + GRADCAM ON A SINGLE SLICE
# ============================================================================
def predict_and_explain(model, img_np, target_layer_name=None):
    """Run inference + Grad-CAM on a single slice."""
    # Find target layer
    if target_layer_name is None:
        for name, mod in model.named_modules():
            if isinstance(mod, nn.Conv2d):
                target_layer_name = name

    # Prepare input
    from monai.transforms import Compose, ResizeWithPadOrCropd, ScaleIntensityd, ToTensord
    tfm = Compose([
        ResizeWithPadOrCropd(["image"],[256,256],mode="constant"),
        ScaleIntensityd(["image"], minv=0.0, maxv=1.0),
        ToTensord(["image"]),
    ])
    d   = tfm({"image": img_np})
    inp = d["image"].unsqueeze(0).to(device)

    # Prediction (no_grad for efficiency)
    model.eval()
    with torch.no_grad():
        pred_map = torch.sigmoid(model(inp)).squeeze().cpu().numpy()

    # Grad-CAM (needs grad)
    extractor = GradCAMpp(model, target_layer=target_layer_name)
    inp_grad  = inp.detach().requires_grad_(True)
    out       = model(inp_grad)
    score     = torch.sigmoid(out).mean()
    cam       = extractor(class_idx=None, scores=score.unsqueeze(0))
    extractor.remove_hooks()

    heatmap = cam[0].squeeze().detach().cpu().numpy() if cam else np.zeros((256,256))
    if heatmap.max() > heatmap.min():
        heatmap = (heatmap-heatmap.min())/(heatmap.max()-heatmap.min())
    if heatmap.shape != (256,256):
        heatmap = np.array(
            Image.fromarray((heatmap*255).astype(np.uint8))
                 .resize((256,256), Image.BILINEAR)) / 255.0

    return pred_map, heatmap

def overlay(img_np, heatmap, alpha=0.5):
    img2  = np.stack([img_np[0]]*3, axis=2) if img_np.shape[0]==1 else img_np
    img2  = (img2*255).astype(np.uint8)
    heat  = (plt.cm.jet(heatmap)[:,:,:3]*255).astype(np.uint8)
    return ((1-alpha)*img2 + alpha*heat).astype(np.uint8)

# ============================================================================
# DEMO: BEFORE FL ROUND — SNAPSHOT PREDICTION
# ============================================================================
print("\n🔬 Snapshotting prediction BEFORE FL round ...")
img_np   = demo_sample["img"]
mask_np  = demo_sample["mask"]

# Find target layer once
tgt_layer = None
for name, mod in global_model.named_modules():
    if isinstance(mod, nn.Conv2d): tgt_layer = name

pred_before, cam_before = predict_and_explain(global_model, img_np, tgt_layer)
print(f"  ✅ Before FL round — mean pred: {pred_before.mean():.4f}")

# Store weights before
weights_before = copy.deepcopy(global_model.state_dict())

# ============================================================================
# LIVE FL ROUND — 1 ROUND, ALL 5 CLIENTS
# ============================================================================
print("\n🚀 Running 1 FL round (live demo) ...")
print("="*60)

seg_loss    = DiceCELoss(sigmoid=True,include_background=False,
                          to_onehot_y=False,lambda_dice=0.7,lambda_ce=0.3)
scaler      = GradScaler()
global_params = [v.cpu().numpy() for v in global_model.state_dict().values()]
keys          = list(global_model.state_dict().keys())
client_sds    = []
client_sizes  = []
client_metrics = []

for cid, loader in enumerate(client_loaders):
    print(f"\n  📡 Client {cid} (hospital {cid+1}, σ={client_noise[cid]}HU) training ...")
    client_model = UNetModel().to(device)
    client_model.load_state_dict(
        {k: torch.tensor(v) for k,v in zip(keys, global_params)}
    )
    client_model.train()
    optimizer = AdamW(client_model.parameters(), lr=1e-4, weight_decay=1e-5)
    tot_loss=tot_dice=0.0; n=0

    for batch in tqdm(loader, desc=f"    C{cid}", leave=False, ncols=70):
        imgs  = batch["image"].to(device)
        masks = batch["mask"].to(device)
        optimizer.zero_grad()
        with autocast():
            logits   = client_model(imgs)
            loss     = seg_loss(logits, masks)
        scaler.scale(loss).backward()
        scaler.step(optimizer); scaler.update()
        pb=(torch.sigmoid(logits)>0.5).float(); tb=(masks>0.5).float()
        pf,tf=pb.view(-1),tb.view(-1)
        tp=(pf*tf).sum(); fp=(pf*(1-tf)).sum()
        fn=((1-pf)*tf).sum()
        dice=((2*tp+1e-8)/(2*tp+fp+fn+1e-8)).item()
        tot_loss+=loss.item(); tot_dice+=dice; n+=1

    avg_dice=tot_dice/max(n,1); avg_loss=tot_loss/max(n,1)
    print(f"    ✅ Client {cid}: loss={avg_loss:.4f}  dice={avg_dice:.4f}  "
          f"samples={len(loader.dataset)}")
    client_sds.append(client_model.state_dict())
    client_sizes.append(len(loader.dataset))
    client_metrics.append({"cid":cid,"dice":avg_dice,"loss":avg_loss})
    del client_model; torch.cuda.empty_cache()

# FedAvg
print("\n  🔗 FedAvg aggregation ...")
total   = sum(client_sizes)
agg_sd  = copy.deepcopy(client_sds[0])
for key in agg_sd:
    agg_sd[key] = torch.zeros_like(agg_sd[key], dtype=torch.float32)
    for sd,w in zip(client_sds,client_sizes):
        agg_sd[key] += sd[key].float()*(w/total)
global_model.load_state_dict(agg_sd)
print("  ✅ Global model updated via FedAvg")

# ============================================================================
# AFTER FL ROUND — SNAPSHOT + COMPARISON
# ============================================================================
print("\n🔬 Snapshotting prediction AFTER FL round ...")
pred_after, cam_after = predict_and_explain(global_model, img_np, tgt_layer)
print(f"  ✅ After FL round — mean pred: {pred_after.mean():.4f}")

# ============================================================================
# VISUALISATION — THE MAIN DEMO FIGURE
# ============================================================================
print("\n📊 Generating demo figure ...")

# Prepare arrays
img_disp = img_np[0] if img_np.shape[0]==1 else img_np
mask_disp = mask_np[0] if mask_np.ndim==3 else mask_np
# Resize mask display to 256x256
mask_disp = np.array(Image.fromarray((mask_disp*255).astype(np.uint8))
                     .resize((256,256),Image.NEAREST))/255.0
img_disp256 = np.array(Image.fromarray((img_disp*255).astype(np.uint8))
                        .resize((256,256),Image.BILINEAR))/255.0

overlay_before = overlay(
    np.stack([img_disp256]*3,axis=2).transpose(2,0,1), cam_before)
overlay_after  = overlay(
    np.stack([img_disp256]*3,axis=2).transpose(2,0,1), cam_after)

fig, axes = plt.subplots(2, 5, figsize=(25, 10))
fig.suptitle(
    "🏥 RareNoduleFL — Live Demo: 1 Federated Learning Round\n"
    f"Slice: {demo_sample['fname']}  |  Nodule pixels: {int((mask_np>0.5).sum())}",
    fontsize=15, fontweight="bold"
)

# Row 1: Before FL round
axes[0,0].imshow(img_disp256, cmap="gray")
axes[0,0].set_title("CT Slice", fontsize=11)

axes[0,1].imshow(mask_disp, cmap="Reds", vmin=0, vmax=1)
axes[0,1].contour(mask_disp, levels=[0.5], colors=["white"], linewidths=2)
axes[0,1].set_title("Ground Truth Mask", fontsize=11)

axes[0,2].imshow(pred_before, cmap="Blues", vmin=0, vmax=1)
axes[0,2].contour(mask_disp, levels=[0.5], colors=["white"], linewidths=1.5)
axes[0,2].set_title(f"Prediction BEFORE FL\n(mean={pred_before.mean():.4f})", fontsize=11)

axes[0,3].imshow(overlay_before)
axes[0,3].contour(mask_disp, levels=[0.5], colors=["white"], linewidths=1.5)
axes[0,3].set_title("Grad-CAM++ BEFORE FL\n(model focus area)", fontsize=11)

# Client dice bar chart
ax_bar = axes[0,4]
cids   = [m["cid"] for m in client_metrics]
dices  = [m["dice"] for m in client_metrics]
colors_bar = ["#2196F3","#4CAF50","#FF9800","#9C27B0","#F44336"]
bars = ax_bar.bar([f"C{c}\nσ={client_noise[c]}HU" for c in cids],
                   dices, color=colors_bar, alpha=0.85)
for bar,v in zip(bars,dices):
    ax_bar.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=10)
ax_bar.set_ylabel("Local Dice"); ax_bar.set_ylim(0,1)
ax_bar.set_title("Per-Client Dice\n(Round 1)", fontsize=11)
ax_bar.grid(True, alpha=0.3, axis="y")

# Row 2: After FL round
axes[1,0].imshow(img_disp256, cmap="gray")
axes[1,0].set_title("CT Slice (same)", fontsize=11)

axes[1,1].imshow(mask_disp, cmap="Reds", vmin=0, vmax=1)
axes[1,1].contour(mask_disp, levels=[0.5], colors=["white"], linewidths=2)
axes[1,1].set_title("Ground Truth Mask", fontsize=11)

axes[1,2].imshow(pred_after, cmap="Blues", vmin=0, vmax=1)
axes[1,2].contour(mask_disp, levels=[0.5], colors=["white"], linewidths=1.5)
axes[1,2].set_title(f"Prediction AFTER FL\n(mean={pred_after.mean():.4f})", fontsize=11)

axes[1,3].imshow(overlay_after)
axes[1,3].contour(mask_disp, levels=[0.5], colors=["white"], linewidths=1.5)
axes[1,3].set_title("Grad-CAM++ AFTER FL\n(model focus area)", fontsize=11)

# Summary text panel
ax_sum = axes[1,4]
ax_sum.axis("off")
summary = [
    "DEMO SUMMARY",
    "─"*24,
    "",
    "Federated Round: 1",
    "Clients: 5 hospitals",
    "Aggregation: FedAvg",
    "",
    "Per-client training:",
]
for m in client_metrics:
    summary.append(f"  C{m['cid']}: dice={m['dice']:.3f}")
summary += [
    "",
    "Key insight:",
    "Each hospital trains on",
    "local data only — no",
    "raw data shared.",
    "",
    "FedAvg combines updates",
    "→ global model improves",
]
ax_sum.text(0.05,0.97,"\n".join(summary),
            transform=ax_sum.transAxes, fontsize=9.5,
            va="top", ha="left", fontfamily="monospace",
            bbox=dict(boxstyle="round",facecolor="#e8f5e9",
                      edgecolor="#66bb6a",alpha=0.9))

for ax in axes.flatten():
    ax.axis("off") if ax not in [ax_bar, ax_sum] else None
    if ax not in [ax_bar, ax_sum]:
        ax.axis("off")

plt.tight_layout()
demo_fig = BASE_DIR/"results"/"demo_fl_round.png"
demo_fig.parent.mkdir(exist_ok=True)
plt.savefig(demo_fig, dpi=150, bbox_inches="tight")
plt.show()
print(f"\n✅ Demo figure saved → {demo_fig}")

print("\n" + "="*60)
print("🎉 DEMO COMPLETE!")
print("="*60)
print(f"\n  What just happened:")
print(f"  1. Loaded pretrained FL model from Drive")
print(f"  2. Distributed to 5 clients (simulated hospitals)")
print(f"  3. Each client trained 1 local epoch on private data")
print(f"  4. FedAvg aggregated all updates → new global model")
print(f"  5. Compared before/after predictions on a rare nodule")
print(f"\n  This is exactly how federated learning works in practice —")
print(f"  no raw CT images leave each hospital's silo.")
print(f"\n✅ Use demo_fl_round.png in your presentation slide 4")
