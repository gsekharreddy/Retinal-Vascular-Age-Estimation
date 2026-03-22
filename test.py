"""
Retinal Biological Age Gap Predictor
======================================
Predicts the biological age GAP between two retinal scans of the same subject.

Input signals (fused):
  1. Geometric stream  — displacement vectors from control point TXT files
                         (dx, dy per landmark → magnitude, angle, stats)
  2. Visual stream     — paired retinal images fed through a Siamese CNN
  3. Fusion head       — MLP that combines both streams → scalar age gap (years)

Dataset / File conventions:
  Ground Truth/   → control_points_{SubjectID}_{img1}_{img2}.txt
  Images/         → {SubjectID}_{img1}.ext  and  {SubjectID}_{img2}.ext
  Masks/          → mask.png  (circular FOV mask, applied to both images)

TXT format  (one row per landmark):
  x1  y1  x2  y2          (space-separated floats)

Label convention:
  The MAGNITUDE of retinal displacement encodes biological change.
  If you have explicit ground-truth age gaps (years), put them in a CSV:
      subject_id, age_gap
  and pass --label_csv path/to/labels.csv
  Otherwise the script runs in SELF-SUPERVISED mode and uses the mean
  displacement magnitude as a proxy target (good for pre-training /
  exploratory work).
"""

import os
import re
import glob
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from torch.amp import autocast
from torch.optim.lr_scheduler import CosineAnnealingLR


# ─────────────────────────────────────────────────────────────────
#  1.  CONFIG
# ─────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(description="Retinal Age Gap Trainer")
    p.add_argument("--data_root",    default=".",          help="Root of dataset")
    p.add_argument("--output_dir",   default="./outputs",  help="Checkpoints & logs")
    p.add_argument("--label_csv",    default=None,
                   help="Optional CSV with columns [subject_id, age_gap]. "
                        "If omitted, mean displacement magnitude is used as proxy.")
    p.add_argument("--img_size",     type=int,   default=512)
    p.add_argument("--batch_size",   type=int,   default=8)
    p.add_argument("--epochs",       type=int,   default=80)
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--val_split",    type=float, default=0.2)
    p.add_argument("--backbone",     default="efficientnet_b3",
                   choices=["efficientnet_b3", "resnet50", "convnext_small"])
    p.add_argument("--geo_only",     action="store_true",
                   help="Use only geometric features (no CNN images needed)")
    p.add_argument("--img_only",     action="store_true",
                   help="Use only Siamese CNN stream (ignore control points)")
    p.add_argument("--use_mask",     action="store_true", default=True)
    p.add_argument("--workers",      type=int,   default=4)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--resume",       default=None)
    p.add_argument("--amp",          action="store_true", default=True)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────
#  2.  CONTROL POINT PARSING & FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────────

def parse_cp_filename(fname: str):
    """
    Extracts (subject_id, img1_idx, img2_idx) from filename like:
      control_points_A03_1_2.txt   →  ('A03', '1', '2')
      12345_control_points_A03_1_2.txt  →  ('A03', '1', '2')
    """
    stem = Path(fname).stem
    # Match pattern: control_points_<SubjectID>_<i>_<j>
    m = re.search(r'control_points_([^_]+)_(\d+)_(\d+)', stem)
    if m:
        return m.group(1), m.group(2), m.group(3)
    raise ValueError(f"Cannot parse subject/pair from filename: {fname}")


def load_control_points(txt_path: str) -> np.ndarray:
    """Returns (N, 4) array: x1 y1 x2 y2 per row."""
    return np.loadtxt(txt_path, dtype=np.float32)


def extract_geo_features(cp: np.ndarray) -> np.ndarray:
    """
    Given (N, 4) control points, compute a rich feature vector:
      Per landmark (N):
        dx, dy           — raw displacement components
        magnitude        — Euclidean displacement
        angle            — direction of displacement (radians)
        norm_x1, norm_y1 — spatial position of source point (normalised 0-1)

    Summary statistics (over all N landmarks):
        mean/std/max of magnitude
        mean/std of |dx|, |dy|
        mean angle, circular variance of angle
        anisotropy = std(dx) / (std(dy) + 1e-6)

    Returns 1-D float32 vector.
    """
    x1, y1 = cp[:, 0], cp[:, 1]
    x2, y2 = cp[:, 2], cp[:, 3]
    dx = x2 - x1
    dy = y2 - y1
    mag  = np.sqrt(dx**2 + dy**2)
    angle = np.arctan2(dy, dx)

    # Normalise positions to [0,1] using data range
    xrange = max(x1.max() - x1.min(), 1.0)
    yrange = max(y1.max() - y1.min(), 1.0)
    nx1 = (x1 - x1.min()) / xrange
    ny1 = (y1 - y1.min()) / yrange

    # Per-landmark features: shape (N, 6)
    per_lm = np.stack([dx, dy, mag, angle, nx1, ny1], axis=1).flatten()

    # Summary statistics
    stats = np.array([
        mag.mean(), mag.std(), mag.max(), mag.min(),
        np.abs(dx).mean(), np.abs(dx).std(),
        np.abs(dy).mean(), np.abs(dy).std(),
        np.sin(angle).mean(), np.cos(angle).mean(),   # circular mean
        1.0 - np.cos(angle - angle.mean()).mean(),     # circular variance
        dx.std() / (dy.std() + 1e-6),                 # anisotropy
        (mag > mag.mean()).sum() / len(mag),           # fraction of large disps
    ], dtype=np.float32)

    return np.concatenate([per_lm, stats]).astype(np.float32)


def scan_control_points(gt_dir: str):
    """
    Scans Ground Truth/ and returns a list of dicts:
      { subject_id, img1_idx, img2_idx, txt_path,
        geo_features (ndarray), proxy_target (float) }
    """
    txt_files = sorted(glob.glob(os.path.join(gt_dir, "*.txt")))
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in: {gt_dir}")

    records = []
    for fpath in txt_files:
        try:
            subj, i1, i2 = parse_cp_filename(fpath)
        except ValueError as e:
            print(f"[WARN] Skipping {fpath}: {e}")
            continue

        cp  = load_control_points(fpath)
        geo = extract_geo_features(cp)
        proxy = float(np.sqrt((cp[:,2]-cp[:,0])**2 +
                              (cp[:,3]-cp[:,1])**2).mean())   # mean disp magnitude

        records.append({
            "subject_id":  subj,
            "img1_idx":    i1,
            "img2_idx":    i2,
            "txt_path":    fpath,
            "geo_features": geo,
            "proxy_target": proxy,
        })

    print(f"[GT] {len(records)} control-point pairs loaded.")
    return records


# ─────────────────────────────────────────────────────────────────
#  3.  IMAGE UTILITIES
# ─────────────────────────────────────────────────────────────────

def build_image_index(img_dir: str) -> dict:
    """Returns {stem_lower: full_path} for all images in img_dir."""
    index = {}
    for p in Path(img_dir).rglob("*"):
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
            index[p.stem.lower()] = str(p)
    return index


def find_image(index: dict, subject_id: str, img_idx: str):
    """
    Tries common naming patterns:
      {SubjectID}_{idx}   |   {idx}_{SubjectID}   |   {SubjectID}{idx}
    Returns path or None.
    """
    candidates = [
        f"{subject_id}_{img_idx}",
        f"{subject_id.lower()}_{img_idx}",
        f"{img_idx}_{subject_id}",
        f"{subject_id}{img_idx}",
        subject_id,   # fallback: just subject id (single image per subject)
    ]
    for c in candidates:
        if c.lower() in index:
            return index[c.lower()]
    return None


def load_mask(mask_dir: str, img_size: int) -> torch.Tensor | None:
    for name in ["mask.png", "feature_mask.png"]:
        path = os.path.join(mask_dir, name)
        if os.path.exists(path):
            m = Image.open(path).convert("L").resize((img_size, img_size), Image.NEAREST)
            t = torch.from_numpy(np.array(m)).float() / 255.0
            return (t > 0.5).float().unsqueeze(0)
    print("[WARN] No mask found.")
    return None


# ─────────────────────────────────────────────────────────────────
#  4.  DATASET
# ─────────────────────────────────────────────────────────────────

class RetinalAgePairDataset(Dataset):
    def __init__(self, records, img_index, transform,
                 mask_tensor=None, geo_only=False, img_only=False):
        self.transform   = transform
        self.mask        = mask_tensor
        self.geo_only    = geo_only
        self.img_only    = img_only

        self.samples = []
        missing_imgs = 0

        for r in records:
            p1 = find_image(img_index, r["subject_id"], r["img1_idx"])
            p2 = find_image(img_index, r["subject_id"], r["img2_idx"])

            if not geo_only and (p1 is None or p2 is None):
                missing_imgs += 1
                if not img_only:
                    # Geo-only fallback for this sample
                    pass
                else:
                    continue   # skip entirely if we need images

            self.samples.append({
                "geo":    torch.from_numpy(r["geo_features"]),
                "target": torch.tensor(r["proxy_target"], dtype=torch.float32),
                "img1":   p1,
                "img2":   p2,
                "subj":   r["subject_id"],
            })

        if missing_imgs:
            print(f"[Dataset] {missing_imgs} pairs missing one/both images "
                  f"(geo features still used).")
        print(f"[Dataset] {len(self.samples)} samples ready.")

    def _load_img(self, path):
        if path is None:
            return torch.zeros(3, 512, 512)   # placeholder when image is missing
        img = Image.open(path).convert("RGB")
        t   = self.transform(img)                    # CPU tensor
        if self.mask is not None:
            t = t * self.mask.cpu()                  # mask must stay CPU in workers
        return t

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s   = self.samples[idx]
        geo = s["geo"]
        tgt = s["target"]

        if self.geo_only:
            return geo, tgt

        img1 = self._load_img(s["img1"])
        img2 = self._load_img(s["img2"])

        if self.img_only:
            return img1, img2, tgt

        return geo, img1, img2, tgt


# ─────────────────────────────────────────────────────────────────
#  5.  MODEL
# ─────────────────────────────────────────────────────────────────

class GeoStream(nn.Module):
    """MLP that processes geometric displacement features."""
    def __init__(self, in_dim: int, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.SiLU(), nn.Dropout(0.3),
            nn.Linear(256, 256),   nn.SiLU(), nn.Dropout(0.2),
            nn.Linear(256, out_dim), nn.SiLU(),
        )
    def forward(self, x):
        return self.net(x)


class SiameseCNN(nn.Module):
    """Shared-weight CNN encoder for retinal image pairs."""
    def __init__(self, backbone: str, pretrained: bool = True):
        super().__init__()
        w = "DEFAULT" if pretrained else None

        if backbone == "efficientnet_b3":
            base = models.efficientnet_b3(weights=w)
            self.out_dim = base.classifier[1].in_features
            base.classifier = nn.Identity()
        elif backbone == "resnet50":
            base = models.resnet50(weights=w)
            self.out_dim = base.fc.in_features
            base.fc = nn.Identity()
        elif backbone == "convnext_small":
            base = models.convnext_small(weights=w)
            self.out_dim = base.classifier[2].in_features
            base.classifier = nn.Identity()

        self.encoder = base

    def forward(self, img1, img2):
        f1 = self.encoder(img1)
        f2 = self.encoder(img2)
        # Difference + absolute difference captures "what changed"
        diff = f2 - f1
        absdiff = torch.abs(diff)
        return torch.cat([diff, absdiff], dim=1)   # (B, 2*out_dim)


class FusionHead(nn.Module):
    def __init__(self, geo_dim, cnn_dim):
        super().__init__()
        total = geo_dim + cnn_dim
        self.net = nn.Sequential(
            nn.Linear(total, 512), nn.SiLU(), nn.Dropout(0.4),
            nn.Linear(512,  128),  nn.SiLU(), nn.Dropout(0.2),
            nn.Linear(128,  1),
        )
    def forward(self, geo_feat, cnn_feat):
        x = torch.cat([geo_feat, cnn_feat], dim=1)
        return self.net(x).squeeze(1)


class RetinalAgeGapModel(nn.Module):
    def __init__(self, geo_in_dim: int, backbone: str,
                 geo_only=False, img_only=False):
        super().__init__()
        self.geo_only = geo_only
        self.img_only = img_only

        if not img_only:
            self.geo_stream = GeoStream(geo_in_dim, out_dim=128)

        if not geo_only:
            self.cnn_stream = SiameseCNN(backbone)
            cnn_out = self.cnn_stream.out_dim * 2   # diff + absdiff

        if geo_only:
            self.head = nn.Sequential(
                nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, 1)
            )
        elif img_only:
            self.head = nn.Sequential(
                nn.Linear(cnn_out, 256), nn.SiLU(), nn.Dropout(0.3),
                nn.Linear(256, 1)
            )
        else:
            self.fusion = FusionHead(128, cnn_out)

    def forward(self, *args):
        if self.geo_only:
            geo = args[0]
            return self.head(self.geo_stream(geo)).squeeze(1)

        if self.img_only:
            img1, img2 = args[0], args[1]
            return self.head(self.cnn_stream(img1, img2)).squeeze(1)

        # Fusion mode
        geo, img1, img2 = args[0], args[1], args[2]
        g = self.geo_stream(geo)
        c = self.cnn_stream(img1, img2)
        return self.fusion(g, c)


# ─────────────────────────────────────────────────────────────────
#  6.  TRAINING UTILITIES
# ─────────────────────────────────────────────────────────────────

class Meter:
    def __init__(self): self.reset()
    def reset(self): self.s = self.n = 0
    def update(self, v, n=1): self.s += v*n; self.n += n
    @property
    def avg(self): return self.s / max(self.n, 1)


def unpack_batch(batch, geo_only, img_only, device):
    if geo_only:
        geo, tgt = batch
        return geo.to(device), None, None, tgt.to(device)
    if img_only:
        img1, img2, tgt = batch
        return None, img1.to(device), img2.to(device), tgt.to(device)
    geo, img1, img2, tgt = batch
    return geo.to(device), img1.to(device), img2.to(device), tgt.to(device)


def run_epoch(model, loader, optimizer, criterion, scaler,
              device, use_amp, train, geo_only, img_only):
    model.train() if train else model.eval()
    loss_m, mae_m = Meter(), Meter()
    ctx = torch.enable_grad() if train else torch.no_grad()

    with ctx:
        for batch in loader:
            geo, img1, img2, tgt = unpack_batch(batch, geo_only, img_only, device)

            with autocast(device_type="cuda", enabled=use_amp):
                if geo_only:
                    pred = model(geo)
                elif img_only:
                    pred = model(img1, img2)
                else:
                    pred = model(geo, img1, img2)
                loss = criterion(pred, tgt)

            if train:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()

            mae = (pred.detach() - tgt).abs().mean().item()
            loss_m.update(loss.item(), tgt.size(0))
            mae_m.update(mae, tgt.size(0))

    return loss_m.avg, mae_m.avg


# ─────────────────────────────────────────────────────────────────
#  7.  MAIN
# ─────────────────────────────────────────────────────────────────

def main():
    args = get_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}" +
          (f"  ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    gt_dir   = os.path.join(args.data_root, "Ground Truth")
    img_dir  = os.path.join(args.data_root, "Images")
    mask_dir = os.path.join(args.data_root, "Masks")

    # ── Control points ─────────────────────────────────────────
    records = scan_control_points(gt_dir)

    # ── Optional explicit age-gap labels ───────────────────────
    if args.label_csv and os.path.exists(args.label_csv):
        label_df = pd.read_csv(args.label_csv)
        label_df.columns = [c.strip().lower() for c in label_df.columns]
        label_map = dict(zip(label_df["subject_id"].astype(str),
                             label_df["age_gap"].astype(float)))
        for r in records:
            if r["subject_id"] in label_map:
                r["proxy_target"] = label_map[r["subject_id"]]
        print(f"[Labels] Loaded explicit age-gap labels for "
              f"{sum(r['subject_id'] in label_map for r in records)} subjects.")
    else:
        print("[Labels] No label CSV found — using mean displacement magnitude "
              "as proxy target (self-supervised mode).")

    # ── Mask ───────────────────────────────────────────────────
    mask_t = None
    if args.use_mask and os.path.isdir(mask_dir):
        mask_t = load_mask(mask_dir, args.img_size)
        if mask_t is not None:
            mask_t = mask_t.to(device)
            print("[Mask] Loaded.")

    # ── Transforms ─────────────────────────────────────────────
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    train_tf = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # ── Image index ────────────────────────────────────────────
    img_index = {}
    if os.path.isdir(img_dir) and not args.geo_only:
        img_index = build_image_index(img_dir)
        print(f"[Images] {len(img_index)} images indexed.")

    # ── Geo feature dim ────────────────────────────────────────
    sample_geo_dim = len(records[0]["geo_features"])
    print(f"[Geo] Feature vector dim: {sample_geo_dim}")

    # ── Datasets ───────────────────────────────────────────────
    n_val   = max(1, int(len(records) * args.val_split))
    n_train = len(records) - n_val

    np.random.shuffle(records)
    train_recs, val_recs = records[:n_train], records[n_train:]

    train_ds = RetinalAgePairDataset(train_recs, img_index, train_tf,
                                     mask_t, args.geo_only, args.img_only)
    val_ds   = RetinalAgePairDataset(val_recs,   img_index, val_tf,
                                     mask_t, args.geo_only, args.img_only)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=args.workers,
                              pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=args.workers,
                              pin_memory=True)

    print(f"[Split] Train: {n_train}  Val: {n_val}")

    # ── Model ──────────────────────────────────────────────────
    model = RetinalAgeGapModel(
        geo_in_dim=sample_geo_dim,
        backbone=args.backbone,
        geo_only=args.geo_only,
        img_only=args.img_only,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    mode_str = ("geo-only" if args.geo_only else
                "img-only" if args.img_only else "fusion")
    print(f"[Model] Mode={mode_str}  Backbone={args.backbone}  "
          f"Params={n_params:,}")

    # ── Optimiser / Loss / Scheduler ───────────────────────────
    criterion = nn.HuberLoss(delta=5.0)
    optimizer = optim.AdamW(model.parameters(),
                             lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    use_amp   = args.amp and device.type == "cuda"
    scaler    = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ── Resume ─────────────────────────────────────────────────
    start_epoch  = 1
    best_val_mae = float("inf")
    if args.resume and os.path.isfile(args.resume):
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        start_epoch  = ck["epoch"] + 1
        best_val_mae = ck.get("best_val_mae", float("inf"))
        print(f"[Resume] Epoch {ck['epoch']}")

    # ── Training loop ──────────────────────────────────────────
    import time
    max_epochs  = min(args.epochs, 10)       # hard cap at 10 epochs
    max_seconds = 30 * 60                    # 30-minute wall-clock limit
    train_start = time.time()

    logs = []
    for epoch in range(start_epoch, max_epochs + 1):
        elapsed = time.time() - train_start
        if elapsed >= max_seconds:
            print(f"\n[Stop] 30-minute wall-clock limit reached "
                  f"({elapsed/60:.1f} min) at epoch {epoch-1}. Stopping early.")
            break
        tr_loss, tr_mae = run_epoch(model, train_loader, optimizer, criterion,
                                    scaler, device, use_amp, train=True,
                                    geo_only=args.geo_only, img_only=args.img_only)
        vl_loss, vl_mae = run_epoch(model, val_loader,   optimizer, criterion,
                                    scaler, device, use_amp, train=False,
                                    geo_only=args.geo_only, img_only=args.img_only)
        scheduler.step()

        is_best = vl_mae < best_val_mae
        if is_best: best_val_mae = vl_mae

        ck = {"epoch": epoch, "model": model.state_dict(),
              "optimizer": optimizer.state_dict(),
              "scheduler": scheduler.state_dict(),
              "best_val_mae": best_val_mae}
        torch.save(ck, os.path.join(args.output_dir, "last.pth"))
        if is_best:
            torch.save(ck, os.path.join(args.output_dir, "best.pth"))

        logs.append({"epoch": epoch, "tr_loss": tr_loss, "tr_mae": tr_mae,
                     "vl_loss": vl_loss, "vl_mae": vl_mae,
                     "lr": optimizer.param_groups[0]["lr"]})

        elapsed_min = (time.time() - train_start) / 60
        star = " ★" if is_best else ""
        print(f"Ep {epoch:03d}/{max_epochs}  "
              f"tr_loss={tr_loss:.4f} tr_MAE={tr_mae:.3f}  "
              f"val_loss={vl_loss:.4f} val_MAE={vl_mae:.3f}  "
              f"[{elapsed_min:.1f}m]{star}")

    pd.DataFrame(logs).to_csv(
        os.path.join(args.output_dir, "training_log.csv"), index=False)
    print(f"\n[Done]  Best Val MAE: {best_val_mae:.3f}")
    print(f"        Outputs → {args.output_dir}")


if __name__ == "__main__":
    main()