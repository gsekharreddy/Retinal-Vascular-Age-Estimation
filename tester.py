"""
Retinal Age Gap — Tester Script
=================================
Loads best.pth, runs inference on 10 random samples, and saves a
visual result grid:  retinal pair  |  displacement arrows  |  prediction card

Usage:
  python retinal_age_tester.py
  python retinal_age_tester.py --data_root ./MyDataset --checkpoint ./outputs/best.pth
  python retinal_age_tester.py --n_samples 10 --out_img results.png --geo_only
"""

import os
import re
import glob
import random
import argparse
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

import torch
import torch.nn as nn
from torchvision import transforms, models

# ── reuse helpers from the trainer (same file expected alongside) ──
import sys
sys.path.insert(0, os.path.dirname(__file__))


# ─────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",   default=".",               help="Dataset root")
    p.add_argument("--checkpoint",  default="./outputs/best.pth")
    p.add_argument("--out_img",     default="./outputs/test_results.png")
    p.add_argument("--n_samples",   type=int, default=10)
    p.add_argument("--img_size",    type=int, default=512)
    p.add_argument("--backbone",    default="efficientnet_b3",
                   choices=["efficientnet_b3","resnet50","convnext_small"])
    p.add_argument("--geo_only",    action="store_true")
    p.add_argument("--img_only",    action="store_true")
    p.add_argument("--seed",        type=int, default=0)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────
#  DATA HELPERS  (self-contained, no trainer import needed)
# ─────────────────────────────────────────────────────────────────

def parse_cp_filename(fname):
    stem = Path(fname).stem
    m = re.search(r'control_points_([^_]+)_(\d+)_(\d+)', stem)
    if m:
        return m.group(1), m.group(2), m.group(3)
    raise ValueError(f"Cannot parse: {fname}")

def load_cp(path):
    return np.loadtxt(path, dtype=np.float32)

def extract_geo(cp):
    x1,y1,x2,y2 = cp[:,0],cp[:,1],cp[:,2],cp[:,3]
    dx,dy = x2-x1, y2-y1
    mag   = np.sqrt(dx**2+dy**2)
    angle = np.arctan2(dy,dx)
    xr = max(x1.max()-x1.min(),1.); yr = max(y1.max()-y1.min(),1.)
    nx1,ny1 = (x1-x1.min())/xr, (y1-y1.min())/yr
    per_lm = np.stack([dx,dy,mag,angle,nx1,ny1],axis=1).flatten()
    stats  = np.array([
        mag.mean(),mag.std(),mag.max(),mag.min(),
        np.abs(dx).mean(),np.abs(dx).std(),
        np.abs(dy).mean(),np.abs(dy).std(),
        np.sin(angle).mean(),np.cos(angle).mean(),
        1.-np.cos(angle-angle.mean()).mean(),
        dx.std()/(dy.std()+1e-6),
        (mag>mag.mean()).sum()/len(mag),
    ],dtype=np.float32)
    return np.concatenate([per_lm,stats]).astype(np.float32)

def build_image_index(img_dir):
    idx={}
    for p in Path(img_dir).rglob("*"):
        if p.suffix.lower() in {".jpg",".jpeg",".png",".bmp",".tif",".tiff"}:
            idx[p.stem.lower()] = str(p)
    return idx

def find_image(index, subject_id, img_idx):
    for c in [f"{subject_id}_{img_idx}", f"{subject_id.lower()}_{img_idx}",
              f"{img_idx}_{subject_id}", f"{subject_id}{img_idx}", subject_id]:
        if c.lower() in index:
            return index[c.lower()]
    return None

def load_mask(mask_dir, img_size):
    for name in ["mask.png","feature_mask.png"]:
        path = os.path.join(mask_dir, name)
        if os.path.exists(path):
            m = Image.open(path).convert("L").resize((img_size,img_size),Image.NEAREST)
            t = torch.from_numpy(np.array(m)).float()/255.
            return (t>0.5).float().unsqueeze(0)
    return None


# ─────────────────────────────────────────────────────────────────
#  MODEL  (copied from trainer — must match exactly)
# ─────────────────────────────────────────────────────────────────

class GeoStream(nn.Module):
    def __init__(self, in_dim, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim,256),nn.SiLU(),nn.Dropout(0.3),
            nn.Linear(256,256),  nn.SiLU(),nn.Dropout(0.2),
            nn.Linear(256,out_dim),nn.SiLU())
    def forward(self,x): return self.net(x)

class SiameseCNN(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        w = "DEFAULT"
        if backbone=="efficientnet_b3":
            base=models.efficientnet_b3(weights=w); self.out_dim=base.classifier[1].in_features; base.classifier=nn.Identity()
        elif backbone=="resnet50":
            base=models.resnet50(weights=w);        self.out_dim=base.fc.in_features;             base.fc=nn.Identity()
        elif backbone=="convnext_small":
            base=models.convnext_small(weights=w);  self.out_dim=base.classifier[2].in_features;  base.classifier=nn.Identity()
        self.encoder=base
    def forward(self,i1,i2):
        f1,f2=self.encoder(i1),self.encoder(i2)
        return torch.cat([f2-f1,torch.abs(f2-f1)],dim=1)

class FusionHead(nn.Module):
    def __init__(self,geo_dim,cnn_dim):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(geo_dim+cnn_dim,512),nn.SiLU(),nn.Dropout(0.4),
            nn.Linear(512,128),nn.SiLU(),nn.Dropout(0.2),nn.Linear(128,1))
    def forward(self,g,c): return self.net(torch.cat([g,c],dim=1)).squeeze(1)

class RetinalAgeGapModel(nn.Module):
    def __init__(self, geo_in_dim, backbone, geo_only=False, img_only=False):
        super().__init__()
        self.geo_only=geo_only; self.img_only=img_only
        if not img_only: self.geo_stream=GeoStream(geo_in_dim)
        if not geo_only:
            self.cnn_stream=SiameseCNN(backbone); cnn_out=self.cnn_stream.out_dim*2
        if geo_only:
            self.head=nn.Sequential(nn.Linear(128,64),nn.SiLU(),nn.Linear(64,1))
        elif img_only:
            self.head=nn.Sequential(nn.Linear(cnn_out,256),nn.SiLU(),nn.Dropout(0.3),nn.Linear(256,1))
        else:
            self.fusion=FusionHead(128,cnn_out)
    def forward(self,*args):
        if self.geo_only:   return self.head(self.geo_stream(args[0])).squeeze(1)
        if self.img_only:   return self.head(self.cnn_stream(args[0],args[1])).squeeze(1)
        return self.fusion(self.geo_stream(args[0]), self.cnn_stream(args[1],args[2]))


# ─────────────────────────────────────────────────────────────────
#  VISUALISATION HELPERS
# ─────────────────────────────────────────────────────────────────

THUMB  = 300          # retinal image thumbnail size
ARROW  = 300          # displacement arrow canvas size
CARD_W = 260          # prediction card width
ROW_H  = THUMB + 10   # row height with small padding
COLS   = THUMB*2 + ARROW + CARD_W + 50   # total width per row
PAD    = 20

# Colour palette
BG       = (15,  17,  26)
CARD_BG  = (28,  32,  48)
ACCENT   = (82, 196, 255)
GOOD     = (80, 220, 130)
WARN     = (255, 180,  60)
BAD      = (255,  90,  90)
WHITE    = (240, 240, 255)
GREY     = (120, 120, 150)


def _font(size):
    """Load a font, fall back to default if not available."""
    for name in ["arial.ttf","Arial.ttf","DejaVuSans.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            pass
    return ImageFont.load_default()


def make_displacement_canvas(cp, size=ARROW):
    """
    Draw control point positions (image 1) and arrows to image 2 positions.
    Points coloured by displacement magnitude (blue=small → red=large).
    """
    canvas = Image.new("RGB", (size, size), (20, 22, 35))
    draw   = ImageDraw.Draw(canvas)

    x1,y1,x2,y2 = cp[:,0],cp[:,1],cp[:,2],cp[:,3]
    mag = np.sqrt((x2-x1)**2+(y2-y1)**2)

    # Normalise coords to canvas
    all_x = np.concatenate([x1,x2]); all_y = np.concatenate([y1,y2])
    xmin,xmax = all_x.min(), all_x.max()+1
    ymin,ymax = all_y.min(), all_y.max()+1
    margin = 30

    def norm(v, lo, hi):
        return margin + (v-lo)/(hi-lo+1e-6)*(size-2*margin)

    nx1 = norm(x1,xmin,xmax); ny1 = norm(y1,ymin,ymax)
    nx2 = norm(x2,xmin,xmax); ny2 = norm(y2,ymin,ymax)

    mag_norm = (mag - mag.min()) / (mag.max()-mag.min()+1e-6)

    for i in range(len(cp)):
        t  = float(mag_norm[i])
        r  = int(40  + 215*t)
        g  = int(160 - 130*t)
        b  = int(255 -  80*t)
        col = (r,g,b)

        ax,ay = float(nx1[i]),float(ny1[i])
        bx,by = float(nx2[i]),float(ny2[i])

        # Arrow line
        draw.line([(ax,ay),(bx,by)], fill=col, width=2)
        # Arrowhead
        import math
        angle = math.atan2(by-ay, bx-ax)
        tip_len = 8
        for da in [0.4,-0.4]:
            ex = bx - tip_len*math.cos(angle+da)
            ey = by - tip_len*math.sin(angle+da)
            draw.line([(bx,by),(ex,ey)], fill=col, width=2)
        # Source dot
        draw.ellipse([(ax-4,ay-4),(ax+4,ay+4)], fill=col, outline=WHITE)

    # Label
    font = _font(11)
    draw.text((6,4), "Displacement Map", font=font, fill=GREY)
    draw.text((6,size-18),
              f"mean={mag.mean():.1f}px  max={mag.max():.1f}px",
              font=font, fill=GREY)
    return canvas


def make_prediction_card(subject_id, pred, proxy, img_size=THUMB):
    """
    Dark card showing subject ID, predicted age gap, and proxy target.
    """
    card = Image.new("RGB", (CARD_W, img_size), CARD_BG)
    draw = ImageDraw.Draw(card)

    f_big   = _font(28)
    f_med   = _font(16)
    f_small = _font(13)

    # Header bar
    draw.rectangle([(0,0),(CARD_W,44)], fill=(35,40,62))
    draw.text((14,12), "SUBJECT", font=f_small, fill=GREY)
    draw.text((14,28), subject_id, font=_font(15), fill=ACCENT)

    # Predicted value
    draw.text((14,66),  "Predicted Age Gap", font=f_small, fill=GREY)
    draw.text((14,86),  f"{pred:.2f}", font=f_big, fill=WHITE)
    draw.text((110,104),"units", font=f_small, fill=GREY)

    # Proxy (ground truth proxy)
    draw.text((14,148), "Proxy Target", font=f_small, fill=GREY)
    draw.text((14,166), f"{proxy:.2f}", font=f_med, fill=GOOD)

    # Error
    err   = abs(pred - proxy)
    pct   = err / (abs(proxy)+1e-6) * 100
    ecol  = GOOD if pct < 15 else (WARN if pct < 35 else BAD)
    draw.text((14,210), "Absolute Error", font=f_small, fill=GREY)
    draw.text((14,228), f"{err:.2f}  ({pct:.0f}%)", font=f_med, fill=ecol)

    # Mini bar: error magnitude
    bar_y  = img_size - 50
    bar_w  = CARD_W - 28
    filled = min(int(bar_w * min(pct/100, 1.0)), bar_w)
    draw.rectangle([(14,bar_y),(14+bar_w,bar_y+12)], fill=(40,44,64), outline=GREY)
    draw.rectangle([(14,bar_y),(14+filled,bar_y+12)], fill=ecol)
    draw.text((14,bar_y+16), "Error magnitude", font=f_small, fill=GREY)

    return card


def make_retinal_thumb(path, size=THUMB, label=""):
    """Load a retinal image and return a labelled thumbnail."""
    if path and os.path.exists(path):
        img = Image.open(path).convert("RGB").resize((size,size), Image.LANCZOS)
    else:
        img = Image.new("RGB",(size,size),(30,30,40))
        d = ImageDraw.Draw(img)
        d.text((size//2-40, size//2-10), "No image", fill=GREY, font=_font(14))

    if label:
        d = ImageDraw.Draw(img)
        tw = 120; th = 22
        d.rectangle([(4,4),(4+tw,4+th)], fill=(0,0,0,180))
        d.text((8,7), label, fill=ACCENT, font=_font(13))
    return img


# ─────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────

def main():
    args = get_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    gt_dir   = os.path.join(args.data_root, "Ground Truth")
    img_dir  = os.path.join(args.data_root, "Images")
    mask_dir = os.path.join(args.data_root, "Masks")

    # ── Scan control points ───────────────────────────────────
    txt_files = sorted(glob.glob(os.path.join(gt_dir, "*.txt")))
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in {gt_dir}")

    all_records = []
    for f in txt_files:
        try:
            subj,i1,i2 = parse_cp_filename(f)
            cp  = load_cp(f)
            geo = extract_geo(cp)
            proxy = float(np.sqrt((cp[:,2]-cp[:,0])**2+(cp[:,3]-cp[:,1])**2).mean())
            all_records.append(dict(subject_id=subj,i1=i1,i2=i2,
                                    cp=cp,geo=geo,proxy=proxy,txt=f))
        except Exception as e:
            print(f"[WARN] {f}: {e}")

    print(f"[GT] {len(all_records)} pairs found.")

    # ── Pick N samples ────────────────────────────────────────
    n = min(args.n_samples, len(all_records))
    samples = random.sample(all_records, n)
    print(f"[Sampled] {n} subjects for testing.")

    # ── Image index + mask ────────────────────────────────────
    img_index = build_image_index(img_dir) if os.path.isdir(img_dir) else {}
    mask_t    = load_mask(mask_dir, args.img_size) if os.path.isdir(mask_dir) else None

    tf = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    # ── Load model ────────────────────────────────────────────
    geo_dim = len(samples[0]["geo"])
    model   = RetinalAgeGapModel(geo_dim, args.backbone,
                                  args.geo_only, args.img_only).to(device)

    if os.path.isfile(args.checkpoint):
        ck = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ck["model"])
        print(f"[Checkpoint] Loaded from {args.checkpoint}  "
              f"(best val MAE={ck.get('best_val_mae',float('nan')):.3f})")
    else:
        print(f"[WARN] No checkpoint at {args.checkpoint} — using random weights.")

    model.eval()

    # ── Inference ─────────────────────────────────────────────
    results = []
    with torch.no_grad():
        for rec in samples:
            geo_t = torch.from_numpy(rec["geo"]).unsqueeze(0).to(device)

            p1_path = find_image(img_index, rec["subject_id"], rec["i1"])
            p2_path = find_image(img_index, rec["subject_id"], rec["i2"])

            def load_img_t(path):
                if path and os.path.exists(path):
                    t = tf(Image.open(path).convert("RGB"))
                    if mask_t is not None:
                        t = t * mask_t.cpu()
                    return t.unsqueeze(0).to(device)
                return torch.zeros(1,3,args.img_size,args.img_size,device=device)

            img1_t = load_img_t(p1_path)
            img2_t = load_img_t(p2_path)

            if args.geo_only:
                pred = model(geo_t).item()
            elif args.img_only:
                pred = model(img1_t, img2_t).item()
            else:
                pred = model(geo_t, img1_t, img2_t).item()

            results.append({**rec, "pred": pred,
                             "p1_path": p1_path, "p2_path": p2_path})
            print(f"  {rec['subject_id']:>8}  pred={pred:7.3f}  "
                  f"proxy={rec['proxy']:7.3f}  "
                  f"err={abs(pred-rec['proxy']):6.3f}")

    # ── Build output image ────────────────────────────────────
    n_rows  = len(results)
    hdr_h   = 70
    total_h = hdr_h + n_rows*(ROW_H + PAD) + PAD
    total_w = PAD + THUMB + 4 + THUMB + 4 + ARROW + 4 + CARD_W + PAD

    canvas = Image.new("RGB", (total_w, total_h), BG)
    draw   = ImageDraw.Draw(canvas)

    # Header
    draw.rectangle([(0,0),(total_w,hdr_h)], fill=(22,26,42))
    draw.text((PAD, 10), "Retinal Age Gap — Test Results",
              font=_font(22), fill=ACCENT)
    mae = np.mean([abs(r["pred"]-r["proxy"]) for r in results])
    draw.text((PAD, 42),
              f"{n} samples  |  Mean Abs Error: {mae:.3f}  |  "
              f"Backbone: {args.backbone}  |  "
              f"Mode: {'geo-only' if args.geo_only else 'img-only' if args.img_only else 'fusion'}",
              font=_font(13), fill=GREY)

    # Column headers
    col_headers = ["Image 1", "Image 2", "Displacement", "Prediction"]
    col_xs      = [PAD, PAD+THUMB+4, PAD+THUMB*2+8, PAD+THUMB*2+ARROW+12]
    for hx,ht in zip(col_xs, col_headers):
        draw.text((hx, hdr_h-18), ht, font=_font(12), fill=GREY)

    # Rows
    for row_i, r in enumerate(results):
        y = hdr_h + PAD + row_i*(ROW_H+PAD)
        x = PAD

        # Image 1
        t1 = make_retinal_thumb(r["p1_path"], THUMB,
                                 f"{r['subject_id']} img{r['i1']}")
        canvas.paste(t1, (x, y))
        x += THUMB + 4

        # Image 2
        t2 = make_retinal_thumb(r["p2_path"], THUMB,
                                 f"{r['subject_id']} img{r['i2']}")
        canvas.paste(t2, (x, y))
        x += THUMB + 4

        # Displacement arrows
        arr = make_displacement_canvas(r["cp"], ARROW).resize(
              (ARROW, THUMB), Image.LANCZOS)
        canvas.paste(arr, (x, y))
        x += ARROW + 4

        # Prediction card
        card = make_prediction_card(r["subject_id"], r["pred"], r["proxy"], THUMB)
        canvas.paste(card, (x, y))

        # Row separator
        draw.line([(PAD, y+ROW_H+PAD//2),(total_w-PAD, y+ROW_H+PAD//2)],
                  fill=(35,40,58), width=1)

    # Footer MAE summary bar
    fy = total_h - 28
    draw.text((PAD, fy), f"Mean Absolute Error: {mae:.4f}   "
              f"Best: {min(abs(r['pred']-r['proxy']) for r in results):.4f}   "
              f"Worst: {max(abs(r['pred']-r['proxy']) for r in results):.4f}",
              font=_font(13), fill=GREY)

    os.makedirs(os.path.dirname(os.path.abspath(args.out_img)), exist_ok=True)
    canvas.save(args.out_img, quality=95)
    print(f"\n[Saved] {args.out_img}  ({total_w}×{total_h}px)")
    print(f"[Summary] Mean Abs Error = {mae:.4f}")


if __name__ == "__main__":
    main()