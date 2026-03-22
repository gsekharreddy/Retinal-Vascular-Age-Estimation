import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np

# ==============================
# CONFIG
# ==============================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
EPOCHS = 20
LR = 1e-4
IMG_SIZE = 224
NUM_WORKERS = 0  # Windows-safe (avoid pickling issues)

# ==============================
# DATASET (ROBUST PAIRING)
# ==============================
class RetinalDataset(Dataset):
    def __init__(self, img_dir, gt_dir, transform=None):
        self.transform = transform

        self.img_paths = sorted(glob.glob(os.path.join(img_dir, "*")))
        self.gt_paths = sorted(glob.glob(os.path.join(gt_dir, "*")))

        print("Images found:", len(self.img_paths))
        print("GT files found:", len(self.gt_paths))

        # Assume 2 images per GT (common in retinal datasets: L/R eyes)
        self.pairs = []
        for i in range(len(self.gt_paths)):
            if 2*i + 1 < len(self.img_paths):
                self.pairs.append((self.img_paths[2*i], self.gt_paths[i]))
                self.pairs.append((self.img_paths[2*i + 1], self.gt_paths[i]))

        print(f"✅ Using paired samples: {len(self.pairs)}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, gt_path = self.pairs[idx]

        # Load image
        img = Image.open(img_path).convert("RGB")

        # Load GT (control points)
        try:
            with open(gt_path, 'r') as f:
                values = [float(x) for x in f.read().split()]
        except Exception as e:
            values = [0.0]

        # ⚠️ Placeholder label (replace with real age if available)
        age = np.mean(values)

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(age, dtype=torch.float32)

# ==============================
# TRANSFORMS
# ==============================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ==============================
# MODEL
# ==============================
def get_model():
    model = models.efficientnet_b0(weights="IMAGENET1K_V1")
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 1)
    return model

# ==============================
# TRAIN LOOP
# ==============================
def train():
    dataset = RetinalDataset("Images", "Ground Truth", transform)

    if len(dataset) == 0:
        raise ValueError("❌ No data pairs found. Check dataset structure.")

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    model = get_model().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for imgs, ages in loader:
            imgs = imgs.to(DEVICE)
            ages = ages.to(DEVICE).unsqueeze(1)

            preds = model(imgs)
            loss = criterion(preds, ages)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), "retinal_age_model.pth")
    print("✅ Model saved: retinal_age_model.pth")

# ==============================
# ENTRY POINT (IMPORTANT FOR WINDOWS)
# ==============================
if __name__ == "__main__":
    train()
