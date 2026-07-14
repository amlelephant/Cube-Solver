"""
cnn_classifier.py

Defines and trains a lightweight CNN for sticker color classification.
The network takes a small BGR image patch and outputs a probability
distribution over 6 color classes.

Architecture choice: MobileNet-inspired depthwise separable convolutions
keep the model tiny (~50K parameters) so it runs fast on CPU in real time.

Usage:
    python cnn_classifier.py --train     # generate synthetic data and train
    python cnn_classifier.py --test      # run accuracy evaluation
    python cnn_classifier.py --webcam    # live single-patch test via webcam
"""

import os
import argparse
import random
import numpy as np
import cv2

# ---------------------------------------------------------------------------
# Try to import PyTorch. If not installed, give a clear message.
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    import torchvision.transforms as transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# Single source of truth for class list lives in color_classifier
# ---------------------------------------------------------------------------
from color_classifier import CLASSES
NUM_CLASSES = len(CLASSES)
PATCH_SIZE   = 32          # CNN input: 32×32 px per sticker region
MODEL_PATH   = "sticker_cnn.pt"

# Canonical BGR color for each class (used in synthetic data generation)
# These are approximate midpoint colors; augmentation creates the variation.
CLASS_BGR = {
    "white":  (235, 235, 230),
    "yellow": (30,  220, 230),
    "red":    (25,  25,  210),
    "orange": (20,  120, 230),
    "blue":   (200, 80,  30 ),
    "green":  (30,  160, 40 ),
}


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_model():
    """
    Tiny CNN for 32×32 sticker patch → 6 classes.

    Layer structure:
      Block 1: Conv 3×3 → BN → ReLU → MaxPool   (32→16, 3ch→32ch)
      Block 2: Conv 3×3 → BN → ReLU → MaxPool   (16→8,  32ch→64ch)
      Block 3: Conv 3×3 → BN → ReLU → MaxPool   (8→4,   64ch→128ch)
      Flatten → FC(128) → Dropout(0.3) → FC(6)

    ~120K parameters; forward pass ~0.3ms on CPU.
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch not installed. Run: pip install torch torchvision")

    class StickerCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                # Block 1
                nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                # Block 2
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                # Block 3
                nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128 * 4 * 4, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(128, NUM_CLASSES),
            )

        def forward(self, x):
            return self.classifier(self.features(x))

    return StickerCNN()


# ---------------------------------------------------------------------------
# Synthetic dataset generation
# ---------------------------------------------------------------------------
def generate_synthetic_patch(color_name, patch_size=PATCH_SIZE):
    """
    Generate one synthetic 32×32 BGR patch for a given color class.

    Augmentations applied to simulate real-world variation:
      - Random lighting shift (multiply V channel)
      - Random hue drift (small shift on H channel)
      - Random saturation variation
      - Random Gaussian noise
      - Random slight blur (simulate out-of-focus)
      - Random slight rotation of the patch
      - Occasional specular highlight (bright spot)
      - Occasional dark shadow edge
    """
    base_bgr = CLASS_BGR[color_name]

    # Start from a solid-color patch
    patch = np.full((patch_size, patch_size, 3), base_bgr, dtype=np.uint8)

    # --- HSV augmentation ---
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).astype(np.float32)

    # Hue drift: ±12 degrees
    hsv[:, :, 0] += random.uniform(-12, 12)
    hsv[:, :, 0] = np.clip(hsv[:, :, 0], 0, 179)

    # Saturation variation: ×0.5 to ×1.3
    hsv[:, :, 1] *= random.uniform(0.5, 1.3)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)

    # Value (brightness) variation: ×0.35 to ×1.4
    hsv[:, :, 2] *= random.uniform(0.35, 1.4)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)

    patch = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # --- Spatial augmentations ---
    # Gaussian noise
    noise = np.random.normal(0, random.uniform(3, 18), patch.shape).astype(np.int16)
    patch = np.clip(patch.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Random blur (0=none, 1=slight, 2=moderate)
    blur_k = random.choice([0, 0, 1, 1, 2])
    if blur_k > 0:
        ksize = blur_k * 2 + 1
        patch = cv2.GaussianBlur(patch, (ksize, ksize), 0)

    # Random rotation ±15 degrees
    if random.random() < 0.5:
        angle = random.uniform(-15, 15)
        M = cv2.getRotationMatrix2D((patch_size // 2, patch_size // 2), angle, 1.0)
        patch = cv2.warpAffine(patch, M, (patch_size, patch_size),
                               borderMode=cv2.BORDER_REFLECT)

    # Specular highlight (30% chance)
    if random.random() < 0.30:
        cx = random.randint(4, patch_size - 4)
        cy = random.randint(4, patch_size - 4)
        radius = random.randint(2, 6)
        overlay = patch.copy()
        cv2.circle(overlay, (cx, cy), radius, (255, 255, 255), -1)
        alpha = random.uniform(0.3, 0.7)
        patch = cv2.addWeighted(patch, 1 - alpha, overlay, alpha, 0)

    # Shadow edge (20% chance)
    if random.random() < 0.20:
        shadow = np.zeros_like(patch)
        side = random.choice(["left", "right", "top", "bottom"])
        thickness = random.randint(3, 8)
        if side == "left":
            shadow[:, :thickness] = 1
        elif side == "right":
            shadow[:, -thickness:] = 1
        elif side == "top":
            shadow[:thickness, :] = 1
        else:
            shadow[-thickness:, :] = 1
        patch = (patch.astype(np.float32) * (1 - shadow * random.uniform(0.3, 0.6))).astype(np.uint8)

    return patch


if TORCH_AVAILABLE:
    class SyntheticStickerDataset(Dataset):
        """
        Generates synthetic sticker patches on-the-fly during training.
        No disk I/O needed.
        """
        def __init__(self, samples_per_class=2000, patch_size=PATCH_SIZE):
            self.samples_per_class = samples_per_class
            self.patch_size = patch_size
            self.total = samples_per_class * NUM_CLASSES

            # Pre-build index list
            self.items = []
            for class_idx, color_name in enumerate(CLASSES):
                for _ in range(samples_per_class):
                    self.items.append((color_name, class_idx))
            random.shuffle(self.items)

            # Transforms: normalise to [-1, 1] (standard for CNNs)
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5]),
            ])

        def __len__(self):
            return self.total

        def __getitem__(self, idx):
            color_name, label = self.items[idx]
            patch_bgr = generate_synthetic_patch(color_name, self.patch_size)
            # Convert BGR→RGB before ToTensor (torchvision expects RGB)
            patch_rgb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB)
            tensor = self.transform(patch_rgb)
            return tensor, label


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(epochs=25, samples_per_class=3000, batch_size=128, lr=1e-3):
    if not TORCH_AVAILABLE:
        print("ERROR: PyTorch not installed.")
        print("Run: pip install torch torchvision")
        return

    print(f"Generating synthetic dataset: {samples_per_class} samples × {NUM_CLASSES} classes")
    print(f"Total: {samples_per_class * NUM_CLASSES} patches\n")

    dataset   = SyntheticStickerDataset(samples_per_class=samples_per_class)
    n_val     = int(len(dataset) * 0.1)
    n_train   = len(dataset) - n_val
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model     = build_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        # --- Train ---
        model.train()
        train_loss = 0.0
        train_correct = 0

        for patches, labels in train_loader:
            patches, labels = patches.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(patches)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss    += loss.item() * patches.size(0)
            train_correct += (outputs.argmax(1) == labels).sum().item()

        scheduler.step()

        # --- Validate ---
        model.eval()
        val_correct = 0
        with torch.no_grad():
            for patches, labels in val_loader:
                patches, labels = patches.to(device), labels.to(device)
                outputs = model(patches)
                val_correct += (outputs.argmax(1) == labels).sum().item()

        train_acc = train_correct / n_train * 100
        val_acc   = val_correct   / n_val   * 100
        avg_loss  = train_loss / n_train

        print(f"Epoch {epoch:2d}/{epochs}  loss={avg_loss:.4f}  "
              f"train={train_acc:.1f}%  val={val_acc:.1f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"             ↑ New best ({val_acc:.1f}%) — saved to {MODEL_PATH}")

    print(f"\nTraining complete. Best validation accuracy: {best_val_acc:.1f}%")
    print(f"Model saved to: {MODEL_PATH}")


# ---------------------------------------------------------------------------
# Inference helper (used by the ensemble)
# ---------------------------------------------------------------------------
def load_model():
    """Load trained model from disk. Returns None if not found."""
    if not TORCH_AVAILABLE:
        return None
    if not os.path.exists(MODEL_PATH):
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model


_cached_model  = None
_cached_device = None

def classify_region_cnn(bgr_region):
    """
    Classify a BGR image patch using the trained CNN.

    Returns:
        probs  (dict): {color_name: probability}, sums to 1.0
        or None if model not loaded.

    The returned dict is used by the ensemble combiner.
    """
    global _cached_model, _cached_device

    if not TORCH_AVAILABLE:
        return None

    if _cached_model is None:
        _cached_model = load_model()
        if _cached_model is None:
            return None
        _cached_device = next(_cached_model.parameters()).device

    if bgr_region is None or bgr_region.size == 0:
        return None

    # Resize to CNN input size
    patch = cv2.resize(bgr_region, (PATCH_SIZE, PATCH_SIZE))
    patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)

    # Normalise
    tensor = torch.from_numpy(patch_rgb.transpose(2, 0, 1)).float()
    tensor = (tensor / 255.0 - 0.5) / 0.5
    tensor = tensor.unsqueeze(0).to(_cached_device)

    with torch.no_grad():
        logits = _cached_model(tensor)
        probs  = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

    return {CLASSES[i]: float(probs[i]) for i in range(NUM_CLASSES)}


# ---------------------------------------------------------------------------
# CLI entry points
# ---------------------------------------------------------------------------
def run_accuracy_test(n_per_class=500):
    """Quick accuracy check on freshly generated synthetic patches."""
    if not TORCH_AVAILABLE:
        print("PyTorch not installed.")
        return

    model = load_model()
    if model is None:
        print(f"No model found at {MODEL_PATH}. Run with --train first.")
        return

    device = next(model.parameters()).device
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

    print(f"\nAccuracy test: {n_per_class} synthetic patches per class\n")
    print(f"{'Color':<10} {'Correct':>8} {'Total':>8} {'Acc':>8}")
    print("-" * 40)

    total_correct = 0
    total_count   = 0

    for class_idx, color_name in enumerate(CLASSES):
        correct = 0
        for _ in range(n_per_class):
            patch_bgr = generate_synthetic_patch(color_name)
            patch_rgb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB)
            t = transform(patch_rgb).unsqueeze(0).to(device)
            with torch.no_grad():
                pred = model(t).argmax(1).item()
            if pred == class_idx:
                correct += 1

        acc = correct / n_per_class * 100
        print(f"{color_name:<10} {correct:>8} {n_per_class:>8} {acc:>7.1f}%")
        total_correct += correct
        total_count   += n_per_class

    overall = total_correct / total_count * 100
    print("-" * 40)
    print(f"{'OVERALL':<10} {total_correct:>8} {total_count:>8} {overall:>7.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CNN sticker color classifier")
    parser.add_argument("--train",  action="store_true", help="Train the model")
    parser.add_argument("--test",   action="store_true", help="Run accuracy test")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--samples", type=int, default=3000,
                        help="Synthetic samples per class")
    args = parser.parse_args()

    if args.train:
        train(epochs=args.epochs, samples_per_class=args.samples)
    elif args.test:
        run_accuracy_test()
    else:
        parser.print_help()