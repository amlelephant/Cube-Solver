"""
train_move_classifier.py

Trains a ResNet-18 to classify which cube face moved and in which direction.

Input:   ORDERED stack of temporal difference images across the move window
         (before→mid_00→mid_01→mid_02→after = 4 signed diffs = 12 channels).
         A single diff is order-free — CW and CCW of the same face are
         time-reversals of each other, so one diff can barely separate
         them. Stacking the diffs in temporal order as input channels
         (early fusion) preserves motion order, which is exactly where
         direction lives. `--diffs 1` reproduces the old single-diff
         input (mid_01 − before) for ablation.
Output:  12 classes — the raw BLE byte (0–11) already in your moves_labeled.jsonl

Classes:
  0  blue   CW      1  blue   CCW
  2  green  CW      3  green  CCW
  4  white  CW      5  white  CCW
  6  yellow CW      7  yellow CCW
  8  red    CW      9  red    CCW
  10 orange CW      11 orange CCW

Usage:
  # Train on one or more session folders:
  python train_move_classifier.py --sessions training_data/solve_*/

  # With specific options:
  python train_move_classifier.py --sessions training_data/solve_*/ \\
      --epochs 40 --batch 32 --output move_classifier.pt

  # Evaluate a trained model:
  python train_move_classifier.py --sessions training_data/solve_*/ \\
      --eval --model move_classifier.pt
"""

import argparse
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
    import torchvision.models as models
    import torchvision.transforms as T
except ImportError:
    sys.exit("PyTorch not installed. Run: pip install torch torchvision")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_CLASSES = 12
IMG_SIZE    = 224    # ResNet-18 standard input size
MODEL_PATH  = "move_classifier.pt"
NUM_DIFFS   = 4      # ordered diffs per sample (4 = full window, 1 = legacy)

# Temporal order of the postprocess_session.py window snapshots
FRAME_ORDER = ["before", "mid_00", "mid_01", "mid_02", "after"]

CLASS_NAMES = [
    "blue-CW",   "blue-CCW",
    "green-CW",  "green-CCW",
    "white-CW",  "white-CCW",
    "yellow-CW", "yellow-CCW",
    "red-CW",    "red-CCW",
    "orange-CW", "orange-CCW",
]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MoveDiffDataset(Dataset):
    """
    Each sample is an ORDERED stack of temporal difference images across
    the move window: consecutive diffs of [before, mid_00, mid_01, mid_02,
    after] → 4 signed diffs → 12 input channels (num_diffs=4), or the
    legacy single mid_01 − before diff (num_diffs=1).

    Each diff is mostly grey (128) where nothing moved, and shows
    bright/dark regions exactly where stickers shifted during the turn.
    The spatial pattern encodes which face turned; the channel ORDER of
    the stacked diffs encodes the motion's temporal direction — a CCW
    turn is (approximately) the CW stack reversed, which a single diff
    cannot represent.

    Label: ble_raw (0-11), read directly from moves_labeled.jsonl.

    Augmentation note on horizontal flipping:
      Flipping a CW motion produces a CCW motion visually.
      Labels 0,2,4,6,8,10 are CW (even) and 1,3,5,7,9,11 are CCW (odd).
      A horizontal flip must XOR the label with 1 to stay correct.
      This doubles the dataset for free with correct labels.
      (The flip mirrors space only — channel order is untouched.)
    """

    def __init__(self, session_dirs: list[Path], augment: bool = True,
                 num_diffs: int = NUM_DIFFS):
        self.samples   = []   # [(frame_paths, label), ...] in temporal order
        self.augment   = augment
        self.num_diffs = num_diffs
        self._load_sessions(session_dirs)

    def _window_keys(self) -> list[str]:
        if self.num_diffs == 1:
            return ["before", "mid_01"]
        return FRAME_ORDER

    def _load_sessions(self, session_dirs: list[Path]):
        skipped = 0
        for session_dir in session_dirs:
            labeled = session_dir / "moves_labeled.jsonl"
            if not labeled.exists():
                print(f"  WARNING: no moves_labeled.jsonl in {session_dir.name} "
                      f"— run postprocess_session.py first")
                continue

            with open(labeled) as f:
                for line in f:
                    m = json.loads(line.strip())

                    label = m.get("ble_raw")
                    if label is None or not (0 <= label <= 11):
                        skipped += 1
                        continue

                    frames = m.get("frames", {})
                    paths  = []
                    for key in self._window_keys():
                        rel = frames.get(key)
                        p   = session_dir / rel if rel else None
                        if p is None or not p.exists():
                            paths = None
                            break
                        paths.append(p)

                    if paths is None:
                        skipped += 1
                        continue

                    self.samples.append((paths, label))

        if skipped:
            print(f"  Skipped {skipped} moves (missing frames or invalid label)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        paths, label = self.samples[idx]

        imgs = [cv2.imread(str(p)) for p in paths]
        if any(i is None for i in imgs):
            # Fallback: neutral (no-motion) stack if a file is unreadable
            stack = np.full((IMG_SIZE, IMG_SIZE, 3 * self.num_diffs), 128,
                            dtype=np.uint8)
            return self._to_tensor(stack), label

        stack = build_diff_stack(imgs)

        # Augmentation
        if self.augment:
            # Horizontal flip: correct label by XOR 1 (CW↔CCW)
            if random.random() < 0.5:
                stack = np.fliplr(stack).copy()
                label = label ^ 1   # even→odd (CW→CCW) and vice versa

            # Brightness / contrast jitter (safe — doesn't change direction)
            stack = self._jitter(stack)

        return self._to_tensor(stack), label

    def _jitter(self, img: np.ndarray) -> np.ndarray:
        """Random brightness and contrast adjustment."""
        alpha = random.uniform(0.8, 1.2)    # contrast
        beta  = random.randint(-15, 15)     # brightness
        img   = np.clip(img.astype(np.float32) * alpha + beta, 0, 255)
        return img.astype(np.uint8)

    def _to_tensor(self, img: np.ndarray) -> torch.Tensor:
        """HWC uint8 (3K channels) → CHW float32, ImageNet stats per diff."""
        return stack_to_tensor(img)

    def class_counts(self) -> dict[int, int]:
        return dict(Counter(label for _, label in self.samples))

    def class_weights(self) -> torch.Tensor:
        """Inverse-frequency weights for weighted loss / sampler."""
        counts = self.class_counts()
        total  = sum(counts.values())
        weights = torch.zeros(NUM_CLASSES)
        for cls, cnt in counts.items():
            weights[cls] = total / (cnt * NUM_CLASSES)
        return weights


def build_diff_stack(frames_bgr: list[np.ndarray]) -> np.ndarray:
    """
    N ordered BGR frames → HWC uint8 stack of N-1 consecutive signed
    diffs (RGB each), 128 = no change. Channel order = temporal order.
    """
    resized = [cv2.resize(f, (IMG_SIZE, IMG_SIZE)).astype(np.int16)
               for f in frames_bgr]
    diffs = []
    for a, b in zip(resized, resized[1:]):
        d = np.clip(b - a + 128, 0, 255).astype(np.uint8)
        diffs.append(cv2.cvtColor(d, cv2.COLOR_BGR2RGB))
    return np.concatenate(diffs, axis=2)


def stack_to_tensor(img: np.ndarray) -> torch.Tensor:
    """HWC uint8 (3K channels) → CHW float32 normalised, stats tiled ×K."""
    k = img.shape[2] // 3
    t = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).repeat(k).view(-1, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).repeat(k).view(-1, 1, 1)
    return (t - mean) / std


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(device: torch.device, num_diffs: int = NUM_DIFFS) -> nn.Module:
    """
    ResNet-18 pretrained on ImageNet, final layer replaced for 12 classes.
    All layers unfrozen — fine-tune the whole network.
    The pretrained features (edges, textures, shapes) transfer well
    to motion-blur patterns in the diff images.

    For num_diffs > 1 the first conv is inflated to 3*num_diffs input
    channels by tiling the pretrained RGB kernels and dividing by
    num_diffs, so initial activation magnitudes match the pretrained
    network (standard channel-inflation trick from two-stream/I3D work).
    """
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    if num_diffs > 1:
        old = model.conv1
        new = nn.Conv2d(3 * num_diffs, old.out_channels,
                        kernel_size=old.kernel_size, stride=old.stride,
                        padding=old.padding, bias=False)
        with torch.no_grad():
            new.weight.copy_(old.weight.repeat(1, num_diffs, 1, 1) / num_diffs)
        model.conv1 = new

    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model.to(device)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    session_dirs = [Path(p) for pattern in args.sessions
                    for p in (Path(".").glob(pattern)
                               if "*" in pattern else [Path(pattern)])]
    session_dirs = [d for d in session_dirs if d.is_dir()]

    if not session_dirs:
        sys.exit("No session directories found. Check --sessions argument.")

    print(f"\n{'='*55}")
    print(f"  Move Classifier — Training")
    print(f"{'='*55}")
    print(f"  Sessions:  {len(session_dirs)}")
    print(f"  Input:     {args.diffs} ordered diff(s) = {3*args.diffs} channels")

    # Build dataset
    full_ds = MoveDiffDataset(session_dirs, augment=True, num_diffs=args.diffs)
    if len(full_ds) == 0:
        sys.exit("No valid samples found. Run postprocess_session.py first.")

    # Class distribution report
    counts = full_ds.class_counts()
    print(f"\n  Class distribution ({len(full_ds)} total samples):")
    for cls in range(NUM_CLASSES):
        n    = counts.get(cls, 0)
        bar  = "█" * min(n // 5, 40)
        warn = "  ← LOW" if n < 50 else ""
        print(f"    [{cls:2d}] {CLASS_NAMES[cls]:<12}  {n:4d}  {bar}{warn}")

    low_classes = [cls for cls in range(NUM_CLASSES) if counts.get(cls, 0) < 50]
    if low_classes:
        print(f"\n  WARNING: Classes {low_classes} have fewer than 50 samples.")
        print(f"           Record more solves or expect lower accuracy on these.")

    # Train / val split (80/20, stratified by class)
    random.seed(42)
    indices_by_class = {c: [] for c in range(NUM_CLASSES)}
    for i, (_, label) in enumerate(full_ds.samples):
        indices_by_class[label].append(i)

    train_idx, val_idx = [], []
    for cls, idxs in indices_by_class.items():
        random.shuffle(idxs)
        split = max(1, int(len(idxs) * 0.8))
        train_idx.extend(idxs[:split])
        val_idx.extend(idxs[split:])

    from torch.utils.data import Subset
    train_ds = Subset(full_ds, train_idx)
    val_ds   = Subset(MoveDiffDataset(session_dirs, augment=False,
                                      num_diffs=args.diffs), val_idx)

    # Weighted sampler — oversample rare classes in each batch
    sample_weights = full_ds.class_weights()
    train_weights  = [sample_weights[full_ds.samples[i][1]].item()
                      for i in train_idx]
    sampler = WeightedRandomSampler(train_weights, len(train_weights))

    train_loader = DataLoader(train_ds, batch_size=args.batch,
                              sampler=sampler, num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch,
                              shuffle=False, num_workers=0)

    print(f"\n  Train: {len(train_ds)}  Val: {len(val_ds)}")
    print(f"  Batch: {args.batch}  Epochs: {args.epochs}")

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"  GPU:   {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print(f"  Device: CPU")

    # Model, loss, optimiser
    model     = build_model(device, num_diffs=args.diffs)
    # Weighted cross-entropy handles class imbalance in the loss
    criterion = nn.CrossEntropyLoss(
        weight=full_ds.class_weights().to(device)
    )
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    best_val_acc = 0.0
    output_path  = Path(args.output)

    print(f"\n  {'Epoch':<6} {'Train Loss':<12} {'Train Acc':<12} {'Val Acc':<10}")
    print(f"  {'-'*44}")

    for epoch in range(1, args.epochs + 1):
        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        train_loss    = 0.0
        train_correct = 0
        train_total   = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out  = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

            train_loss    += loss.item() * imgs.size(0)
            train_correct += (out.argmax(1) == labels).sum().item()
            train_total   += imgs.size(0)

        scheduler.step()

        train_acc  = train_correct / train_total * 100
        avg_loss   = train_loss / train_total

        # ── Validate ───────────────────────────────────────────────────────
        model.eval()
        val_correct = 0
        val_total   = 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                out = model(imgs)
                val_correct += (out.argmax(1) == labels).sum().item()
                val_total   += imgs.size(0)

        val_acc = val_correct / val_total * 100 if val_total > 0 else 0.0

        marker = " ← best" if val_acc > best_val_acc else ""
        print(f"  {epoch:<6} {avg_loss:<12.4f} {train_acc:<11.1f}% {val_acc:.1f}%{marker}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch":     epoch,
                "state_dict":model.state_dict(),
                "val_acc":   val_acc,
                "class_names":CLASS_NAMES,
                "num_diffs": args.diffs,
            }, output_path)

    print(f"\n  Best val accuracy: {best_val_acc:.1f}%")
    print(f"  Model saved to:    {output_path}")

    # Final per-class breakdown on val set
    _per_class_eval(model, val_loader, device)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _per_class_eval(model: nn.Module, loader: DataLoader,
                    device: torch.device):
    model.eval()
    correct_by_class = Counter()
    total_by_class   = Counter()

    with torch.no_grad():
        for imgs, labels in loader:
            imgs   = imgs.to(device)
            preds  = model(imgs).argmax(1).cpu()
            for pred, label in zip(preds.numpy(), labels.numpy()):
                total_by_class[label]   += 1
                if pred == label:
                    correct_by_class[label] += 1

    print(f"\n  Per-class accuracy on validation set:")
    print(f"  {'Class':<14} {'Correct':>8} {'Total':>7} {'Acc':>8}")
    print(f"  {'-'*42}")
    for cls in range(NUM_CLASSES):
        n   = total_by_class[cls]
        c   = correct_by_class[cls]
        acc = c / n * 100 if n > 0 else 0.0
        bar = "▓" * int(acc / 5)
        print(f"  {CLASS_NAMES[cls]:<14} {c:>8} {n:>7}  {acc:>5.1f}%  {bar}")


def evaluate(args):
    """
    Evaluate on the VALIDATION split only — same 80/20 stratified split
    used during training.  Evaluating on all data inflates numbers because
    it includes training examples.
    """
    session_dirs = [Path(p) for pattern in args.sessions
                    for p in (Path(".").glob(pattern)
                               if "*" in pattern else [Path(pattern)])]
    session_dirs = [d for d in session_dirs if d.is_dir()]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt   = torch.load(args.model, map_location=device)
    num_diffs = ckpt.get("num_diffs", 1)   # pre-stack checkpoints were 1-diff

    full_ds = MoveDiffDataset(session_dirs, augment=False, num_diffs=num_diffs)
    if len(full_ds) == 0:
        sys.exit("No samples found.")

    # Reproduce the exact same stratified val split as training
    random.seed(42)
    indices_by_class = {c: [] for c in range(NUM_CLASSES)}
    for i, (_, label) in enumerate(full_ds.samples):
        indices_by_class[label].append(i)

    val_idx = []
    for cls, idxs in indices_by_class.items():
        random.shuffle(idxs)
        split = max(1, int(len(idxs) * 0.8))
        val_idx.extend(idxs[split:])   # same 20% held out during training

    from torch.utils.data import Subset
    val_ds  = Subset(full_ds, val_idx)
    loader  = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0)

    model = build_model(device, num_diffs=num_diffs)
    model.load_state_dict(ckpt["state_dict"])
    print(f"\nLoaded model from {args.model}  (epoch {ckpt['epoch']}, "
          f"{num_diffs}-diff input, val_acc={ckpt['val_acc']:.1f}% at save time)")
    print(f"Evaluating on {len(val_ds)} held-out validation samples "
          f"(20% stratified split, same as training)\n")

    _per_class_eval(model, loader, device)


# ---------------------------------------------------------------------------
# Inference helper (import and use in other scripts)
# ---------------------------------------------------------------------------

_inference_model  = None
_inference_device = None
_inference_diffs  = NUM_DIFFS

def load_for_inference(model_path: str = MODEL_PATH):
    """Load the trained model for use in a live pipeline."""
    global _inference_model, _inference_device, _inference_diffs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt   = torch.load(model_path, map_location=device)
    num_diffs = ckpt.get("num_diffs", 1)   # pre-stack checkpoints were 1-diff
    model  = build_model(device, num_diffs=num_diffs)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    _inference_model  = model
    _inference_device = device
    _inference_diffs  = num_diffs
    return model, device


def predict(frames_bgr: list[np.ndarray],
            model_path: str = MODEL_PATH) -> tuple[int, str, float]:
    """
    Predict which face moved (and direction) from an ordered move window.

    Parameters
    ----------
    frames_bgr : ordered BGR frames spanning the move — ideally the
                 5-frame window [before, mid_00, mid_01, mid_02, after].
                 Automatically subsampled to what the loaded model
                 expects (a 1-diff model uses first + middle frame).

    Returns
    -------
    (class_id, class_name, confidence)
    """
    global _inference_model, _inference_device, _inference_diffs

    if _inference_model is None:
        load_for_inference(model_path)

    need = _inference_diffs + 1
    n    = len(frames_bgr)
    if n < 2:
        raise ValueError("predict() needs at least 2 frames in temporal order")
    # Evenly resample the window to the frame count the model expects
    # (duplicates allowed — a repeated frame yields a neutral diff).
    idxs   = [round(i * (n - 1) / (need - 1)) for i in range(need)]
    frames = [frames_bgr[i] for i in idxs]

    t = stack_to_tensor(build_diff_stack(frames))
    t = t.unsqueeze(0).to(_inference_device)

    with torch.no_grad():
        logits = _inference_model(t)
        probs  = torch.softmax(logits, dim=1).squeeze().cpu()

    cls  = probs.argmax().item()
    conf = probs[cls].item()
    return cls, CLASS_NAMES[cls], round(conf, 3)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train ResNet-18 to classify cube face moves from frame diffs"
    )
    parser.add_argument("--sessions", nargs="+", required=True,
                        help="Session folder(s) — supports globs: training_data/solve_*/")
    parser.add_argument("--epochs",  type=int,   default=40)
    parser.add_argument("--batch",   type=int,   default=32)
    parser.add_argument("--lr",      type=float, default=1e-4)
    parser.add_argument("--diffs",   type=int,   default=NUM_DIFFS, choices=[1, 4],
                        help="Ordered diffs per sample: 4 = full-window stack "
                             "(default), 1 = legacy single diff (ablation)")
    parser.add_argument("--output",  type=str,   default=MODEL_PATH,
                        help="Where to save the best model")
    parser.add_argument("--eval",    action="store_true",
                        help="Evaluate an existing model instead of training")
    parser.add_argument("--model",   type=str,   default=MODEL_PATH,
                        help="Model to evaluate (used with --eval)")
    args = parser.parse_args()

    if args.eval:
        evaluate(args)
    else:
        train(args)