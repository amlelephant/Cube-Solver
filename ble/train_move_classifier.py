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
Output:  12 classes. Two label modes (--labels):

  wca (default) — camera-relative moved layer from the orientation
         tracker (U, U', D, D', L, L', R, R', F, F', B, B').
         WHY: the color of the turning face lives in a handful of sticker
         pixels, but WHERE the motion happened is the easiest feature in
         a diff image — so a color-labeled model shortcuts to "motion at
         top = yellow", which only holds for the recorder's habitual
         grip. Camera-relative labels make position the *correct* answer
         instead of a spurious one. Face color is recovered downstream
         from appearance (facelet state / cube detector), not from motion.

  ble — the raw BLE byte (0-11), color-relative; kept as the ablation
         baseline to quantify exactly what the wca retarget buys:
         0/1 blue CW/CCW   2/3 green   4/5 white
         6/7 yellow        8/9 red     10/11 orange

Validation: whole sessions are held out by default (--holdout session).
Moves inside one solve are near-duplicates of each other — ~420ms apart,
same lighting, same grip, same cube position, often the same face — so the
older per-move random split (--holdout random, kept for comparison) puts
near-identical frames on both sides and reports memorization as accuracy.
The gap between the two modes is the leakage.

Cropping: if a session folder contains crops.json (produced by
cache_crops.py using the cv/detection cube detector), every frame of a
move is cropped to that move's shared cube box before the diff stack is
built — the cube fills the input instead of ~30% of it, and background
pixels stop diluting the signal. Sessions without crops.json train on
full frames.

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

WCA_CLASS_NAMES = ["U", "U'", "D", "D'", "L", "L'", "R", "R'", "F", "F'",
                   "B", "B'"]
WCA_INDEX = {n: i for i, n in enumerate(WCA_CLASS_NAMES)}

# Horizontal-flip label maps. A mirrored clip shows a real move, but not
# the same one — flipping doubles the dataset only if the label is fixed:
#   ble: face color is unchanged in a mirror, direction reverses → XOR 1.
#   wca: direction reverses AND the left/right layers swap sides, so
#        L <-> R' and L' <-> R while U/D/F/B just get primed.
BLE_FLIP = {i: i ^ 1 for i in range(12)}
WCA_FLIP = {0: 1, 1: 0,          # U  <-> U'
            2: 3, 3: 2,          # D  <-> D'
            4: 7, 7: 4,          # L  <-> R'
            5: 6, 6: 5,          # L' <-> R
            8: 9, 9: 8,          # F  <-> F'
            10: 11, 11: 10}      # B  <-> B'

LABEL_MODES = {
    "wca": (WCA_CLASS_NAMES, WCA_FLIP),
    "ble": (CLASS_NAMES, BLE_FLIP),
}


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

    Label: wca_notation mapped to 0-11 (default), or ble_raw (0-11) in
    ble mode — see module docstring for why wca is the default.

    If the session has a crops.json (from cache_crops.py), all frames of
    a move are cropped to the move's shared cube box before diffing.

    Augmentation note on horizontal flipping:
      Flipping a CW motion produces a CCW motion visually, so the label
      must be remapped through the mode's flip map (see BLE_FLIP /
      WCA_FLIP above). This doubles the dataset for free.
      (The flip mirrors space only — channel order is untouched.)
    """

    def __init__(self, session_dirs: list[Path], augment: bool = True,
                 num_diffs: int = NUM_DIFFS, label_mode: str = "wca"):
        self.samples     = []   # [(frame_paths, label, crop_box), ...]
        # Session name per sample, index-aligned with self.samples. Kept
        # parallel rather than as a 4th tuple field so existing
        # (paths, label, box) unpacking stays valid.
        self.sample_session = []
        self.augment     = augment
        self.num_diffs   = num_diffs
        self.label_mode  = label_mode
        self.class_names, self.flip_map = LABEL_MODES[label_mode]
        self.cropped_sessions   = 0
        self.uncropped_sessions = 0
        self._load_sessions(session_dirs)

    def _label_of(self, m: dict) -> int | None:
        if self.label_mode == "ble":
            label = m.get("ble_raw")
            return label if label is not None and 0 <= label <= 11 else None
        return WCA_INDEX.get(m.get("wca_notation") or "")

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

            crop_boxes = {}
            crops_file = session_dir / "crops.json"
            if crops_file.exists():
                crop_boxes = json.loads(crops_file.read_text())["boxes"]
                self.cropped_sessions += 1
            else:
                self.uncropped_sessions += 1

            with open(labeled) as f:
                for line in f:
                    m = json.loads(line.strip())

                    label = self._label_of(m)
                    if label is None:
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

                    box = crop_boxes.get(f"move_{m['move_num']:04d}")
                    self.samples.append((paths, label,
                                         tuple(box) if box else None))
                    self.sample_session.append(session_dir.name)

        if skipped:
            print(f"  Skipped {skipped} moves (missing frames or invalid label)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        paths, label, box = self.samples[idx]

        imgs = [cv2.imread(str(p)) for p in paths]
        if any(i is None for i in imgs):
            # Fallback: neutral (no-motion) stack if a file is unreadable
            stack = np.full((IMG_SIZE, IMG_SIZE, 3 * self.num_diffs), 128,
                            dtype=np.uint8)
            return self._to_tensor(stack), label

        if box is not None:
            x1, y1, x2, y2 = box
            x1, y1 = max(0, x1), max(0, y1)
            if x2 - x1 >= 8 and y2 - y1 >= 8:
                imgs = [i[y1:y2, x1:x2] for i in imgs]

        stack = build_diff_stack(imgs)

        # Augmentation
        if self.augment:
            # Horizontal flip: remap label through the mode's flip map
            if random.random() < 0.5:
                stack = np.fliplr(stack).copy()
                label = self.flip_map[label]

            # Small random shift + scale: breaks residual "exact pixel
            # position" memorization without changing which layer moved
            stack = self._shift_scale(stack)

            # Brightness / contrast jitter (safe — doesn't change direction)
            stack = self._jitter(stack)

        return self._to_tensor(stack), label

    def _jitter(self, img: np.ndarray) -> np.ndarray:
        """Random brightness and contrast adjustment."""
        alpha = random.uniform(0.8, 1.2)    # contrast
        beta  = random.randint(-15, 15)     # brightness
        img   = np.clip(img.astype(np.float32) * alpha + beta, 0, 255)
        return img.astype(np.uint8)

    def _shift_scale(self, img: np.ndarray) -> np.ndarray:
        """
        Random translation (±8%) and scale (0.9-1.1), identical across
        all channels so temporal structure is preserved. Border fills
        with 128 — the diff stack's no-motion neutral.
        """
        h, w = img.shape[:2]
        s    = random.uniform(0.9, 1.1)
        tx   = random.uniform(-0.08, 0.08) * w
        ty   = random.uniform(-0.08, 0.08) * h
        M = np.float32([[s, 0, tx + (1 - s) * w / 2],
                        [0, s, ty + (1 - s) * h / 2]])
        # warpAffine handles at most 4 channels — warp per 3-channel diff
        out = np.empty_like(img)
        for k in range(img.shape[2] // 3):
            out[:, :, 3*k:3*k+3] = cv2.warpAffine(
                img[:, :, 3*k:3*k+3], M, (w, h),
                borderMode=cv2.BORDER_CONSTANT, borderValue=(128, 128, 128))
        return out

    def _to_tensor(self, img: np.ndarray) -> torch.Tensor:
        """HWC uint8 (3K channels) → CHW float32, ImageNet stats per diff."""
        return stack_to_tensor(img)

    def class_counts(self) -> dict[int, int]:
        return dict(Counter(label for _, label, _ in self.samples))

    def class_weights(self) -> torch.Tensor:
        """Inverse-frequency weights for weighted loss / sampler."""
        counts = self.class_counts()
        total  = sum(counts.values())
        weights = torch.zeros(NUM_CLASSES)
        for cls, cnt in counts.items():
            weights[cls] = total / (cnt * NUM_CLASSES)
        return weights


# ---------------------------------------------------------------------------
# Train / val splitting
# ---------------------------------------------------------------------------

def split_indices(ds: MoveDiffDataset, holdout: str = "session",
                  val_sessions: int | None = None,
                  seed: int = 42) -> tuple[list[int], list[int], list[str]]:
    """
    Split sample indices into (train_idx, val_idx, held_out_session_names).

    holdout="session" (default) — hold out WHOLE sessions. This is the
        honest measure of whether the model generalizes to a new recording.
        Moves within one solve are heavily correlated: consecutive moves sit
        ~420ms apart under near-identical lighting, grip and cube position,
        and are often the same face. A per-move random split therefore puts
        near-duplicate frames on both sides and reports a number closer to
        memorization than generalization.

    holdout="random" — the legacy per-move stratified 80/20 split. Kept for
        comparison against previously reported numbers; the gap between the
        two modes IS the leakage.

    Session mode does not stratify (it can't — a session contains whatever
    faces that solve happened to use), so rare classes may land entirely in
    train and get zero val samples. That is reported, not silently averaged
    away.
    """
    if holdout == "random":
        rng = random.Random(seed)
        by_class = {c: [] for c in range(NUM_CLASSES)}
        for i, (_, label, _) in enumerate(ds.samples):
            by_class[label].append(i)

        train_idx, val_idx = [], []
        for idxs in by_class.values():
            rng.shuffle(idxs)
            split = max(1, int(len(idxs) * 0.8))
            train_idx.extend(idxs[:split])
            val_idx.extend(idxs[split:])
        return train_idx, val_idx, []

    sessions = sorted(set(ds.sample_session))
    if len(sessions) < 2:
        sys.exit(f"Session holdout needs >= 2 sessions, found {len(sessions)}. "
                 f"Record more solves, or use --holdout random (leaky).")

    n_val = val_sessions if val_sessions is not None \
        else max(1, round(len(sessions) * 0.2))
    n_val = min(n_val, len(sessions) - 1)   # always leave >= 1 for training

    rng = random.Random(seed)
    shuffled = list(sessions)
    rng.shuffle(shuffled)
    held_out = set(shuffled[:n_val])

    train_idx, val_idx = [], []
    for i, sess in enumerate(ds.sample_session):
        (val_idx if sess in held_out else train_idx).append(i)

    return train_idx, val_idx, sorted(held_out)


def report_split(ds: MoveDiffDataset, train_idx: list[int], val_idx: list[int],
                 held_out: list[str], class_names: list[str]) -> None:
    """Print which sessions were held out and how val class coverage landed."""
    if held_out:
        print(f"\n  Holdout: {len(held_out)} whole session(s) — "
              f"no move from these appears in training:")
        for name in held_out:
            n = sum(1 for s in ds.sample_session if s == name)
            print(f"    {name}  ({n} moves)")
    else:
        print(f"\n  Holdout: per-move random split (LEAKY — val moves come "
              f"from the same solves as train)")

    val_counts = Counter(ds.samples[i][1] for i in val_idx)
    absent = [class_names[c] for c in range(NUM_CLASSES)
              if val_counts.get(c, 0) == 0]
    if absent:
        print(f"\n  NOTE: {len(absent)} class(es) have no validation samples "
              f"({', '.join(absent)}).")
        print(f"        Val accuracy says nothing about them. Held-out "
              f"sessions can't be stratified —")
        print(f"        raise --val-sessions or record solves covering these "
              f"faces.")


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

def build_model(device: torch.device, num_diffs: int = NUM_DIFFS,
                dropout: float = 0.4) -> nn.Module:
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

    model.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(model.fc.in_features, NUM_CLASSES),
    )
    return model.to(device)


def _load_state(model: nn.Module, state_dict: dict) -> None:
    """
    Load a checkpoint, remapping pre-dropout checkpoints (bare fc.Linear)
    onto the current Sequential(Dropout, Linear) head.
    """
    if "fc.weight" in state_dict:
        state_dict = dict(state_dict)
        state_dict["fc.1.weight"] = state_dict.pop("fc.weight")
        state_dict["fc.1.bias"]   = state_dict.pop("fc.bias")
    model.load_state_dict(state_dict)


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
    print(f"  Labels:    {args.labels} "
          f"({'camera-relative layer' if args.labels == 'wca' else 'cube-relative color'})")

    # Build dataset
    full_ds = MoveDiffDataset(session_dirs, augment=True, num_diffs=args.diffs,
                              label_mode=args.labels)
    if len(full_ds) == 0:
        sys.exit("No valid samples found. Run postprocess_session.py first.")

    class_names = full_ds.class_names
    print(f"  Crops:     {full_ds.cropped_sessions} session(s) cached, "
          f"{full_ds.uncropped_sessions} full-frame"
          + ("  ← run cache_crops.py" if full_ds.uncropped_sessions else ""))

    # Class distribution report
    counts = full_ds.class_counts()
    print(f"\n  Class distribution ({len(full_ds)} total samples):")
    for cls in range(NUM_CLASSES):
        n    = counts.get(cls, 0)
        bar  = "█" * min(n // 5, 40)
        warn = "  ← LOW" if n < 50 else ""
        print(f"    [{cls:2d}] {class_names[cls]:<12}  {n:4d}  {bar}{warn}")

    low_classes = [cls for cls in range(NUM_CLASSES) if counts.get(cls, 0) < 50]
    if low_classes:
        print(f"\n  WARNING: Classes {low_classes} have fewer than 50 samples.")
        print(f"           Record more solves or expect lower accuracy on these.")

    # Train / val split — whole sessions held out by default (see
    # split_indices docstring for why a per-move split overstates accuracy)
    train_idx, val_idx, held_out = split_indices(
        full_ds, holdout=args.holdout, val_sessions=args.val_sessions)
    report_split(full_ds, train_idx, val_idx, held_out, class_names)

    from torch.utils.data import Subset
    train_ds = Subset(full_ds, train_idx)
    val_ds   = Subset(MoveDiffDataset(session_dirs, augment=False,
                                      num_diffs=args.diffs,
                                      label_mode=args.labels), val_idx)

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
    model     = build_model(device, num_diffs=args.diffs,
                            dropout=args.dropout)
    # Weighted cross-entropy handles class imbalance in the loss
    criterion = nn.CrossEntropyLoss(
        weight=full_ds.class_weights().to(device)
    )
    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    best_val_acc = 0.0
    best_epoch   = 0
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
            best_epoch   = epoch
            torch.save({
                "epoch":     epoch,
                "state_dict":model.state_dict(),
                "val_acc":   val_acc,
                "class_names":class_names,
                "label_mode":args.labels,
                "num_diffs": args.diffs,
                "dropout":   args.dropout,
                "holdout":   args.holdout,
                "val_session_names": held_out,
            }, output_path)

        # Early stopping — val has plateaued, further epochs only overfit
        if epoch - best_epoch >= args.patience:
            print(f"  Early stop: no val improvement in {args.patience} epochs")
            break

    print(f"\n  Best val accuracy: {best_val_acc:.1f}%")
    print(f"  Model saved to:    {output_path}")

    # Final per-class breakdown on val set (best checkpoint, not last epoch)
    ckpt = torch.load(output_path, map_location=device)
    _load_state(model, ckpt["state_dict"])
    _per_class_eval(model, val_loader, device, class_names)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _per_class_eval(model: nn.Module, loader: DataLoader,
                    device: torch.device,
                    class_names: list[str] = CLASS_NAMES):
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
        print(f"  {class_names[cls]:<14} {c:>8} {n:>7}  {acc:>5.1f}%  {bar}")


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
    num_diffs   = ckpt.get("num_diffs", 1)  # pre-stack checkpoints were 1-diff
    label_mode  = ckpt.get("label_mode", "ble")  # pre-wca checkpoints
    class_names = ckpt.get("class_names", LABEL_MODES[label_mode][0])

    full_ds = MoveDiffDataset(session_dirs, augment=False, num_diffs=num_diffs,
                              label_mode=label_mode)
    if len(full_ds) == 0:
        sys.exit("No samples found.")

    # Reproduce training's val split. Prefer the session names recorded in
    # the checkpoint — that is exact even if --sessions globs differently
    # now than it did at training time. Fall back to re-deriving the split.
    ckpt_sessions = ckpt.get("val_session_names")
    holdout       = ckpt.get("holdout", "random")  # pre-holdout checkpoints

    if ckpt_sessions:
        held_out = [s for s in ckpt_sessions if s in set(full_ds.sample_session)]
        if len(held_out) < len(ckpt_sessions):
            missing = set(ckpt_sessions) - set(held_out)
            print(f"\nWARNING: {len(missing)} held-out session(s) from training "
                  f"are not in --sessions: {', '.join(sorted(missing))}")
        if not held_out:
            sys.exit("None of the checkpoint's held-out sessions are present. "
                     "Point --sessions at the folders used for training.")
        val_idx = [i for i, s in enumerate(full_ds.sample_session)
                   if s in set(held_out)]
    else:
        _, val_idx, held_out = split_indices(full_ds, holdout=holdout,
                                             val_sessions=args.val_sessions)

    from torch.utils.data import Subset
    val_ds  = Subset(full_ds, val_idx)
    loader  = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0)

    model = build_model(device, num_diffs=num_diffs)
    _load_state(model, ckpt["state_dict"])
    print(f"\nLoaded model from {args.model}  (epoch {ckpt['epoch']}, "
          f"{num_diffs}-diff input, {label_mode} labels, "
          f"val_acc={ckpt['val_acc']:.1f}% at save time)")

    if holdout == "session":
        print(f"Evaluating on {len(val_ds)} moves from "
              f"{len(held_out)} fully held-out session(s): "
              f"{', '.join(held_out)}")
    else:
        print(f"Evaluating on {len(val_ds)} samples from a per-move random "
              f"split (LEAKY — retrain with --holdout session for an honest "
              f"number)")
    print()

    _per_class_eval(model, loader, device, class_names)


# ---------------------------------------------------------------------------
# Inference helper (import and use in other scripts)
# ---------------------------------------------------------------------------

_inference_model  = None
_inference_device = None
_inference_diffs  = NUM_DIFFS
_inference_names  = CLASS_NAMES

def load_for_inference(model_path: str = MODEL_PATH):
    """Load the trained model for use in a live pipeline."""
    global _inference_model, _inference_device, _inference_diffs, \
           _inference_names
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt   = torch.load(model_path, map_location=device)
    num_diffs  = ckpt.get("num_diffs", 1)  # pre-stack checkpoints were 1-diff
    label_mode = ckpt.get("label_mode", "ble")
    model  = build_model(device, num_diffs=num_diffs)
    _load_state(model, ckpt["state_dict"])
    model.eval()
    _inference_model  = model
    _inference_device = device
    _inference_diffs  = num_diffs
    _inference_names  = ckpt.get("class_names", LABEL_MODES[label_mode][0])
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
    return cls, _inference_names[cls], round(conf, 3)


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
    parser.add_argument("--labels",  type=str,   default="wca",
                        choices=["wca", "ble"],
                        help="Label target: wca = camera-relative layer "
                             "(default — position is a sound feature), "
                             "ble = cube-relative color (ablation baseline)")
    parser.add_argument("--holdout", type=str, default="session",
                        choices=["session", "random"],
                        help="Validation split: session = hold out whole "
                             "solves (default, honest); random = legacy "
                             "per-move 80/20 (leaky — val moves come from "
                             "the same solves as train)")
    parser.add_argument("--val-sessions", type=int, default=None,
                        help="How many sessions to hold out with "
                             "--holdout session (default: 20%% of them)")
    parser.add_argument("--dropout", type=float, default=0.4,
                        help="Dropout before the classifier head")
    parser.add_argument("--weight-decay", type=float, default=1e-2,
                        help="AdamW weight decay")
    parser.add_argument("--patience", type=int,  default=12,
                        help="Early-stop after this many epochs without "
                             "val improvement")
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