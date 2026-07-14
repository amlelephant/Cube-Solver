"""
motion_detector.py

Determines whether the cube is currently in motion or at rest.
Used as the controller that switches between:
  - Frame-stacking data collection (MOVING)
  - State-reading / partner's algorithm (STILL)

Two backends, selectable at runtime:

  Backend A — Classical frame delta (default)
    No model, no training data, runs in ~0ms.
    Compares pixel values between consecutive cube crops.
    Reliable for most conditions. Use this first.

  Backend B — CNN classifier (optional, trained)
    Small MobileNetV3 binary classifier.
    More robust to lighting flicker and camera jitter.
    Train with: python motion_detector.py --train
    Use when frame delta produces too many false positives.

Both backends operate on the CUBE CROP ONLY (from YOLO bounding box),
not the full frame. Body movement, background changes, and other motion
outside the cube region are ignored.

Output states:
  MOVING  — cube face(s) are actively rotating
  STILL   — cube is stationary, safe to read state
  NO_CUBE — YOLO has not found the cube in this frame

Usage:
    from motion_detector import MotionDetector, CubeState

    detector = MotionDetector(backend="delta")  # or "cnn"
    detector.start()

    while True:
        ret, frame = cap.read()
        cube_crop = get_cube_crop(frame)  # from cube_detector.py
        state = detector.update(cube_crop)

        if state == CubeState.MOVING:
            collect_motion_frame(frame)
        elif state == CubeState.STILL:
            read_cube_state(frame)

Training the CNN backend:
    python motion_detector.py --train --data training_data/
    python motion_detector.py --test --webcam
"""

import cv2
import numpy as np
import os
import argparse
from enum import Enum
from collections import deque
from dataclasses import dataclass
from typing import Optional

# ---------------------------------------------------------------------------
# State enum
# ---------------------------------------------------------------------------

class CubeState(Enum):
    STILL   = "STILL"
    MOVING  = "MOVING"
    NO_CUBE = "NO_CUBE"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Frame delta backend
DELTA_THRESHOLD       = 20.0    # mean absolute pixel diff to trigger MOVING
                                # lower = more sensitive, higher = less sensitive
                                # tune this first if getting false positives

# How many consecutive frames must agree before state changes
# Prevents single-frame flickers from switching state
STILL_CONFIRM_FRAMES  = 3      # need N STILL readings to declare STILL
MOVING_CONFIRM_FRAMES = 2      # need N MOVING readings to declare MOVING

# CNN backend
CNN_MODEL_PATH        = "motion_cnn.pt"
CNN_INPUT_SIZE        = (64, 64)
CNN_CONF_THRESHOLD    = 0.65   # below this = uncertain, use STILL as safe default

# Crop resize for both backends (smaller = faster)
PROCESS_SIZE          = (128, 128)


# ---------------------------------------------------------------------------
# Frame delta backend
# ---------------------------------------------------------------------------

class DeltaDetector:
    """
    Computes mean absolute difference between consecutive cube crops.
    If diff exceeds DELTA_THRESHOLD, cube is MOVING.

    Robust enough for most conditions. Fails on:
    - Lighting that flickers (fluorescent bulb hum)
    - Camera auto-exposure hunting
    Both produce a constant low-level diff that can exceed threshold.
    Fix: raise DELTA_THRESHOLD slightly, or switch to CNN backend.
    """

    def __init__(self, threshold: float = DELTA_THRESHOLD):
        self.threshold  = threshold
        self._prev_gray = None

    def reset(self):
        self._prev_gray = None

    def compute(self, crop: np.ndarray) -> tuple[CubeState, float]:
        """
        Returns (raw_state, diff_value).
        raw_state is not yet smoothed — caller applies confirmation logic.
        """
        if crop is None or crop.size == 0:
            return CubeState.NO_CUBE, 0.0

        # Resize and convert to grayscale
        small = cv2.resize(crop, PROCESS_SIZE)
        gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)

        if self._prev_gray is None:
            self._prev_gray = gray
            return CubeState.STILL, 0.0

        diff  = np.mean(np.abs(gray - self._prev_gray))
        self._prev_gray = gray

        state = CubeState.MOVING if diff > self.threshold else CubeState.STILL
        return state, float(diff)


# ---------------------------------------------------------------------------
# CNN backend
# ---------------------------------------------------------------------------

def build_motion_cnn():
    """
    Tiny binary classifier for STILL vs MOVING.
    Input: (3, 64, 64) RGB crop
    Output: 2 logits [STILL, MOVING]

    Architecture: 4 conv blocks + global average pool + linear.
    ~80K parameters. Runs in <2ms on CPU.
    """
    try:
        import torch.nn as nn
        class MotionCNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.features = nn.Sequential(
                    # 64 → 32
                    nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16),
                    nn.ReLU(inplace=True), nn.MaxPool2d(2),
                    # 32 → 16
                    nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True), nn.MaxPool2d(2),
                    # 16 → 8
                    nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True), nn.MaxPool2d(2),
                    # 8 → 4
                    nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d(1),
                )
                self.classifier = nn.Sequential(
                    nn.Flatten(),
                    nn.Dropout(0.3),
                    nn.Linear(64, 2),
                )

            def forward(self, x):
                return self.classifier(self.features(x))

        return MotionCNN()
    except ImportError:
        return None


class CNNDetector:
    """
    CNN-based motion classifier. Falls back to DeltaDetector if model
    file not found or PyTorch not installed.
    """

    def __init__(self):
        self._model  = None
        self._device = None
        self._loaded = False

    def load(self) -> bool:
        try:
            import torch
            if not os.path.exists(CNN_MODEL_PATH):
                return False
            model = build_motion_cnn()
            if model is None:
                return False
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.load_state_dict(torch.load(CNN_MODEL_PATH, map_location=device))
            model.eval()
            self._model  = model
            self._device = device
            self._loaded = True
            return True
        except Exception as e:
            print(f"[motion_detector] CNN load failed: {e}")
            return False

    def compute(self, crop: np.ndarray) -> tuple[CubeState, float]:
        if not self._loaded or crop is None or crop.size == 0:
            return CubeState.NO_CUBE, 0.0

        import torch
        small = cv2.resize(crop, CNN_INPUT_SIZE)
        rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        t     = torch.from_numpy(rgb.transpose(2, 0, 1)).float()
        t     = (t / 255.0 - 0.5) / 0.5
        t     = t.unsqueeze(0).to(self._device)

        with torch.no_grad():
            logits = self._model(t)
            probs  = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

        moving_prob = float(probs[1])
        if moving_prob > CNN_CONF_THRESHOLD:
            return CubeState.MOVING, moving_prob
        elif moving_prob < (1 - CNN_CONF_THRESHOLD):
            return CubeState.STILL, 1.0 - moving_prob
        else:
            # Uncertain — default to STILL (safer for state reading)
            return CubeState.STILL, 0.5


# ---------------------------------------------------------------------------
# Main MotionDetector class
# ---------------------------------------------------------------------------

@dataclass
class MotionResult:
    state:       CubeState
    raw_diff:    float        # raw diff value (delta) or moving_prob (CNN)
    confirmed:   bool         # False if still accumulating confirmation frames
    frame_count: int          # total frames processed


class MotionDetector:
    """
    Controller that outputs confirmed CubeState for each frame.

    Applies confirmation hysteresis to prevent state flickering:
    - Must see MOVING for MOVING_CONFIRM_FRAMES consecutive frames → declare MOVING
    - Must see STILL for STILL_CONFIRM_FRAMES consecutive frames → declare STILL
    - In-between: hold the current confirmed state

    This means there is a small lag at transitions (~3-5 frames = 100-170ms at 30fps).
    This is intentional: we want to avoid starting frame collection on a
    single noisy frame and stopping collection before the move completes.
    """

    def __init__(self,
                 backend:   str   = "delta",
                 threshold: float = DELTA_THRESHOLD):
        """
        backend: "delta" or "cnn"
        threshold: only used by delta backend
        """
        self.backend          = backend
        self._delta           = DeltaDetector(threshold)
        self._cnn             = CNNDetector()
        self._confirmed_state = CubeState.STILL
        self._pending_buffer  = deque(maxlen=max(STILL_CONFIRM_FRAMES,
                                                  MOVING_CONFIRM_FRAMES))
        self._frame_count     = 0
        self._use_cnn         = False

        if backend == "cnn":
            loaded = self._cnn.load()
            if loaded:
                self._use_cnn = True
                print("[motion_detector] CNN backend loaded.")
            else:
                print("[motion_detector] CNN model not found, "
                      "falling back to delta backend.")
                print(f"                  Train with: "
                      f"python motion_detector.py --train")

    def reset(self):
        """Call between solves to clear state."""
        self._delta.reset()
        self._confirmed_state = CubeState.STILL
        self._pending_buffer.clear()
        self._frame_count = 0

    def update(self, cube_crop: Optional[np.ndarray]) -> MotionResult:
        """
        Process one frame. Returns the current confirmed state.

        cube_crop: BGR image of the cube region only (from YOLO crop).
                   Pass None if YOLO did not find the cube this frame.
        """
        self._frame_count += 1

        if cube_crop is None or cube_crop.size == 0:
            return MotionResult(CubeState.NO_CUBE, 0.0, True, self._frame_count)

        # Get raw reading from active backend
        if self._use_cnn:
            raw_state, diff = self._cnn.compute(cube_crop)
        else:
            raw_state, diff = self._delta.compute(cube_crop)

        # Accumulate in confirmation buffer
        self._pending_buffer.append(raw_state)

        # Check for MOVING confirmation (fast to trigger)
        moving_count = sum(1 for s in self._pending_buffer
                           if s == CubeState.MOVING)
        if moving_count >= MOVING_CONFIRM_FRAMES:
            self._confirmed_state = CubeState.MOVING

        # Check for STILL confirmation (slower to trigger — wait for settle)
        still_count = sum(1 for s in self._pending_buffer
                          if s == CubeState.STILL)
        if still_count >= STILL_CONFIRM_FRAMES:
            self._confirmed_state = CubeState.STILL

        confirmed = (moving_count >= MOVING_CONFIRM_FRAMES or
                     still_count  >= STILL_CONFIRM_FRAMES)

        return MotionResult(self._confirmed_state, diff,
                            confirmed, self._frame_count)

    @property
    def current_state(self) -> CubeState:
        return self._confirmed_state


# ---------------------------------------------------------------------------
# CNN training
# ---------------------------------------------------------------------------

def train_cnn(data_dir: str, epochs: int = 20, batch_size: int = 32):
    """
    Train the CNN motion classifier from labeled data.

    Expected data directory structure:
        data_dir/
            STILL/
                frame_0001.jpg
                frame_0002.jpg
                ...
            MOVING/
                frame_0001.jpg
                ...

    Generate this structure with: python label_motion.py --data training_data/
    """
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, Dataset
    except ImportError:
        print("PyTorch required for CNN training.")
        return

    class MotionDataset(Dataset):
        def __init__(self, root):
            self.samples = []
            for label_idx, class_name in enumerate(["STILL", "MOVING"]):
                class_dir = os.path.join(root, class_name)
                if not os.path.exists(class_dir):
                    print(f"WARNING: {class_dir} not found")
                    continue
                for fname in os.listdir(class_dir):
                    if fname.lower().endswith((".jpg", ".png", ".jpeg")):
                        self.samples.append(
                            (os.path.join(class_dir, fname), label_idx)
                        )

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            path, label = self.samples[idx]
            img = cv2.imread(path)
            if img is None:
                img = np.zeros((*CNN_INPUT_SIZE, 3), dtype=np.uint8)
            img = cv2.resize(img, CNN_INPUT_SIZE)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            t   = torch.from_numpy(img.transpose(2, 0, 1)).float()
            t   = (t / 255.0 - 0.5) / 0.5
            return t, label

    dataset = MotionDataset(data_dir)
    if len(dataset) < 20:
        print(f"Too few samples ({len(dataset)}). "
              f"Generate with: python label_motion.py --data training_data/")
        return

    n_val   = max(1, int(len(dataset) * 0.15))
    n_train = len(dataset) - n_val
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model     = build_motion_cnn().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    still_count  = sum(1 for _, l in dataset.samples if l == 0)
    moving_count = sum(1 for _, l in dataset.samples if l == 1)
    print(f"\nDataset: {len(dataset)} samples "
          f"(STILL={still_count}, MOVING={moving_count})")
    print(f"Device:  {device}\n")

    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        train_correct = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            train_correct += (model(x).argmax(1) == y).sum().item()
        scheduler.step()

        model.eval()
        val_correct = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                val_correct += (model(x).argmax(1) == y).sum().item()

        train_acc = train_correct / n_train * 100
        val_acc   = val_correct   / n_val   * 100
        print(f"Epoch {epoch:2d}/{epochs}  "
              f"train={train_acc:.1f}%  val={val_acc:.1f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), CNN_MODEL_PATH)
            print(f"           ↑ saved ({val_acc:.1f}%)")

    print(f"\nBest val accuracy: {best_val_acc:.1f}%")
    print(f"Model saved to:    {CNN_MODEL_PATH}")


# ---------------------------------------------------------------------------
# Label generation helper
# ---------------------------------------------------------------------------

def label_from_ble_sessions(data_dir: str, output_dir: str,
                             moving_window_s: float = 0.25,
                             still_min_gap_s: float = 0.6):
    """
    Auto-generate STILL/MOVING labeled crops from record_training.py sessions.

    For each solve session in data_dir:
      - Frames within ±moving_window_s of a BLE move event → MOVING
      - Frames more than still_min_gap_s from any event    → STILL

    Requires: cube_detector.py (YOLO) to crop frames correctly.

    Usage:
        python motion_detector.py --label --data training_data/ --output motion_labels/
    """
    import json
    from pathlib import Path

    try:
        from cube_detector import detect_and_extract
    except ImportError:
        print("cube_detector.py required for cropping.")
        return

    still_dir  = os.path.join(output_dir, "STILL")
    moving_dir = os.path.join(output_dir, "MOVING")
    os.makedirs(still_dir,  exist_ok=True)
    os.makedirs(moving_dir, exist_ok=True)

    sessions = [d for d in Path(data_dir).iterdir() if d.is_dir()]
    total_still  = 0
    total_moving = 0

    for session in sessions:
        moves_file  = session / "moves.jsonl"
        frames_dir  = session / "frames"

        if not moves_file.exists():
            continue

        # Load move timestamps
        move_times = []
        with open(moves_file) as f:
            for line in f:
                m = json.loads(line)
                move_times.append(m["timestamp"])

        if not move_times:
            continue

        # Label before/after frames from each move
        for fname in sorted(frames_dir.glob("*.jpg")):
            img = cv2.imread(str(fname))
            if img is None:
                continue

            # Get frame timestamp from filename or metadata
            # Frames are named move_NNNN_before.jpg / move_NNNN_after.jpg
            stem = fname.stem
            # "before" frames are just before a move → MOVING
            # "after" frames are after settle → STILL
            # This is a simplification — see note below

            result = detect_and_extract(img)
            if not result["found"] or not result["patches"]:
                continue

            # Crop the cube region
            quad = result["quad"]
            if quad is None:
                continue
            x1 = int(min(quad[:, 0]))
            y1 = int(min(quad[:, 1]))
            x2 = int(max(quad[:, 0]))
            y2 = int(max(quad[:, 1]))
            crop = img[max(0,y1):y2, max(0,x1):x2]
            if crop.size == 0:
                continue

            crop_resized = cv2.resize(crop, CNN_INPUT_SIZE)

            if "before" in stem:
                # Frame just before a move — likely transitioning or still
                # Label as STILL (the cube was at rest just before the move fired)
                out = os.path.join(still_dir, f"{session.name}_{stem}.jpg")
                cv2.imwrite(out, crop_resized)
                total_still += 1
            elif "after" in stem:
                # Frame after settle — STILL
                out = os.path.join(still_dir, f"{session.name}_{stem}.jpg")
                cv2.imwrite(out, crop_resized)
                total_still += 1

        # For MOVING labels we need intermediate frames
        # These come from frames captured during the BLE event window
        # In the current record_training.py, we don't store mid-move frames
        # TODO: update record_training.py to also save 3-4 mid-move frames
        # per move event (captured 50-100ms after the event fires)

    print(f"\nLabeled: STILL={total_still}, MOVING={total_moving}")
    print(f"Output:  {output_dir}/")
    print()
    print("NOTE: MOVING frames require updating record_training.py to capture")
    print("mid-move frames. See update_recorder() in this file.")


def update_recorder_note():
    """
    Documents the change needed in record_training.py to capture mid-move frames.
    These become MOVING labels for CNN training.

    In record_training.py, inside record_move(), after receiving the BLE event:

        # Immediately capture 4 frames — these are during or just after motion
        moving_frames = []
        for i in range(4):
            await asyncio.sleep(0.03)   # ~30ms spacing
            ts, frame = cam.read_latest()
            moving_frames.append((ts, frame))
            fname = self.save_frame(frame, f"{label}_mid_{i:02d}")

    These mid frames are labeled MOVING. The before/after frames are STILL.
    """
    pass  # this is documentation only


# ---------------------------------------------------------------------------
# Live test / demo
# ---------------------------------------------------------------------------

def run_live_test(backend: str = "delta", threshold: float = DELTA_THRESHOLD):
    """
    Open webcam, run motion detector, display result on screen.
    Press Q to quit, T to print current threshold calibration info.
    """
    try:
        from cube_detector import detect_and_extract, reset_tracker
    except ImportError:
        print("cube_detector.py required for live test.")
        return

    detector = MotionDetector(backend=backend, threshold=threshold)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    diff_history = deque(maxlen=60)   # for calibration display

    print(f"\nLive motion test — backend={backend}")
    print("Hold cube still, then move it. Watch the state label.")
    print("Press T to print diff statistics. Q to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        # Get cube crop
        result    = detect_and_extract(frame)
        debug     = result["debug_frame"]
        cube_crop = None

        if result["found"] and result["quad"] is not None:
            q  = result["quad"]
            x1 = int(min(q[:, 0]))
            y1 = int(min(q[:, 1]))
            x2 = int(max(q[:, 0]))
            y2 = int(max(q[:, 1]))
            cube_crop = frame[max(0,y1):y2, max(0,x1):x2]

        motion = detector.update(cube_crop)
        diff_history.append(motion.raw_diff)

        # Draw state overlay
        state_color = {
            CubeState.MOVING:  (0,  80, 255),
            CubeState.STILL:   (0, 220,  60),
            CubeState.NO_CUBE: (80,  80,  80),
        }[motion.state]

        label = f"{motion.state.value}  diff={motion.raw_diff:.2f}"
        cv2.putText(debug, label, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, state_color, 2)
        cv2.putText(debug, f"frame={motion.frame_count}", (10, 72),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)

        # Diff bar
        bar_w = min(int(motion.raw_diff * 8), 400)
        cv2.rectangle(debug, (10, 85), (10 + bar_w, 100), state_color, -1)
        cv2.rectangle(debug, (10, 85), (10 + int(threshold * 8), 100),
                      (255, 255, 255), 1)

        cv2.imshow("Motion Detector", debug)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("t") and diff_history:
            arr = list(diff_history)
            print(f"Last 60 frames — "
                  f"min={min(arr):.2f}  max={max(arr):.2f}  "
                  f"mean={np.mean(arr):.2f}  std={np.std(arr):.2f}")
            print(f"Current threshold: {threshold}")
            print(f"Suggested threshold (mean+2std): "
                  f"{np.mean(arr) + 2*np.std(arr):.2f}")

    cap.release()
    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Motion detector for cube solving")
    parser.add_argument("--test",      action="store_true",
                        help="Run live webcam test")
    parser.add_argument("--train",     action="store_true",
                        help="Train the CNN backend")
    parser.add_argument("--label",     action="store_true",
                        help="Generate STILL/MOVING labels from BLE sessions")
    parser.add_argument("--backend",   type=str, default="delta",
                        choices=["delta", "cnn"])
    parser.add_argument("--threshold", type=float, default=DELTA_THRESHOLD)
    parser.add_argument("--data",      type=str, default="training_data/")
    parser.add_argument("--output",    type=str, default="motion_labels/")
    parser.add_argument("--epochs",    type=int, default=20)
    args = parser.parse_args()

    if args.test:
        run_live_test(args.backend, args.threshold)
    elif args.train:
        train_cnn(args.output, epochs=args.epochs)
    elif args.label:
        label_from_ble_sessions(args.data, args.output)
    else:
        parser.print_help()