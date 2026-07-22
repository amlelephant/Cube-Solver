"""
prepare_data.py

One-time preprocessing: turn a recorded session's raw frames/ directory
into a compact array the onset detector can train on fast.

For each session it writes  <session>/detector_stream.npz  containing:
    frames     uint8 (N, 96, 96)  grayscale, cube-cropped, downscaled
    onset_idx  int32 (M,)         frame index of each BLE move event
    onset_ts   float64 (M,)       original BLE timestamps (for reference)
    fps        float
    name       str

Why precompute
--------------
Training samples 96-frame clips at random. Decoding 96 JPEGs per sample on
every __getitem__ makes the data loader the bottleneck (Windows runs
num_workers=0 here). A whole session at 96x96 grayscale is ~15MB, so every
session fits in memory and clip sampling becomes an array slice.

Cropping
--------
Uses the same cv/detection cube detector as cache_crops.py, but the unit
is different: cache_crops.py resolves ONE box per move window, whereas
this needs a box for every frame of a continuous stream.

Detection runs every --crop-stride frames, the resulting box sequence is
median-smoothed and then linearly interpolated to every frame. That tracks
the cube slowly drifting around the frame as you turn it, without
injecting the detector's frame-to-frame jitter — jitter would look like
global motion, which is exactly the signal the detector is trying to read
(same rule as crop_utils' module docstring, applied over time).

Sessions where the cube is never detected fall back to a centered square
crop, so a missing/failed detector degrades quality rather than blocking
training.

Usage:
    python prepare_data.py --sessions ../training_data/solve_*/
    python prepare_data.py --sessions ../training_data/solve_*/ --force
    python prepare_data.py --sessions ../training_data/solve_*/ --no-crop
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

# crop_utils lives in ble/ — same sys.path bootstrap pattern the cv/
# cross-topic scripts use (see CLAUDE.md).
_BLE_DIR = Path(__file__).resolve().parents[1]
if str(_BLE_DIR) not in sys.path:
    sys.path.insert(0, str(_BLE_DIR))

FRAME_SIZE   = 96    # must match model.FRAME_SIZE
CROP_STRIDE  = 10    # detect every Nth frame (~3Hz at 30fps)
SMOOTH_WIN   = 5     # rolling median over this many detections (~1.7s)
STREAM_FILE  = "detector_stream.npz"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path) if l.strip()]


def center_square(shape) -> tuple:
    """Fallback crop: largest centered square. Keeps aspect ratio sane."""
    h, w = shape[:2]
    side = min(h, w)
    x, y = (w - side) // 2, (h - side) // 2
    return (x, y, x + side, y + side)


def _rolling_median(arr: np.ndarray, win: int) -> np.ndarray:
    """Median filter along axis 0, edge-padded. arr is (N, 4)."""
    if len(arr) < 2 or win < 2:
        return arr
    half = win // 2
    padded = np.pad(arr, ((half, half), (0, 0)), mode="edge")
    return np.stack([np.median(padded[i:i + win], axis=0)
                     for i in range(len(arr))])


def per_frame_boxes(detector, load_frame, n_frames: int,
                    stride: int = CROP_STRIDE) -> tuple[np.ndarray, int]:
    """
    A crop box for every frame: detect every `stride` frames, median-smooth
    the detections, then linearly interpolate across all frames.

    `load_frame` is a callable i -> BGR image (or None). Taking a callable
    rather than a list of paths lets live capture reuse this unchanged on
    in-memory frames, so inference crops exactly the way training cropped.

    Returns (boxes (N, 4) int, n_detected). Falls back to a centered square
    for the whole session if nothing is ever detected.
    """
    from crop_utils import detect_box, square_with_margin

    probe_idx, raw = [], []
    shape = None
    for i in range(0, n_frames, stride):
        img = load_frame(i)
        if img is None:
            continue
        shape = img.shape
        box = detect_box(detector, img)
        if box is not None:
            probe_idx.append(i)
            raw.append(box)

    if shape is None:
        raise RuntimeError("no readable frames")

    if not raw:
        fallback = center_square(shape)
        return np.tile(fallback, (n_frames, 1)).astype(np.int32), 0

    smoothed = _rolling_median(np.array(raw, dtype=np.float32), SMOOTH_WIN)

    # Interpolate each coordinate across every frame index
    all_idx = np.arange(n_frames)
    interp = np.stack([np.interp(all_idx, probe_idx, smoothed[:, c])
                       for c in range(4)], axis=1)

    boxes = np.array([square_with_margin(tuple(b), shape) for b in interp],
                     dtype=np.int32)
    return boxes, len(raw)


def crop_to_box(img: np.ndarray, box) -> np.ndarray:
    """Clamp `box` to the image and crop; returns img unchanged if degenerate."""
    x1, y1, x2, y2 = box
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(img.shape[1], int(x2)), min(img.shape[0], int(y2))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return img
    return img[y1:y2, x1:x2]


def build_gray_stream(load_frame, boxes: np.ndarray, n_frames: int
                      ) -> np.ndarray:
    """
    (N, FRAME_SIZE, FRAME_SIZE) uint8 grayscale, each frame cropped to its
    box. This is the detector's input format — shared by offline
    preparation and live capture so the two cannot drift apart.

    An unreadable frame repeats its predecessor, which produces a
    zero-motion diff rather than a spurious one.
    """
    out = np.empty((n_frames, FRAME_SIZE, FRAME_SIZE), dtype=np.uint8)
    for i in range(n_frames):
        img = load_frame(i)
        if img is None:
            out[i] = out[i - 1] if i > 0 else 0
            continue
        # Branch on channel COUNT, not ndim. Importing ultralytics
        # monkeypatches cv2.imread (ultralytics.utils.patches) to return
        # im[..., None] for 2-D reads, so an IMREAD_GRAYSCALE load comes back
        # (H, W, 1) rather than (H, W) — but only once a detector has been
        # loaded, which is why a detector-free read of the same file looks
        # fine. An ndim check then feeds 1 channel into BGR2GRAY and throws.
        if img.ndim == 3:
            img = img[:, :, 0] if img.shape[2] == 1 else \
                cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        out[i] = cv2.resize(crop_to_box(img, boxes[i]),
                            (FRAME_SIZE, FRAME_SIZE),
                            interpolation=cv2.INTER_AREA)
    return out


def prepare_session(session_dir: Path, detector, force: bool = False) -> bool:
    out_path = session_dir / STREAM_FILE
    if out_path.exists() and not force:
        print(f"  {session_dir.name}: {STREAM_FILE} exists, skipping "
              f"(--force to redo)")
        return True

    frames_dir = session_dir / "frames"
    frames_idx = session_dir / "frames.jsonl"
    moves_path = session_dir / "moves.jsonl"

    if not frames_dir.is_dir() or not any(frames_dir.iterdir()):
        print(f"  {session_dir.name}: no frames/ — SKIPPED.")
        print(f"      This session was postprocessed without --keep-frames, "
              f"so the continuous")
        print(f"      stream is gone. Only labeled/ move snapshots survive, "
              f"which cannot")
        print(f"      train an onset detector (no inter-move negatives). "
              f"Re-record.")
        return False

    if not frames_idx.exists() or not moves_path.exists():
        print(f"  {session_dir.name}: missing frames.jsonl or moves.jsonl, "
              f"skipping")
        return False

    frame_recs = load_jsonl(frames_idx)
    moves      = load_jsonl(moves_path)

    paths = [frames_dir / r["file"] for r in frame_recs]
    keep  = [i for i, p in enumerate(paths) if p.exists()]
    if len(keep) < len(paths):
        print(f"  {session_dir.name}: {len(paths) - len(keep)} indexed frames "
              f"missing on disk, using the {len(keep)} present")
        frame_recs = [frame_recs[i] for i in keep]
        paths      = [paths[i] for i in keep]

    n = len(paths)
    if n < 2:
        print(f"  {session_dir.name}: fewer than 2 frames, skipping")
        return False

    ts  = np.array([r["ts"] for r in frame_recs], dtype=np.float64)
    dur = ts[-1] - ts[0]
    fps = n / dur if dur > 0 else 30.0

    # Crop boxes
    load_color = lambda i: cv2.imread(str(paths[i]))
    if detector is not None:
        boxes, n_det = per_frame_boxes(detector, load_color, n)
        crop_note = f"{n_det} detections -> per-frame boxes"
        if n_det == 0:
            crop_note = "cube never detected -> centered square"
    else:
        probe = cv2.imread(str(paths[0]))
        boxes = np.tile(center_square(probe.shape), (n, 1)).astype(np.int32)
        crop_note = "no detector -> centered square"

    # Decode, crop, downscale (grayscale read is cheaper than color here)
    out = build_gray_stream(
        lambda i: cv2.imread(str(paths[i]), cv2.IMREAD_GRAYSCALE), boxes, n)

    # Map each BLE move timestamp to its nearest frame index
    move_ts   = np.array([m["timestamp"] for m in moves], dtype=np.float64)
    onset_idx = np.clip(np.searchsorted(ts, move_ts), 0, n - 1)
    left      = np.clip(onset_idx - 1, 0, n - 1)
    take_left = np.abs(ts[left] - move_ts) < np.abs(ts[onset_idx] - move_ts)
    onset_idx = np.where(take_left, left, onset_idx).astype(np.int32)

    align_ms = np.abs(ts[onset_idx] - move_ts) * 1000
    dupes    = len(onset_idx) - len(np.unique(onset_idx))

    np.savez_compressed(out_path, frames=out, onset_idx=onset_idx,
                        onset_ts=move_ts, fps=fps, name=session_dir.name)

    size_mb = out_path.stat().st_size / 1e6
    print(f"  {session_dir.name}: {n} frames, {len(moves)} onsets, "
          f"{fps:.1f}fps, {size_mb:.0f}MB")
    print(f"      crop: {crop_note}")
    print(f"      onset->frame alignment: median {np.median(align_ms):.0f}ms, "
          f"max {align_ms.max():.0f}ms")
    if dupes:
        print(f"      NOTE: {dupes} onset(s) share a frame with another — "
              f"two-handed simultaneous")
        print(f"            moves closer together than one frame at "
              f"{fps:.0f}fps. Unresolvable here;")
        print(f"            see decode.py on the resulting recall ceiling.")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess sessions into detector_stream.npz")
    parser.add_argument("--sessions", nargs="+", required=True,
                        help="Session folder(s) — supports globs: "
                             "../training_data/solve_*/")
    parser.add_argument("--force", action="store_true",
                        help="Rebuild even if detector_stream.npz exists")
    parser.add_argument("--no-crop", action="store_true",
                        help="Skip cube detection; use a centered square crop")
    args = parser.parse_args()

    session_dirs = [Path(p) for pattern in args.sessions
                    for p in (Path(".").glob(pattern)
                              if "*" in pattern else [Path(pattern)])]
    session_dirs = [d for d in session_dirs if d.is_dir()]
    if not session_dirs:
        sys.exit("No session directories found. Check --sessions.")

    detector = None
    if not args.no_crop:
        from crop_utils import load_detector
        detector = load_detector()
        if detector is None:
            print("WARNING: cube detector unavailable (needs ultralytics + "
                  "cv/detection/detect_full_cube.pt).\n"
                  "         Falling back to centered square crops — the cube "
                  "will fill less of\n"
                  "         the input. Pass --no-crop to silence this.\n")

    print(f"\nPreparing {len(session_dirs)} session(s)...")
    ok = sum(prepare_session(d, detector, force=args.force)
             for d in sorted(session_dirs))
    print(f"\nDone: {ok}/{len(session_dirs)} session(s) ready.")
    if ok == 0:
        sys.exit("No sessions prepared — nothing to train on.")
