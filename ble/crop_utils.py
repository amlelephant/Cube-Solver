"""
crop_utils.py

Shared cube-crop helpers for move-classifier training (cache_crops.py)
and live inference (live_test.py).

Uses the fine-tuned YOLO cube detector from cv/detection/ to find the
cube bounding box, then expands it to a square with margin so the crop
contains the whole cube plus a little context.

The one rule that matters: ALL frames of one move window are cropped with
the SAME box (median of per-frame detections). Cropping each frame with
its own box would inject the detector's frame-to-frame jitter straight
into the temporal diffs the classifier trains on — fake global motion.
"""

import sys
from pathlib import Path

import cv2
import numpy as np

# cube_detector lives in cv/detection/ — same sys.path bootstrap pattern
# the cv/ cross-topic scripts use (see CLAUDE.md).
_DETECTION_DIR = Path(__file__).resolve().parents[1] / "cv" / "detection"
if str(_DETECTION_DIR) not in sys.path:
    sys.path.insert(0, str(_DETECTION_DIR))

from cube_detector import (  # noqa: E402
    FINE_TUNED_MODEL, normalize_lighting, MIN_BOX_FRAC, MAX_BOX_FRAC,
)

# Lower than the live pipeline's YOLO_CONF (0.50): for cropping, a
# slightly loose box beats no box — mid-turn motion blur drops confidence.
CROP_CONF   = 0.35

# Extra context around the detected box, per side, as a fraction of the
# box's larger dimension.
CROP_MARGIN = 0.12


def load_detector():
    """Load the fine-tuned cube detector once. None if unavailable."""
    try:
        from ultralytics import YOLO
    except ImportError:
        return None
    if not Path(FINE_TUNED_MODEL).exists():
        return None
    return YOLO(FINE_TUNED_MODEL)


def detect_box(model, frame):
    """Best cube box (x1, y1, x2, y2) in pixels, or None."""
    fh, fw = frame.shape[:2]
    results = model(normalize_lighting(frame), verbose=False, conf=CROP_CONF)
    best, best_conf = None, 0.0
    for r in results:
        if r.boxes is None:
            continue
        for b in r.boxes:
            conf = float(b.conf[0])
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            frac = (x2 - x1) * (y2 - y1) / (fw * fh)
            if frac < MIN_BOX_FRAC or frac > MAX_BOX_FRAC:
                continue
            if conf > best_conf:
                best, best_conf = (x1, y1, x2, y2), conf
    return best


def median_box(boxes):
    """Per-coordinate median of a list of (x1, y1, x2, y2) boxes."""
    a = np.array(boxes, dtype=np.float32)
    return tuple(int(round(v)) for v in np.median(a, axis=0))


def square_with_margin(box, frame_shape, margin=CROP_MARGIN):
    """Expand box by margin and pad to square, clamped inside the frame."""
    fh, fw = frame_shape[:2]
    x1, y1, x2, y2 = box
    side = max(x2 - x1, y2 - y1) * (1 + 2 * margin)
    side = min(side, fw, fh)
    half = side / 2
    cx = min(max((x1 + x2) / 2, half), fw - half)
    cy = min(max((y1 + y2) / 2, half), fh - half)
    return (int(round(cx - half)), int(round(cy - half)),
            int(round(cx + half)), int(round(cy + half)))


def move_crop_box(model, frames, margin=CROP_MARGIN):
    """One shared square crop box for all frames of a move, or None."""
    boxes = [b for b in (detect_box(model, f) for f in frames)
             if b is not None]
    if not boxes:
        return None
    return square_with_margin(median_box(boxes), frames[0].shape, margin)


def crop(frame, box):
    """Crop frame to box; returns the frame unchanged on a degenerate box."""
    x1, y1, x2, y2 = box
    x1, y1 = max(0, x1), max(0, y1)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return frame
    return frame[y1:y2, x1:x2]
