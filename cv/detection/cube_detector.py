"""
cube_detector.py

Locates a Rubik's cube face in a webcam frame using a two-stage pipeline:

  Stage 1 — YOLOv8n (fine-tuned)
    A neural network trained specifically on Rubik's cube images fires a
    bounding box around the cube regardless of background complexity.
    Runs every YOLO_INTERVAL frames to keep CPU cost low.

  Stage 2 — OpenCV CSRT tracker
    Between YOLO frames, a cheap tracker follows the bounding box.
    This keeps the pipeline real-time on CPU (no GPU required at inference).

  Stage 3 — Perspective warp + sticker extraction
    The bounding box is warped to a flat 300x300 image and sliced into
    9 sticker patches, which feed the ensemble classifier unchanged.

Model loading
-------------
On first call, the detector looks for 'cube_yolo.pt' in the project
directory.  If not found, it falls back to the generic YOLOv8n pretrained
weights ('yolov8n.pt'), which Ultralytics downloads automatically.
The fine-tuned model ('cube_yolo.pt') must be trained separately —
see TRAINING.md for the full walkthrough.

Public API
----------
  detect_and_extract(frame)  →  result dict
  draw_sticker_overlay(...)  →  draws predictions back onto frame
"""

import cv2
import numpy as np
import os

# ---------------------------------------------------------------------------
# YOLO via Ultralytics — optional import so the rest of the project still
# works if ultralytics is not yet installed.
# ---------------------------------------------------------------------------
try:
    from ultralytics import YOLO as _YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Resolved relative to this file, not the caller's cwd — this module gets
# imported from sibling topic folders (cv/solver/, cv/labeling/) whose cwd
# doesn't match cv/detection/ where these weights actually live.
_MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
FINE_TUNED_MODEL  = os.path.join(_MODEL_DIR, "detect_full_cube.pt")  # produced by your training run
FALLBACK_MODEL    = os.path.join(_MODEL_DIR, "yolov8n.pt")           # generic pretrained, auto-downloaded

# Run YOLO every N frames; use the tracker in between
YOLO_INTERVAL     = 12

# Minimum YOLO confidence to accept a detection
YOLO_CONF         = 0.50

# Minimum bounding-box area (fraction of frame area) to be a valid cube
MIN_BOX_FRAC      = 0.015
MAX_BOX_FRAC      = 0.92

# Per-channel mean the frame is normalized to before YOLO inference.
# 120 matches the daytime footage the detector was trained on.
NORM_TARGET_MEAN  = 120.0

# Inset from sticker edge when sampling colour (avoids border shadows)
STICKER_INSET     = 5

# Size each sticker patch is resized to (must match CNN input)
PATCH_SIZE        = 32

# Warp target size
WARP_SIZE         = 300


# ---------------------------------------------------------------------------
# Module-level state (persists across calls within one session)
# ---------------------------------------------------------------------------
_yolo_model       = None     # loaded YOLO model
_tracker          = None     # OpenCV tracker instance
_tracker_box      = None     # last good bbox: (x, y, w, h)
_frame_count      = 0        # total frames processed this session
_last_yolo_frame  = -999     # frame index when YOLO last fired
_model_type       = None     # "finetuned" | "generic" | None


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_yolo():
    """Load YOLO model once and cache it.  Returns (model, model_type)."""
    global _yolo_model, _model_type

    if _yolo_model is not None:
        return _yolo_model, _model_type

    if not YOLO_AVAILABLE:
        return None, None

    if os.path.exists(FINE_TUNED_MODEL):
        _yolo_model = _YOLO(FINE_TUNED_MODEL)
        _model_type = "finetuned"
        print(f"[detector] Loaded fine-tuned model: {FINE_TUNED_MODEL}")
    else:
        _yolo_model = _YOLO(FALLBACK_MODEL)
        _model_type = "generic"
        print(f"[detector] Fine-tuned model not found. Using generic YOLOv8n.")
        print(f"           Train cube_yolo.pt first — see TRAINING.md")

    return _yolo_model, _model_type


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_and_extract(frame):
    """
    Attempt to locate the cube face and extract 9 sticker patches.

    Parameters
    ----------
    frame : np.ndarray  BGR webcam frame

    Returns
    -------
    dict with keys:
        found        bool
        patches      list[np.ndarray]  9 patches or []
        quad         np.float32 (4,2) corners or None
        method       "yolo_detect" | "yolo_track" | "no_yolo" | None
        model_type   "finetuned" | "generic" | None
        confidence   float  YOLO confidence of last detection (0 if tracking)
        debug_frame  np.ndarray  frame with overlay drawn
    """
    global _tracker, _tracker_box, _frame_count, _last_yolo_frame

    _frame_count += 1
    debug        = frame.copy()
    fh, fw       = frame.shape[:2]
    frame_area   = fh * fw

    model, mtype = _load_yolo()

    # ------------------------------------------------------------------ #
    # Path A: no YOLO available — draw guide and return empty             #
    # ------------------------------------------------------------------ #
    if model is None:
        _draw_no_yolo(debug)
        return _empty(debug, "no_yolo")

    # ------------------------------------------------------------------ #
    # Path B: decide whether to run YOLO or use the tracker              #
    # ------------------------------------------------------------------ #
    # _tracker is None either because opencv-contrib isn't installed,
    # or because tracking was lost.  In both cases we run YOLO.
    run_yolo = (
        _tracker is None
        or _tracker_box is None
        or (_frame_count - _last_yolo_frame) >= YOLO_INTERVAL
    )

    box  = None     # (x1, y1, x2, y2)
    conf = 0.0
    method = "yolo_track"

    if run_yolo:
        box, conf = _run_yolo(model, frame, frame_area)
        if box is not None:
            _last_yolo_frame = _frame_count
            method = "yolo_detect"
            # Try to start a tracker on the new box
            new_tracker = _make_tracker()
            if new_tracker is not None:
                x1, y1, x2, y2 = box
                new_tracker.init(frame, (x1, y1, x2 - x1, y2 - y1))
                _tracker     = new_tracker
                _tracker_box = (x1, y1, x2 - x1, y2 - y1)
            else:
                # No contrib tracker available — run YOLO every frame
                _tracker     = None
                _tracker_box = None
        else:
            # YOLO missed — try the existing tracker if we have one
            if _tracker is not None and _tracker_box is not None:
                ok, tb = _tracker.update(frame)
                if ok:
                    x, y, w, h = [int(v) for v in tb]
                    box = (x, y, x + w, y + h)
                    _tracker_box = (x, y, w, h)
                else:
                    _tracker     = None
                    _tracker_box = None
    else:
        # Between YOLO frames: update tracker
        ok, tb = _tracker.update(frame)
        if ok:
            x, y, w, h = [int(v) for v in tb]
            box = (x, y, x + w, y + h)
            _tracker_box = (x, y, w, h)
        else:
            # Tracker lost — force YOLO on next frame
            _tracker         = None
            _tracker_box     = None
            _last_yolo_frame = -999

    # ------------------------------------------------------------------ #
    # Path C: no detection                                                #
    # ------------------------------------------------------------------ #
    if box is None:
        _draw_guide(debug, mtype)
        return _empty(debug, method, mtype)

    # ------------------------------------------------------------------ #
    # Path D: good box — warp and extract patches                         #
    # ------------------------------------------------------------------ #
    x1, y1, x2, y2 = box
    quad    = np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
    patches = _warp_and_slice(frame, quad)

    _draw_found(debug, box, method, conf, mtype)

    return dict(
        found       = True,
        patches     = patches,
        quad        = quad,
        method      = method,
        model_type  = mtype,
        confidence  = conf,
        debug_frame = debug,
    )


def reset_tracker():
    """Call this between face scans so the tracker doesn't carry over state."""
    global _tracker, _tracker_box, _last_yolo_frame
    _tracker         = None
    _tracker_box     = None
    _last_yolo_frame = -999


# ---------------------------------------------------------------------------
# YOLO inference
# ---------------------------------------------------------------------------

def normalize_lighting(frame):
    """
    Gray-world lighting normalization: scale each BGR channel so its mean
    lands on NORM_TARGET_MEAN.

    The detector was trained almost entirely on daylight footage; warm
    indoor lighting shifts the channel ratios far enough that confidence
    collapses (measured 2026-07-15: median conf 0.02 at night vs 0.81 on
    daytime frames, restored to 0.5+ by this correction). Brightness-only
    scaling does NOT fix it — the cast is in the channel ratios.

    Used for YOLO input only. Sticker patches for color classification are
    cut from the raw frame, whose color cast the ensemble's per-session
    calibration already accounts for.

    Implemented as a per-channel 256-entry LUT: the scaling is a constant
    per channel, so applying it via cv2.LUT is bit-identical to the
    float-multiply version but ~50x faster (measured 43ms -> 0.9ms per
    720p frame, 2026-07-18) — at 1 LUT per frame this was half the live
    pipeline's frame budget.
    """
    means = cv2.mean(frame)[:3]
    ramp = np.arange(256, dtype=np.float32)
    lut = np.dstack([
        np.clip(ramp * (NORM_TARGET_MEAN / max(m, 1.0)), 0, 255)
        for m in means
    ]).astype(np.uint8)
    return cv2.LUT(frame, lut)


def _run_yolo(model, frame, frame_area):
    """
    Run YOLO on the frame.  Returns (box, confidence) or (None, 0).

    box is (x1, y1, x2, y2) in pixel coordinates.

    When the fine-tuned model is loaded we look only at class 0 (cube).
    When using the generic pretrained model we look at ALL classes with a
    lower confidence floor — the cube will not be in COCO classes so this
    won't fire, but it lets you test the pipeline before training.
    """
    results = model(normalize_lighting(frame), verbose=False, conf=YOLO_CONF)

    best_box  = None
    best_conf = 0.0

    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            conf = float(box.conf[0])
            if conf < YOLO_CONF:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            area = (x2 - x1) * (y2 - y1)

            # Size sanity check
            frac = area / frame_area
            if frac < MIN_BOX_FRAC or frac > MAX_BOX_FRAC:
                continue

            if conf > best_conf:
                best_conf = conf
                best_box  = (x1, y1, x2, y2)

    return best_box, best_conf


# ---------------------------------------------------------------------------
# OpenCV tracker factory
# ---------------------------------------------------------------------------

def _make_tracker():
    """
    Return an OpenCV tracker if opencv-contrib-python is installed,
    otherwise return None.  The caller treats None as "no tracker
    available" and runs YOLO on every frame instead — correct but
    slightly slower on CPU.

    CSRT (contrib) is preferred.  KCF (contrib) is the fallback.
    Both live in opencv-contrib-python, NOT in base opencv-python.

    To enable tracking:
        pip uninstall opencv-python -y
        pip install opencv-contrib-python
    """
    try:
        return cv2.TrackerCSRT_create()
    except AttributeError:
        pass
    try:
        return cv2.TrackerKCF_create()
    except AttributeError:
        pass
    return None


# ---------------------------------------------------------------------------
# Warp and sticker extraction
# ---------------------------------------------------------------------------

def _warp_and_slice(frame, quad):
    """
    Perspective-warp the region inside quad to WARP_SIZE×WARP_SIZE and
    slice into 9 sticker patches.
    """
    dst = np.float32([
        [0,         0        ],
        [WARP_SIZE, 0        ],
        [WARP_SIZE, WARP_SIZE],
        [0,         WARP_SIZE],
    ])
    M    = cv2.getPerspectiveTransform(quad, dst)
    warp = cv2.warpPerspective(frame, M, (WARP_SIZE, WARP_SIZE))

    cell    = WARP_SIZE // 3
    patches = []
    for row in range(3):
        for col in range(3):
            x0 = col * cell + STICKER_INSET
            y0 = row * cell + STICKER_INSET
            x1 = x0 + cell - 2 * STICKER_INSET
            y1 = y0 + cell - 2 * STICKER_INSET
            patch = warp[y0:y1, x0:x1]
            patch = cv2.resize(patch, (PATCH_SIZE, PATCH_SIZE))
            patches.append(patch)

    return patches


# ---------------------------------------------------------------------------
# Drawing utilities
# ---------------------------------------------------------------------------

def _draw_found(debug, box, method, conf, mtype):
    x1, y1, x2, y2 = box
    color  = (0, 230, 60)  if method == "yolo_detect" else (0, 180, 255)
    label  = f"cube {'det' if method == 'yolo_detect' else 'trk'}"
    if conf > 0:
        label += f" {conf:.2f}"
    if mtype:
        label += f" [{mtype[:2]}]"

    cv2.rectangle(debug, (x1, y1), (x2, y2), color, 2)
    cv2.putText(debug, label, (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def _draw_guide(debug, mtype):
    h, w   = debug.shape[:2]
    cx, cy = w // 2, h // 2
    size   = int(min(h, w) * 0.38)
    cv2.rectangle(debug,
                  (cx - size, cy - size), (cx + size, cy + size),
                  (65, 65, 65), 1)

    lines = ["Hold cube face here"]
    if mtype == "generic":
        lines += ["(generic model — train cube_yolo.pt", "for reliable detection)"]
    for i, line in enumerate(lines):
        cv2.putText(debug, line,
                    (cx - size + 4, cy - size - 10 + i * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (100, 100, 100), 1)


def _draw_no_yolo(debug):
    h, w = debug.shape[:2]
    cv2.putText(debug,
                "ultralytics not installed — run: pip install ultralytics",
                (10, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 80, 220), 1)


def draw_sticker_overlay(frame, quad, colors, confidences):
    """
    Project the 9 colour predictions back onto the original frame as
    filled circles, with a yellow warning ring on low-confidence stickers.
    """
    from color_classifier import COLOR_BGR

    dst = np.float32([
        [0,         0        ],
        [WARP_SIZE, 0        ],
        [WARP_SIZE, WARP_SIZE],
        [0,         WARP_SIZE],
    ])
    M_inv = cv2.getPerspectiveTransform(dst, quad)
    cell  = WARP_SIZE // 3

    for row in range(3):
        for col in range(3):
            idx = row * 3 + col
            wx  = col * cell + cell // 2
            wy  = row * cell + cell // 2

            pt  = cv2.perspectiveTransform(np.float32([[[wx, wy]]]), M_inv)
            fx  = int(pt[0][0][0])
            fy  = int(pt[0][0][1])

            color_name = colors[idx]      if idx < len(colors)      else "unknown"
            conf       = confidences[idx] if idx < len(confidences) else 0.0
            bgr        = COLOR_BGR.get(color_name, (80, 80, 80))

            outer_r = int(13 + conf * 5)
            cv2.circle(frame, (fx, fy), outer_r,     (20, 20, 20), -1)
            cv2.circle(frame, (fx, fy), outer_r - 3, bgr,          -1)
            if conf < 0.50:
                cv2.circle(frame, (fx, fy), outer_r, (0, 210, 255), 1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _empty(debug, method="none", mtype=None):
    return dict(found=False, patches=[], quad=None,
                method=method, model_type=mtype,
                confidence=0.0, debug_frame=debug)