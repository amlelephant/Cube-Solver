# Cube Solver — Computer-Vision Speedcube Verification

A computer-vision pipeline that verifies a Rubik's Cube solve using nothing but
a webcam: it scans the cube's six faces before and after a solve, reconstructs
the full 54-sticker state from each scan, and validates the result with a
group-theory solve check. No smart cube or special hardware required.

This is the verification core for **CubeArena**, a planned competitive
platform for speedcubing (see [`docs/VISION.md`](docs/VISION.md) for the full
product vision — the vision doc describes a much larger platform than exists
today; treat it as direction, not a status report).

## Why this matters

Every existing "verified" speedcubing platform requires a smart cube — an
expensive piece of hardware most cubers don't own. A webcam-only verification
path is strictly more accessible: anyone with a laptop camera can compete on
a leaderboard with some confidence the result is real, without buying
anything. That accessibility gap is the actual product bet here.

**Honest limitation:** webcam-based scan verification does not stop a cube
swap (solving a second cube out of frame and presenting it as the scrambled
one's result). It does verify that the scanned end state is a legitimately
solved cube reachable from the scanned start scramble — which is meaningfully
more rigor than a self-reported time, and is the right tradeoff for a v1 that
doesn't require anyone to own extra hardware.

## How it works (MVP path)

1. **Cube face detection** (`cv/cube_detector.py`) — a fine-tuned YOLOv8
   model locates the cube in frame, an OpenCV CSRT tracker follows it between
   detection frames to stay real-time on CPU, and the located face is
   perspective-warped into a flat image and sliced into 9 sticker patches.
2. **Sticker color classification** (`cv/ensemble.py`,
   `cv/cnn_classifier.py`, `cv/color_classifier.py`) — each sticker patch is
   classified by both a small CNN and an HSV-threshold classifier; the two
   are combined for a confidence-scored color call per sticker.
3. **Full-state scan** (`cv/state_finder.py`) — walks the user through
   scanning all six faces (U/R/F/D/L/B), assembles the 54-sticker state
   string, and validates it against cube group theory using `kociemba`.
4. **Verification** — run the scan once on the scrambled cube and once after
   the solve; a valid end state that resolves to "solved" is the verification
   signal that feeds the leaderboard.

## Repo structure

- **`cv/`** — the MVP: cube detection, sticker classification, and the
  full-state scanner (`state_finder.py`). This is the active development
  focus.
- **`ble/`** — smart-cube (GoCube/Rubik's Connected) Bluetooth integration
  and a per-move classifier trained on webcam + BLE move data. This is a
  secondary, convenience-oriented verification path for users who do own
  smart-cube hardware — not required for the MVP.
- **`legacy/`** — superseded prototypes and experiments kept for reference
  (an earlier standalone cube detector, an HSV-only classifier baseline
  without the CNN ensemble, an old BLE integration draft, a hand-tracking
  experiment). Not part of the active pipeline.
- **`docs/`** — product vision doc.

## Setup

```
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

`kociemba` is required but not bundled by default on all platforms — see the
comment in `requirements.txt` if the install fails; `kociemba2` is a
pure-Python fallback that `state_finder.py` and `test_install.py` both
already detect automatically.

## Running it

From inside `cv/`:

```
python test_install.py     # verifies dependencies, models, and runs a smoke test — start here
python state_finder.py     # full 6-face scan + solve verification demo
```

Model weights (`*.pt`) referenced by these scripts are trained artifacts and
are not committed to this repo (see `.gitignore`) — `test_install.py` will
tell you which ones are missing and how to train them
(`cv/train.py` for the YOLO detector, `cv/cnn_classifier.py --train` for the
sticker classifier).

The optional smart-cube path lives in `ble/`; see the docstring at the top
of `ble/cube_ble.py` for usage.

## Status

Early-stage prototype. The detection and classification models are trained
on a limited dataset (one user, limited lighting/camera conditions) and have
not been validated for generalization to a broad user base — that's the next
real milestone, not a solved problem.
