# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A computer-vision pipeline that verifies a Rubik's Cube solve from webcam
scans alone (no smart cube required): locate the cube, classify all 54
stickers, validate the scanned state with group theory. This is the MVP and
primary focus — see `README.md` for the pitch. `docs/VISION.md` is a much
larger product vision doc ("CubeArena") kept for direction; most of it (web
app, matchmaking, ratings, tournaments) is not implemented here.

**Priority order, deliberately**: the CV pipeline (`cv/`) is the point of
this project and demonstrates the ML/CV work. The smart-cube BLE integration
(`ble/`) is a secondary convenience path for users who happen to own
compatible hardware — useful, but not what this repo is trying to prove.
Don't let BLE work crowd out CV work without being asked.

## Setup

- Python 3.13, venv at `.venv/` (`.venv\Scripts\activate` on Windows).
- `pip install -r requirements.txt` — pinned to what's proven working in
  `.venv`. Cube-state solve validation (`cv/solver/state_finder.py`,
  `cv/test_install.py`) uses the vendored pure-Python solver at
  `cv/solver/twophase/` (MIT licensed — see its `LICENSE.txt`), **not** the
  `kociemba` PyPI package, which ships no Windows wheel and needs a C
  compiler to build from source; a prior version of this code assumed a
  `kociemba2` PyPI fallback existed — it doesn't, `pip install kociemba2`
  404s. `twophase`'s first `solve()` call precomputes pruning tables
  (~30-60s), cached afterward to `cv/solver/tables.json` (gitignored).
- No test runner is configured. The one real test,
  `ble/tests/test_ble_threading.py`, is run with `python -m pytest tests/`
  from inside `ble/` (it inserts `ble/` onto `sys.path` itself).
- `cv/` is split into topic subfolders — `detection/`, `classification/`,
  `labeling/`, `solver/` (see Repo structure below) — and scripts expect to
  be run **from inside their own subfolder** (e.g. `cd cv/solver && python
  state_finder.py`; `cd cv/detection && python train.py`), same as `ble/`.
  They load models and datasets via bare relative filenames (e.g.
  `cv/detection/cube_detector.py` loads `"detect_full_cube.pt"`, not a path
  relative to `__file__`), so keep model weights and dataset folders flat
  alongside the script that references them within its topic subfolder —
  don't move a model without updating the constant that points at it. The
  three files whose imports cross a topic boundary (`cv/solver/state_finder.py`,
  `cv/labeling/autolabel.py`, `cv/test_install.py`) carry a small `sys.path`
  bootstrap at the top that adds the sibling topic folder(s) they need —
  keep that bootstrap in sync if you move a module between topics.
  `cv/test_install.py` itself stays at `cv/` root (it's a cross-cutting
  smoke test spanning every topic, not owned by one).

## Repo structure

- **`cv/`** — the MVP pipeline, split into topic subfolders (reorganized
  2026-07-15 — see below). Cross-topic imports (e.g. `state_finder.py`
  reaching into `detection/`) go through a small `sys.path` bootstrap — see
  Setup above — rather than package-style `from cv.detection.cube_detector
  import ...` imports.
  - **`detection/`** — locates the cube/face in a frame.
    - `cube_detector.py` — YOLOv8 (`detect_full_cube.pt`, fine-tuned; falls
      back to stock `yolov8n.pt`) locates the cube; an OpenCV CSRT tracker
      follows it between detection frames for real-time CPU performance;
      the located face is perspective-warped and sliced into 9 sticker
      patches. Public API: `detect_and_extract(frame)`, `draw_sticker_overlay(...)`.
    - `continuity_guard.py` — pure logic (never touches camera/model) that
      turns a stream of per-frame detections into the solve-continuity
      report described in `docs/ROADMAP.md` §2.1 (uniqueness, presence,
      trajectory-continuity checks against the swap attack); wraps
      `cube_detector` via `detect_cubes`/`dedup_boxes`. `test_continuity_guard.py`
      is its unit test.
    - `motion_detector.py` — MOVING/STILL/NO_CUBE gating (pixel-diff
      `DeltaDetector` by default, optional CNN `CNNDetector`) deciding
      whether a frame is worth feeding into detection/classification.
    - `train.py` — trains the YOLO detector, writes `cube_yolo.pt`.
    - `diagnose.py` — diagnoses a trained YOLO model (class count, mAP from
      `runs/detect/cube_yolo/results.csv`, live inference sanity checks).
    - `test_square.py` — standalone classical-CV (contour-based)
      face-square finder; experiment, not wired into the pipeline.
    - `detect_full_cube.pt`, `cube_yolo.pt`, `yolov8n.pt`, `yolo26n.pt` —
      YOLO weights (gitignored). Note: `cube_detector.py` actually loads
      `detect_full_cube.pt`, while `train.py`/`diagnose.py`/`test_install.py`
      all reference `cube_yolo.pt` — two different fine-tuned weight files
      that currently coexist; a pre-existing mismatch, not yet unified.
    - `cube_dataset/`, `face_dataset/` — Roboflow-exported YOLO training
      sets (gitignored — large, not source).
    - `training_runs/` — archived historical `ultralytics` training run
      output (gitignored; new training runs create a fresh `runs/` here on
      their own).
  - **`classification/`** — per-sticker color classification.
    - `color_classifier.py` — HSV-threshold classifier with confidence
      scoring (`classify_hsv`); also defines `CLASSES`, the canonical
      6-color list other modules import.
    - `cnn_classifier.py` — small CNN sticker classifier (`sticker_cnn.pt`,
      `MODEL_PATH`), trained via `python cnn_classifier.py --train` (on
      synthetically generated sticker patches — no external dataset folder).
    - `ensemble.py` — combines the CNN and HSV classifiers
      (`ensemble_classify`, `classify_face`) for a confidence-scored call
      per sticker; this ensemble is what makes the classifier meaningfully
      more robust than the HSV-only baseline in `legacy/hsv_only_classifier/`.
  - **`labeling/`** — dataset capture + labeling tools.
    - `record_frames.py` — lightweight webcam recorder for cube-**detection**
      training data only: no BLE, no move tracking, just continuous frames
      into `sessions/solve_<timestamp>/frames/`. Use this instead of
      `ble/record_training.py` when you don't need a BLE-synced move dataset.
    - `autolabel.py` — model-assisted bounding-box labeling: `propose`
      auto-labels confident frames (via `detection/continuity_guard.py`)
      and linearly interpolates through short detection gaps; you review by
      *deleting* wrong previews (keeping = accepting); `finalize` writes a
      YOLO train/val dataset. See its module docstring for the full flow.
    - `label_review.py` — general image-review UI (Left/Right to scroll, D
      to pull a bad image — and its matching YOLO label, if any — into a
      `deleted/` subfolder). Not specific to `autolabel.py` output; works on
      any `images/` (`+labels/`) folder, e.g. `detection/face_dataset/train/images`.
    - `autolabel_out/`, `sessions/` — generated output (gitignored).
  - **`solver/`** — the MVP demo entry point + solve validation.
    - `state_finder.py` — **the MVP demo entry point.** Live webcam UI: scan
      all 6 faces (SPACE to capture), assemble the 54-sticker state,
      validate with the vendored `twophase` solver. This is the shape the
      actual verification feature should grow from (scan scramble → scan
      solved end state → confirm valid solve).
    - `twophase/` — vendored pure-Python two-phase solver (from
      [tcbegley/cube-solver](https://github.com/tcbegley/cube-solver), MIT
      licensed, see its `LICENSE.txt`). Zero third-party dependencies.
      `solve()` call signature and 54-char URFDLB facelet-string format
      match the `kociemba` package it replaces.
    - `tables.json` — precomputed pruning tables cache (gitignored).
  - `test_install.py` — dependency + pipeline smoke test spanning all four
    subfolders above, with no webcam or model weights required. Stays at
    `cv/` root since it isn't owned by one topic. Run this first when
    setting up.

- **`ble/`** — smart-cube (GoCube / Rubik's Connected) integration, secondary
  to the MVP.
  - `cube_ble.py` — connects over Bluetooth (`bleak`), streams `MoveEvent`s
    and full `CubeState`. Vendor protocol constants and framing
    (checksum, `0x2A`/`0x0D 0x0A` prefix/suffix) documented inline.
  - `win_compat.py` — **must be the first import** in any script using
    `bleak` on Windows (forces MTA COM threading before `pywin32`/OpenCV/etc.
    can silently set STA, which breaks bleak's async callbacks). See its
    docstring for the full explanation. `cube_ble.py` already does this
    correctly — preserve the import order if you touch it.
  - `orientation_tracker.py` — maps cube-relative BLE move events (reported
    by center color) to camera-relative WCA notation via a virtual cube
    model, with IMU drift correction.
  - `record_training.py` → `postprocess_session.py` → `train_move_classifier.py`
    → `live_test.py` — the pipeline for recording BLE+webcam sessions,
    aligning move timestamps to frames, training a ResNet-18 move
    classifier on temporal diff images, and testing it live. Sessions land
    in `ble/training_data/solve_<timestamp>/` (gitignored, large).

- **`legacy/`** — superseded code, kept for reference, not part of the
  active pipeline:
  - `cube_detection_standalone/` — an earlier, nearly-identical standalone
    detector (differs from `cv/detection/cube_detector.py` only in tuned constants —
    model filename, confidence thresholds); superseded once the detector
    was integrated into the full ensemble pipeline.
  - `hsv_only_classifier/` — the pre-ensemble, CNN-free color classifier.
  - `old_ble_copy/` — an earlier draft of `cube_ble.py` without thread-safe
    event scheduling.
  - `old_models/`, `tmp_repro.py` — old checkpoints and a scratch repro
    script.
  - `hand_tracking_experiment/` — a `mediapipe` hand-tracking spike,
    unrelated to the current direction.

- **`docs/VISION.md`** — the full CubeArena product spec. Reference only.

## Notes from the reorganization (2026-07-14)

This repo was previously organized as several parallel, copy-pasted
experiment folders (`classification cnn and cv2/`, `classification only cv2/`,
`cube detection/`, `full_move/`) with genuinely duplicated modules that had
diverged (two different `cube_ble.py`, two `color_classifier.py`, and
`win_compat.py`/`orientation_tracker.py` living in a folder that didn't
actually use them while `full_move/` imported them by bare name and silently
depended on that other folder being on `sys.path`). It's now consolidated
into `cv/` (MVP), `ble/` (secondary), and `legacy/` (archived), with the
duplication resolved by picking the more complete implementation in each
case. If something seems to reference a module that "should" be here but
isn't, check `legacy/` before assuming it's missing.

No commits existed before this reorganization — `git log` is the source of
truth for anything after this point.

## Known accuracy issues and fixes in progress

**Detection reliability** — `cube_detector.py`'s two-stage pipeline (YOLO
every `YOLO_INTERVAL` frames, CSRT/KCF tracker in between) requires
`opencv-contrib-python`, not base `opencv-python` — `cv2.TrackerCSRT_create`
only exists in contrib. With base `opencv-python` installed, tracker
creation silently returns `None`, `run_yolo` is permanently `True`, and YOLO
runs unsmoothed on every frame with no bridging of momentary confidence
dips. This was misconfigured until 2026-07-14; `requirements.txt` now pins
`opencv-contrib-python`. If detection ever feels flaky again, check
`pip show opencv-python` isn't shadowing it (the two packages can't coexist
— uninstall the base one first).

The face detector itself is also still only trained on 270 images
(`cv/detection/face_dataset/`) and struggles disproportionately with the orange (L)
face specifically — plausibly a data-imbalance issue, not yet root-caused
by inspecting the actual training images. More/varied training data is the
real fix; not yet done.

**Orange/red color calibration** — `ensemble.py`'s orange/red tiebreak
(the hardest color pair — orange reads redder than its nominal hue under
most non-daylight lighting) is calibrated per session via
`ensemble.calibrate()`. Originally this was only fed from a successful
full L-face scan in `state_finder.py`'s main loop — which meant it could
never activate in practice when detecting the orange face was itself the
thing failing (the exact case it was needed for). Fixed by adding
`run_calibration()`, a dedicated pre-scan step that samples a fixed box in
the raw frame directly (no face detection involved at all) for the R and L
center stickers before the 6-face scan begins. Don't reintroduce a
calibration path that depends on `cube_detector` succeeding first — that's
the bug that made the original fix inert.
