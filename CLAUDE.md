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
  `.venv`. Cube-state solve validation (`cv/state_finder.py`,
  `cv/test_install.py`) uses the vendored pure-Python solver at
  `cv/twophase/` (MIT licensed — see its `LICENSE.txt`), **not** the
  `kociemba` PyPI package, which ships no Windows wheel and needs a C
  compiler to build from source; a prior version of this code assumed a
  `kociemba2` PyPI fallback existed — it doesn't, `pip install kociemba2`
  404s. `twophase`'s first `solve()` call precomputes pruning tables
  (~30-60s), cached afterward to `cv/tables.json` (gitignored).
- No test runner is configured. The one real test,
  `ble/tests/test_ble_threading.py`, is run with `python -m pytest tests/`
  from inside `ble/` (it inserts `ble/` onto `sys.path` itself).
- Scripts expect to be run **from inside their own folder** (`cv/` or
  `ble/`), not from the repo root — they load models and datasets via bare
  relative filenames (e.g. `cv/cube_detector.py` loads `"detect_full_cube.pt"`,
  not a path relative to `__file__`). Keep model weights and dataset folders
  flat alongside the scripts that reference them; don't nest them into a
  `models/`/`datasets/` subfolder without also updating the path constants.

## Repo structure

- **`cv/`** — the MVP pipeline.
  - `cube_detector.py` — YOLOv8 (`detect_full_cube.pt`, fine-tuned; falls
    back to stock `yolov8n.pt`) locates the cube; an OpenCV CSRT tracker
    follows it between detection frames for real-time CPU performance;
    the located face is perspective-warped and sliced into 9 sticker
    patches. Public API: `detect_and_extract(frame)`, `draw_sticker_overlay(...)`.
  - `color_classifier.py` — HSV-threshold classifier with confidence
    scoring (`classify_hsv`); also defines `CLASSES`, the canonical 6-color
    list other modules import.
  - `cnn_classifier.py` — small CNN sticker classifier
    (`sticker_cnn.pt`, `MODEL_PATH`), trained via `python cnn_classifier.py --train`.
  - `ensemble.py` — combines the CNN and HSV classifiers
    (`ensemble_classify`, `classify_face`) for a confidence-scored call per
    sticker; this ensemble is what makes the classifier meaningfully more
    robust than the HSV-only baseline in `legacy/hsv_only_classifier/`.
  - `state_finder.py` — **the MVP demo entry point.** Live webcam UI: scan
    all 6 faces (SPACE to capture), assemble the 54-sticker state, validate
    with the vendored `twophase` solver. This is the shape the actual
    verification feature should grow from (scan scramble → scan solved end
    state → confirm valid solve).
  - `twophase/` — vendored pure-Python two-phase solver (from
    [tcbegley/cube-solver](https://github.com/tcbegley/cube-solver), MIT
    licensed, see its `LICENSE.txt`). Zero third-party dependencies. `solve()`
    call signature and 54-char URFDLB facelet-string format match the
    `kociemba` package it replaces.
  - `test_install.py` — dependency + pipeline smoke test with no webcam or
    model weights required. Run this first when setting up.
  - `motion_detector.py`, `train.py`, `diagnose.py`, `test_square.py` —
    supporting/training utilities.
  - `cube_dataset/`, `face_dataset/` — Roboflow-exported YOLO training sets
    for the detector (gitignored — large, not source).
  - `training_runs/` — archived historical `ultralytics` training run
    output, consolidated from three separate folders during reorganization
    (gitignored; not read by any script by path — new training runs will
    create a fresh `cv/runs/` on their own).

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
    detector (differs from `cv/cube_detector.py` only in tuned constants —
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
