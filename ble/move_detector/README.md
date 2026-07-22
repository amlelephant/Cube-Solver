# move_detector

Finds **when** a move happens; says nothing about **which** move it was.

That split is the point. The existing `ble/train_move_classifier.py` is
good at naming a move given a correctly-centred window, but at inference
those windows came from `live_test.py`'s `MotionGate` — a global
frame-delta threshold that closed a window only after 133ms of confirmed
stillness. During a real solve the cube never comes to rest (median
inter-move gap 420ms, p10 180ms), so windows merged, and a whole-cube
rotation produced a *larger* delta than a layer turn so phantom windows
fired. The classifier was being fed garbage and its live behaviour said
almost nothing about its actual accuracy.

This replaces that gate with a learned per-frame onset score, supervised
for free by the BLE move timestamps you already record.

## Pipeline

```
record_training.py                     # webcam + BLE, keep the raw frames
  └─ postprocess_session.py --keep-frames
       └─ prepare_data.py              # frames/ -> detector_stream.npz
            └─ train.py                # CNN encoder + TCN -> onset score
                 └─ decode.py          # peak-pick -> windows -> classifier
```

```bash
python prepare_data.py --sessions ../training_data/solve_*/
python train.py        --sessions ../training_data/solve_*/
python train.py        --sessions ../training_data/solve_*/ --eval --model move_detector.pt
```

Run these from inside `move_detector/`, same convention as the rest of the
repo.

## Why peak-picking instead of a state machine

A threshold gate asks "is there motion", then uses the falling edge as a
boundary — which requires stillness between moves. Peak-picking asks
"where is motion most turn-like", which does not. Two moves 200ms apart
are two peaks inside one continuous hump; a state machine sees one
uninterrupted `MOVING` run.

It also fixes the rotation problem for free. A `y` rotation has no BLE
event, so every frame of it is labelled negative during training. The
model learns rotations are not turns; no threshold can.

## Why a TCN and not the ConvLSTM

The temporal-emphasis intuition behind the ConvLSTM is right, but with
~750 onsets from a handful of same-day sessions a ConvLSTM over raw frames
has enough capacity to memorise grip and lighting, and you then cannot
tell "wrong architecture" from "not enough data". This is 0.58M
parameters, trains in minutes, and gives a baseline to beat. Swapping in a
ConvLSTM later is a change to one module (`model.py`).

The stronger case for a recurrent model is the **classifier**, not the
detector: `train_move_classifier.py` collapses time into 12 input channels,
and its worst class by a wide margin is `U'` — a *direction* error, which
is exactly what weak temporal modelling predicts.

## Known ceiling: two-handed moves

1.8% of consecutive move pairs arrive under 100ms apart, and **97% of
those are on different faces** (`U→R'`, `R→U'`). Those are simultaneous
two-handed turns, not fast sequential ones — they overlap in time, so no
temporal model at any frame rate splits them into two peaks. Expect recall
to cap near 98%.

Do not try to fix this in the detector. Fix it downstream: a
group-theoretic decode against the scanned start and end states can insert
a move that was never seen. Which matters anyway — at 92% per-move
accuracy over a 60-move solve, raw sequence accuracy is 0.6%, so the
solver constraint is doing the heavy lifting regardless.

## Files

| file | role |
|---|---|
| `prepare_data.py` | `frames/` → `detector_stream.npz` (crop, downscale, onset indices) |
| `dataset.py` | clip sampling, Gaussian targets, augmentation, session splitting |
| `model.py` | `FrameEncoder` + TCN, and `score_stream()` for whole-session inference |
| `decode.py` | peak-picking, onset F1, and `onset_windows()` handoff to the classifier |
| `train.py` | training loop, session holdout, threshold tuning |

## Notes

- **Session holdout is mandatory**, not optional. Clips from one session
  overlap and share lighting, grip and background — a random clip split
  reports memorisation. Same reasoning as `--holdout session` in
  `train_move_classifier.py`.
- **Model selection is by onset F1, not loss.** Per-frame loss is dominated
  by the easy negatives between moves; a model can improve its loss while
  getting worse at placing exactly one peak per turn.
- **Crop boxes are interpolated, not per-frame detected.** Detector jitter
  between frames looks like global motion, which is the signal being read.
  Detection runs every 10th frame, is median-smoothed, then interpolated.
- **The model is non-causal** — frame *t* sees ~1s either side. That suits
  buffer-the-solve-then-analyse. True streaming would need causal padding
  and would cost accuracy at the leading edge.
