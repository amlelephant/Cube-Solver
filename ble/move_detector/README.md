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
record_training.py                     # webcam + BLE, keeps the raw frames
  └─ postprocess_session.py            # frames/ kept by default
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

## Testing in a new environment

Cross-environment performance is the number that matters and the one most
easily faked, so this path is worth following exactly. Measured on
2026-07-22, moving to new lighting cost ~11 points of speed-matched recall
until sessions from that environment were in training — a same-room score
says very little about a new room.

```bash
# 1. record in the new environment (needs the BLE cube for ground truth)
cd ..  &&  python record_training.py
python postprocess_session.py --session training_data/solve_<stamp>
cd move_detector
python prepare_data.py --sessions ../training_data/solve_<stamp>/

# 2. score the deployed model on it — it has never seen this session
python train.py --eval --model move_detector.pt \
                --sessions ../training_data/solve_<stamp>/
```

`--eval` prints which sessions it considers **never seen in any form** and
scores only those plus explicitly held-out ones; sessions the checkpoint
trained on are skipped unless you pass `--allow-train-sessions`. Cite the
unseen number. If a checkpoint predates 2026-07-22 it does not record its
training set at all, and `--eval` will say so rather than guess.

Without a BLE cube there is no ground truth, so use `live_detect.py` for a
qualitative look instead — it reads threshold and `min_sep` from the
checkpoint, so it matches however the model was tuned:

```bash
python live_detect.py --detector move_detector.pt
```

To then *fix* a new environment rather than just measure it, add its
sessions to training and hold one out per environment, so the reported
number stays cross-environment rather than quietly becoming within-one:

```bash
python train.py --sessions ../training_data/solve_*/ --holdout session \
                --val-session-names solve_<old_env> solve_<new_env>
```

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

## Crowded moves: mostly fixable, and mostly in the detector

This section used to say that sub-100ms pairs are two-handed simultaneous
turns which no temporal model can split, and to fix it downstream rather
than in the detector. That was measured on the 2026-07-21 sessions, which
contain almost no fast moves (2.5% of pairs under 100ms), and it was
wrong. Against faster footage — 28% of held-out onsets have a neighbour
under 150ms — most crowded pairs turn out to be separable, and two
detector-side changes recovered most of the loss:

| change | sub-150ms recall |
|---|---|
| baseline (`MIN_SEP=3`, `sigma=2`) | 73.8% |
| `MIN_SEP=2` | 78.6% |
| `MIN_SEP=2` + `sigma=1` | 83.5% |

Both were mis-set for the same reason: they were tuned on slow footage.
Every sub-150ms miss scored 0.80-0.98 at the ground-truth frame, so the
model saw those turns and the *decoder* was discarding them; and at
`sigma=2` the Gaussian targets for onsets 1-2 frames apart merge into one
blob, so the supervision never asked for two peaks. See `decode.py`'s
header for the full measurement.

A residual floor is real — 14 of the 17 remaining sub-150ms misses are on
different faces, consistent with genuinely overlapping two-handed turns.
Recover those downstream: a group-theoretic decode against the scanned
start and end states can insert a move that was never seen. That matters
regardless — at 92% per-move accuracy over a 60-move solve, raw sequence
accuracy is 0.6%, so the solver constraint is doing the heavy lifting
either way.

The general lesson, since it will recur: **a decoder constant tuned on one
speed regime silently becomes a ceiling in another**, and it looks exactly
like a model limitation from the outside.

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
