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

## Directory layout (reorganized 2026-08-03)

This folder used to accumulate every checkpoint, sweep result, and log
flat at its root — over 150 files, with no indication of when anything
was produced. It's now:

    checkpoints/          every *.pt — model checkpoints
    results/<YYYY-MM-DD>/ sweep/audit/eval *.json output, dated by when
                          the run that produced it happened
    logs/<YYYY-MM-DD>/    the matching *.log files, same dating
    cache/                reconstruct_tables.npz, crop_contact_sheet.png
    <everything else>     *.py, *.md, *.txt, *.sh stay at this root

Filenames were preserved during the move (only their directory changed),
so every exact filename cited in GAMEPLAN.md / ALGORITHM_PRIOR.md /
ACCURACY_TARGET.md / PATH_TO_VERIFICATION.md still identifies the same
file — just look for it under `results/<some-date>/` or `logs/<some-date>/`
rather than bare at this root. New sweeps should write their `--out` into
`results/<today>/` / `logs/<today>/` by the same convention rather than
back into the root.

The old first-generation ResNet-18 classifier track (`ble/train_move_*`'s
predecessor experiments: encoding sweeps, optical-flow probes, and their
checkpoints/logs) was archived out of `ble/` root to `legacy/move_classifier_rnd/`
the same day — see that folder if you're looking for something that
"should" be in `ble/` but isn't; `train_move_classifier.py` and
`encodings_move.py` stayed at `ble/` root since they're still live imports
of the current pipeline (`prepare_data.py`, `algorithms.py`, `live_detect.py`,
`window_audit.py` all import constants from them).

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
python train.py        --sessions ../training_data/solve_*/ --eval --model checkpoints/move_detector.pt
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
python train.py --eval --model checkpoints/move_detector.pt \
                --sessions ../training_data/solve_<stamp>/
```

`--eval` prints which sessions it considers **never seen in any form** and
scores only those plus explicitly held-out ones; sessions the checkpoint
trained on are skipped unless you pass `--allow-train-sessions`. Cite the
unseen number. If a checkpoint predates 2026-07-22 it does not record its
training set at all, and `--eval` will say so rather than guess.

### The full suite: `env_suite.py`

`--eval` scores the detector alone and needs a BLE cube for ground truth.
To measure the **whole pipeline** in a new room, on any cube:

```bash
python env_suite.py --name kitchen-evening        # 3 takes, saves a scorecard
python env_suite.py --compare scorecards/*.json   # read them side by side
```

Ground truth is a prescribed scramble, so no smart cube is needed and the
number describes the cube you are actually holding. Three takes, each
probing an axis that has broken this pipeline before rather than three
repetitions of an easy case: **steady** (baseline), **fast** (crowded
moves — peaks 1-2 frames apart, the detector's known weak spot, which
aggregate F1 hides), and **offset** (cube in a frame corner, separating
"the room is dark" from "YOLO cannot find the cube here").

Each take reports capture fps, the crop's luminance / contrast /
sharpness, then the three-way stage split (detector recall, classifier
accuracy, end-to-end) and what the decode recovers on top. Two details
that decide whether the comparison means anything:

- **fps is printed first and gates everything.** A dimmer room makes the
  webcam expose longer and drop frames, and below 30fps the detector
  *misses* moves rather than inventing them — so "this room is worse" and
  "this room is dimmer so capture got slower" look identical unless fps is
  on the scorecard.
- **the within-environment spread is printed too.** One environment varies
  several points across its own takes; a cross-environment difference
  smaller than that is not yet a difference.

`--session` scores recorded sessions instead of the webcam, which is how
you get a baseline row for the training environment without re-recording
it — the columns come from the same code, so the rows are comparable.
`--compare` then attributes any gap to capture rate, detector or
classifier, in that order.

#### What it found, and what fixed it (2026-07-22)

Scored on sessions held out from *both* classifiers, so every cell is
honest and the rows differ only by weights:

| classifier | 07-21 (trained env) | 07-22 (new env) |
|---|---|---|
| `move_classifier_all20.pt` | 91.5% | **84.3%** |
| `move_classifier_all23.pt` | 92.4% | **92.7%** |

The detector was at 100% recall in every cell — the entire gap was the
classifier, which is what `--compare` reported unprompted. The suite also
says *why* the new environment is harder in numbers rather than adjectives:
its crop sharpness is 107-144 against 318-381 on 07-21, i.e. materially
more motion blur, which is exactly what degrades a classifier reading
temporal diffs.

Retraining on all 23 sessions closed the gap (+8.4 points on 07-22) with
no cost on the old environment (+0.9). Note the holdout had to be *named*
to see any of this: `all20` reported a healthy 94.3%, but on a validation
split that happened to contain no 07-22 session at all. Two headline
accuracies are not comparable unless their holdouts span the same
environments — hence `--val-session-names`, one session per environment,
on both trainers.

Without a BLE cube there is no ground truth, so use `live_detect.py` for a
qualitative look instead — it reads threshold and `min_sep` from the
checkpoint, so it matches however the model was tuned:

```bash
python live_detect.py --detector checkpoints/move_detector_all28.pt
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
and its worst classes remain `U'`, `F` and `F'` — *direction* errors, which
is exactly what weak temporal modelling predicts. That survived retraining
on 23 sessions (`move_classifier_all23.pt`: `U'` 86.7%, `F` 83.3%,
`F'` 78.9% against 97-100% for the D and L layers), so it is a shape
problem rather than a data-volume problem.

#### But direction is not the binding error live (2026-07-23)

Read the above only as a statement about **in-distribution** data. The
first live `--ble` take says the error *axis* changes completely once you
leave it, and it is worth knowing which problem you are buying before
rearchitecting for one of them:

| regime | substitutions | temporal inverse | adjacent face | opposite face |
|---|---|---|---|---|
| prescribed scrambles, slow (3 takes) | 13 | **69%** | 31% | 0% |
| free solves, long (3 takes) | 63 | **24%** | **67%** | 10% |
| 12 recorded sessions (in training) | 47 | **40%** | — | — |

Same models, same room. On slow scrambles and on the recorded sessions the
residual is direction confusion — the hard case that survives when
everything else works. On a real 100-move solve the classifier fails at a
more basic level: it names the *wrong face*, often confidently (`F`→`B` at
0.99, `L'`→`R` at 0.98, `R'`→`U'` at 1.00).

(The first version of this table read 6% temporal on solves; that was a
single take. Across three it is 24% — still outweighed by adjacent-face
2.7 to 1, but not the near-zero one take suggested. A single live take
does not establish an error mix.)

The rotation explanation was tested and rejected — see the `--ble` caveat
above; the errors are isolated, not contiguous. So this is generalisation,
not labelling. Per-move accuracy was **72.7%** on the scramble and **74.0%**
on the solve, against 92-98% on recorded sessions: an ~19-point drop, far
larger than the ~8 points a new environment cost in the 07-22 measurement.

The practical consequence: **an architecture change that encodes time more
richly targets 6% of the errors that matter for long solves.** Whatever is
done to the classifier next should be justified against the adjacent-face
number, and re-measured with a saved live take (`--save`) rather than a
held-out recorded session, or it will optimise the regime that already
works.

#### Data volume WAS the lever, at a bigger increment (2026-07-24)

The very next retrain (`move_classifier_all27.pt`, +4 sessions from one more
`--ble` sitting) concluded data volume was *not* the fix: it scored worse
than the deployed `move_classifier_all23.pt` on every session held out by
both. Six more live sessions later (`move_classifier_all39.pt`, +12 sessions
from five more `--ble --save` sittings, same day) that conclusion reversed.
Scored on the identical 5-session cross-environment holdout — three
recorded sessions all23 and all27 also never trained on, plus the two live
sessions all23 predates entirely — per-move accuracy:

| classifier | 20260720 | 20260721 | 20260722 | 20260723 (live) | 20260724 (live) | aggregate |
|---|---|---|---|---|---|---|
| `move_classifier_all23.pt` | 97.2% | 93.2% | 90.4% | **7.1%** | **8.7%** | 61.5% |
| `move_classifier_all27.pt` | 95.8% | 88.4% | 83.2% | 74.5% | 59.5% | 79.6% |
| `move_classifier_all39.pt` | 95.8% | 89.7% | **91.0%** | **85.7%** | **79.4%** | **88.0%** |

`all23`'s 7-9% on live sessions is not a typo — it predates every `--ble`
take that exists and was never going to generalise to a regime it had never
seen once. `all27`'s +4 sessions from a single sitting recovered some of
that but cost accuracy on 20260721/20260722, reading exactly like "more
data of the same kind doesn't help" if you stopped there. `all39`'s +12
sessions from five sittings recovered the rest of the live gap **and**
stopped costing recorded-session accuracy (20260722 improved to a new
best). The earlier conclusion wasn't wrong given what it measured — one
sitting is not enough sessions to separate "this data doesn't help" from
"this data doesn't help yet" — but read it as bounded by sample size, not
as a general result. `move_classifier_all39.pt` is now `CLASSIFIER_PATH`.

The detector got the same treatment for the same reason: the sessions
these five sittings recorded had never been through `prepare_data.py` at
all (`--session` and `--ble` scoring don't need the cache;
`move_detector/train.py` does), so the deployed `checkpoints/move_detector.pt` — a
same-day "final fit" on 12 sessions with no held-out score of its own — had
literally never been measured against 20260723/20260724 footage. Once
prepared and scored, it read 89.8% F1 across all 16 sessions from those two
days. Retrained as `checkpoints/move_detector_all28.pt` (28 frame-bearing sessions,
`--holdout session` with one name per recording day including 20260724):
93.9% F1 / 95.8% recall aggregate on that four-day holdout, and it beats
the old detector head-to-head on the two sessions neither had trained on
(95.8% vs 91.2% F1, 92.5% vs 91.6% F1). Now `DETECTOR_PATH`.

#### Correction, same day: the 7-9% number above was mostly a broken eval, not a broken model

The table two paragraphs up is misleading and the "data volume WAS the
lever" headline is wrong about *why*. `train_move_classifier.py` crops
every frame to the move's cube box **if the session has a `crops.json`**
(from `cache_crops.py`) and silently trains on the **full, uncropped
frame** otherwise. `cache_crops.py` had only ever been run on the 23
recorded sessions — every one of the 16 sessions from the five 07-23/07-24
`--ble --save` sittings (the entire live-regime holdout in the table above)
trained full-frame, while inference (`live_detect.py`,
`move_detector/prepare_data.py`) **always** crops via the cube detector.
So the table above wasn't measuring "distribution shift from live
footage" cleanly — for the two live holdout sessions specifically, it was
also comparing every classifier's full-frame-uncropped behaviour against a
cropped inference-shaped input, on top of whatever real content gap
existed.

Re-scored after running `cache_crops.py` on all 39 sessions (so the
holdout images are cropped for every checkpoint, cleanly separating "did
this classifier ever see live content" from "was this eval apples to
apples"):

| classifier | 20260720 | 20260721 | 20260722 | 20260723 (live) | 20260724 (live) | aggregate |
|---|---|---|---|---|---|---|
| `move_classifier_all23.pt` (unchanged weights, fair eval) | 97.2% | 93.2% | 90.4% | 90.8% | 84.1% | **90.6%** |
| `move_classifier_all27.pt` | 95.8% | 88.4% | 83.2% | 81.6% | 73.0% | 83.6% |
| `move_classifier_all39.pt` (trained on a cropped/uncropped mix) | 95.8% | 89.7% | 91.0% | 93.9% | 78.6% | 89.1% |
| `move_classifier_all39_cropped.pt` (all 39 sessions, all cropped) | 97.2% | 94.5% | 94.0% | 93.9% | 94.4% | **94.6%** |

`all23` alone — the same weights, zero retraining — jumps from 61.5% to
90.6% aggregate just by fixing the eval. It was never anywhere near as bad
on live footage as the first table claimed; it was being scored on images
it was never trained to read regardless of content. Notice also that
`all39` (mixed cropped/uncropped training data) scores *worse* than plain
`all23` on 20260721 in this fair comparison (89.7% vs 93.2%) — training on
inconsistent scale seems to cost accuracy even on in-distribution content,
not just on the mismatched sessions themselves.

`move_classifier_all39_cropped.pt` — all 39 sessions, all of them cropped
— is the real result: 94.6% aggregate, and it's the only checkpoint that
doesn't have a weak day. It's now `CLASSIFIER_PATH`, replacing
`move_classifier_all39.pt`.

**End-to-end effect (detector + classifier + decode, not just isolated
classifier accuracy):** re-run on the same 9 sessions as the error-type
breakdown below, raw per-move accuracy went from 77.9% to **90.9%**, and
for the first time one session (`solve_20260724_094947_solve`) actually
verified end to end with an exact reconstruction — the decode had never
once engaged before this, because every prior raw error rate (15-45 errors
per session) was far outside its ~6-error repair budget. Substitution
errors alone fell from 206 to 68 across those 9 sessions (-67%), and within
that shrunk pool, adjacent-face confusion fell further (127→33, -74%) than
direction confusion (70→30, -57%) — consistent with cropping specifically
restoring the spatial resolution needed to tell *which* face moved, not
just making everything uniformly a bit better. Detector errors (missed +
phantom onsets) didn't change, since the detector wasn't touched — they're
now the **majority** of remaining errors (55%, up from 30%), which is
where the next round of work should go, not the classifier.

**Lesson for next time:** when a retrain "fixes" a regime, check what else
changed about the data before crediting the fix to the thing you meant to
change (here: session count / "more live data"). `cache_crops.py`'s
full-frame fallback is silent — it prints a one-line summary
(`N session(s) cached, M full-frame`) buried in the training log's header
and nothing else calls it out. Grep for it before trusting a
cross-environment number: `grep "full-frame" training_run_*.log`.

### The window mismatch, and fixing it by training on it (2026-07-26)

Detector errors became the majority of what's left (above), but the
classifier still had one more lever: `window_audit.py` scored 2456 moves by
how many of their 5 window slots (`before`/`mid_00`/`mid_01`/`mid_02`/`after`)
differ from the exact window `postprocess_session.py` built at training time,
and grouped accuracy by that count:

```
differing   0      1      2      3      4      5
trainer   98.9%  98.6%  99.5%  98.2%  97.6% 100.0%   (same moves, canonical window)
inference 98.9%  97.8%  95.8%  95.8%  91.9%  91.9%
```

The trainer row is flat — these aren't harder moves — so the window itself
costs accuracy: inference anchors a window on the **frame** the detector's
peak landed on, while training anchored on the **sub-frame BLE timestamp**,
and that quantization flips the nearest-frame choice for offsets that sit
near a frame midpoint (the classifier's `mid_*` slots do, at 30fps).

First attempt: make `onset_windows`/`window_from_anchor` pick frames the
same way `postprocess_session.py` does — nearest frame *in time*, not index
arithmetic against a mean fps (`decode.py`, `frame_times` argument). It
raised exact window agreement with the trainer's choice from 11% of moves to
33% (mean slots differing 1.78 → 1.38) but bought only +0.3 points, not
significant (McNemar p=0.40, end-to-end moved -0.4, p=0.31). Kept anyway
since it's the correct construction and a prerequisite for any sub-frame
fix, but it doesn't close the gap: the residual is the anchor itself
(frame- vs sub-frame-timestamp), not the frame-selection rule, and that's
irreducible from the inference side without a sub-frame onset regression
head.

So instead: stop asking inference to reproduce the exact training window
and train the classifier to tolerate the shift it will actually be handed.
`train_move_classifier.py --anchor-jitter` rebuilds each training sample's
window at a randomly sampled anchor offset (`ANCHOR_JITTER_PMF`, in frames)
instead of always the canonical one, sampled from the **live** anchor-offset
distribution (`metric_audit.py`, 577 matched live moves) rather than the
narrower recorded-session one — live is ~2x wider (52% exact vs 72%) and is
the regime that actually needs the tolerance. Retrained on the identical 39
sessions and 5 named holdout sessions as `all39_cropped` so augmentation is
the only variable: `move_classifier_all39_jitter.pt`, 94.7% on that holdout
(all39_cropped: 94.6% — no in-distribution cost).

Scored on the 10 saved live takes neither model trained on
(`metric_audit.py`, same detector, so both are scored against an identical
set of found moves):

| | `all39_cropped` | `all39_jitter` |
|---|---|---|
| classifier of found | 81.5% | **85.8%** |
| free solves only | 78.6% | **83.2%** |
| end-to-end | 69.5% | **73.2%** |

McNemar on the paired moves: 54 fixed, 29 broken, p=0.008 — the first
statistically significant classifier improvement measured on live takes in
this investigation (`all27`/`all39` and the crop fix were sized well past
significance already; this one is smaller and needed the test). It does
**not** help the two takes whose detector did worst
(`solve_20260724_134516_solve` at 69% recall, `solve_20260726_100142_solve`
with 14 phantoms) — a bad onset stream is a different failure mode and
jitter only touches the classifier. `move_classifier_all39_jitter.pt` is
now `CLASSIFIER_PATH`.

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

## The group-theoretic decode (`reconstruct.py`)

The downstream decode mentioned above now exists: given the start state,
the solved end state, and the noisy detected sequence, `reconstruct.py`
beam-searches over substitute/delete/insert/rotate edits (costed by the
classifier's softmax, deletion priced by onset strength) and verifies
candidate stories against the cube group. Measured on the 12
frame-bearing sessions against BLE ground truth (2026-07-22, deployed
detector + `move_classifier_all23.pt`):

- **8/12 sessions verified solved, 7 exactly matching the BLE move list**
  (up from 7 and 5 on the previous `move_classifier_all20.pt` — the
  retrain converted two near-misses into exact reconstructions).
- Per-move accuracy across all 12: raw 97.1% → **97.8%** after the
  decode; on the sessions that verify, 98.9% → **99.9%**, 7 of 8 exact.
  The decode is worth ~1 point where it closes and nothing where it does
  not, which is the honest shape of a constraint-based repair.
- The 4 unsolved sessions still carry double-digit classifier errors,
  many confidently wrong and non-inverse (`F'`→`B` at 0.91). That is
  squarely outside the measured envelope — see below — so they need a
  better classifier or mid-solve state anchors, not a wider beam.

The envelope is stated in cost, not error count, because that is what
actually binds — see the "measured limit" table under `verify_solve.py`.

Run it with `python reconstruct.py --session ../training_data/solve_*/`;
`--selftest` and `--synthetic` need no models or sessions.

## Verifying a solve end to end (`verify_solve.py`)

`reconstruct.py` answers "is there a move sequence consistent with this
video and these two cube states". `verify_solve.py` asks the question that
decides whether that answer is worth anything, and is the live test to run
when checking the whole pipeline in a new environment.

One sitting, two recordings:

1. **scramble — measurement.** You are shown a random quarter-turn scramble
   and perform it on a solved cube. Every move is known, so this gives
   honest detector-recall / classifier-accuracy / end-to-end numbers on
   *your* cube with no BLE hardware, plus the decode's reconstruction
   against a start and end state that are both known exactly.
2. **solve — verification.** You solve that cube. There is no per-move
   ground truth, which is the real use case; the endpoints carry the
   entire claim.

```bash
python verify_solve.py                       # 20-move scramble, then solve
python verify_solve.py --scramble 25 --seed 7
python verify_solve.py --ble --front blue --top yellow       # + ground truth
python verify_solve.py --session ../training_data/solve_*/   # rehearsal
```

### `--ble`: the smart cube as ground truth, not as input

Both phases rest on an assumption they cannot check alone — phase 1 that
you performed the printed scramble correctly, phase 2 that a verdict is
enough to debug from. Neither survives a 60-move solve nobody remembers
afterwards. `--ble` logs every turn the cube feels on the same wall clock
as the frames, which turns "something was wrong" into "move 12 read `D'`,
you turned `D`":

- the cube is checked to actually **be solved** before phase 1 starts;
- phase 1 prints performed-vs-prescribed and verifies against what you
  **actually did**, so a mis-performed scramble stops looking like a
  pipeline error (it was previously indistinguishable, and it silently
  invalidates phase 2's claimed start state);
- **phase 2 gets per-move truth for the free solve** — detector recall,
  classifier accuracy, end-to-end and decode gain. There is no other way
  to get those numbers on a real solve;
- the cube is asked whether it is really solved at the end, which the
  decode cannot establish on its own.

The stream is a **label, never an input** — the detector, classifier and
decode never see it, so the claim stays "decided from webcam frames alone".
Without `--ble` every number is still produced, just against the weaker
assumption. `--front`/`--top` name the orientation you will hold, so BLE
moves land in the same camera-relative frame the classifier uses.

**Caveat that matters: the truth cannot see whole-cube rotations.** The cube
does not report `x`/`y`/`z` as BLE events (`orientation_tracker.py` §"Whole-cube
rotation detection"), and `cube_ble.py` disables the IMU quaternion stream, so
nothing ever calls `notify_reoriented()`. Rotate mid-take and the label frame
silently stays where it was calibrated, scoring a correct classifier wrong from
that point on. It is detectable rather than invisible — a rotation corrupts
*every* subsequent label, so it appears as one long unbroken run of errors,
whereas classifier errors are isolated and separated by correct stretches.
Check which shape you have before believing a bad number (measured 2026-07-23:
a 100-move solve came back with 15 error runs of length ≤2 separated by up to
10 correct moves — not a rotation).

**Second caveat: `end_solved`/`end_facelets_wrong` in a saved take's
`ble_meta.json` cannot currently be trusted.** All five `--ble --save`
sittings on 2026-07-24 reported the free solve ending NOT SOLVED, at a
*constant* 36 facelets wrong regardless of solve length (70-134 moves) or
scramble. That constancy is the tell: an actually-unsolved cube would not
land on the same wrong-count five times running. Composing each session's
own recorded move word (`moves.jsonl`) from the scramble's end state with
`reconstruct.seq_to_state` confirms all five really did reach solved — the
bug is in `BleTruth.snapshot_state()` / `cube_ble.CubeState.to_kociemba()`
(the live `get_state()` query), not in the move-event stream that trains
the models. Likely cause: `_parse_state`'s `face_order` (`cube_ble.py`,
the byte-block-to-face-name mapping for the STATE message, separate from
the move-event parsing) doesn't match the firmware's actual layout — but
the saved `end_state` strings are not uniform per 9-byte block even though
the cube was genuinely solved, so it is more than a simple relabelling and
has not been root-caused from logs alone; it needs a live session with raw
payload bytes captured. Until fixed, cross-check a "NOT SOLVED" verdict
against the move log before trusting it, the way this paragraph just did.

### `--save`: keep the take

```bash
python verify_solve.py --ble --front blue --top yellow --save
```

Writes both takes to `../training_data/solve_<stamp>_{scramble,solve}/` in
exactly `record_training.py`'s layout — `frames/`, `frames.jsonl`,
`moves.jsonl`, `config.json` — so `postprocess_session.py`,
`prepare_data.py`, `train_move_classifier.py` and `--session` all consume
them with no special-casing. Original frames and timestamps are saved, never
the resampled ordering, and the write happens *before* analysis so a bug
downstream cannot cost you a take.

This is not optional bookkeeping. A live take is the only source of data for
the regime that actually breaks this pipeline — long solves, fast regrips,
whatever room you are standing in — and the failures are the ones worth
keeping. Without `--ble` there are no move labels, so the take is usable for
the detector but not for classifier training.

The `--session` form replays recorded sessions through the identical
verification path with the BLE move list as truth. Use it to check the
wiring before spending a live take, and to re-measure after retraining
either model — it needs no camera and no cube.

### The falsifiability sweep is the point

A verifier that has only ever seen genuine solves cannot be evaluated:
"VERIFIED" is not a result until you know what the same video says to a
claim that is false. Every run therefore re-decodes the *same* onsets
against deliberately wrong start states — 1, 2, 4 and 8 quarter turns off
the truth, plus a cube that was never scrambled and a cube scrambled by
something else entirely.

On `solve_20260721_103149` (deployed detector + `move_classifier_all20.pt`,
the classifier deployed at the time of this measurement)
the true claim verified at cost 6.17 and reconstructed the BLE move list
**exactly** — raw 95.6% per-move, 100% after the decode, at a cost
identical to the ground-truth path — and all six wrong claims were
rejected.

Read that second number with the fifth column of the table, which is why
it is printed. If the true claim decodes to word `W`, a claim `d` quarter
turns away is solved by `w⁻¹·W` — the same accepts plus `d` insertions — so
a valid story provably exists at `cost(W) + d·C_INS` whether or not the
beam finds it. The beam did not find them. The near decoys are rejected by
the **search**, not by the cost model, whose own separation is only 4.0 per
quarter turn of claim error. Operationally the system does reject them,
which is what a verifier has to do; it is not evidence that no cheap wrong
story exists, and reporting it as such would be the easiest way to
oversell this pipeline.

That was measured, not assumed: the 1-move-off decoy on that session is
rejected at beam 4000 and at 16000, and **accepted at beam 64000 at cost
10.17** — exactly `6.17 + C_INS`, the constructive bound the table printed.

### What this means for missed moves

The same measurement bounds the decode's headline job, inventing a move
the camera never saw. At the deployed beam of 4000:

| case | recovered? |
|---|---|
| one missed move, nothing else wrong, n = 20 / 40 / 60 / 90 (synthetic) | yes, exactly |
| that plus 1, 2 or 4 synthetic substitutions, n = 90 | yes, exactly |
| one onset deleted from the real `103149` replay (4 genuine classifier errors, true path 6.17 → 10.17) | **no** — needs beam 64000 |

Sequence length is not the limit and edit count is not the limit; the
true story's total **cost** is. A story costing ~6 survives 90 onsets at
beam 4000, one costing ~10 does not, and each insertion is a flat 4.0. So
there is roughly one insertion of headroom beyond whatever the
classifier's own mistakes already cost — which is another way of saying
the same thing the session results say: **the classifier is the binding
constraint, and buying it down buys insertion recovery for free.** Note
also that synthetic noise is optimistic here (its errors are cheap to
override); quote `--synthetic` envelopes as upper bounds.

## Files

| file | role |
|---|---|
| `prepare_data.py` | `frames/` → `detector_stream.npz` (crop, downscale, onset indices) |
| `dataset.py` | clip sampling, Gaussian targets, augmentation, session splitting |
| `model.py` | `FrameEncoder` + TCN, and `score_stream()` for whole-session inference |
| `decode.py` | peak-picking, onset F1, `align_sequences()`, and `onset_windows()` handoff to the classifier |
| `train.py` | training loop, session holdout, threshold tuning |
| `live_detect.py` | end-to-end detector + classifier, live or replayed against BLE truth |
| `reconstruct.py` | group-theoretic decode of the detected sequence against known start/end states — beam search over substitute/delete/insert/rotate edits, costed by the classifier softmax, verified with the vendored solver's cube model. `--selftest`, `--synthetic`, `--session` |
| `verify_solve.py` | the end-to-end claim: prescribed scramble (measurement) + free solve (verification), with a falsifiability sweep against wrong claims. `--ble` adds per-move ground truth from the smart cube; `--session` rehearses it offline on recorded sessions |
| `ble_truth.py` | smart cube as a live ground-truth **label** for `verify_solve.py --ble` — background BLE thread, wall-clock move log windowable to a take, orientation-free solved check. Never an input to the pipeline |
| `env_suite.py` | cross-environment scorecard — several prescribed-scramble takes probing speed/placement, scored per stage, saved to `scorecards/` and compared across rooms with `--compare` |

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
