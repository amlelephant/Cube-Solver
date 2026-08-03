# Ground-truth artifacts in the move pipeline

Investigation of 2026-07-30, prompted by three hypotheses about why the move
model underperforms: (1) occasional bad detector bounding boxes shifting the
crop and causing adjacent-face errors, (2) middle-slice moves arriving from
BLE as two near-simultaneous face events, (3) the same problem for two-layer
(wide) moves.

Verdicts up front:

| # | Hypothesis | Verdict |
|---|---|---|
| 2 | Slice moves decompose into two BLE events | **CONFIRMED, and larger than stated** — it is one instance of a general BLE-timestamp-collision problem that accounts for 6 of the 11 unverified sessions |
| 1 | Crop-box shift degrades the classifier | **Premise confirmed, mechanism refuted** — shift costs real accuracy, but it produces *direction* errors and detector phantoms, not adjacent-face errors |
| 3 | Wide moves break the labels | **NOT PRESENT in this corpus** (16/16 endpoint checks pass), but the orientation tracker's rotation handling is dead code, so it is a latent bug the moment anyone rotates the cube |

Everything below was measured on the 55 sessions in `ble/training_data/`,
against `move_joint_seed0` replays and `results/2026-07-28/verify_joint_seed0_full.json`.

---

## 1. The BLE clock is quantized to 30 ms

This is the root fact the rest of section 2 falls out of. Over all 4,586
inter-move gaps in the corpus:

- **98.3%** land within 3 ms of an exact multiple of 30 ms
  (uniform-random control: 20.5%)
- median residual off a 30 ms multiple: **0.22 ms**
- against the 33.3 ms camera frame period, only 11.9% align — so this is the
  cube's notification tick, not an artifact of the webcam

The gap histogram is a comb: dense bins at 0, 30, 60, 90, 120, 150, 180, 210,
240, 270, 300 ms with essentially zero mass between the teeth.

Two consequences:

1. **Every move timestamp carries up to ±15 ms (±0.45 frame) of quantization
   error** before any other noise. `dataset.SIGMA` is 1.0 *frame*, so the
   Gaussian onset target is placed with label noise approaching half its own
   width. (This is very likely why the anchor-jitter retrain helped — it was
   augmenting against label noise that is really in the data.)
2. **Moves in the same tick get dt = 0 exactly**, which is where section 2
   starts.

## 2. Timestamp collisions — the dominant measurable defect

### 2.1 Slice moves are real and behave exactly as predicted

An `M` turn rotates the core, so relative to the core the R and L layers turn
in the same spatial direction. The cube reports that as `R` + `L'` (or `R'` +
`L`) — one physical motion, two events.

Searching for same-axis opposite-face pairs within 200 ms:

- **46 of 46 are slice-patterned** (one primed, one not — the same spatial
  direction). **Zero** are the opposed `R`+`L` pattern that a coincidental
  fast pair would sometimes produce. At 46/46 this is not ambiguous.
- 31 of them are at **dt = 0.0 ms** — the same BLE packet.
- They concentrate in 5 sessions, at 12–18% of that session's moves:
  `20260724_095506_solve`, `20260724_103307_solve`, `20260724_134516_solve`,
  `20260725_134744_solve`, `20260729_221809_solve`.

The *labels are not wrong*: applying each session's ground-truth sequence to a
solved cube reproduces the solve (see §4), so `R`+`L'` is a state-correct
encoding of `M`. The damage is entirely temporal and visual.

### 2.2 Slices are 17% of a bigger problem

Any two ground-truth moves landing within 2 frames of each other collide.
Across the corpus there are **234 such pairs — 10.1% of all 4,641 moves**
are in one (§2.3 splits these into 91 unreportable and 143 merely hard):

| kind | count | median dt | same frame |
|---|---|---|---|
| slice pair (M/E/S decomposed) | 40 (17.1%) | 0.0 ms | 32 |
| genuinely distinct fast moves | 173 (73.9%) | 60.1 ms | 7 |
| same face (half-turn as 2 quarters) | 21 (9.0%) | 60.1 ms | 1 |

So slices are the sharpest case but not the bulk. The bulk is fast finger-trick
pairs two ticks apart.

### 2.3 Why a collision is a miss — and which ones are truly forced

The 234 pairs above split into two tiers, and the distinction is real.
`peak_pick` accepts a peak only if it is `>= MIN_SEP` frames from every peak
already accepted, so:

| gap | pairs | % of moves | status |
|---|---|---|---|
| 0 frames | 40 | 1.7% | **unresolvable** |
| 1 frame | 51 | 2.2% | **unresolvable** |
| 2 frames | 143 | 6.2% | crowded — allowed, but needs a dip between the peaks |

**91 pairs (3.9% of all moves) are genuinely unreportable.** Both ends of the
pipeline collapse them to one event:

- **Training**: `dataset._build_target` is `y[i] = max over onsets`. Two
  onsets on the same frame produce a target *bit-identical* to one onset. The
  supervision never asks for two peaks.
- **Inference**: `peak_pick`'s refractory rule accepts at most one of them.
  `decode.py` already documents that two onsets one frame apart "can never
  both be one" — it calls this the decoder's hard floor.

The 143 pairs at exactly 2 frames are *hard, not impossible* — the model has
to put a resolvable dip between two peaks 67ms apart. An earlier draft of
this document called all 234 structural; that overstated it. The empirical
findings below are unaffected, since they are measured from actual misses
rather than predicted from the gap.

`decode.onset_collisions()` returns the two tiers, and `metric_audit.py` /
`verify_joint.py` now report misses split by them.

`prepare_data.py` already prints a NOTE counting duplicate onsets and calls
them "two-handed simultaneous moves". For the 40 slice pairs that attribution
is wrong: they are one-handed single motions, and no amount of temporal
sharpening can resolve a motion that only happened once. `decode.py`'s
"14 of the 17 remaining sub-150 ms misses are on different faces, consistent
with genuinely overlapping two-handed turns" is reading the slice signature
and assigning it the wrong cause.

### 2.4 This accounts for 6 of the 11 unverified sessions

`gt_path_cost` decomposed by subtracting collision-forced misses at
`C_INS = 4.0`. `collMiss` = missed ground-truth moves that were in a collision
(i.e. structurally unmatchable); `residIU` = leftover cost in insertion units:

| session | gtcost | miss | sub | ph | collMiss | residual | residIU | |
|---|---|---|---|---|---|---|---|---|
| `20260721_102711` | 58.7 | 3 | 11 | 4 | 0 | 58.7 | 14.68 | model |
| `20260724_134516_solve` | 52.7 | 13 | 1 | 0 | 13 | **0.7** | **0.17** | collision |
| `20260725_134744_solve` | 48.0 | 12 | 0 | 0 | 12 | **0.0** | **0.00** | collision |
| `20260722_100959` | 46.6 | 7 | 0 | 5 | 7 | 18.6 | 4.65 | model |
| `20260722_101225` | 42.7 | 6 | 6 | 3 | 1 | 38.7 | 9.67 | model |
| `20260724_103307_solve` | 40.0 | 10 | 0 | 0 | 10 | **0.0** | **0.00** | collision |
| `20260724_100120_solve` | 36.6 | 2 | 1 | 6 | 2 | 28.6 | 7.16 | model |
| `20260724_095506_solve` | 35.9 | 8 | 0 | 1 | 8 | **3.9** | **0.97** | collision |
| `20260726_165044_solve` | 28.2 | 6 | 0 | 1 | 6 | **4.2** | **1.06** | collision |
| `20260723_105530_solve` | 20.3 | 2 | 1 | 3 | 1 | 16.3 | 4.09 | model |
| `20260725_180216_solve` | 20.0 | 5 | 0 | 0 | 5 | **0.0** | **0.00** | collision |

Six sessions land at **0.00–1.06 insertion units** of residual — inside the
~1-unit envelope that verifies in practice. Three of them are *exactly* zero:
their entire `gt_path_cost` is collision-forced insertions.

Supporting correlations, restricted to the 26 solve-phase sessions so length
is not doing the work:

- verified sessions: mean collision rate **3.23%**, mean miss rate 0.66%
- unverified sessions: mean collision rate **8.55%**, mean miss rate 5.28%
- `corr(collision rate, miss rate) = 0.829`
- `partial corr(collision rate, gt_path_cost | length) = 0.574`

**This prediction was measured, and it was too optimistic — see §6.** The
table above credits *every* collision-forced miss to a single mechanism, but
OP_SLICE only addresses the 17% of collisions that are actual slices (§2.2).
The rest are fast distinct moves two ticks apart, which a slice transition
correctly does nothing for. The measured gain was +2 sessions, not the ~+6
this arithmetic implies. The decomposition is still the right way to see
*where* the cost lives; it is not a forecast of what any one fix recovers.

### 2.5 What it does *not* overturn

The four sessions in the honest subset (`102711`, `101225`, `105530_solve`,
`100120_solve`) are all **model**-driven, with 4.1–14.7 IU of residual after
collisions are accounted for. The decoder-sprint conclusion — that the honest
subset's bottleneck is not reachable by decoder tuning — stands untouched.
What changes is the *full-sweep* number, which was being depressed by a
measurement artifact on sessions the honest subset never included.

### 2.6 Training-side damage

Slice pairs also corrupt classifier supervision, though less severely than
expected. Of the 17 slice pairs in sessions with `moves_labeled.jsonl`:
12 have partially overlapping windows, 5 disjoint, **0 fully identical** —
`postprocess_session.move_window`'s sandwiching pulls them apart. So the
classifier gets two heavily-overlapping views of one motion with two
contradictory labels, rather than the same image twice.

Corpus-wide, 75.8% of the 3,529 classifier windows are squeezed below the
nominal 400 ms, and 7.6% are under 150 ms.

This leakage is visible in the confusion matrix. Over 41 substitutions in the
seed0 replays, the single most common specific confusion is **`L` → `R'`, 5
occurrences** — which is precisely "the model reported the other half of the
slice."

## 3. Crop-box shift (H1): premise yes, mechanism no

### 3.1 Correlational test — underpowered, not significant

Isolating *impulsive* box error (deviation of the per-move `crops.json` centre
from a 7-move median-filtered trajectory; real cube translation is smooth,
detector error is not) and splitting by outcome:

| outcome | n | median | mean | p90 |
|---|---|---|---|---|
| correct | 2626 | 0.0381 | 0.0507 | 0.107 |
| substitution | 23 | 0.0383 | 0.0761 | 0.212 |

Mann-Whitney p = 0.24. The medians are identical; only the tail differs
(top-decile residual has 2.26% substitution rate vs ~0.9% elsewhere, on 6
errors). **No usable evidence either way at n = 23.**

### 3.2 Causal test — shift the box deliberately

Ran the full detector + classifier over 3 sessions (392 ground-truth moves)
with the YOLO boxes displaced by a fixed fraction of the box side. Boxes are
computed once and reused, so the only variable is placement:

| shift (dx, dy) | move acc | end-to-end | miss | phantom | same-face err | adjacent err |
|---|---|---|---|---|---|---|
| (0.00, 0.00) | **94.1%** | 89.8% | 18 | 18 | 12 | 10 |
| (+0.05, 0.00) | 92.8% | 88.8% | 17 | 26 | 17 | 10 |
| (+0.10, 0.00) | 92.5% | 88.3% | 18 | 31 | 17 | 10 |
| (−0.10, 0.00) | 91.4% | 86.2% | 22 | 25 | 19 | 13 |
| (0.00, +0.10) | **95.9%** | 90.3% | 23 | 24 | 9 | 6 |
| (+0.15, 0.00) | 92.3% | 85.2% | 30 | 31 | 20 | 7 |

Three findings:

1. **Horizontal placement matters, vertical does not.** ±0.10 horizontally
   costs 1.6–2.7 points of move accuracy; +0.10 vertically costs nothing
   (it measured 1.8 points *better*, i.e. noise). The pipeline is genuinely
   sensitive to where the cube sits horizontally in the crop.
2. **The detector degrades faster than the classifier.** Phantoms nearly
   double (18 → 31) and misses rise (18 → 30) well before move accuracy moves
   much. If crop placement is hurting, it is hurting onset detection first.
3. **The predicted mechanism does not appear.** Adjacent-face errors are flat
   across every shift (10, 10, 10, 13, 6, 7). What grows is *same-face,
   wrong-direction* errors (12 → 20). So a shifted box does not push the model
   to an adjacent face; it makes it lose the turn direction.

Natural jitter for comparison: median impulsive residual ≈ 0.038 of the crop
side, p90 ≈ 0.107, worst observed move-to-move jump 0.83. So real jitter does
reach the magnitudes tested here, but only in the tail.

**Net**: crop placement is a real lever, mostly on the *detector*, and worth
attacking via horizontal crop-jitter augmentation. It is not the explanation
for adjacent-face classification errors.

### 3.3 Where adjacent-face errors actually come from

Structure of all 41 substitutions in the seed0 replays:

| structure | count | share |
|---|---|---|
| same face, wrong direction | 12 | 29.3% |
| adjacent face (x→y axis) | 9 | 22.0% |
| adjacent face (z→y axis) | 8 | 19.5% |
| opposite face, same x-axis | 6 | 14.6% |
| adjacent face (y→x axis) | 3 | 7.3% |
| opposite face, same z-axis | 2 | 4.9% |
| adjacent face (z→x axis) | 1 | 2.4% |

Adjacent-face is 51% in total, and 41% of everything is an axis error *into
y* (i.e. predicted `U`/`D` when the truth was an x- or z-axis face). `U` is
over-predicted: `R→U'`, `R'→U`, `F'→U'`, `B→U`, `F→U`, `B'→U` together are 11
of 41. That is a class-prior/appearance issue, not a geometry-of-the-crop
issue.

## 4. Wide moves and rotations (H3): absent here, latent bug regardless

A wide turn is `Fw = z B'` — a whole-cube rotation plus a single *opposite*
face event, not two events. It would show up as an untracked reorientation.

**Direct test**: for each of the 16 paired scramble/solve sessions, apply the
scramble's ground-truth sequence to a solved `CubieCube`, then the solve's.
If any reorientation went untracked, the result cannot be solved.

**16 of 16 return to solved**, including all four slice-heavy sessions. So in
this corpus there are no wide moves and no cube rotations, and the slice
encoding is state-correct. (The remaining 23 sessions have no paired scramble
and cannot be checked this way.)

**But the machinery to handle it does not work.** `OrientationTracker` exposes
`set_orientation_from_imu`, `notify_reoriented`, `FaceMap.apply_whole_rotation`
and `FaceMap.apply_face_rotation` — and grep shows **none of them is called
anywhere in the repo**. The face map is frozen at `calibrate()` for the whole
session; `front_color`/`top_color` are constant in all 55 sessions, which is a
property of the code, not of the cubing. The moment a session contains a
rotation or a wide move, every subsequent label is silently wrong and the
endpoint check above is the only thing that would catch it.

Worth wiring up before any live/product capture, and worth running the §4
endpoint check as a preflight on every new session pair.

---

## 5. OP_SLICE — the decoder-side fix, and what it took

Implemented 2026-07-30 in `reconstruct.py`, gated behind `--slices` and
defaulting to exact prior behaviour (`run_selftest` asserts bit-identical
results with the flag off).

The transition itself is trivial: at each onset, additionally offer "class k
AND its slice partner", applying the composed `SLICE_VECS[k]` and emitting
both move names. Getting it to actually *work* took two further pieces, and
both are worth recording because neither was obvious.

### 5.1 The posterior already knows where the slices are

Offered at every onset, OP_SLICE is 12 extra near-free transitions per step.
The beam floods with equally cheap nonsense and loses the true story even
when that story costs almost nothing — measured on `103307_solve`, where the
intended path exists at cost 8.0 and the beam missed it at width 64000.

The signal that fixes it comes from the training-label defect in §2.6. A
slice's two halves were labelled on heavily overlapping windows, so the
classifier learned to *hedge* between them. Over 3,527 matched onsets:

| | partner-class probability |
|---|---|
| slice onsets (n=33) | median **0.359**, min > 0.1 |
| every other onset (n=3494) | median **0.00004** |

Four orders of magnitude, p = 2e-23. At a 0.1 gate: **33/33 slice onsets
caught, 12/3494 false positives**. On `103307_solve` the gate opens 9 of 88
onsets, 8 of which are real slices, with none missed.

The poison is the detector. Nothing else in the pipeline had to change to
find it.

### 5.2 The consistency ladder has to know slices exist

Even with a tight gate and a cheap path, the session still failed. Tracking
the true hypothesis through the beam showed it being evicted at onset 25 of
88 while doing a *conf-1.000 accept* — out-ranked, not out-costed.

The cause: `_Beam.suffix` builds the reference story by applying every
remaining detection as a single turn. A slice-using truth therefore never has
residual `SOLVED` and never lands in `rep1`, so it sits at `RESID_CAP` for the
whole decode while wrong-but-one-plain-edit-away stories score level 1 and
rank above it. Fixed by having the reference story read gated onsets as
slices (`_Beam._ref_move`) and by adding "toggle onset q between one turn and
a slice" to `rep1`'s one-edit repair set.

With both pieces, `103307_solve` verifies **exactly** at beam 4000 in 22.5s —
having previously failed at beam 16000 in 130s.

### 5.3 Two bounds in `_rank` are unsound with slices

`_rank`'s capacity bound assumes each remaining onset supplies one quarter
turn (budget doubles with slices) and its parity bound assumes the remaining
turn count has parity `remaining % 2` (a slice is two turns, so parity
becomes unconstrained). Both are adjusted under `slices=True`.

### 5.4 Falsifiability

`slices`, `c_slice` and `slice_rows` are threaded into the decoy path in
`verify_solve.decode_claim` and `verify_joint`'s `vs_args`. A cost model more
generous to the true claim than to its decoys would inflate every verdict
without making any of them more true. **The full sweep re-measures this; do
not quote a verified count from a run whose decoys were priced differently.**

## 6. Measured result

Full sweep, 42 colour-prepped sessions, both checkpoints, baseline and
`--slices` run back to back in the same sitting (baselines were re-measured,
not recalled), at the corrected `SLICE_GATE = 0.10`:

| | seed 0 baseline | seed 0 `--slices` | seed 1 baseline | seed 1 `--slices` |
|---|---|---|---|---|
| VERIFIED | 30/42 | **32/42** | 30/42 | **32/42** |
| EXACT | 26/42 | 27/42 | 28/42 | 28/42 |
| regressions | — | **none** | — | **none** |

**Both seeds flip the same two sessions**, which is the result replicating
rather than two independent coin flips.

Only five sessions changed `gt_path_cost` at all, and every one of them
contains slices — the gate is inert everywhere else, which is exactly the
intended behaviour:

| session | gt_path_cost | outcome |
|---|---|---|
| `20260724_103307_solve` | 40.0 → 20.0 | **newly VERIFIED** (exact) |
| `20260724_095506_solve` | 35.9 → 15.9 | **newly VERIFIED** |
| `20260724_134516_solve` | 52.7 → 35.2 | still short |
| `20260725_134744_solve` | 48.0 → 28.0 | still short |
| `20260722_100644` | 14.7 → 12.2 | already verified |

`20260729_221809_solve` (10 slice pairs) is not in the sweep — it has no
colour stream yet. Running `prepare_data.py --color` on it would add a third
genuinely slice-heavy session to the measurement.

### Falsifiability is not weakened

The check that decides whether these verdicts mean anything. Decoys are
priced under the identical cost model:

| decoy | baseline | `--slices` |
|---|---|---|
| cube was never scrambled | 3/30 | 3/32 |
| start state 1 move off | 29/30 | 28/32 |
| start state 2 moves off | 25/30 | 24/32 |
| start state 4 moves off | 20/30 | 19/32 |
| true claim strictly cheapest | 29/30 | **31/32** |

The fraud that actually matters — a cube that was never scrambled — is
accepted on the same 3 sessions either way. Near-miss decoys are accepted at
slightly *lower* rates. So OP_SLICE buys verifications without buying
credulity, which is the only way the +2 counts.

### Seed-1 replication caught a regression the seed-0 run did not

Replicating on `checkpoints/move_joint_seed1.pt` gave 30/42 → 31/42: the **same two
sessions flipped**, but `solve_20260728_233139_scramble` — a 20-move
scramble that verified in baseline — stopped verifying.

Cause: `SLICE_GATE` had been set to 0.05, looser than the 0.10 the separation
was actually measured at. On that session the gate opened at a single
*uncertain* onset (argmax `F` 0.539, partner `B'` **0.064**) — not a slice,
just a low-confidence call. The extra transition plus the slice-aware ranking
changes were enough to lose an otherwise easy decode.

Corrected to `SLICE_GATE = 0.10`. There is no recall margin to buy by going
lower: every observed slice onset sits at ≥ 0.1 with median 0.36, so
loosening only admits false positives.

Re-measured at 0.10, seed 1 goes 30/42 → **32/42 with no regressions** — the
scramble verifies again and both real flips are retained. That is the table
above.

The lesson generalises past this flag: a single-seed sweep did not surface
this, and it was a *regression on the easy case*, which is exactly the class
of damage a headline "+2 verified" hides.

### Honest reading

+2 of 42 is a small, precisely-targeted gain, and it is worth exactly what it
looks like. What it does establish:

* the mechanism is confirmed end-to-end, not just in the cost arithmetic —
  the two flips are the two sessions whose cost the mechanism actually
  changed, and both decode to the true sequence;
* nothing else moved, which is the signature of a fix that is not just
  loosening the decoder;
* the honest-subset sessions (`102711`, `100959`, `101225`, `105530_solve`,
  `100120_solve`) are untouched, with `gt_path_cost` identical to the
  decimal. Their failures are substitution-driven and this changes nothing
  about them. The decoder-sprint conclusion stands.

The two slice sessions that did not flip both improved by 7-8 slices' worth
of cost and are still 28-35 over. They carry additional non-slice error mass;
a lower `--c-slice` would close some of the remainder, but that trade must be
re-measured against the falsifiability table above rather than assumed.

## Suggested order of work

1. **Add the §4 endpoint check as a postprocess gate.** Cheapest thing here,
   catches the entire H3 class, and would have caught it retroactively.
2. ~~**Recompose slice pairs.**~~ DONE, decoder-side — §5, §6. 30/42 → 32/42
   with no regressions and no falsifiability cost.
3. ~~**Stop charging the model for collisions in the metrics.**~~ DONE —
   `decode.onset_collisions`, reported by `metric_audit.py` and
   `verify_joint.py`.
4. **Horizontal crop-jitter augmentation** for the onset detector (not the
   classifier — the detector is what degrades first).
5. **Quantization-aware onset targets.** The label is a 30 ms box, not a point;
   a target reflecting that is better matched to the data than a σ=1-frame
   Gaussian on a point estimate. This is the one that addresses the *other*
   83% of collisions, which OP_SLICE deliberately does not touch.
6. **The model-side M/E/S vocabulary** (classes 12 → 18, per-axis move
   algebra, GT recomposition, retrain). Still open. It would fix the training
   poison in §2.6 rather than routing around it at decode time — though note
   that poison is currently what makes the slice gate work (§5.1), so this
   change would need SLICE_GATE replaced by an explicit class, not merely
   retuned.
