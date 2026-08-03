# Label frame drift, and the end state the cube lies about

Two ground-truth defects found 2026-08-03, one of which silently corrupts
training labels. This is the plan and the evidence behind it.

---

## Issue A — the recorded end state is the cube's claim, not a fact

### What is actually wrong

`record_training.py` and `verify_solve.py` both ask the cube for its full
54-sticker state at the end of a session and write it to `ble_meta.json`
as `end_state`. The cube's own state tracking drifts (a flat battery does
it; see `ble_truth.solved_report`'s docstring and the 2026-07-23 incident),
so that field is frequently wrong — and it is wrong in the direction that
matters, reporting NOT SOLVED at the end of a solve that plainly ended
solved.

### What is NOT wrong, checked before assuming

Nothing in the training or verification math consumes it. The backtrack
targets the solved state, everywhere:

| consumer | target | source of the start state |
|---|---|---|
| `session_check.check_pair` | `CubieCube()`, i.e. solved | a fresh solved cube |
| `reconstruct.start_from_gt` | `SOLVED` | `inverse(product(gt))` |
| `verify_joint.decode_moves` | `RC.SOLVED.copy()` | `start_from_gt(gt)` |
| `verify_solve.verify_claim` | passed in; `SOLVED` for a solve | scanned / derived |

So the endpoint gate and every decode are immune. This is a provenance and
workflow defect, not a correctness one, and the fix should not pretend
otherwise.

### Where it does bite

1. **It looks authoritative.** A field called `end_state` in a file called
   `ble_meta.json` reads as ground truth. Anyone building on this corpus
   later will believe it.
2. **It can block a recording.** `verify_solve.py`'s phase-1 gate calls
   `truth.solved_report()` before the scramble and, if the reviewer does
   not type `go`, `sys.exit("Still not solved. Aborted.")`. On a lying cube
   that aborts a perfectly good take. The code already knows this is
   possible — it prints "if the cube in your hand LOOKS solved, believe
   your eyes" when more than 24 facelets are off — and then blocks anyway.
3. **The end-of-phase-2 advisory** prints "a VERIFIED verdict below would
   be wrong on the facts" off the same unreliable reading.

### Fix

- Rename the recorded field to `cube_reported_end_state` and add
  `end_state_trusted: false`. Never call the cube's claim `end_state`.
- Add `derived_end_state`: the recorded start state with the recorded move
  word applied. That is the honest end state, and it is exactly what every
  consumer above already computes for itself.
- Add `derived_end_solved` alongside, so the one question that matters is
  answerable from the file without recomputing the group product.
- Make the phase-1 gate advisory when the facelet count says the cube is
  lying (>24 wrong is not a partly-finished solve — one quarter turn moves
  6 facelets). Warn, do not abort.

---

## Issue B — the label frame drifts after a middle-slice turn

This is the one that corrupts training data.

### Mechanism

The smart cube reports face turns **relative to its core**. Turning the M
slice rotates the core, so the R and L outer layers — which did not move in
space — appear to the cube to have turned backwards, and it emits the pair
`R` + `L'`. `GROUND_TRUTH_ARTIFACTS.md` measured this: 46 of 46 same-axis
opposite-face pairs within 200ms carry exactly that signature, 31 sharing a
BLE timestamp exactly.

That pair is a correct description of the state change **in the cube's own
frame**. What it does not say is that the four centres on that axis have
moved. The cube's coordinate system is defined by centre colour, so from
that moment the colour→camera-position map is stale, and every subsequent
move is reported in a frame that has rotated relative to the camera.

`OrientationTracker` freezes its face map at `calibrate()`.
`FaceMap.apply_whole_rotation` exists and is never called (see
`ble-orientation-tracker-dead-code`). So nothing corrects for it.

Do a slice twice and the centres make a half turn: the face physically on
top now has the old bottom centre's colour, the cube calls a turn of it
`D`, and the camera plainly sees `U`. That is the failure as observed at
the cube, and the vision model is the thing that is right.

### Why `session_check.py` passing 16/16 is not evidence against this

`check_pair` applies the labels to a `CubieCube`, whose model has **no
centres** — it is center-relative by construction. Today's labels are
cube-frame, so the product is correct and the check passes. It is blind to
camera-frame drift by construction, and cannot be otherwise.

`session_check.py`'s docstring currently claims these passes "incidentally
validate the slice-move encoding". They validate the *state* half of that
claim and are silent on the *camera-frame label* half. That sentence needs
correcting; it is the reason this went unnoticed.

### The correction, derived

Doing M (the slice between L and R, turning in the L direction) rotates the
core in the L direction, i.e. by `x'`, and the cube emits `(R, L')`. So the
core rotation is the **inverse of the whole-cube rotation in the unprimed
reported face's own direction**:

| reported pair | physical | core rotation |
|---|---|---|
| `R`, `L'` | M | `x'` |
| `L`, `R'` | M' | `x` |
| `U`, `D'` | E' | `y'` |
| `D`, `U'` | E | `y` |
| `F`, `B'` | S | `z'` |
| `B`, `F'` | S' | `z` |

with the face→rotation map `U→y, D→y', R→x, L→x', F→z, B→z'`.

### Evidence

The vision model only ever sees the camera, so whichever labelling it
agrees with is the camera-relative truth. Scored with the SHIPPED
implementation (`slice_frame.annotate`), on the span of moves the
correction actually changes, held-out sessions only
(`solve_20260729_221809_solve`, `solve_20260730_113054_solve`, 24 moves
pooled):

| | seed 0 | seed 1 |
|---|---|---|
| cube-frame (today) | 62.5% | 45.8% |
| camera-frame (fixed) | **79.2%** | **62.5%** |

Replicated on both seeds, +16.7 points on each.

On the four sessions the checkpoints **trained on**, the correction makes
agreement *worse* in all six seed/session cells (86.7→53.3, 100.0→66.7,
91.7→83.3, 100.0→66.7 on seed 0; same direction on seed 1). That is the
expected signature rather than a contradiction: those models were fitted
to the corrupted labels and reproduce the corruption.

### Honest limits of that evidence

- **38 post-slice held-out moves.** Small. The direction replicates across
  two independent seeds, which is what makes it actionable, but the point
  estimate is not precise.
- **The SIGN of the rotation is barely tested.** Every accumulated
  orientation in this corpus lies in a 4-element subgroup that keeps
  returning to the identity, and the sequences are dominated by half turns,
  which are self-inverse. The derived rule and its inverse therefore score
  identically on 11 of 12 session/seed cells. The one cell that separates
  them (`solve_20260725_134744_solve`, seed 1) favours the derived sign,
  65.0% vs 60.0%. New footage containing an odd number of same-axis slices
  would settle it properly.
- **Slices cluster in the last layer** (first slice at move 100-132 of
  120-152), so "after the first slice" is also "the fast, crowded tail".
  That confounds the *before vs after* drop, but not the *today vs
  corrected* comparison, which scores the same moves against the same
  predictions.

### Scope — smaller than it first looks, and worth stating precisely

7 of 76 sessions contain slices: 56 non-overlapping slice pairs over 5,968
moves. But the mislabelled span is NOT everything after the first slice.
The drift is a rotation that composes, and these algorithms do slices in
pairs (`M2`), so the accumulated orientation keeps returning to the
identity — every one of the 7 sessions ends back at the identity frame.
Only the moves that fall inside a non-identity span are wrong:

    26 moves corpus-wide, 2-5 per affected session — 0.44% of all moves.

An earlier estimate in this document said ~140 moves (2.3%) by assuming the
drift persists once it starts. It does not, and the corrected figure is
5x smaller. Those 26 moves are 100% wrong today, they sit in the last-layer
phase the algorithm prior operates on, and fixing them is free — but this
is a correctness fix, not a headline accuracy lever, and quoting it as one
would be wrong.

### What a slice's OWN two labels should be is a separate problem

The camera sees ONE motion; the 12-class vocabulary has no symbol for it,
and the two labels emitted for the pair name two layers that did not move
in space. That is the known label poison whose fingerprint is the slice
gate (`slice-gate-from-posterior`, p=2e-23). The real fix is a vocabulary
that can say `M`/`E`/`S`, which is a model change. **Out of scope here.**
This plan fixes only the drift AFTER the pair, and leaves the pair's own
two labels exactly as they are.

---

## Plan

Ordered so that each step is verifiable before the next depends on it.

1. **`ble/slice_frame.py`** — pure logic, no BLE, no torch: slice-pair
   detection, the rotation table above, and a `CameraFrame` that maps a
   cube-frame move name to its camera-frame name under the running
   orientation. Everything else calls this, so the live path and the
   offline backfill cannot drift apart. Unit-testable without hardware,
   which matters because the BLE paths are not testable here at all.

2. **`ble/backfill_camera_frame.py`** — add `camera_notation` and
   `orientation` to every existing session's `moves.jsonl`, in place,
   idempotent. `wca_notation` is left ALONE: it is cube-frame, several
   consumers depend on that, and silently changing its meaning is how this
   class of bug happens in the first place.

3. **`prepare_data.py`** — build `onset_class` from `camera_notation` when
   present, and record `label_frame` in the stream so a re-prepared corpus
   is distinguishable from an old one. This is the step that actually fixes
   training.

4. **`session_check.py`** — correct the over-claiming docstring, and report
   per-session slice counts and net drift so the gate stops being silent
   about the thing it cannot see.

5. **`orientation_tracker.py`** — wire slice detection into the live path
   so newly recorded sessions are correct at the source. Cannot be tested
   without a cube; keeping the logic in `slice_frame.py` is what makes that
   acceptable.

6. **Issue A** — end-state provenance in `record_training.py` and
   `verify_solve.py`, and unblock the phase-1 gate.

### Deliberately not done

- **Re-training.** The corpus changes under step 3, and what that is worth
  is a measurement, not an assumption. Prepare first, measure, then decide.
- **The decoder's frame.** `reconstruct.start_from_gt` builds the start
  state from the cube-frame word, while the model emits camera-frame
  predictions; on a slice session those disagree after the first slice.
  `reconstruct.py` already carries the machinery for this (`--rotations`,
  persistent orientation, `rotate_sigma`, `camera_class_for`) and disables
  it because "the recorded sessions hold one orientation throughout" — a
  premise now known to be false for these 7 sessions. Enabling and
  measuring it is a follow-up, not part of this fix.
- **`M`/`E`/`S` as first-class classes.** The real fix for slice labelling,
  and a model change.
