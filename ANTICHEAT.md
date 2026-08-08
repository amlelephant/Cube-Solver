# Anticheat

**Status:** count gate built and measured 2026-08-05 (`ble/move_detector/anticheat_gate.py`).
Owns every `VERIFIED` verdict (LAUNCH_ROADMAP.md Track C).

## Threat model

Everything physical reduces to three attacks. Naming them this way matters,
because two of them are not swaps and swap detection therefore cannot see
them:

| # | Attack | Caught by |
|---|---|---|
| A | **Substitution** — swap in a solved cube | count floor + continuity guard + appearance jump |
| B | **Solving outside the timed window** — either follow a solver's solution, or stop the timer unsolved and solve before scanning | count floor (below) and the post-stop test |
| C | **Video injection / virtual camera** — no cube analysis catches this | challenge-response only (§4) |

## 1. The move-count gate — the core of A and B

`ble/move_detector/anticheat_gate.py`. The idea (this file's earlier "new
option") was to reuse the move model the Coach tier needs anyway, and ask it
only the question it is good at.

**Why counting and not identification.** Move-by-move verification is dead —
a cliff at ~98–99% per-move accuracy that the decoder ceiling sits below
(`ACCURACY_TARGET.md`). But that cliff is about *which* move. Counting
discards substitutions entirely and is sensitive only to insertions and
deletions, so it runs on the model's strong axis. MER is an over-estimate of
counting error.

### 1.1 The floor is above God's number, and the metric is the trap

A legit solve is tens of moves; a substituted cube needs none. That alone
defeats A. It does not defeat B's first form: run the scramble through a
solver and execute the answer. Every move genuinely happens and one cube is
on camera throughout.

But a solver's output is **bounded** and a human method is not. So the floor
sits above the longest solution any solver would produce.

> **God's number is 20 in the HALF-turn metric. This model's alphabet is 12
> classes — 6 faces × 2 directions — so R2 decodes as R,R and everything
> here is QUARTER-turn metric, where God's number is 26.** A floor set at
> "20-something" would sit *below* the cheat ceiling and catch nothing. If
> R2-as-its-own-class ever lands, this constant moves back to 20 and the
> floor must be recalibrated with it.

The operative ceiling is not 26 but **30**: a cheat runs two-phase
(Kociemba), not an optimal solver, and two-phase returns ~19–22 HTM ≈ 25–30
QTM. Corroborated by the recorded scramble sessions, which are the same
object and measure 20–28 QTM true.

Constants: `SOLVER_CEILING_QTM = 30`, `MIN_OBSERVED_MOVES = 32`,
`LEGIT_FLOOR_QTM = 45`.

### 1.2 Calibrating against what we actually observe

The gate sees *predicted* moves and the model under-counts, more so the
faster the solve. Under-counting pushes a cheat further below the floor
(harmless) and a legit solve toward it (the false-DQ risk). Worst-case
retention from `speed_sim_*_blur.json`:

| TPS | 2 | 3 | 4 | 5 | 6 | 8 | 10 |
|---|---|---|---|---|---|---|---|
| retention (worst session) | 0.98 | 0.91 | 0.90 | 0.84 | 0.81 | 0.63 | 0.50 |
| a 45-QTM solve reads as | 44 | 41 | 40 | 38 | 36 | 28 | 22 |

Above the separation limit the bands genuinely overlap and **no threshold
separates them**, so the gate abstains there rather than guessing. That limit
is *derived* from the floor (`separation_tps_limit()`), not hardcoded, so
changing the floor moves it automatically.

**Speed augmentation moved that limit from 7.11 to 9.62 TPS** (2026-08-05).
Training on time-warped clips (`--speed-aug`, `dataset.speed_warp_block`)
raised worst-case retention at every level on both seeds:

| TPS | seed 0 | seed 1 |
|---|---|---|
| 5 | 0.836 → 0.967 | 0.757 → 0.875 |
| 6 | 0.809 → 0.954 | 0.763 → 0.842 |
| 8 | 0.632 → 0.882 | 0.612 → 0.796 |
| 10 | 0.500 → 0.770 | 0.428 → 0.691 |

`RETENTION_FLOOR` takes the **worse of the two seeds** at each level, because
a gate that must never false-DQ cannot be calibrated on the luckier one.
9.62 TPS is a ~4.7s 45-move solve against a ~3.1s world record, so the "too
fast to read" abstain case now covers only the very fastest humans alive
rather than everyone under 6.3 seconds.

Those constants describe `move_ctc_spd_s0/s1`. **If verification ever runs a
different checkpoint, re-measure with `speed_sim.py --blur` and replace
them** — a gate calibrated on a model that is not the one running is worse
than no calibration, because it looks calibrated.

### 1.3 Measured, both seeds, 2026-08-06

`python anticheat_gate.py score --ctc checkpoints/move_ctc_spd_s{0,1}.pt`

**Re-run 2026-08-06 on the SPEED-AUGMENTED checkpoints.** The previous
numbers here were measured on `move_ctc_aug44_s{0,1}`, which is not what
`RETENTION_FLOOR` describes — so the gate had never actually been scored on
the model its own constants are calibrated for. That is precisely the
checkpoint-coupling trap §1.2 warns about, and it survived in this file for a
day. The corrected figures are below; the old ones are kept in the last
column because a re-measurement with nothing to compare against is half a
result.

The corpus splits into two halves that measure the two opposite errors, and
the second half is free:

| | seed 0 | seed 1 | was (aug44) |
|---|---|---|---|
| `*_solve` verified | **36/36** | **36/36** | 36/36 |
| false DQ | **0** | **0** | 0 |
| min headroom above the floor | +39 | +37 | +34 / +38 |
| `*_scramble` caught | **21/21** | **21/21** | 21/21 |
| lowest legit reads | 71 | 69 | 66 |
| highest proxy reads | 28 | 28 | 28 |
| separation gap | **43** | **41** | 38 |

- **`*_solve`** — legit solves. Held out from training: 10/10 verified on
  both seeds, minimum headroom +39 / +46.
- **`*_scramble`** — a machine-generated ~20-HTM sequence executed by hand on
  camera. That is *structurally identical* to solver-following. They were
  recorded as scrambles and never designed for this, which makes them an
  uncontaminated proxy. Held out: 3/3 caught on both seeds.

So speed augmentation did not cost the gate anything and widened the
separation gap by 3–5 moves. The headline outcome — 36/36 verified, 21/21
caught — is identical on both seeds and on both checkpoint families, which is
the strongest statement available from this corpus: the result is not seed
luck and not an artifact of one training recipe.

**What this does not establish.** The catch side is a proxy, not a recorded
attack — it shows the count separates a short machine sequence from a human
solve, not that someone deliberately trying to beat this gate fails. The
proxy is sound for *counting* specifically (a scramble and its inverse have
the same move count) and would not be for an appearance statistic.

### 1.4 The post-timer window (B's second form)

Stop the timer on an unsolved cube, solve it, then scan. No substitution, so
nothing in §2 sees it.

The obvious spec — "require zero moves between timer-stop and scan-complete"
— does not survive measurement. **Phantom onsets accumulate with window
length**, so a fixed constant is a test whose false-positive rate depends on
how long the user takes to present their cube. Measured on move-free session
tails (held-out, seed 0): mean 0.45 phantoms/10s, worst 3.62/10s. Over a 40s
scan that worst rate is ~14 phantoms; a constant limit of 2 would reject
almost everyone.

So the test is rate-based (`POST_STOP_PHANTOM_RATE = 4.0`/10s), and it is
really the same discriminant pointed the other way: a hidden solve costs ≥36
observed moves, an honest user accrues only phantoms.

**A product constraint falls out of this.** The allowance grows at 0.4
moves/s while a real solve reads ≥36, so beyond ~90s the allowance swallows
an entire hidden solve and the test has no power left.
`POST_STOP_MAX_WINDOW_S = 60` keeps a 12-move margin. **If the UI ever lets
the post-timer scan run longer than 60s, this test silently stops working**
— the gate abstains past that rather than passing.

Still uncalibrated: the available tails are 1–3s and the real window is tens
of seconds, so the rate is an extrapolation. Needs real
timer-stop-to-scan-complete footage.

### 1.5 Bad light must abstain, never reject

Measured 2026-08-05: in low light the **onset detector** loses 40+ points
while the classifier loses only ~5. The count *is* the detector's output, so
bad light makes the count read low — which would fabricate a `too_few_moves`
DQ out of a model failure.

The gate is therefore three-tier (`VERIFIED` / `REJECTED` / `REVIEW`), unlike
`continuity_guard.py`, which is deliberately two-tier because every signal it
has is swap-indicative. Two conditions here make evidence *unreadable* rather
than incriminating — bad light, and too-fast-to-separate — and abstain
conditions are checked **before** reject conditions so an unreadable solve is
never rejected on evidence just declared unreadable. Continuity/appearance
evidence is the exception: it is independent of the move model and still
legible when the move model is not.

## 2. Substitution — the appearance and continuity arms

- `cv/detection/continuity_guard.py` — uniqueness, presence, trajectory
  continuity. Two-tier by design.
- `cv/detection/swap_check.py` — persistent (not adjacent-frame) appearance
  change. Adjacent-frame distance is the wrong statistic: every turn
  recolours a face and blur spikes it, but those changes are transient. A
  substitution is large *and* persistent.

Known open hole: **the table-edge swap** (cube leaves frame under the table
edge mid-frame). `cv/labeling/record_attack.py` records these; one session
exists (`table_edge_20260804_175814`), which is not enough to tune against.

Known dead end: appearance-based swap detection alone — a legit solve
*completing* is the biggest appearance jump in a session.

## 3. What each arm actually owns

The three arms are independent, which is the point: a cheat has to beat all
of them, and they fail in different conditions.

| Arm | Sees | Blind to |
|---|---|---|
| Count gate | too few moves, moves after the timer | **a swap that arrives with enough moves** (below), anything at ≥7 TPS, anything in bad light |
| Continuity guard | cube leaving/duplicating/teleporting | a swap fully inside the frame |
| Appearance | a different cube before vs after | a swap to a visually identical cube |

**The one physical attack that survives the count gate**, stated plainly
because it is easy to believe the gate closed more than it did: make a full
solve's worth of plausible moves on camera *without solving*, then swap in a
solved cube. That reads as ~50 moves, clears the floor of 32, and the count
gate passes it — fifty seconds of flailing is not required, eight will do,
which makes it a world-class time. Nothing about the move count distinguishes
it from a real solve, because the move count is genuinely real; only the
*outcome* is faked.

Continuity and appearance own this case entirely, which is why the recorded
attack sessions still matter and why they should be **this shape
specifically** rather than generic swap variety. It is also the reason the
count gate can never be the only arm.

The paid tier's move stream is **corroborating evidence** only — it never
issues a verdict (LAUNCH_ROADMAP C4).

### 3.1 Trying it: `ble/move_detector/live_anticheat.py`

All three arms live, in one window, ending in a real `adjudicate()` verdict:

```bash
cd ble/move_detector
python live_anticheat.py                                   # defaults to move_ctc_spd_s0
python live_anticheat.py --ctc checkpoints/move_ctc_spd_s1.pt --camera 1
```

SPACE walks the state machine — start the solve, stop the timer, finish the
scan — then it decodes and adjudicates. R for another attempt, S to save the
evidence bundle, Q to quit.

**The count is deferred, not per-frame, and that is not a shortcut.**
`per_frame_boxes` median-smooths the crop detections and interpolates across
the window, so the box for frame 40 depends on detections at frame 50.
Cropping each frame with only its own box would inject the detector's
frame-to-frame jitter into the temporal diffs — the fake global motion the
model is trying to read. So frames are buffered and counted once, when the
window closes. It also happens to be how the product behaves.

Verified against the offline path (`anticheat_gate.py score`) on three
sessions, exact agreement on all three — which is what establishes that the
live crop/encode path reproduces training rather than merely resembling it:

| Session | Offline | Live | Verdict |
|---|---|---|---|
| `solve_20260721_101219` | 83 | 83 | — |
| `solve_20260721_102432` | 137 | 137 | verified |
| `solve_20260723_104750_scramble` | 28 | 28 | rejected, `solver_following` |
| `solve_20260724_094947_scramble` | 26 | 26 | rejected, `solver_following` |

Two things it refuses to guess at, both of which would otherwise produce a
confident wrong number:

* **Capture rate.** The model's receptive field is measured in FRAMES, so
  fps is a temporal scale factor: at 15fps a move spans half the frames
  anything in training did, and the solve reads as far faster than the clock
  says. Outside ±25% of 30fps the count is suppressed and the attempt
  abstains to `review` — the raw count is still recorded in the bundle so
  you can see what it would have said.
* **Checkpoint coupling.** `RETENTION_FLOOR`, the abstain band and every
  headroom number were measured on the speed-augmented checkpoints. Loading
  one trained without speed augmentation prints a banner, because a gate
  calibrated on a model that is not the one running is worse than no
  calibration — it looks calibrated.

The substitution meter stays display-only here, as it is in
`live_guard_test.py`: its threshold is uncalibrated on the attack side, and
wiring an uncalibrated bar into a verdict is how a false DQ ships.

**Frame rate.** The buffer is JPEG-encoded, and that is a frame-rate fix
rather than a disk one. Raw, a 640×360 frame is 675 KB — 21 MB/s, and 1.24 GB
of live Python objects across a 60-second solve. The allocation churn is what
made capture *degrade over an attempt*: fine at the start, falling apart by
the end, which is the signature of the buffer and not of any per-frame cost.
Encoded it is 41 KB (16.6× smaller, 75 MB for the same minute) for 1.18 ms of
encode. Decode happens post-hoc where there is no clock. Counts are unchanged
through the encode — all four sessions above still match exactly.

Measured after: **34.7 fps sustained** against `detect_cubes`'s own 37 fps
ceiling, so the whole harness — guard, swap meter, encode, HUD — costs 1.8 ms
a frame on top of the detector that both tools were always paying for.

## 4. Camera injection

Not addressed by any of the above and no cube analysis can address it. Every
scramble is server-issued and unique, so a pre-recorded video cannot match a
fresh scramble. The residual attack is switching sources mid-solve to conceal
a swap. Defence is **challenge-response**: the server demands a specific face
at a random moment, and the verification face order is randomised per solve
so no recording can satisfy it. Not built.

## 5. Next

1. Record the attack that survives the count gate — **enough moves, then
   swap**, including the table-edge variant — and tune continuity and
   appearance against it. Solver-following recordings are *not* a priority:
   the scramble sessions are a structural proxy (21/21 caught) and the bound
   on a solver's output is mathematical rather than empirical.
2. Record real timer-stop-to-scan footage and calibrate the phantom rate.
3. Wire `adjudicate()` into `verify_solve.py`'s live path and into the
   server-side re-verification worker (A3). The verdict function is pure and
   takes plain data specifically so both can run it and agree.
4. Enforce `POST_STOP_MAX_WINDOW_S` in the capture UI.
