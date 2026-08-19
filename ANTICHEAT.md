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
  substitution is large *and* persistent. **Measured dead** — see §2.2.
- `cv/detection/solved_check.py` — **was the cube solved at the timer stop**
  (2026-08-10). The arm that closes §3's surviving attack; see §2.1.

### 2.1 Solved-at-stop: asking a question that has an answer

Appearance-based substitution detection asks "is this the same cube", which
is unanswerable when the attacker's second cube is the same brand under the
same light — and that is the cube they will use. `solved_check.py` asks
instead:

> at the moment the timer stopped, was the cube on camera solved?

A cheat that makes a full solve's worth of moves without solving has a
scrambled cube on camera at `t_stop`, and is caught **before** any
substitution happens. Nothing about cube identity is needed.

**The statistic is colour fragmentation, not colour identity.** A solved cube
shows 1–3 *solid* faces; a scrambled one is a mosaic of the same colours. So
count the distinct solid regions (`n_regions`). No facelet registration is
involved — which matters, because `cube_detector` boxes the whole cube and
its 3×3 slice only lands on real facelets when a face is held flat-on, and at
the timer stop the cube is at whatever angle the solver's hands left it.

**Opposite faces are merged into three axis classes (W/Y, R/O, B/G), and this
is free.** A solved cube can never show both faces of an opposite pair, so
merging costs nothing on the solved side — while deleting every hard colour
pair in this project at once, orange/red above all. The cost lands only on
the scrambled side, where the mosaic reads slightly *less* fragmented than it
is; that makes the test conservative, so its error is a miss and never a
false DQ.

**Measured, held out by date** (`*_solve` sessions end solved, `*_scramble`
end scrambled — the same structural proxy the count gate's 21/21 rests on,
and here it is even more direct, since the question literally is "is this
cube solved"). Threshold `n_regions > 4`:

| date cut | held-out catch | held-out false DQ |
|---|---|---|
| before 2026-07-28 | 71% (10/14) | **0** (0/14) |
| before 2026-07-31 | 90% (9/10) | **0** (0/10) |
| before 2026-08-02 | 100% (8/8) | **0** (0/8) |

Read the catch column as "roughly three quarters to all": each holdout has
8–14 attacks, so one session is 7–12 points and the spread between those
cells is noise. The false-DQ column is the one making a claim. In-sample the
solved median is **3 regions — exactly the three visible faces**, against 5
for scrambled.

`n_regions` ships rather than the better-looking `fragmentation`, which peaks
higher in sample and false-DQs in 10 of 15 swept settings when held out. Its
threshold is pinned by the single worst legit solve, making it an
`extreme`-kind statistic — judge those on their worst column, per the metric
robustness work.

**Known residual, diagnosed not guessed:** in warm evening light the
*background* classifies as one enormous solid region, the wall becomes the
largest blob, and a scrambled cube scores as solved. Three geometric fixes
were tried and all measured worse than no mask (hexagon 80%, ellipse 60%,
tighter centre crop held no line at all) — the cube fills most of its own
box, so any mask that removes background removes cube in proportion. The real
fix is a cube *segmentation* rather than a box. Until then this is the same
evening-lighting cliff as everywhere else, and the response is abstention.

**It is evadable** by presenting the cube badly at the stop. That path leads
to `REVIEW`, not `VERIFIED`, which is the correct handling of deliberate
unreadability.

#### 2.1a The 0-false-DQ result does not reach the live call (2026-08-10)

Everything above is measured by `solved_check.py score`, and re-running it
fresh reproduces it exactly: **0 false DQ on 28 legit solves, 20/28 caught.**
The number is real. What it is a property of is narrower than it looks.

That measurement reads **`tail_window`** — 1.5 s ending 0.5 s before the last
frame of the recording — with boxes from **`trajectory.npz`**, i.e. raw
`continuity_guard.detect_cubes` output. The live call does neither.
`live_anticheat.stop_window_solved` and `verify_solve._solved_at_stop` read
the frames **straddling the timer stop**, with boxes from
`prepare_data.per_frame_boxes`, which squares the box and adds
`crop_utils.CROP_MARGIN = 0.12`.

Both differences matter, and `ble/move_detector/stop_window_check.py` runs
the 2×2 to say how much (same 28+28 sessions, same `solved_at`, same
palette — only these two factors move):

| | false DQ / 28 legit | caught / 28 |
|---|---|---|
| **[A]** tail window + trajectory boxes — *the published cell* | **0** | 20 |
| **[B]** tail window + per-frame boxes (+12% margin) | 2 | 17 |
| **[C]** timer stop + trajectory boxes | 2 | 23 |
| **[D]** timer stop + per-frame boxes — **what the live gate does** | **3** | 18 |

Each factor costs ~2 false DQ on its own; together, 3 — **10.7%, against a
launch requirement of under 1%.**

**Why it is this fragile.** The operating point is chosen as one step above
the worst observed legit session (worst legit = exactly `4.000` regions,
threshold "> 4"). By construction it has **no false-DQ margin at all**, so
any change to its input spends margin it does not have. That is a property
of how the threshold was derived, not a flaw in the statistic.

**Retuning does not rescue it.** Give each cell its own zero-false-DQ
threshold and the catch rate is the price:

| cell | zero-false-DQ threshold | catch |
|---|---|---|
| [A] tail + trajectory | > 4 | **20/28 (71%)** |
| [B] tail + per-frame | > 5 | 10/28 (36%) |
| [C] stop + trajectory | > 6 | 11/28 (39%) |
| [D] stop + per-frame | > 5 | 13/28 (46%) |

So at the timer stop, honestly tuned, the arm catches **under half** — not
the 71–100% in the table above.

One genuinely encouraging detail: the *window* move helps the catch side. At
the shipped threshold, [C] catches 23/28 against [A]'s 20/28 — a scrambled
cube reads as more obviously scrambled at the stop than at the end of the
recording, which is what one would hope. It is the false-DQ side, and the
zero-margin threshold, that the move breaks.

**What this does NOT mean.** It does not mean raise `SOLVED_MAX_REGIONS` —
that hands the attack back exactly the margin the test exists to take away.
The options are to re-derive the threshold on what the live call actually
feeds (accepting ~46% catch), or to move the live evaluation to a window
where the cube is deliberately presented — which is the post-stop scan
window, already covered on the custody side, so evaluating there costs the
attacker nothing they did not already have. Neither is done.

**Caveats on this measurement**, in both directions: n = 28 per side; the
"timer stop" is simulated at `last_onset + 1.0 s` rather than a real button
press, chosen generous in the direction that helps the test; and `*_scramble`
tails remain a proxy for the attack rather than a recording of it.

### 2.2 What is dead, and must not be re-proposed

Appearance-*magnitude* substitution detection. Over 68 legit sessions the
persistent-change statistic runs median 0.472 / p90 0.587 / **max 0.857** on
a 0–1 scale, so a zero-false-positive threshold sits near 0.94: there is no
gap to put a threshold in. The cause is decisive — **the largest persistent
appearance change in an honest session is the solve completing**, mixed
colours to solid faces, the exact transformation a swap produces at the exact
moment it would produce it. Sessions ending solved peak at 0.542 at 82%
through; sessions ending scrambled peak at 0.384 at 40% through.

That also explains 2 live false DQs: `continuity_guard` DQs at
`SIG_DIST=0.45`, below the 0.542 median peak of a legit finishing solve.

§2.1 is the successor: not how *big* the change was, but whether the cube was
lawfully solved when the clock stopped.

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
| Count gate | too few moves, moves after the timer | a swap that arrives with enough moves, anything at ≥9.6 TPS, anything in bad light |
| Continuity guard | cube leaving/duplicating/teleporting | a swap fully inside the frame |
| **Solved-at-stop** | **a cube that was not solved when the clock stopped** | a cube presented unreadably at the stop (→ review); evening light |
| **Post-stop custody** | **the cube changing identity between the stop and the scan** | a swap to a visually identical cube *within* the tracked trajectory |
| Appearance magnitude | — | everything; measured dead, §2.2 |

**The attack that used to survive the count gate.** Make a full solve's worth
of plausible moves on camera *without solving*, then swap in a solved cube.
That reads as ~50 moves, clears the floor of 32, and the count gate passes it
— eight seconds of flailing suffices, which makes it a world-class time.
Nothing about the move count distinguishes it from a real solve, because the
move count is genuinely real; only the *outcome* is faked.

**Closed as of 2026-08-10, by two arms that only work together.** The attack
has a seam: the substitution must happen *after* the timer stops, because a
solved cube presented earlier would have to be turned for the remaining
"solve" and would come apart. So:

1. **Solved-at-stop** (§2.1) says the cube was already solved when the clock
   stopped. The flailer's cube is not, and this fires before any swap.
2. **Post-stop custody** — a `ContinuityGuard` over the scan window *alone* —
   says it is still the same tracked object when the scan validates it.

Either one alone leaves the swap a window to occur in; together there is
none. Custody is demanded only over the post-stop window rather than the
whole attempt on purpose: that window is short and deliberate, the cube is
being *presented* rather than manipulated, and no hands cross it — whereas
whole-session continuity at DQ strength is exactly what drives the measured
10.1% false-DQ rate.

What is left is the residual in §2.1: a cheat who deliberately obscures the
cube at the stop reaches `REVIEW` rather than `VERIFIED`, and evening light
degrades the arm. Neither is a pass.

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

## 3.2 Can the move CLASSIFIER gate, not just the count?

Measured 2026-08-10, `ble/move_detector/solve_distance.py`. The idea: replay
the decoded moves from the server-issued scramble and DQ if the end state is
too far from solved — a hard maximum on "distance to an actual solve". It is
the natural next question once the count gate exists, and the answer is
asymmetric.

**As a DQ rule it catches nothing.** 4 of 24 honest solves replay to
`n_solved = 0`, as scrambled as a uniformly random sequence, so a threshold
that never false-DQs sits at 0 and DQs 0/4800 of every null.

The mechanism is exact: `n_solved` is very nearly a deterministic function of
the decode's **edit distance**, not of honesty.

| edit distance | 0 | 1 | 2 | 3 | 6–7 | 10–11 | 13+ |
|---|---|---|---|---|---|---|---|
| n_solved | 20 | 12 | 7, 9 | 6 | 2–4 | 3, 5 | 0–2 |

All seven 1-error sessions land on exactly 12. A mid-sequence error does not
nudge the end state, it **conjugates** it, and a conjugated quarter turn
scatters. The first error costs ~8 cubies and the signal is inside the random
floor by ~5 errors — while daytime word error already puts a typical solve at
1–3 errors and the tail at 30+. This is the verification accuracy cliff
arriving through a new door.

**As positive evidence it survives**, because the direction is not symmetric:
landing *near* solved cannot happen by accident. Threshold pinned above every
null, so zero false accepts:

| null model | best of 4800 | honest solves verified |
|---|---|---|
| random flail | 5 | **13/24 = 54%** |
| decoy scramble | 6 | 12/24 = 50% |
| partial solve | 17 | 2/24 = 8% |

**The weak cell is the other arm's strong cell.** A cheat who stops before
the last layer has a visibly unsolved cube at the timer stop, which is what
§2.1 catches trivially. The two are complementary rather than competing.

Not wired into `adjudicate()`. Seed 1 caches only 6 sessions — the 6 hardest
for seed 0 — and on those both seeds give a median of 1.0, so the 54% is
carried by the easier sessions and the statistic is dead on the hardest ones
for *both* seeds. A fast path that collapses on the hard cases is a
measurement, not yet an arm.

## 4. Camera injection

Not addressed by any of the above and no cube analysis can address it. Every
scramble is server-issued and unique, so a pre-recorded video cannot match a
fresh scramble. The residual attack is switching sources mid-solve to conceal
a swap. Defence is **challenge-response**: the server demands a specific face
at a random moment, and the verification face order is randomised per solve
so no recording can satisfy it. Not built.

## 5. Next

1. Record the enough-moves-then-swap attack, including the table-edge
   variant. The count gate cannot see it and §2.1 now covers it on a
   *proxy*; a recorded attack is what turns that into a measurement. Note
   what the proxy already establishes and what it does not: `*_scramble`
   tails are genuinely "a cube that had real moves performed and is not
   solved", so the solved-at-stop side is well proxied — the **custody** side
   is not proxied at all and has never seen a real swap.
   Solver-following recordings remain a non-priority: the scramble sessions
   proxy that (21/21) and the bound on a solver's output is mathematical.
2. Record real timer-stop-to-scan footage and calibrate the phantom rate.
3. Wire `adjudicate()` into `verify_solve.py`'s live path and into the
   server-side re-verification worker (A3). The verdict function is pure and
   takes plain data specifically so both can run it and agree.
4. Enforce `POST_STOP_MAX_WINDOW_S` in the capture UI.
