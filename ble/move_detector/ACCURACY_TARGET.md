# What accuracy do we actually need?

`TODO.md` item 4 calls this the first goal: *"We need to test what % accuracy
we need to actually end up with a verification on, lets say 90% of solves.
This will set our goal and give us a clearer path forward."*

Measured 2026-07-30 on the 42-session seed-0 sweep (`results/2026-07-31/slice_on_seed0.json`),
27 of which are solve-length (≥60 moves). Simulation study in
`accuracy_target.py`.

---

## 1. The answer: ~98% raw per-move accuracy

Verification is not a gradual function of accuracy. It is a cliff, and the
recorded corpus lands on both sides of it:

| raw per-move accuracy | sessions | verified |
|---|---|---|
| ≥ 98% | 15 | **15/15 (100%)** |
| 95 – 98% | 4 | **0/4** |
| 90 – 95% | 5 | 1/5 |
| < 90% | 3 | 1/3 |

Nothing between 95% and 98% verified. So the working target is **≈98% raw
per-move accuracy**, and "90% of solves verify" is not a separate, softer
goal — at 98% you get ~100%, and a few points below it you get ~0%.

This is why chasing incremental accuracy has felt unrewarding. Between 94%
and 97% almost nothing changes downstream, which matches the four
interventions recorded in [[decoder-sprint-exhausted]] and
[[jitter-retrain-decoder-flat]] that each moved move-level metrics without
moving a single verified session.

## 2. Where we actually are — the number that matters

The headline 32/42 is dominated by sessions the model was trained on. Split
by the checkpoint's own session lists:

| group | n | raw acc | verified | miss rate | sub rate |
|---|---|---|---|---|---|
| trained on | 22 | 97.2% | **77%** | 2.69% | 0.08% |
| val (same-day holdout) | 4 | 94.7% | **0%** | 2.18% | 3.07% |
| unseen environment | 1 | 78.2% | **0%** | 11.29% | 10.48% |

**Held-out verification is 0/5.** Quote that, not 32/42, when describing what
the pipeline can do for a user whose solve it has never seen. The trained-on
number is a memorisation ceiling, not a capability.

Note the small n on the held-out rows, and see [[named-holdouts-cross-env]]:
random session holdouts already flatter cross-environment performance, so
0/5 is if anything the optimistic reading.

This is the same effect `TODO.md` item 4 describes ("testing so well in
environments of same day to training but failing on different conditions").
Quantified: **the gap between 94.7% and 98% is the entire product problem.**
It is ~3.3 accuracy points, on held-out data.

## 3. Which errors to attack

Total error mass over the 27 solve-length sessions splits:

| channel | share of error mass | mean rate |
|---|---|---|
| miss (dropped move) | **60%** | 2.93% |
| phantom (invented move) | 21% | 1.04% |
| substitution (named wrong) | 19% | 0.91% |

**81% of what goes wrong is an insertion/deletion problem, not a naming
problem.** `TODO.md` item 3 says exactly this from the other direction
("The biggest issues with the algorithims is not the correct identification
but insertions and deletions"), and item 3's suspected cause — "the onset
detector is missing moves or possibly merging them together if the onset peak
is too close in time to another (which could be possible since the algos are
fast)" — is confirmed and quantified in `GROUND_TRUTH_ARTIFACTS.md`:

* the BLE clock is quantised to a 30 ms tick, so 10.1% of moves land within
  2 frames of a neighbour;
* 91 pairs (3.9% of all moves) are closer than `MIN_SEP` and **no peak picker
  can report both**;
* of those, 17% are middle-slice turns — one physical motion the cube reports
  as two events. Those are now recovered by `reconstruct.py --slices`
  (+2 sessions, no falsifiability cost).

The remaining ~83% of collisions are genuinely fast distinct moves, and they
are the single largest identified block of the miss channel.

## 4. Simulation

Isolating cause from correlation needs more samples than the corpus has —
only four sessions sit in the 95-98% band. `accuracy_target.py` drives the
real decoder with synthetic onsets at the measured error mix (19/60/21) and
the measured confidence profile.

### 4.1 The curve confirms the cliff

20 trials per point, 120 moves (the corpus median), beam 4000, no retry:

| raw acc | verified | exact |
|---|---|---|
| 99.9% | 100% | 100% |
| 99.6% | 95–100% | 90–100% |
| 99.0% | 90–95% | 65–80% |
| 98.3% | 85% | 75% |
| 98.0% | 60% | 35% |
| 96.0% | 15% | 5% |
| 95.1% | 10% | 10% |

**For ≥90% of solves to verify, raw accuracy must be ≈99%** — a point higher
than the corpus's coarse bins suggested. Between 98% and 96% verification
collapses from 60% to 15%. The cliff is real, and simulation puts its edge
just below 98%.

### 4.2 The channels are NOT equivalent — and this is the real result

Same error mass, routed entirely through one channel:

| channel | rate | raw acc | verified |
|---|---|---|---|
| substitution | 3.0% | 97.0% | **100%** |
| miss | 3.0% | 96.6% | **45%** |
| phantom | 3.0% | **100.0%** | **40%** |
| substitution | 2.0% | 97.6% | 100% |
| miss | 2.0% | 98.0% | 65% |
| phantom | 2.0% | 100.0% | 65% |
| substitution | 1.0% | 98.9% | 100% |
| miss | 1.0% | 98.9% | 90% |
| phantom | 1.0% | 100.0% | 90% |

Two conclusions, both of which redirect the project:

**Substitutions are very nearly free.** At 3% substitution — triple the real
rate — every single trial still verified. The decoder has a group-theoretic
constraint that pins down *which* move a mis-named onset must have been; it
does not have anything comparable for a move that was never reported. Effort
spent making the classifier name moves better is being spent on the one error
channel that does not cost verifications.

**Raw per-move accuracy is the wrong target metric.** Phantoms do not reduce
raw accuracy *at all* — the phantom rows sit at 100.0% — yet 3% phantoms drop
verification to 40%. A model can post a perfect accuracy number and verify
two solves in five. Any target expressed as "% accuracy" is measuring
something the outcome is only loosely coupled to.

### 4.3 Solve length — no effect detected

At 1.0% error (measured mix): 60 moves 100%, 90 moves 100%, 120 moves 90%,
160 moves 90%, 200 moves 100%.

Non-monotonic, so **there is no length effect to report at n=20** — the 90%
cells are two trials failing out of twenty, which the 200-move cell then
contradicts. Whatever length costs, it is smaller than this study can see and
far smaller than the channel effect in §4.2. Do not quote a length penalty
from this table.

## 5. The goal, stated usefully

Not "98% accuracy". Accuracy is the wrong unit (§4.2). The target that
actually predicts verification is a pair of **onset-count** rates:

> **miss rate ≤ ~1% and phantom rate ≤ ~1%, on held-out sessions.**
> Substitution rate is not currently a constraint at all.

Against that target, measured over the 27 solve-length sessions. The
distribution is heavily skewed — a few bad sessions pull the mean well above
the typical session — so both statistics are given:

| channel | median session | mean | target |
|---|---|---|---|
| miss | 1.50% | 2.93% | ≤ 1% |
| phantom | 0.72% | 1.04% | ≤ 1% |
| substitution | 0.00% | 0.91% | (unconstrained) |

The typical session needs its miss rate cut by about **1.5x**; the mean is
~3x out because of the tail. Use the median for "what does a normal solve
need" and the mean for "what does the corpus average" — they answer different
questions and the 2x difference between them matters.

Sanity check that the simulation and the corpus agree: the median session's
rates sum to ~2.2% total error, which §4.1 puts at ~85% verified; trained-on
sessions actually verify 77%. Close enough to trust the model. (Using the
means instead would predict ~15%, which is why the skew has to be handled
explicitly rather than averaged away.)

So the problem reduces to **the miss rate, on held-out data**. Nothing else
in the error budget needs to move.

### 5.1 Threshold tuning cannot get there — measured, not assumed

The obvious cheap lever: the onset threshold was tuned for F1, which weights
misses and phantoms equally, while the deployed operating point runs them at
better than 2:1 skewed toward misses. If they cost about the same downstream
(§4.2 says they do), that looked like free ground.

`threshold_sweep.py` re-peak-picks the cached posteriorgrams across
thresholds — no decoding needed to find the balance point. Seed 0, 27
solve-length sessions, 3,312 onsets:

| threshold | miss | phantom | sub | F1 |
|---|---|---|---|---|
| 0.10 | 3.08% | 2.63% | 1.18% | 0.971 |
| 0.20 | 3.32% | 1.60% | 1.15% | 0.975 |
| 0.30 | 3.44% | 1.18% | 1.12% | 0.977 |
| 0.40 | 3.71% | 0.85% | 1.06% | **0.977** |
| **0.50 (deployed)** | 4.02% | 0.60% | 1.00% | 0.977 |
| 0.60 | 4.89% | 0.36% | 0.88% | 0.973 |
| 0.70 | 7.07% | 0.18% | 0.69% | 0.962 |

**The hypothesis is wrong.** Going 0.50 → 0.10 buys 0.94 points of miss for
2.03 points of phantom — a 2:1 trade *against*, when the two cost about the
same. Break-even is around 0.30 (−0.58 miss for +0.58 phantom), and below
that it degrades. `min_sep` 1 and 2 score identically, confirming
`decode.py`'s claim that the strict-local-max requirement, not the
refractory window, is the binding constraint.

**The deployed threshold is already near-optimal.** There is nothing to win
here, which is worth knowing before spending modelling effort on it.

### 5.2 The miss target is below the structural floor

Of 3,312 ground-truth onsets, 148 (4.47%) sit in pairs closer than `MIN_SEP`.
At most one member of each pair can ever be reported, so:

> **the miss rate has a structural floor of ~2.23%.**

The target in §5 is ≤1%. **The floor is more than double the target.** No
threshold, no beam width and no better classifier reaches it — and at the
deployed threshold the floor is already ~56% of the observed 4.02% miss rate.

This is the single most important consequence of today's work. It means
"reduce the miss rate" is not a tuning problem at all. Either the onset
representation changes so that two events 0-1 frames apart can both be
emitted (peak-picking on a 1-D curve structurally cannot), or the ground
truth's 30 ms quantisation stops collapsing them — and only the first is
under our control at inference time.

### What follows

1. **Stop optimising classifier naming accuracy.** At 3% substitution — triple
   the real rate — 20 of 20 simulated solves still verified. This is the
   quantitative reason the four interventions in
   [[decoder-sprint-exhausted]] / [[jitter-retrain-decoder-flat]] moved
   move-level metrics and zero verified sessions.
2. **Everything is the onset detector now.** Misses and phantoms are both its
   output. The largest identified block of the miss channel is onset
   collisions (§3): 3.9% of moves are in a pair the peak picker cannot
   separate, which is by itself larger than the entire 1% miss budget.
3. **Report miss/phantom rates, not accuracy,** in every future comparison.
   `metric_audit.py` and `verify_joint.py` already split misses into
   forced/crowded/clean (`decode.onset_collisions`) — that breakdown is now
   the primary metric, not a footnote.
4. **The remaining levers, ranked by §5.1 and §5.2.** Threshold tuning is
   dead (§5.1). What is left has to attack the floor itself:
   * **Replace peak-picking with a representation that can emit two events
     at one time index.** A 1-D onset curve plus a strict-local-max rule
     cannot, by construction. A per-frame *count* head (0/1/2 events) or a
     CTC-style decode over the posteriorgram both can, and both reuse the
     existing joint model's trunk. This is the only lever that addresses the
     2.23% floor directly.
   * **Quantisation-aware onset targets** — the label is a 30 ms box, not a
     point. Helps the ~83% of collisions that are not slices, and is cheap.
   * Horizontal crop-jitter augmentation for the detector (from the crop
     study) — smaller, but miss-channel and independent.
5. **Verification is not the only product.** `TODO.md` item 1 notes the
   anticheat/verification split: a solve can be a confirmed win without a
   move-by-move reconstruction. This cliff applies to the *paid*
   move-recording feature; the free verification tier has a different and
   much cheaper bar, and the two should be costed separately.

---

Related: `GROUND_TRUTH_ARTIFACTS.md` (collision mechanism, OP_SLICE),
`PATH_TO_VERIFICATION.md`, `MODEL_REWORK_PLAN.md`.
