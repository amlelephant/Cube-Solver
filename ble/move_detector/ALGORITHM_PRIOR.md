# Algorithm-aware decoding — design plan

Status: **the RECOGNISER in this document failed (§6) and stays failed. A
different method — generative, state-gated, CTC-scored — was gated on
2026-08-01 and PASSED on both seeds: see §7 and `algorithm_gate.py`.**
Written 2026-07-25, §6 added the same day, §7 added 2026-08-01.

> **Read §6 before §2-5.** The matcher exists and is correct
> (`algorithms.py`, `--selftest`, `--step1`), but the measurement it was
> built to gate says the correction payoff is bounded at ~21% of the
> classifier half of the error budget. The design below is preserved
> because the reasoning is still sound and the orientation half is
> untested — but do not implement step 2 on the strength of §1's numbers
> alone. §1 measured coverage on ground-truth words; §6 measured it where
> the errors actually are, and the two disagree for a structural reason.

Add a *local* prior to the group-theoretic decode (`reconstruct.py`) by
recognising known algorithms in the detected move sequence, at all 24 cube
orientations. Two payoffs: it repairs classifier errors inside recognised
stretches, and it infers the camera-to-cube orientation — including a
mid-solve rotation, which the pipeline currently cannot see at all.

---

## 1. Why this is the right next move

`reconstruct.py`'s only constraint is the pair of endpoints. That constraint
is **global and terminal**: it discriminates nothing until the very end of
the sequence, which is exactly why the ranking heuristic needed the REP1
ladder and why the beam has to be 4000 wide. The module docstring already
names the fix:

> if real sessions hit them, the productive fix is a better classifier or
> **mid-solve state anchors**, not more beam.

An algorithm match is a mid-solve anchor. It is evidence about moves 40–52
that arrives *at* move 40, rather than at move 152.

It is also the right time. Per `[[classifier-crop-mismatch]]`, after the
crop fix the remaining error is 55% detector (miss + phantom) and 45%
classifier — and the decode's envelope is roughly *one insertion beyond
whatever the classifier's mistakes already cost*. Anything that lowers the
true story's cost buys envelope directly.

### The evidence that it will work

Measured on the 35 solve-length BLE ground-truth sessions (2026-07-25):

| measurement | value |
|---|---|
| 4-move windows belonging to a 4-gram seen ≥3× | **65.6%** |
| moves covered exactly by a 40-entry mined library | **32.0%** (range 0–83%) |
| same library, same solves, **moves shuffled** | **0.0%** |

The top recurring n-grams are textbook, not incidental:

```
67x  R U R' U'                                 sexy move
24x  F R U R' U' F'                            edge orientation
16x  R U R' U' R' F R R U' R' U' R  ...        T-perm (R2 as R R)
14x  R U' R U R U R U' R' U'                   U-perm family
```

The **0.0% null** is the number that matters. Exact matches of length ≥6
essentially never occur by chance, so a matcher has enormous headroom to
be made fuzzy (tolerating 1–2 substitutions) before precision degrades.

Solve lengths are 65–216 moves, which is layer-by-layer / beginner
territory, not 55-move CFOP. That is *favourable*: a small library repeated
many times, rather than a 78-case PLL/OLL set each seen once.

Caveat to carry into the write-up: the 32% figure used a library mined from
these same sessions and is therefore optimistic. See §5.

---

## 2. Design — a prior, never a rewrite

The tempting version is a pre-pass: match algorithms, hard-correct the
sequence, then decode. **Do not build that.** It commits greedily, a false
match injects errors the decoder can no longer question, and it
double-counts the classifier's evidence.

Instead the matcher reshapes the **cost rows** and decides nothing. This is
the same shape as the existing `BLEND_INV` prior in `onset_costs()` — prior
mass blended into a softmax, scaled by that onset's uncertainty, never a
constraint. The beam still searches the same space; the endpoint constraint
remains the only hard truth; a `VERIFIED` verdict still means what it meant.

```
                     ┌─────────────────────────────┐
 cost_rows (n,12) ───┤ match_algorithms()          │
                     │  24 orientations x mirror   │
                     │  + null hypothesis          │
                     └──────────────┬──────────────┘
                                    │ matches
                     ┌──────────────▼──────────────┐
 cost_rows ─────────►│ apply_prior()               ├──► cost_rows'
                     └─────────────────────────────┘         │
                                                              ▼
                                                    reconstruct.decode()
                                                       (unchanged)
```

### 2.1 Matching

For algorithm `a` (length `L`) at orientation `g` starting at onset `i`:

```
score(i, a, g) = Σ_j  cost_rows[i+j][ camera_class_for(g, a_j) ]
```

This is the negative log-likelihood, under the classifier's own softmax, of
having observed that algorithm there. It uses the **full softmax**, not the
argmax — a stretch the classifier read as `R U R' F'` with `U'` a close
second on the last onset scores well against `R U R' U'`, which is precisely
the case worth catching.

Reuse, don't rebuild: `reconstruct.py` already has the orientation
machinery — `_SIGMAS`, `rotate_sigma()`, `camera_class_for(sid, k)` — because
`--rotations` needed it. Store the library once in the cube frame and expand
through `camera_class_for`.

**Mirrors are not orientations.** A left-handed F2L insert is a different
algorithm, not a rotated one; it must be a library expansion (reflect
face L↔R, invert every turn), not one of the 24.

**Half turns.** Algorithms are written with `R2`; the classifier emits only
quarter turns, and the detector may fire once or twice for one physical
half-turn motion. Store the library expanded to quarter turns (`R2` → `R R`)
and let the decoder's existing delete/insert edits absorb the count
mismatch.

Cost: `M × 24 × 2 × L` vectorised ops over length `n`. At `M=200`, `L≈12`,
`n≈150` that is ~10⁵ vector ops — well under a second, and it runs once per
session, not per beam step.

### 2.2 The null hypothesis is the whole ballgame

Most of a solve — cross, pair setup, regrips, AUF — is **not** algorithmic.
A matcher without a properly priced "no algorithm here" hypothesis will
hallucinate algorithms across the other 68% and actively make things worse.

So: at each position, softmax the shortlist of candidate matches covering it
*together with* a null hypothesis at a fixed prior cost, and let the prior
weight at that position be the posterior mass on non-null. Calibrate the
null's cost so that the shuffled-word false-match rate stays at the measured
0% floor.

### 2.3 Applying the prior

For onset `i` with posterior weight `w_i` on predicted class `k_i`:

```
q  = (1 - w_i)·p  +  w_i·onehot(k_i)      then renormalise, then -log
```

Two hard rules, both learned the expensive way elsewhere in this pipeline:

- **Cap `w_i`.** Never let the prior alone drive an acceptance cost to zero.
  A confidently-wrong classifier stretch plus an unbounded prior locks the
  error in permanently.
- **Scale `w_i` by the onset's uncertainty**, exactly as `BLEND_INV` does.
  The prior should move a coin-flip onset a long way and a 0.99-confidence
  onset barely at all.

### 2.4 Orientation inference — the second, separable payoff

`verify_solve.py` states the limitation plainly:

> the truth is blind to whole-cube rotations … a rotation corrupts EVERY
> following label, so it appears as one unbroken run of errors

The orientation posterior falls out of matching for free: aggregate match
scores per `sigma` over a sliding window. A **change** in the argmax sigma
partway through the sequence is the signature of an unrecorded rotation, and
it says *where*. Today `--rotations` brute-forces rotation insertions at
every position with no evidence guiding it — this turns that into a handful
of evidence-backed candidate positions, which is the difference between a
search that is affordable and one that is not.

Build this as a **separate, separately-ablatable output**. It is useful even
if the correction prior disappoints, and conflating the two would make the
result unreadable.

---

## 3. Where the code goes

New `ble/move_detector/algorithms.py`:

| function | returns |
|---|---|
| `LIBRARY` | cube-frame words + provenance for each |
| `expand(library)` | (word, sigma, mirrored) for all 24 × 2 |
| `match_algorithms(cost_rows, ...)` | `[Match(i, alg, sigma, score, posterior)]` |
| `apply_prior(cost_rows, matches, w_max)` | reshaped `cost_rows` |
| `orientation_profile(matches, n)` | `(n, 24)` posterior over sigmas |

One hook, in `reconstruct.costs_from_moves()` — its docstring already calls
itself *"the one place the classifier softmax and the detector's onset
strength are turned into edit costs, shared by session replay and the live
verification so the two cannot drift apart."* That is exactly the right
insertion point and it keeps replay and live identical by construction.

Flags: `--algo-prior W` (**default 0.0 = off**) and `--algo-orientation` on
`reconstruct.py` and `verify_solve.py`. Ship it off, turn it on once §5
says it earns its place.

---

## 4. Risks

**The library encodes one person's habits.** It is layer-by-layer shaped
because the recorded solver is. A CFOP or Roux solver gets a prior that
matches nothing — acceptable (it degrades to today's behaviour) — but it
must be *verified* to degrade rather than mislead. Test on a deliberately
non-matching sequence.

**A wrong reading can be reinforced.** Mitigated by §2.3's two rules, but
this is the failure mode to watch for in the ablation: check whether any
session gets *worse*, not just whether the mean improves.

**Evaluation circularity.** A library mined from the test sessions measures
nothing. The library must come from a public CFOP/LBL algorithm set, or —
if mined — mined **only from training sessions**, with the holdout untouched
and that fact stated next to the number. This is the same discipline
`[[named-holdouts-cross-env]]` records.

**It changes what `VERIFIED` means.** The falsifiability sweep's separation
margin is denominated in cost units. A prior that makes the true story
cheaper also makes near-decoys cheaper, since they share most of their
moves. The decoy separation **must** be re-measured, not assumed to carry
over.

---

## 5. Validation, in order — each step gates the next

**Step 0 — done (§1).** Is there algorithmic structure to exploit? Yes:
65.6% of 4-windows recur, 32% exact coverage, 0% shuffled null.

**Step 1 — matcher precision on ground truth, no decoder involved.**
Run the matcher on BLE ground-truth words. Measure recall (fraction of moves
covered) and the false-match rate on length-matched shuffles. If fuzzy
matching cannot hold the null near zero, stop here — the rest cannot work.

**Step 2 — raw per-move accuracy** on the 5-session cross-environment
holdout (`solve_20260720_142006`, `20260721_102711`, `20260722_101225`,
`20260723_105530_solve`, `20260724_100120_solve`), prior on vs off. Report
per-session, not just the mean.

**Step 3 — end-to-end** on the 9 representative sessions: verified count,
exact reconstructions, and true-story cost (the binding quantity — see the
`reconstruct.py` docstring). Current baseline to beat: raw per-move 90.9%,
one exact reconstruction.

**Step 4 — re-run the falsifiability sweep** and report the new decoy
separation alongside the accuracy gain. A gain that shrinks the margin to
zero is not a gain.

**Step 5 — ablate orientation inference separately**, on a deliberately
rotated take.

---

## 6. Outcome of step 1 (2026-07-25)

Built as `algorithms.py`. `--selftest` covers the orientation enumeration,
the mirror map (cross-checked against `train_move_classifier.WCA_FLIP`),
rotation round-tripping, the substitution budget, and library dedup.

### 6a. Coverage vs null, on ground-truth words — passed, but misleading

37 curated face-turn-only algorithms, 98 entries after dedup modulo
rotation and mirror, 4704 variants. 35 solve-length sessions, 3841 moves.

| min_len | budget | real | shuffled null | scramble-legal null | precision |
|---|---|---|---|---|---|
| 4 | 0 | 33.0% | 3.5% | 3.8% | 89.6% |
| 4 | 1 | 56.5% | 34.8% | 26.9% | 67.7% |
| 4 | 2 | 83.7% | 87.8% | 88.9% | 48.5% |
| 6 | 0 | 19.2% | 0.0% | 0.0% | **100%** |
| 7 | 1 | 21.9% | 1.2% | 0.2% | 99.2% |
| 8 | 2 | 34.3% | 6.5% | 3.7% | 90.3% |
| 9 | 2 | 13.2% | 0.0% | 0.0% | **100%** |

This produced one durable design finding. **A flat substitution budget is
the wrong shape**: two tolerated errors in a 4-move trigger is a 50% error
rate and matches almost anything, while the same two in a 15-move T-perm is
13% and is exactly the case worth catching. Hence
`budget(L) = min(MAX_SUBS, floor(L·SUB_RATE))` — 0 below length 7, 1 to 13,
2 above.

### 6b. Correction potential, on real classifier softmaxes — FAILED

§6a measures the matcher on ground-truth words, where an exact match agrees
with the truth *by construction*. That says nothing about whether the prior
would help. The real question needs real softmaxes, which
`reconstruct_replay.json` already caches: 9 sessions from
`move_classifier_all39_cropped.pt`, 68 classifier errors, predictions
aligned to BLE truth first.

Mean coverage was healthy — **36.5%**, up to 49.6% on one session. The
corrections were not:

| budget | errors any match spans | proposes truth | proposes another wrong | drags a CORRECT onset off |
|---|---|---|---|---|
| rate .15, cap 2 | 14/68 (21%) | 6 (9%) | **0** | 29 |
| rate .25, cap 3 | 53/68 (78%) | 26 (38%) | 27 | 487 |
| rate .50, cap 6 | 68/68 (100%) | 65 (96%) | 68 | 1001 |

At the tight operating point the matcher is genuinely precise — it proposes
zero wrong moves — but it only *reaches* 21% of errors. Loosening the budget
to reach them costs 487 proposals against onsets the classifier already got
right, to gain 26: a 19:1 harm ratio.

**The binding constraint is reach, and it is structural, not a tuning
failure.** Recognising a 15-move algorithm requires its neighbourhood to be
read correctly, and a classifier error is precisely where the neighbourhood
is not. At ~95% per-move accuracy a 15-move window is clean only ~46% of the
time, and the errors concentrate in the windows that are not. Algorithm
matching therefore self-selects onto the parts of the solve that were
already right.

This was checked against the obvious confound: reach is measured over **all**
matches, not the greedy cover, because `select_cover` is cheapest-first and
so is biased away from windows that must spend substitution budget. The 21%
is the unbiased figure.

### 6c. A correction I owe §2.3

§2.3 claimed reinforcement was a payoff — that matches agreeing with the
classifier would lower the true story's cost and buy decode envelope. **That
is wrong.** `onset_costs` is relative to the argmax, so accepting the argmax
already costs 0; reinforcing it moves nothing. The cost only falls where the
prior points *against* the argmax, which is the same 21%-reach case. There is
no separate reinforcement payoff.

### 6d. Orientation inference — also weak, for a different reason

All 9 takes held one orientation, so `sigma = 0` is the known answer.
At the tight operating point the top sigma was correct in **5/9** sessions,
with the winning share only 20-75% — no dominant orientation. The cause is
library ambiguity rather than reach: with 4704 variants an observed window
often matches algorithm X at one orientation and algorithm Y at another, so
the votes spread. A much smaller, longer-only library might sharpen this;
it has not been tried.

### 6b-2. Window matching (seed-and-extend) — the right method, same ceiling

§6b tested WHOLE-algorithm matching: a 15-move T-perm had to fit the
substitution budget end to end or be rejected entirely. That is the wrong
unit, and it was a real flaw in the first attempt. The fix is seed-and-extend
(`match_windows`): find the stretch that IS read cleanly, use it to anchor
the algorithm's identity and alignment, then score the whole overlap and
reach into the dirty part. `--selftest` asserts the distinguishing case — a
T-perm fragment with its tail destroyed matches, and whole-algorithm matching
rejects that same fragment.

Two findings came out of it, one methodological and one negative.

**Agreement rate is the wrong acceptance test; mean cost per move is the
right one.** Measured over accepted windows against BLE truth, ranked by how
much of the matched algorithm the solver actually performed:

| observable | true identifications | false ones |
|---|---|---|
| mean cost/move | **all <= 0.07** | **all >= 0.27** |
| agreement rate | 0.93, 1.00, 0.92, ... | 0.80, 0.88, 0.75, ... — overlapping |

Cost separates cleanly; agreement rate does not, and actually inverts the
ranking on the decisive pair. A 15-move T-perm with ONE disagreement where
the classifier's second choice was the T-perm's move (cost/move 0.06) is a
true match and a free correction; one with three disagreements the classifier
was confident about (cost/move 1.11) is false — and agreement rate scores the
false one higher (0.80 vs 0.93). Cost is uncertainty-weighted by
construction, which is exactly the property needed. Same reasoning as §2.3's
"scale by the onset's uncertainty", arrived at from the other direction.

**But the ceiling did not move.** At the safe operating point (seed >= 7,
accept at cost/move <= 0.25) the matcher is *perfectly* precise — across 9
sessions it proposes **3 corrections, 0 corruptions, 0 wrong moves, and fires
on 0% of scramble-legal null words**. It simply does not reach far: 4-9% of
errors depending on how far the seed is loosened, versus 21% for the looser
whole-algorithm setting that came with harm attached.

| method | reach | fix | corrupt | break | null |
|---|---|---|---|---|---|
| whole-algorithm, tight | 21% | 2 | 0 | 2 | 0 |
| whole-algorithm, loose | 78% | 26 | 27 | 487 | — |
| **window, cost-accepted** | **6%** | **3** | **0** | **0** | **0** |
| window, loose seed | 18% | 9 | 0 | 6 | 430 |

And the decisive one: **all 3 corrections land in sessions the classifier
trained on. On the 4 classifier-held-out sessions it is 0 fixes / 0 breaks
out of 45 errors.**

So the user-suggested method is strictly better — it converted a net-zero
result with harm into a small strictly-positive result with none — but the
magnitude is not what blocks it. The ceiling is the same one §6b found, now
expressed exactly: to be *confident* a stretch is a given algorithm you need
the classifier to have read it well, and `cost/move <= 0.25` is that
requirement written down. Where the requirement holds there is little left to
fix; where there is much to fix the requirement fails.

### 6e. Where this leaves it

Do not build steps 2-5 for the correction payoff. Its ceiling is ~21% of the
*classifier* half of the error budget, and per `[[classifier-crop-mismatch]]`
the classifier half is now the smaller one — detector miss/phantom is 55% of
remaining error. The realistic upper bound on the whole idea is a few percent
of remaining errors, for a large amount of machinery.

What is worth keeping:

- `algorithms.py` itself — correct, tested, and the honest measurement
  harness for any future attempt (`--step1` re-runs everything above).
- `match_windows` (seed-and-extend) as the method. It is strictly better
  than whole-algorithm matching and it is **safe at its default operating
  point**: zero corruptions, zero wrong proposals, zero null firing. If it
  is ever wired into `costs_from_moves`, it cannot make things worse — it
  will simply do very little until the classifier improves.
- The cost-vs-agreement finding (§6b-2), which is the one result here that
  generalises beyond this feature: any "is this stretch really X" test in
  this pipeline should be uncertainty-weighted, not count-based.
- The length-scaled budget finding (§6a).
- The reach diagnostic, which is the number any future version of this idea
  has to beat first. It is printed by `--step1b` as the CEILING line.

If it is revisited, the productive order is: fix the detector first (it is
the majority of error), re-measure reach at higher per-move accuracy — reach
should rise superlinearly, since it depends on a whole window being clean —
and only then reconsider. A 99%-accurate classifier makes a 15-move window
clean 86% of the time instead of 46%.

---

## 7. The generative re-ask (2026-08-01) — PASSED

§6 killed a *recogniser*: read the predicted string, find a known algorithm
in it. Its ceiling was reach, and reach was structural — recognising a
15-move algorithm needs its neighbourhood read cleanly, and a classifier
error is exactly where it is not, so the matcher self-selects onto the
parts of the solve that were already right.

The re-ask inverts the direction, on a user suggestion framed as abstract
interpretation: stop inferring per move, infer over a *range*. Do not
recognise anything — **generate**. `reconstruct.py`'s beam already carries a
concrete cube state per hypothesis, so

    alpha : cube state -> (cross done, F2L complete, LL oriented, solved)

is computable for free at every step. Wherever a hypothesis' abstract state
says "F2L complete", the small set of library words that raise the abstract
state can be proposed as whole transitions and scored against the
posteriorgram. There is no reach: coverage of the last layer is 100% by
construction, and a span full of dropped onsets scores no differently from
a clean one.

Two things made this newly answerable, neither of which existed in §6:

* **CTC.** `ctc_logp` gives log P(word | frames) marginalised over every
  alignment. §6's fixed-offset scorer was destroyed by a single dropped
  onset; this is indifferent to them. `train_ctc.py` postdates §6.
* **A measured grammar.** Replaying BLE truth through the cube group over
  the 20 scramble-paired sessions: every solve is one 49-78 quarter-turn
  F2L-building stretch, then 2-7 last-layer chunks that each break the F2L
  predicate and restore it within 5-20 quarter turns. The two lengths do
  not overlap, so the boundary needs no tuning. **The 95 chunks collapse to
  14 distinct words; the top 8 cover 90.5%**, and the structure holds
  unchanged on the cross-day sessions no checkpoint has seen.

### 7a. Gate results — `algorithm_gate.py`, both seeds

Library mined from the CTC checkpoint's own 13 analysable *training*
sessions only. Held-out = the 4 val sessions plus the 07-29/30/31 takes
recorded after the checkpoint (7 sessions, 34 chunks).

| | seed 0 | seed 1 |
|---|---|---|
| top-1 identification, all 95 chunks | 97.9% | 94.7% |
| top-1, spans the plain CTC decode got WRONG | 90.9% | 82.1% |
| top-1, held-out sessions | 97.1% | 91.2% |
| top-1, held-out AND dirty | 94.7% | 84.2% |
| true word present in the mined library | 98.9% | 98.9% |

Identification does **not** require the span to have been read correctly —
that is the entire result, and it is what §6 could not do.

**Null (does the candidate set fire on non-algorithms), nll/move:**

| population | seed 0 median | seed 1 median |
|---|---|---|
| real last-layer chunks | 0.00 | 0.04 |
| F2L spans of equal length | 11.49 | 10.03 |
| the true span, frames shuffled | 8.87 | 7.73 |

A single threshold accepts 97.9% / 98.9% of real chunks and 0.4% of nulls
on both seeds. That is the precision §6's matcher never had at usable
reach.

### 7b. Boundary tolerance — and why the scorer must be LOCAL

The spans above come from ground-truth onset frames. In deployment they
come from a beam hypothesis, so the binding question is how wrong the
boundary may be. Translating the whole span: identification holds to
±10 frames (~±0.33 s), degrades at ±20, collapses at ±40.

Widening the span so it brackets neighbouring AUF turns splits the two
scorers apart, and the split reproduces §6's own lesson exactly:

| frames added each side | 0 | 10 | 20 | 40 | 80 |
|---|---|---|---|---|---|
| `ctc_logp` (global), seed 0 | 97.9% | 97.9% | 96.8% | 68.4% | 34.7% |
| `ctc_logp_local`, seed 0 | 93.7% | 93.7% | 93.7% | 93.7% | 77.9% |
| `ctc_logp_local`, seed 0 held out | 85.3% | 85.3% | 85.3% | 85.3% | 64.7% |

Global CTC must explain every frame of the window, so one extra bracketed
turn destroys the true word's score — the same failure as the first
indel matcher, which "spent its whole budget on insertions skipping
unrelated moves *before* the true algorithm start". `ctc_logp_local` gives
free leading and trailing skip and is **flat out to ±40 frames**, at a
~4-6 point cost when the boundary happens to be exact. **Use the local
scorer.** The strict one is only correct with oracle boundaries, which
nothing downstream will ever have.

### 7c. What it buys, stated honestly

| | seed 0 | seed 1 |
|---|---|---|
| ground-truth moves collapsible into one transition (held out) | 41% | 37% |
| mis-read moves falling inside an identified chunk (held out) | 39% | 27% |
| removed insertion cost at `C_INS` (held out) | ~220 | ~156 |

**A correction to the claim that motivated this.** Error was measured as
78% concentrated in the last layer — but that was the *peak-picking* arm
replayed on sessions it trained on. Re-derived here under CTC on held-out
sessions, the error rate inside last-layer chunks is 16.0% / 14.6% against
18.2% / 19.5% outside: **essentially uniform, marginally lower inside.**
CTC's phantom reduction appears to have already taken the last layer's
excess. So this is not "targets where the errors are". It is "collapses
~40% of the sequence into single transitions that carry ~30-40% of the
errors with them" — a search-and-budget argument, not a targeting one.

### 7d. Standing caveats

* The library is one person's repertoire. It must be shown to *degrade*
  rather than mislead on a solver it does not fit — §4's risk, untested.
* One orientation throughout the corpus; no rotation or mirror expansion is
  exercised by this gate.
* Only the 20 scramble-paired sessions can be analysed at all: the
  trajectory needs a verified start state, and deriving one by walking back
  from "it must have ended solved" assumes the answer (it silently produces
  garbage on the unpaired early sessions, which is how their exclusion was
  found).
* Nothing here has touched `reconstruct.py`. The decode-side risks in §4
  stand undiminished, in particular that **the falsifiability sweep must be
  re-run**: a prior that makes the true story cheaper makes near-decoys
  cheaper too. The one hopeful sign is that a never-scrambled cube does not
  traverse the abstract lattice at all, so this may tighten that check
  rather than loosen it — but that is a hypothesis, not a measurement.
* The two OP_SLICE lessons will both apply to any wiring-in: gate the new
  transition hard (here the abstract state is the gate, which is tighter
  than `SLICE_GATE` ever was), and teach the ranking heuristic about it or
  the true hypothesis gets out-ranked and evicted while doing nothing wrong.

---

## 8. Wired into the beam (2026-08-01/02) — forward inert, backward small

§7 gated the idea; this is what happened when it was actually built into
`reconstruct.py`. Two arms, both measured on the 12 held-out sessions of
each CTC seed, baseline and arm run back to back in the same sitting.

### 8a. Forward OP_ALGO — inert, and measurably so

A multi-onset beam transition (`OP_ALGO`, with a per-hypothesis debt
counter so one step can consume many onsets without breaking the
one-stage-per-onset trace), gated on the abstract state: the same face's
first two layers complete before AND after.

**Result: seed 0 +0 verified, seed 1 −1.** The mechanism is not subtle and
was measured directly rather than inferred: **the gate never admitted a
single hypothesis on 8/12 and 9/12 sessions**, despite 57–513 candidates
being available on each. Left to right, the decoder never holds a
hypothesis whose first two layers are right, because a median 60% of the
true story's cost is spent *before* the last layer begins:

| session | cost before LL | cost in LL | % before |
|---|---|---|---|
| `105530_solve` | 6.1 | 0.0 | 100% |
| `100120_solve` | 8.9 | 4.3 | 67% |
| `111941_solve` | 20.8 | 16.2 | 56% |
| `113054_solve` | 12.9 | 26.9 | 32% |
| `221809_solve` | 127.0 | 75.0 | 63% |

This is the §6 reach ceiling in a new form. The recogniser needed the
algorithm read cleanly; the forward transition needs *everything before the
algorithm* read cleanly. **Do not revive the forward arm.**

### 8b. Backward-first — the architecture that works

Proposed by the user on seeing 8a: run the algorithm finder FIRST, from the
end. The final state is known exactly, so peel the last algorithm off the
tail, undo it, peel the one before, and continue while the abstract state
stays inside the last layer. What falls out is a concrete cube state at the
F2L boundary — a mid-solve anchor — and the ordinary decoder then runs
`start -> anchor` over roughly half the onsets with BOTH endpoints pinned.
Needs no correct prefix, which is exactly what 8a lacked.

`peel_backward` + `decode_backward_first` in `algorithm_gate.py`. Three
backward transitions, all required to keep the abstract state in the last
layer: peel a library algorithm; peel one turn *of the face whose F2L is
complete* (AUF and regripping — at most 2 options, not 12, which is what
stops the peel becoming a second general decoder); delete a phantom.

| | seed 0 | seed 1 |
|---|---|---|
| verified, baseline | 4/12 | 4/12 |
| verified, backward | **5/12** | 4/12 |
| regressions | none | none |
| post-decode accuracy vs baseline | 88.5% vs 88.3% | 88.3% vs 88.1% |

The seed-0 gain is `solve_20260724_100120_solve` — in `CLASSIFIER_UNSEEN`,
the honest subset — where the peel identified 4 algorithms, emitted 54
moves, anchored at onset 82, and produced a **100.0% exact** reconstruction
from a 98.5% raw sequence. Real, but **unreplicated**: seed 1 gains nothing.

### 8c. Post-decode accuracy — the metric that actually answers "can we ship"

Verification is a cliff and moves rarely; the graded question is how close
the decoded word gets to what was performed. `decode()` now always returns
`best_effort_moves` (the MAP story over all onsets, whether or not it
reaches solved) so failing sessions are scorable at all.

**The pooled mean is bimodal and quoting it is misleading.** Split by
capture time, 12 held-out sessions per seed:

| regime | n | raw | baseline decode | backward |
|---|---|---|---|---|
| daytime (<18h), seed 0 | 8 | 94.7% | 95.9% | **96.1%** |
| daytime (<18h), seed 1 | 8 | 94.8% | 95.5% | **95.6%** |
| evening (>=18h), seed 0 | 4 | 73.0% | 73.3% | 73.3% |
| evening (>=18h), seed 1 | 4 | 73.3% | 73.4% | 73.6% |

**The >95% bar is met in daytime on both seeds and missed by 22 points in
the evening.** Two further readings matter:

* The decode contributes ~+1.2 points in daytime and ~+0.3 in the evening.
  Where the observation is bad enough, group theory cannot rescue it —
  which is the information-budget argument showing up as a measurement.
* Closing the evening gap is a MODEL problem. No decoder change touches it,
  and the whole algorithm-prior line moved the daytime figure by 0.2 points.
  See `[[evening-lighting-gap]]` and `[[widened-augmentation-result]]`.

### 8d. Two bugs worth not repeating

**Cost independent of a free parameter.** `ctc_logp_local` skips leading and
trailing frames for free, so candidate cost did not depend on `k`, every
onset-count tied at exactly `c_algo`, and the tie broke arbitrarily — an
8-move word was selected at `k=11`, swallowing three real onsets and
declaring them silent. That corrupts the reference story the ranking ladder
is built from and cost a session that previously verified. Fixed by pricing
the surplus onsets as what they are: phantoms, at `C_DEL` each.

**Scoring a truncated word.** `decode_backward_first` overwrote `moves` with
prefix + peel but left `best_effort_moves` as the prefix-only word inherited
from the sub-decode. That scored an exactly-correct reconstruction at 59.7%
(80 of 134 moves compared against all 134) and read convincingly as a false
verification — the precise failure the falsifiability work exists to catch.
Any derived field must be updated wherever its source is.

### 8e. Still outstanding

* Decoy sweep against the BACKWARD arm. Run against the forward arm only
  (near-miss decoys 3/4 -> 2/4, never-scrambled 0/4 both arms, on 4
  sessions). A pinned mid-solve anchor changes what VERIFIED means far more
  than the forward transition did — it hands the decoder for free the kind
  of anchor the scan-based proposal in `PATH_TO_VERIFICATION.md` §8 was
  rejected for. **The seed-0 gain must not be quoted as a verification
  result until this is run.**
* The peel is all-or-nothing across 3–6 algorithms at ~94% per-chunk
  recall, so one unidentifiable algorithm blocks the whole chain
  (`111941_solve`, `113054_solve` expose no anchor although the true one
  exists). Letting it skip one link for a generic cost is the obvious fix.
* `peel_backward` should not run on scramble takes at all — there is no last
  layer, and it wastes anchors finding near-free AUF peels.

---

## 9. Why the peel is inert — diagnosed 2026-08-02 (`peel_diag.py`)

§8e guessed the backward peel underperforms because "one unidentifiable
algorithm blocks the whole chain". That guess was checked and is **only
one third of the story, and not the largest third.** `peel_diag.py` reads
the cached posteriorgrams (no model load) and reports, per session,
whether the true F2L-boundary anchor is generated, at what rank, and where
the backward search loses it.

The symptom in `algo6_s0/s1.json`: the peel exposes a real anchor in **1
of 24 session-runs**; everywhere else it stops 1–2 onsets from the end
having peeled nothing, despite 57–513 candidates being available.

### 9a. Generation is not the bottleneck

On the 7 analysable held-out sessions (seed 0), the true anchor is
**generated in 4**. The peel is not timid — it takes over 1,800 algorithm
transitions per step at full frontier and reaches depth 5–7. So the
mechanism is not "the search never tries".

| session | true anchor | rank | in shortlist |
|---|---|---|---|
| `105530_solve` | found, onset 69 | 44 / 4437 | yes |
| `100120_solve` | found, onset 70 | 376 / 11258 | yes |
| `221809_solve` | found, onset 61 | **8211 / 17841** | **no** |
| `113054_solve` | found | — | yes |
| `111941_solve` | **never generated** | — | — |
| `211018_solve` | never generated | — | — |
| `213559_solve` | never generated | — | — |

Three distinct failure modes, which need three different fixes and had
been conflated into one:

1. **Selection (the big one).** `decode_backward_first` only uses an
   anchor whose prefix decode *verifies*. When none does it falls through
   to a second loop that takes the first shortlisted anchor — always the
   zero-cost do-nothing one — so the peel's entire output is discarded.
   8 of 12 held-out sessions per seed take that path, including two where
   the true anchor was sitting in the shortlist.
2. **Library gap at the tail.** Held-out chunk coverage is **97%** (33 of
   34 executions are in the 13-word mined library), so coverage is *not*
   a systemic problem — but the one missing word is `111941_solve`'s
   **final** chunk, and the peel chains backward from solved, so a single
   break at the tail is fatal for the whole session. At ~5.5 chunks a
   solve, 97% per-chunk coverage is only ~85% per-session.
3. **Identification failure in poor light.** `211018` / `213559` are the
   evening takes; every chunk is in the library and the peel still misses
   the anchor, wandering to onset 5 of 55 on `211018`. This is the
   `[[evening-lighting-gap]]` model problem reaching the decoder, and no
   decoder change addresses it.

### 9b. Cost cannot arbitrate the best-effort path — structurally

Worth stating separately because it is the reason (1) is not a one-line
fix. On the verified path, two candidate stories both explain every onset
and both reach solved, so their costs are comparable and cheapest-wins is
sound. A **best-effort** story is under no such obligation: it accepts the
argmax wherever it likes, which costs 0, and is never charged for being
wrong. So the do-nothing anchor is always the cheapest, and any
cost-ranked selection re-derives the current (broken) behaviour.

This is `shortlist_anchors`' documented problem one level up — it fixed
the *ordering* of anchors by trading depth against cost on a Pareto
frontier, but the final selection in `decode_backward_first` still uses
raw total cost, which is sound only on the verified path.

What can arbitrate is the peel's **own identification confidence**: the
evidence the verified path gets from reaching the anchor, which the
best-effort path does not have. `peel_backward` now carries `n_algo` and
`max_nll` (the worst per-move identification cost among an anchor's
links) for exactly this. The aggregate is a **max, not a mean**, because
the chain is all-or-nothing — every link must be the right word for the
anchor state to be right, so one bad link invalidates the anchor however
clean its neighbours are.

### 9c. The ceiling, measured before building the rule — and it is short

`peel_diag.py --oracle` decodes the prefix for **every** shortlisted
anchor and scores each implied reconstruction against ground truth. That
is the ceiling any selection rule can reach, with perfect hindsight, over
the anchors the peel currently surfaces. Seed 0, 12 held-out sessions:

| regime | n | base | **oracle ceiling** | headroom |
|---|---|---|---|---|
| daytime (<18h) | 8 | 95.88% | **96.53%** | +0.65 |
| evening (>=18h) | 4 | 73.28% | **76.38%** | +3.10 |

**Anchor selection cannot reach the 97% daytime ship gate even played
perfectly.** Only 2 of 8 daytime sessions have any headroom at all
(`113054_solve` +3.7, `100120_solve` +1.5); the other six are exactly
zero, including the 07-21/07-22 takes the mined library barely matches.
The headroom is ~5x larger in the evening — i.e. concentrated in the
regime whose fix is data, not decoding.

And the obvious rule is much worse than nothing:

| rule | pooled acc |
|---|---|
| do-nothing (current behaviour) | 88.3% |
| cheapest total cost | 87.0% |
| **deepest anchor with >=1 algorithm** | **81.9%** |
| oracle | 89.8% |

"Deepest with an algorithm" wins on 3 sessions and loses on 5, worst
`211018_solve` 57.1% -> 44.0%. So the realistic gain sits between roughly
zero and +0.65, and a threshold tuned to capture it would be fitted to
two daytime sessions. `TRUST_NLL` is left at a deliberately conservative
0.5, where it fired on 0 of the sessions tested — the arm is present and
measurable, not deployed on the strength of a hoped-for gain.

### 9d. The rule was built, deployed as an arm, and it LOSES

§9c said the realistic gain sits between roughly zero and +0.65. It is
below zero, **on both seeds**. `algo_sweep.py` now carries the rule as a
fourth arm (`acc_trust`, `results/2026-08-02/algo7_s0.json` / `results/2026-08-02/algo7_s1.json`), sharing every
decode with the backward arm so the column isolates the selection rule and
nothing else. 12 held-out sessions per seed:

| regime | raw | baseline | forward | backward | **trusted** |
|---|---|---|---|---|---|
| daytime s0 (n=8) | 94.7% | 95.9% | 95.9% | **96.1%** | **95.5%** |
| daytime s1 (n=8) | 94.8% | 95.5% | 95.1% | **95.6%** | **95.3%** |
| evening s0 (n=4) | 73.0% | 73.3% | 73.3% | 73.3% | 73.5% |
| evening s1 (n=4) | 73.3% | 73.4% | 73.4% | 73.6% | 73.6% |

**Daytime −0.6 / −0.3 against the arm it replaces; no seed gains.** And
the harm reproduces on the same session: `solve_20260721_102711` loses
**−5.2** (seed 0) and **−4.6** (seed 1). It is one of the six daytime
sessions §9c measured at *exactly zero* oracle headroom — every anchor
available there is at best neutral, so committing to one could only lose.
Seed 1 also shows the upside is real but small and inconsistent
(`113054_solve` +2.3, a session seed 0's threshold never fired on).

The rule had no way to know which case it was in, which is the point:
identification confidence says a peel is *probably the right word*, not
that using it beats what the plain decode already had.

`TRUST_NLL` is therefore **None (off) by default**. The machinery stays
because it is the honest harness for any future selection rule — pass a
value and `best_effort_moves_trusted` is computed and scored — not because
a better threshold is expected. Anything above ~0.4 fires on `102711`.

**What this measures, stated precisely — it is narrower than it looks.**
The oracle is the ceiling of **selection**, given current **generation**:
it picks the best of the ~40 anchors the peel actually surfaces. Two
things it therefore does NOT bound, and neither should be reported as
closed:

* **Anchors never generated.** `221809_solve`'s true anchor exists at rank
  8211 and is not shortlisted, so its 75.7% ceiling is an understatement
  (evening, so low priority).
* **The library gap.** `111941_solve` reads as "zero headroom" only
  because its final chunk is missing from the library, so no good anchor
  is ever built (§9a). Its true last layer is 52 of 106 moves; recovering
  it would move that session from 92.5% toward ~96–100%, i.e. **+0.4 to
  +0.9 on the 8-session daytime mean** on its own. That is the same order
  as the entire selection ceiling, from one session, and it is why the
  skip-one-link work in `GAMEPLAN.md` D1 is still worth doing.

So the honest summary is: **selection is close to exhausted (+0.65);
generation is not, and generation is where D1 points.** Realistic combined
best case is ~97.1% daytime — at the gate, not comfortably past it — and
that still leaves evening at ~76% where only data helps. With §6
(recogniser: reach ceiling) and §8a (forward transition: gate never
admits) already closed, D1 is the last decode-side idea in this line with
an unmeasured ceiling. Everything after it is model and data — see
`GAMEPLAN.md` §5 and `[[evening-lighting-gap]]`.

Related: `README.md` (this folder) for the pipeline and its measured
numbers; `reconstruct.py`'s module docstring for the cost model and the
measured decode envelope; `algorithm_gate.py` for §7–8; `algo_sweep.py` for
the four-arm harness; `peel_diag.py` for §9 and for the anchor-selection
ceiling (`--oracle`).
