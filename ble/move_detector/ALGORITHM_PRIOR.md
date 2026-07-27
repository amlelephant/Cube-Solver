# Algorithm-aware decoding — design plan

Status: **step 1 built and RUN. It did not pass — see §6. Steps 2-5 are on
hold.** Written 2026-07-25, outcome added the same day.

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

Related: `README.md` (this folder) for the pipeline and its measured
numbers; `reconstruct.py`'s module docstring for the cost model and the
measured decode envelope.
