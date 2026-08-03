# GAMEPLAN — ship threshold, decode rework, lighting (2026-08-02)

The live executable plan. Supersedes `MODEL_REWORK_PLAN.md` as the "what
next" doc (that doc's phases are done: the joint model, CTC, and the
algorithm gate all shipped). Written for a cheaper model to execute:
every step names its files, commands, and an accept/reject gate. Read
`ALGORITHM_PRIOR.md` §6–8 and `ACCURACY_TARGET.md` before touching the
decoder — they are the record of what is already measured dead.

Roles, fixed by the user and not up for re-litigation:

* **Decode = accuracy, not verification.** The decode's job is to make the
  recorded move list as close to what was performed as possible. A solve
  being *legitimate* is the anticheat's job (separate work stream).
* **Ship metric = post-decode word accuracy vs BLE truth, on held-out
  sessions, split by lighting regime, never pooled** (the pooled mean is
  bimodal and hides the one-dimensional lighting split — see
  `results/2026-08-01/algo6_s0.json` / `results/2026-08-02/algo6_s1.json`).

---

## 1. Where we are (all measured, 12 held-out sessions × 2 CTC seeds)

| regime | raw | post-decode (backward arm) |
|---|---|---|
| daytime, seed 0 | 94.7% | **96.1%** |
| daytime, seed 1 | 94.8% | **95.6%** |
| evening, seed 0 | 73.0% | 73.3% |
| evening, seed 1 | 73.3% | 73.6% |

The decode currently contributes ~+1.2 points daytime, ~+0.3 evening.
The entire algorithm-prior line (forward + backward arms) moved daytime
by 0.2 points.

### 1b. A third regime axis: solve SPEED (measured 2026-08-02)

Time-of-day and recording-day were the known axes. There is a third, and
it was drifting unmonitored: **the solver has roughly doubled in speed in
12 days** — 1.30 moves/s on 07-21 to 2.83 on 08-02, with adjacent-onset
crowding rising from 0.9% to ~18%.

| group | onsets | adj pairs ≤2 frames | **same-class ≤2 frames** |
|---|---|---|---|
| training (27 solves) | 3312 | 5.9% | 0.57% |
| held-out eval | 582 | **18.4%** | **1.55%** |
| new 08-02 | 346 | 17.9% | 2.60% |

Three things follow, and the third changes how §1's table reads:

* **The training corpus is stale**, not merely small — it is weighted
  toward a regime the user no longer solves in, and today's 3 solves are
  the first training data at the holdout's speed. **But do not promote
  that into a predicted accuracy gain.** Tested directly: crowding vs
  post-decode accuracy correlates −0.84 across all 12 held-out sessions,
  which looks decisive and is **confounded** — the three worst sessions
  are evening *and* crowded. Within daytime alone it is r = −0.45,
  **p = 0.26 (n=8)**, and the pattern is contradicted session by session:
  the two *least* crowded daytime solves (`102711` 0.7%, `101225` 1.7%)
  score 91.4% and 94.4%, among the worst, while the 8.3%-crowded
  `100120_solve` scores 100%. Crowding does not explain the daytime
  deficit. The distribution shift is a fact about the corpus; its effect
  on accuracy is unproven.
* **The cross-day gap is confounded.** The proximity ladder in
  [[holdout-proximity-ladder]] reads its decline as day/environment
  distance, but the more distant sessions are also the faster, more
  crowded ones. Do not attribute the whole gap to generalization.
* **There is a structural ceiling below 100%, and it is CTC-specific.**
  `ACCURACY_TARGET.md` §5.2's 4.47%/2.23% collision floor is a
  PEAK-PICKING result and does not transfer. CTC's only structural miss is
  two adjacent onsets of the **same class** within ~2 frames, which it
  cannot emit without an intervening blank.

  **Compute this the way the ship metric is computed or it is wrong.**
  Pooled over the 8 never-trained sessions the floor is 1.55% (ceiling
  98.4%) — but the ship metric is a MEAN OF PER-SESSION accuracies over
  the sweep's 12 sessions, which include the four slower, less crowded val
  takes. On that footing:

  | regime | mean per-session CTC ceiling | current | gap |
  |---|---|---|---|
  | daytime (n=8) | **99.2%** | 96.1% | 3.1 pts |
  | evening (n=4) | 99.1% | 73.3% | 25.8 pts |

  So the 97% target sits **2.2 points below the structural ceiling** —
  demanding, but the floor is *not* the binding constraint at 97%, and an
  earlier draft of this section wrongly said it nearly was (it compared a
  pooled onset-weighted floor against a mean-of-sessions metric, over two
  different session sets). Per session the crowded takes are the tight
  ones: `113054_solve` ceiling 97.7% scoring 90.9%, `111941_solve` 98.1%
  scoring 92.5%.

  At 30fps and rising solve speed that ceiling does fall — today's fastest
  take, `124134_solve`, already carries a 4.46% same-class floor (ceiling
  95.5%). Capture frame rate is worth costing before it binds.

## 2. Ship threshold — the recommendation

**Adopt 97% as the daytime target, but do not make it the ship blocker.
The ship gate should be a bundle:**

1. **Daytime held-out post-decode mean ≥ 95%** — already met on both
   seeds (96.1 / 95.6). This is the bar the product stands on.
2. **Per-session floor ≥ 90%** on daytime held-out — a mean can hide one
   catastrophic session; a user only experiences their own solve. Check
   the per-session rows in the sweep JSON, don't add new machinery.
3. **A capture-time lighting check in the product** (§5.3). Measured:
   adding a lamp is worth +21 points end-to-end for free. This is the
   single highest-leverage "feature" available and it costs a UI prompt.
4. **Evening is not a ship blocker** provided (3) exists and the §5 data
   work is scheduled. Ship v1 as "works in reasonable light, tells you
   when light is not reasonable."

**Update 2026-08-02:** this recommendation is now load-bearing rather
than cautious. The decode rework in §4 was measured and its *ceiling* is
~96.5% daytime — so a hard 97% blocker would block ship on work that no
decoder change can deliver. 97% stays the target; §5's evening data is
the route to it.

Why 97% should be the target but not the blocker:

* Between 95% and 97% nothing changes *downstream* — the verification
  cliff sits at ~98–99% raw (`ACCURACY_TARGET.md` §4.1) and verification
  is the anticheat's job now anyway. The difference a user sees is ~5
  wrong moves vs ~3.5 in a 120-move solve. Real, but not ship-or-don't.
* The distance from 96.1 to 97 is small enough to be a *next milestone*
  rather than a blocker — but §4's measurement narrowed how it can be
  closed. Anchor work tops out at +0.65 (§4's result box), so the route is
  §5's data plus D4's adaptive LM fusion, not the decoder rework this doc
  originally led with.
* Chasing past 97 on the current corpus is chasing the miss-channel
  structural floor and the info budget — diminishing and known.

**Quote rules for any future ship-gate number** (these have each burned a
session's work when violated): held-out sessions only, both seeds, split
by regime, raw alongside post-decode so the decode's own contribution is
visible, and re-run the baseline fresh in the same sitting
([[feedback-reverify-cited-baselines]]).

Note on "91% end to end live": that number and the 96% above are
different metrics (live word-alignment scoring vs replay post-decode; the
live scorer also misattributes detector errors — see
[[live-scoring-misattributes]]). Before ship, run one live session and
its replay through the SAME scorer to confirm the two agree; do not mix
them in one sentence otherwise.

---

## 3. Why the decode adds only ~1 point — the mechanism, so the fix makes sense

The user's read is correct: the decode is all-or-nothing. Mechanically:

* The endpoint constraint is **global and terminal**. It discriminates
  nothing until the last onset; a session that can't complete a full
  consistent path to solved gets only generic MAP smoothing (~+0.2).
  All of the decode's power is concentrated in sessions that were nearly
  right already.
* The error mass is ~60% misses / 21% phantoms / 19% substitutions.
  Substitutions are the one channel the decoder fixes nearly for free
  (the group constraint pins down *which* move a mis-named onset was) —
  and they're the smallest channel. A **miss** forces the decoder to
  invent a move with no posterior evidence: a 12-way guess whose only
  support is an endpoint 80 moves away. That is why "small individual
  errors that should be incredibly easy to fix" aren't: they are easy to
  fix **exactly when there is a nearby anchor**, and today the only
  anchor is the end of the solve.
* The algorithm work specifically: the forward arm was measured **inert**
  (the F2L gate never admits a hypothesis, because a median 60% of the
  true story's cost is spent before the last layer begins —
  `ALGORITHM_PRIOR.md` §8a). The backward peel works but is
  **all-or-nothing across 3–6 chunks at ~94% per-chunk identification**:
  one unidentifiable chunk and the whole anchor is lost (0.94⁴ ≈ 78%
  chain success, worse with more chunks). So "+1 point" is not an
  implementation failure — it is the measured shape of a decoder with
  exactly one anchor.

**Therefore the rework is one idea: raise anchor density.** Every anchor
converts a 120-onset global problem into short segments where the
constraint arrives every 15–40 onsets — the regime where the beam is
known to recover ~6 errors reliably (`reconstruct.py` measured limits).

## 4. Decode rework — implementation steps, each gated

> **RESULT, 2026-08-02 — read before executing any of §4.** The core
> premise of D1/D2 was tested and the decode path **cannot reach the 97%
> daytime target**. `peel_diag.py --oracle` scores every anchor the peel
> surfaces against ground truth, which is the ceiling any selection rule
> can reach with perfect hindsight:
>
> | regime | n | base | oracle ceiling | headroom |
> |---|---|---|---|---|
> | daytime | 8 | 95.88% | **96.53%** | +0.65 |
> | evening | 4 | 73.28% | **76.38%** | +3.10 |
>
> Six of eight daytime sessions have *exactly zero* anchor headroom, and
> the obvious selection rule ("deepest anchor with an algorithm") is net
> **−6.4 points**. So D2 is worth at most +0.65 and realistically less,
> and the headroom is 5x larger in the evening — the regime §5 addresses.
>
> **Read the scope of that number carefully.** It bounds anchor
> *selection*, given current anchor *generation*. It does **not** bound
> D1: `111941_solve` shows as zero-headroom only because its final chunk
> is missing from the library, so no good anchor is ever built. Its last
> layer is 52 of 106 moves, and recovering it moves that one session from
> 92.5% toward ~96–100% — **+0.4 to +0.9 on the daytime mean by itself**,
> the same order as the whole selection ceiling.
>
> So: **selection is nearly exhausted; generation is not.** Do D1. The
> realistic combined best case is ~97.1% daytime, which is *at* the gate
> rather than past it, and leaves evening at ~76% where only §5 helps.
> Full record: `ALGORITHM_PRIOR.md` §9.

Standard harness for every step: `cd ble/move_detector`, then

    python algo_sweep.py --ctc checkpoints/move_ctc_s0.pt
    python algo_sweep.py --ctc checkpoints/move_ctc_s1.pt --out results/2026-08-01/algo_sweep_s1.json

both seeds, same sitting as the baseline, compare post-decode accuracy
split by regime. No step ships if either seed regresses daytime by more
than 0.3 (the observed seed-noise scale on this metric).

### D1 — harden the backward peel

**Superseded in part by measurement (2026-08-02).** `peel_diag.py` was
built to check §8e's "one unidentifiable algorithm blocks the chain"
diagnosis before implementing against it, and the diagnosis was wrong in
its particulars. See `ALGORITHM_PRIOR.md` §9 for the full record. What
the peel actually does on the 7 analysable held-out sessions (seed 0):

* it **generates the true anchor in 4 of 7** — generation is not the
  bottleneck, and `113054_solve`'s anchor is in the shortlist already;
* the peel's output is then **discarded** on every session that does not
  verify (8 of 12 per seed), because the fallback takes the cheapest
  anchor, which is the do-nothing one. **This is the largest single
  defect and it is a selection bug, not a search one** — hence D2 below
  is now the first thing to do, not the second;
* `111941_solve` fails for a different reason: its **final** chunk is the
  one held-out word missing from the mined library (coverage is 97%,
  33/34 executions). Because the peel chains backward from solved, one
  missing word at the tail is fatal for the whole session;
* `211018` / `213559` are the evening takes — every chunk is in the
  library and identification still fails. That is
  `[[evening-lighting-gap]]` reaching the decoder; §5 is the fix, not D1.

So the remaining D1 work, in order:

* **The library gap has no cheap decoder fix. Two candidate fixes were
  reasoned through and both fail; neither should be built.**

  *A generic skip* — let the peel skip one unidentified span by peeling
  ordinary quarter turns with the abstract gate suspended, closing when
  F2L is restored. **No:** an anchor built that way is correct only if the
  skipped span was *read* correctly, which is exactly what failed. F2L
  restoration prunes some wrong readings, but the last-layer subgroup has
  ~62k elements so many survive — and unlike a library peel they carry
  **no identification confidence**, so nothing can rank them. That is
  §9b's unrankable-anchor problem reintroduced without the one signal that
  solves it.

  *A public OLL/PLL set* — **also no, and the reason is structural enough
  to be worth knowing.** `build_candidates` scores a library entry by
  `ctc_logp_local(window, labels)`: it matches the **executed word**, not
  the resulting permutation. Checked on the actual missing entry —
  `L' U' L F L' U' L U L F' L2 U L` produces a G-perm permutation (3
  corners and 3 edges cycled, orientation preserved) but is **not** the
  textbook Ga or Gb, on either side of any AUF. Adding textbook algorithms
  adds *words this solver never performs*, so it would not have fixed
  `111941_solve`, while inflating the candidate set and with it the null
  risk `algorithm_gate.py` currently controls.

  **The library is therefore user-specific by construction.** What
  actually closes the gap is more of *this* solver's own sessions — which
  §5's recording work supplies anyway — and, for a new user, an enrollment
  phase that mines their repertoire before the feature is trusted. That is
  a product decision, and it is the honest answer to §7d's untested "one
  person's repertoire" risk. **It is not a decoder change**, which means
  D1 is not a quick win either.
* **Never peel scramble takes.** Currently inert rather than harmful
  (scramble takes generate 0 candidates, so `n_algo` is always 0 and no
  trusted anchor can fire), but it still pays up to 40 prefix decodes per
  scramble session for nothing. Short-circuit when `cands` is empty.

### D2 — use the peel when nothing verifies (**do this first**)

Promoted ahead of D1 by the §9 measurement: the peel already finds good
anchors and then throws them away. Implemented 2026-08-02 as a fourth arm
in `algo_sweep.py` (`acc_trust`), sharing every decode with the backward
arm and differing only in which best-effort story is reported:

* `peel_backward` now carries `n_algo` and `max_nll` per anchor — the
  WORST per-link identification cost, a max not a mean because the chain
  is all-or-nothing (one bad link invalidates the anchor state however
  clean its neighbours are);
* when nothing verifies, `_best_effort_anchor` takes the **deepest anchor
  whose every link identified below `TRUST_NLL`**, falling back to the
  previous story when none qualifies;
* **cost cannot be the selection rule here, structurally** — a
  best-effort story accepts the argmax at cost 0 and is never charged for
  being wrong, so the do-nothing anchor is always cheapest. §9b.

**RESULT: it fails its own gate. Do not ship it; it is off by default.**
Seed 0, 12 held-out sessions (`results/2026-08-02/algo7_s0.json`), daytime **96.1% → 95.5%**.
It fired twice: +0.9 on an evening take, **−5.2 on `102711`**
(91.4% → 86.2%) — one of the six daytime sessions with *zero* oracle
headroom, where any committed anchor can only lose. `TRUST_NLL = None`.

The machinery is kept as the measurement harness (pass `--trust-nll` to
re-measure), and the lesson generalises: identification confidence says a
peeled span is *probably the right word*; it does not say that using it
beats what the plain decode already had. A selection rule needs the
second thing and nothing in the peel provides it.

This arm cannot affect verification or the decoy sweep at all — it only
ever rewrites a field carrying `solved: False` — so no falsifiability
re-run was required.

### D3 — segmented decode (the larger change)

Every accepted last-layer chunk already implies TWO concrete cube states
(before and after the chunk — the peel computes them). Today only the
deepest one is used, as a single anchor. Instead:

* Build the full anchor chain from the peel: `start → A1 → A2 → … → solved`.
* Run `reconstruct.decode()` **independently per segment** with pinned
  endpoints (the machinery exists: `decode_between`). A segment that
  fails to connect falls back to its own best-effort MAP — **it no longer
  zeroes the whole solve**. Concatenate `best_effort_moves` across
  segments in order.
* Watch for the two §8d bug shapes while wiring: any cost must actually
  depend on every parameter selected over, and every derived field
  (`best_effort_moves`) must be rebuilt wherever its source changes.

Accept gate: daytime held-out post-decode, both seeds. Expectation to
beat: decode delta grows from +1.2; the sessions with 3+ identified
chunks should move most. Report per-session so the mechanism is visible.

**Fencing rule (this resolves §8e's open decoy worry):** segmented
output feeds the **accuracy tier only**. `VERIFIED` verdicts must never
be emitted from a segment-anchored decode until the decoy sweep is
re-run against it — an anchor handed to the decoder for free is exactly
the thing the falsifiability work exists to distrust. Keep the
verification path (anticheat) on the un-anchored decoder unchanged. This
is cheap to enforce: a flag on the result dict, asserted where verdicts
are printed.

### D3 — graded output schema

Replace the binary shape of the result with per-segment status:
`exact` / `repaired (n edits)` / `best-effort`, plus overall post-decode
accuracy. This is what the paying user sees per solve, and it makes the
decode's contribution measurable per segment instead of per solve. Pure
plumbing; no gate beyond "the numbers reconcile with D2's sweep output."

### D4 — adaptive LM fusion (optional, after D2)

`move_lm.py` fusion helps consistently cross-day (MER 17.3→9.8 / 17.1→12.2
by seed) and flips sign same-day because a single global `beta` cannot
serve both regimes ([[lm-fusion-regime-dependent]]). Make `beta` adaptive:
run the greedy CTC decode first, read the del/ins balance
(deletion-dominated → large beta, balanced → small), set beta from that,
then run the fused beam. Judge on cross-day held-outs, both seeds, never
on the same-day val. Accept gate: daytime post-decode up on both seeds,
evening not down.

### Do-not list (all measured dead; do not re-attempt)

* Forward `OP_ALGO` beam transition — gate never admits (§8a).
* Recogniser-style algorithm matching on the predicted string — reach
  ceiling is structural (§6).
* Onset threshold tuning — deployed threshold already near-optimal
  (`ACCURACY_TARGET.md` §5.1).
* Diff-baseline/noise matching for evening — measured flat twice.
* Statistical OOD lighting gates from image stats — measured
  anti-correlated; the clock is the only working predictor.
* Ranking decoder variants on MER alone — MER and downstream quality
  disagree in sign ([[lm-fusion-regime-dependent]]).

---

## 4b. Checkpoint lineage and the permanent holdout (2026-08-02)

Three CTC generations now coexist and they differ in **two** dimensions.
Comparing across the wrong pair silently confounds augmentation with data,
which is how this project has drawn wrong conclusions before:

| checkpoint | augmentation | train / val | notes |
|---|---|---|---|
| `move_ctc_s0/s1.pt` | narrow (pre 07-31) | 38 / 4 | the deployed default |
| `move_ctc_aug_s0/s1.pt` | **widened** `AUG_*` | 38 / 4 | same split, aug only |
| `move_ctc_aug44_s0/s1.pt` | widened | **44** / 4 | adds the 6 08-02 takes |

`dataset.py` now holds the widened ranges, so **any fresh training run is
augmented** — which means the controlled baseline for `aug44` is `aug`,
never `move_ctc_s0`. Comparing `aug44` to `move_ctc_s0` changes data and
augmentation at once and answers nothing.

**The permanent held-out set — 8 sessions, never trained on by anything:**

    solve_20260729_221809_scramble / _solve      evening
    solve_20260730_111941_scramble / _solve      daytime
    solve_20260730_113054_scramble / _solve      daytime
    solve_20260731_211018_solve                  evening
    solve_20260731_213559_solve                  evening (lamp added)

Enforced by name in `make_session_list.py`, not by a glob, because a glob
over `solve_*` sweeps them in and every "held out" number in this repo —
`algo6/7_s*.json`, `ALGORITHM_PRIOR` §9, the ship-gate table in §1 — is
denominated in these sessions. Training on them converts all of it into
memorisation numbers at once ([[named-holdouts-cross-env]]).

The 4 validation sessions (`102711`, `101225`, `105530_solve`,
`100120_solve`) are unchanged across all three generations *on purpose* —
that is what makes the comparison controlled. They are same-day-adjacent
and therefore flattering ([[holdout-proximity-ladder]]); val MER is an
early-stopping signal, not a result. Report on the 8 above.

Retrain recipe:

    python prepare_data.py --sessions ../training_data/solve_<new>*/ --color
    python make_session_list.py
    python train_ctc.py --sessions $(cat train_sessions.txt | tr "\n" " ") \
        --val-session-names solve_20260721_102711 solve_20260722_101225 \
            solve_20260723_105530_solve solve_20260724_100120_solve \
        --seed 0 --output move_ctc_<tag>_s0.pt

Then evaluate old-vs-new on identical input, cheap first:

    python eval_lighting.py --models checkpoints/move_ctc_aug_s0.pt checkpoints/move_ctc_aug_s1.pt \
        checkpoints/move_ctc_aug44_s0.pt checkpoints/move_ctc_aug44_s1.pt --out results/2026-08-02/lighting_eval_v2.json

`eval_lighting.py` excludes the union of every passed checkpoint's
sessions, so passing old and new together lands exactly on the 8 above.
Only if that moves is the hour-per-seed `algo_sweep.py` worth running for
post-decode ship numbers.

Sessions with no usable frames — the 11 `solve_20260720_14*` takes have
`frames=0` — are not a preparation failure to chase; there is nothing in
them to prepare.

---

## 5. Lighting — the answer is data, not a pixel transform

**Recommendation: record real evening sessions. Do not build a pixel
transform.** The evidence is already in:

* Every surgical pixel-level intervention tried has measured **flat**:
  per-frame Gaussian noise to restore the diff-luma floor (56.0/57.1/
  56.0/54.8% across sigmas), exposure/colour/crop all ruled out as
  mechanisms individually. The mechanism is NOT isolated, and when the
  mechanism is unknown, a targeted transform is aimed at a guess.
* Widened photometric augmentation — the broad-spectrum *synthetic*
  version of "more evening data" — bought +5.2 evening for −3.0 morning,
  replicated on 2 seeds. Augmentation substitutes for light and stops
  helping once light is present (the lamp-lit take was flat). Real data
  in the regime is the same lever without paying the morning tax.
* The corpus is 87% daytime (54/62 sessions between 09:00–18:00). A
  22-point regime gap over a 6%-representation regime is the textbook
  data-hole signature, confirmed causal by the lamp intervention
  (+21 points, same room/cube/evening).

### 5.1 Recording plan (human task — can start tonight)

10–15 evening sessions across **at least 3 different evenings**, varying
the lamp: none / one lamp / overhead, so the new data spans the regime
rather than one point in it. Keep 3–4 sessions (one full evening, mixed
lamp states) OUT of training as the permanent evening holdout — from a
**different evening** than any training session, per
[[named-holdouts-cross-env]]. Use the existing
`record_training.py → postprocess_session.py` flow unchanged.

### 5.2 Retrain and evaluate

Retrain the CTC arm, 2 seeds, keeping the widened `dataset.AUG_*` ranges
(they help most exactly where light is worst, and evening data may not
cover every dark condition users have). Evaluate:

    python eval_lighting.py --models move_ctc_evening_s0.pt move_ctc_evening_s1.pt

against the held-out evening takes AND the morning holdouts — the accept
gate is two-sided: evening up substantially (expect to recover a large
share of the 22 points), morning within 1 point of current. Then re-run
the §4 sweep for the ship-gate numbers.

### 5.3 Product mitigation (ship regardless of 5.1/5.2)

A capture-time lighting prompt. Do NOT build it from image statistics
(measured not to work). Two signals that do work:

* Clock time — crude, honest, free (evening = suggest a lamp).
* Live onset-peak health: the measured signature of bad light is true
  onsets peaking weak (77% above fg 0.5 vs 98% in daylight). A rolling
  "weak-peak fraction" during the first seconds of capture is a direct
  readout of the actual failure channel — threshold it loosely and
  suggest more light. This is a readout of the model's own posteriorgram,
  not an image-stat gate, so the earlier negative result does not apply.
  Prototype in `lighting_check.py`; verify it separates the known takes
  (56% / 63% evening vs 90%+ morning) before wiring it into
  `verify_solve.py`.

---

## 6. Order of operations

**Revised 2026-08-02 after §4's ceiling measurement.** The original order
put the decode work first; it is now demoted, because its ceiling was
measured below the target and both of D1's candidate fixes were reasoned
out as non-viable (§4 D1). Data work is the critical path.

1. **This week (human, calendar-bound):** §5.1 evening recordings. Nothing
   model-side can start without them and they are worth more than every
   decoder change in §4 combined — the evening regime is 22 points down
   and has 5x the anchor headroom of daytime.
2. **Also human, and cheap:** more *daytime* sessions too. §4 D1 found the
   algorithm library is user-specific by construction (it matches executed
   words, not permutations), so library coverage — 97% of held-out chunks
   today, and one tail miss costs a whole session — improves only with
   more of this solver's own solves.
3. **Code, small and already built:** the D2 trusted-anchor arm is in
   `algo_sweep.py` as `acc_trust` and is measurable now. Ship it only if a
   two-seed sweep shows it non-negative; its ceiling is +0.65 daytime, so
   treat a flat result as the expected outcome, not a failure to debug.
4. **Code, worth doing on its own merits:** D3's graded per-segment output.
   It is what the paying user sees, and its value does not depend on any
   accuracy gain.
5. **When the evening corpus lands:** §5.2 retrain, re-measure the ship
   gate on both regimes with `algo_sweep.py`, both seeds.
6. **Ship** when §2's bundle holds: daytime ≥95 mean (met at 96.1), ≥90
   per-session floor, lighting prompt in place. 97 daytime is the first
   post-ship milestone and it arrives via step 1, not via §4.
