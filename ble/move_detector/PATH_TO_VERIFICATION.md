# Path to verification — decoder-first roadmap (rev 2)

> **SUPERSEDED for planning purposes (2026-07-28):** the §5 decoder
> sprint was executed and exhausted (see §0's sprint-result block). The
> live roadmap is now **`MODEL_REWORK_PLAN.md`** (rev 3) — model-side
> work: oracle attribution, RAFT-gated flow retest, and the
> frame-synchronous posteriorgram model (Path 4, scoped). This document
> remains the authoritative record of everything already tried and
> rejected — read §4 (do-NOT-reinvent inventory) and §8 (rejection
> records) before proposing anything.

Rev 2 written 2026-07-27, superseding rev 1's primary recommendation.
Purpose: a single document a model (or engineer) can pick up cold and
execute against. Rev 1 recommended scan-based mid-solve checkpoints
(Path 1); that was **rejected on review** — see §8 for the record — and
the primary lever is now a set of decoder-efficiency changes (§5) that
run entirely on already-recorded data. Read this fully before changing
anything; several "obvious" moves were already tried and the evidence is
cited so you don't repeat the experiment.

**Rule for whoever runs this plan**: never cite a baseline number from
this doc, from `README.md`, or from memory — re-measure it fresh, in the
same sitting as whatever you're comparing it to. Every number below has a
command next to it. A wrong "X didn't help" conclusion was already
shipped once from citing a stale number (see `ALGORITHM_PRIOR.md` and
the `feedback-reverify-cited-baselines` incident).

**Sprint result (2026-07-27, same sitting as this note — SUPERSEDES the
rest of this document's framing, read this first):** D1, D2, and D3 are
all implemented (`reconstruct.py`/`verify_solve.py`, behind flags that
default to exact prior behaviour, `--selftest`-verified) and were each
measured, independently, against the fresh baseline in §2/§3 below. **All
three are flat: honest subset stays at 1/6 verified, all-sessions stays
at 19/40, in every configuration tried** (D1 at `--candidate-threshold`
0.15, D2 at the measured-confusion `--blend-inv 0.093 --blend-adj 0.107`,
D3 both single-meet and `--meet-sweep`). It is the literal same one honest
session (`solve_20260726_165044_scramble`, a 22-move scramble-phase
take — not a solve) that verifies under every condition; none of the
other 5 honest sessions move.

**This is not an ambiguous or noisy null result — it's now explained
quantitatively.** `gt_path_cost` (the true story's cost under each
condition's own cost model) for the 5 stubborn honest sessions is
**36–119**, i.e. **9–30 insertion-units** (`gt_path_cost / C_INS`) beyond
the proven-working envelope of roughly 1 (see the "Measured limits" note
above). No beam width, cost recalibration, or search restructuring can
plausibly close a 9–30x gap — per §2's information-budget argument, these
sessions' raw accuracy (75.5–91.8% over 96–187 onsets) puts them
structurally outside what ANY decoder can identify, not just this one.
This is the §2 argument confirmed with real numbers, not just predicted.
Per D4's own gate below (§5), these are MODEL failures, not search
failures — **D4 (two-edit ladder) is not warranted and was not built.**

**Consequence for §6-8 below:** the "cheap decoder levers first" strategy
this document argued for has now been genuinely executed and exhausted,
not merely proposed. The honest verified rate is unmoved after FOUR
independent interventions total (three from this sprint, plus the prior
jitter-retrain — see `jitter-retrain-decoder-flat` memory). The
remaining levers are all on the model-accuracy or protocol side: Path 2
(incremental ML — same caveat as ever, unlikely to be enough alone),
Path 4 (architecture rework — its trigger condition in §8 is now
genuinely met, with a real baseline to beat: 1/6, and a quantified gap to
close, ~9-30x per session), or reopening the mid-solve-anchor question
with a mechanism that doesn't depend on the rejected full-scan approach
(§8's Path 1 record). **Not yet done and still mandatory before drawing
final conclusions:** re-running the falsifiability sweep (§7) under each
lever's cost-model changes — only spot-checked, not exhaustively verified,
before this sprint's time budget ran out. Do that first if picking this
back up.

---

## 0. TL;DR

- Honest generalization rate (classifier-unseen sessions, full pipeline,
  camera-only): **1/6 verified, 0/6 exact**, re-measured fresh 2026-07-27
  (`logs/2026-07-27/baseline_rev2_pre.log`) and reproduced identically across four
  independent conditions since (see the sprint-result block above). The
  all-sessions number (19/40) is inflated by sessions the classifier
  trained on — don't quote it.
- Four consecutive significant interventions (three model/cost-model
  changes below, plus the prior jitter retrain) moved move-level accuracy
  or cost-model calibration and left the verified rate flat every time.
  The binding constraint is the decoder's repair budget vs. the error
  mass a 60–150-move solve accumulates — see the information-budget
  analysis in §2, now confirmed with real per-session numbers (§0 above):
  the honest failures sit 9–30 insertion-units outside the envelope, not
  a close miss.
- Rev 1's answer (mid-solve scan checkpoints) is **rejected**: the 6-face
  scan is slow, manual, and itself error-prone (orange/red, lock-on), so
  chaining 5–7 of them per solve multiplies scan-failure exposure and
  wrecks UX. Record kept in §8.
- **Decoder levers (§5): four levers, ordered by cost-of-trying, ALL
  MEASURED AND REJECTED 2026-07-27** — (D1) feed sub-threshold detector peaks into the
  decoder as cheap-to-ignore candidates instead of blind flat-cost
  insertions; (D2) recalibrate the acceptance prior to the *measured*
  live confusion (adjacent-face), replacing the temporal-inverse-only
  blend; (D3) bidirectional meet-in-the-middle decoding, which roughly
  doubles the surviving cost budget at the same beam; (D4, contingent)
  extend the exact repair-lookahead ladder from one future edit to two.
  D1+D2 are cost-model changes (~days); D3 is a search change (~3–5
  days); all run on the existing 40 recorded sessions — no new data, no
  scanning, no retraining.
- These levers push the decoder *toward* its information-theoretic limit;
  they cannot pass it. If the honest verified rate is still far short of
  target after §5, the remaining levers are model accuracy (Path 2 /
  Path 4 in §8) — and that decision will finally rest on a decoder that
  is genuinely exhausted, which is the point.
- Still mandatory regardless of path: the statistically rigorous
  false-accept harness (§7). Any change to costs or search **must**
  re-run the falsifiability sweep — a decoder that corrects more
  aggressively is also, by construction, closer to accepting decoys.

---

## 1. What "verification" has to mean

Two-sided claim; every knob (beam, costs, thresholds) trades one side
against the other. Never report one side alone:

- **True-accept rate** (recall on honest solves) — currently ~17% honest.
- **False-accept rate** (security) — currently *anecdotal*: the
  falsifiability sweep has only run in depth on one session; 0/6 decoys
  bounds nothing (95% upper bound ≈ 39%). See §7.

**MVP target (straw man — confirm with the user, §11):** true-accept
≥ 90% on model-unseen sessions across ≥ 3 environments; false-accept
bounded below ~2% at 95% confidence (~150 decoy attempts, 0 accepted).
Literal 100% true-accept is not well-posed for real users (cube leaves
frame, off-camera fiddling); 90–95% with a sane flagged-review fallback
is the realistic bar.

---

## 2. The information budget — why 75% accuracy is unreachable and 90% is the knife edge

Back-of-envelope, but the conclusion is robust to the roughness.

The only hard constraints on a whole-solve decode are the two endpoint
states. The cube group has ~4.3×10¹⁹ states, so "the accepted sequence
must compose to scramble→solved" carries **log₂(4.3×10¹⁹) ≈ 65 bits**.
Each move is one of 12 quarter-turn classes (~3.6 bits), so the endpoint
constraint is worth roughly 16–18 moves' worth of information — no more,
no matter how clever the search.

Against that, the observation uncertainty at per-move accuracy *p* over
*n* moves is roughly n·[H(1−p) + (1−p)·log₂(a)] bits, where a ≈ 5
plausible confusions per error:

| regime | uncertainty | vs. 65-bit constraint | outcome |
|---|---|---|---|
| 75% acc, 100 moves | ~140 bits | 2× over budget | ~10²⁰ wrong sequences also hit the endpoints; decoys indistinguishable from truth. **No decoder fixes this.** |
| 90% acc, 100 moves | ~70 bits | ≈ balanced | knife edge — exactly where the pipeline sits, and why p=0.008 model gains flipped zero sessions |
| 90% acc, 20 moves | ~14 bits | 4× under budget | massively overdetermined — why short/clean sessions already verify reliably |

Three consequences to internalize before touching code:

1. **At 75% the failure isn't "search can't find the truth" — it's that
   the truth is no longer identifiable.** An oracle search would return
   ~10²⁰ candidates of comparable cost, and the decoy cost-separation
   (~4.0/quarter-turn today) collapses toward zero. Verification's
   security half dies before its usability half does.
2. **At the current ~90% regime, search quality and cost calibration are
   worth real bits.** Sitting at the knife edge means sessions are lost
   by small margins — a truth evicted from the beam, a miss priced at a
   flat 4.0 when the detector actually saw a sub-threshold peak. That is
   exactly what §5 attacks, and why it can move sessions that model
   accuracy couldn't.
3. **The levers in §5 approach the limit; they do not move it.** Only
   higher per-move accuracy (or more constraints — rejected §8) moves
   the limit itself.

---

## 3. Current measured baseline (re-verify before use — see rule above)

As measured 2026-07-26/27, deployed detector `checkpoints/move_detector_all28.pt` +
classifier `move_classifier_all39_jitter.pt`:

| metric | value | source |
|---|---|---|
| Detector recall, recorded sessions (aggregate) | ~96-97% | `metric_audit_heldout.log` |
| Detector recall, live free solves | 78-84% (two outlier takes: 69-70%) | `ma_live2.log`, `live-scoring-misattributes` memory |
| Classifier of found moves, live | 85.8% | `detector-window-mismatch` memory |
| End-to-end, live free solves | 73.2% | same |
| End-to-end, live prescribed scrambles | ~91% | `live-scoring-misattributes` memory |
| Decoder: 40-session sweep, all sessions | 19/40 verified, 10/40 exact | `reconstruct_all_jitter.log` |
| **Decoder: classifier-UNSEEN sessions (honest)** | **1/6 verified, 0/6 exact** | same |
| Cost envelope | true story at cost ~6 survives beam 4000; ~10 needs 64000 | `reconstruct.py` docstring, "Measured limits" |

Re-run commands (from `ble/move_detector/`):
```bash
python reconstruct.py --session ../training_data/solve_*/   # decoder-level, the number that matters
python metric_audit.py --sessions ../training_data/solve_*/ # move-level
python verify_solve.py --session ../training_data/solve_<stamp>_*/  # single-session incl. falsifiability sweep
```
The honest subset is hardcoded as `CLASSIFIER_UNSEEN` in `reconstruct.py`
(~line 1147) and reported separately by the summary — quote that line,
not the aggregate.

---

## 4. What the decoder already has — do NOT re-invent these

Read `reconstruct.py`'s module docstring in full before coding. Inventory
of machinery that already exists (rev 1 of this doc and an earlier review
both mistakenly proposed some of these as "new" ideas):

- **Pattern-database lower bounds** (BFS tables over twist/flip/udslice
  coordinates, `h_light`/`h_full`, cached in `cache/reconstruct_tables.npz`) —
  already used as an admissible capacity penalty in `_Beam._rank`, plus a
  corner-parity bound. Note the capacity term only fires near the tail
  (`remaining + max_end_ins < 14`) because PDB distances cap at ~12 —
  this is inherent to PDB bounds, not an oversight.
- **Exact one-edit repair lookahead** (the `rep1` suffix-consistency
  ladder) — closed-form hash set of every residual repairable by one
  future insert/delete/substitution. Ranks, never prunes. Grading
  "almost consistent" with PDB distances was tried and measured useless
  (h(rel) ≈ 20±3 for any ≥1-error hypothesis) — don't retry it.
- **Score-priced deletions** (`score_del_costs`, `DEL_SCORE_W`) — deleting
  a confident onset already costs more than deleting a weak one.
- **Stratified beam** (`_Beam._stratified_topk`) — protects
  expensive-but-consistent hypotheses from cheap floods. Do not add a
  cost-based pre-truncation before ranking; that exact bug was observed
  and is documented in `_Beam.merge`.
- **Arbitrary-endpoint decoding** (`decode_between`) — any start→end pair
  reduces to decode-to-solved in a shifted frame. D3 builds on this.
- **Search-vs-model failure diagnostic** (`gt_path_cost`) — if the decode
  result costs MORE than the ground-truth path, the beam lost the truth
  (search failure); if LESS, the cost model preferred a wrong story
  (model failure). **Use this to attribute every failure before and
  after each lever below** — it tells you which lever can help.

---

## 5. PRIMARY PATH — decoder levers, in order of cost-of-trying

All four run on existing recorded sessions. After EACH lever: re-run
`--selftest`, the 40-session sweep (honest subset is the headline), the
`gt_path_cost` failure attribution, and — for any cost-model change —
the falsifiability sweep (§7). Report decode wall-time alongside accuracy
(`decode_seconds` is already in the result dict); a lever that doubles
compute for nothing is a regression.

### D1. Soft-onset lattice: feed sub-threshold detector peaks to the decoder

**Evidence this helps:** detector misses are 55% of remaining raw error
(§6, `detector-window-mismatch`), and `live_detect.py`'s miss diagnostic
already buckets them: `subthreshold` misses (peak ≥ 0.10 but below the
operating point) are moves the model *saw* and the threshold discarded.
Today the decoder can only repair a miss with a flat-cost
(`C_INS = 4.0`), position-blind, 12-way-blind insertion — one insertion
exhausts the whole measured budget. But for a sub-threshold peak the
detector knows the position AND the classifier can supply a full softmax
— vastly cheaper information-wise. This is standard lattice decoding
(ASR/OCR); the pipeline currently throws the lattice away.

**Implementation sketch:**
- In the replay path (`_load_replay` → `live_detect.analyse`), lower the
  peak-picking threshold for *decoder input only* (e.g. sweep candidate
  thresholds 0.20/0.25/0.30 against the deployed operating point;
  `peak_pick` and the threshold plumbing already exist in
  `decode.py`/`live_detect.py`). Classify each extra candidate onset as
  usual so it has a real softmax.
- **The crux — reprice deletion for the new weak candidates.** If a weak
  candidate costs the normal `C_DEL = 2.6` to ignore, every false peak
  taxes the true story ~2.6 and the change is a net loss. Deletion cost
  should approximate −ln P(real | peak score): near **zero** for
  candidates below the old operating point (they're mostly noise — the
  measured fp odds concentrate at weak scores, see the `onset_costs`/
  `DEL_SCORE_W` comments), ramping to the current `C_DEL`+`DEL_SCORE_W`
  scale above it. Concretely: extend `score_del_costs` so `strength`
  is measured from the *candidate* threshold but an onset below the old
  operating point gets a deletion cost floor near 0.2–0.5, not `C_DEL`.
  Calibrate the ramp from the measured precision-vs-score curve
  (`metric_audit.py` / the fp-odds numbers in the docstring), not by
  feel.
- Cache note: replay caches to `<session>/reconstruct_replay.json` keyed
  by model names — a threshold change needs `--refresh-cache` or a new
  cache key, or you'll silently measure the old lattice.

**Accept/reject:** does the honest-subset verified count rise, with
falsifiability sweep unchanged-or-better? Secondary: do `gt_path_cost`
attributions show former miss-caused failures (true path paying `C_INS`)
converting to cheap accepts?
**Effort:** 1–2 days. **Risk:** low — purely additive information; the
failure mode (weak-candidate flood) is visible as decode-time blowup and
is bounded by the deletion-floor calibration.

**MEASURED 2026-07-27 — REJECTED (flat).** Implemented as
`--candidate-threshold`/`--del-floor`. First test accidentally used
`--candidate-threshold 0.25`, which turned out to equal the deployed
detector's own tuned operating threshold (`0.25000...6`, not the ~0.4–0.5
this section assumed from older docstring notes) — a no-op, re-run at
`--candidate-threshold 0.15` for a genuine test. That DID add real
candidates (e.g. 154→172, 185→187, 96→104 onsets on the honest sessions)
and raw accuracy barely moved (88.3%→88.2%), confirming the weak
candidates were mostly correctly absorbed near-free rather than flooding
the beam — the mechanism works as designed. **But honest-subset verified
count: 1/6, unchanged from baseline.** Consistent with §0's finding: the
5 stubborn sessions are 9–30 insertion-units out of budget (`gt_path_cost`
36–119) — recovering a handful of the detector's misses at near-zero cost
cannot close a gap that size. Falsifiability sweep NOT re-run (flag this
if reviving this lever). Command:
`python reconstruct.py --session ../training_data/solve_*/
--candidate-threshold 0.15 --beam 4000`.

### D2. Confusion-calibrated acceptance prior (adjacent-face, not just inverse)

**Evidence this helps:** `onset_costs` blends the softmax toward the
**temporal inverse** of the argmax (`BLEND_INV = 0.20`) because that was
the dominant confusion on recorded sessions. But the measured *live*
error axis is **adjacent face, not inverse** (`classifier-error-axis-live`
memory). On live-regime sessions the prior is pointing at the wrong
confusion: accepting the true move when the classifier said an adjacent
face is priced as expensive as accepting an arbitrary wrong move.

**Implementation sketch:**
- Measure the confusion matrix P(true class | predicted class) on
  classifier-unseen sessions only (`metric_audit.py` output; unseen-only,
  or you'll underestimate confusion).
- Replace the hand-built inverse-only blend in `onset_costs` with a blend
  toward the measured confusion-conditional distribution (keep the
  uncertainty scaling `u = 1 − p_max` — it's load-bearing; the raw
  softmax is overconfident exactly on its errors). Keep `--blend-inv`
  behavior reproducible behind a flag so A/B is one command.
- Beware overfitting: 6 unseen sessions is a small sample for a 12×12
  matrix. Pool structure (face-adjacency + inverse + direction classes)
  rather than fitting 144 free cells.

**Accept/reject:** honest-subset verified count; falsifiability sweep
MUST be re-run — softening acceptance costs is precisely the kind of
change that erodes decoy separation. If decoy margin shrinks materially,
the lever is rejected regardless of true-accept gains.
**Effort:** 1–2 days. **Risk:** medium on the security side (measurable,
gated by the sweep), low on effort.

**MEASURED 2026-07-27 — REJECTED (flat).** `--confusion` measured 46 subs
on 5 of the 6 unseen sessions (one has no `frames/`): inverse 45.7%,
adjacent 52.2% (adjacent genuinely edges out inverse, confirming the
memory this section cites), opposite 2.2%. Suggested split
`--blend-inv 0.093 --blend-adj 0.107` tested on the full 40-session sweep:
**honest 1/6, all-sessions 19/40 — identical to baseline**, per-move
accuracy within 0.1pt of baseline both overall and honest. Same
9–30-insertion-unit gap as D1 explains it — reallocating ~0.2 of prior
mass between two confusion types is far too small an effect to close it.
Falsifiability NOT re-run (the honest verified session didn't change, but
this is still a real gap — see §0). Command:
`python reconstruct.py --session ../training_data/solve_*/ --blend-inv
0.093 --blend-adj 0.107`.

### D3. Bidirectional meet-in-the-middle decode

**Evidence this helps:** the measured envelope is about the true story's
*cumulative* cost surviving beam eviction — cost ~6 survives at beam
4000, ~10 needs 64000 (docstring, "Measured limits"; the state constraint
only bites near an endpoint, which is why long flat stretches evict the
truth). Decoding forward from the scramble AND backward from solved,
meeting near the middle, splits the true story's cost across two halves
(~half each) — each half sits comfortably inside the envelope that
already works. Effectively doubles the repair budget at the same beam,
and both halves get an endpoint constraint nearby instead of one of them
being 100 moves from any anchor.

**Implementation (done 2026-07-27 — this is what actually shipped, not
just the sketch):**
- Backward pass = decode the *reversed* onset sequence with *inverted*
  move classes (`_reverse_invert_rows`: permute each cost row by `INV12`,
  reverse the list) starting NATIVELY from `end_state` — no frame-shift
  trick needed (unlike `decode_between`), because both `start_state` and
  `end_state` are already known; only the MEETING point is unknown. The
  group identity `CLASS_VECS[INV12[k]] == inverse(CLASS_VECS[k])` (already
  asserted in `--selftest`) makes the backward beam's live states after
  consuming `[meet, n)` land on exactly the same quantity the forward
  beam's states land on after `[0, meet)`: candidate values of
  `state_meet`. See `decode_bidirectional`'s docstring for the full
  derivation.
- `decode()`'s accumulation loop was factored out into `_run_beam` (byte-
  identical behaviour verified by the pre-existing `--selftest` suite
  passing unchanged) so both the single-pass and bidirectional paths run
  the literal same loop, not a parallel reimplementation that could drift.
- Join: both final beams are already deduplicated-unique internally
  (`_Beam.merge` dedupes every round), and `_hash_mat` is seeded
  identically (`0xC0BE`) across every `_Beam` instance, so a hash match is
  a genuine state match (confirmed with a real equality check on any
  collision, not just trusted). `decode_bidirectional_sweep` tries meet
  points at n/3, n/2, 2n/3 and keeps the cheapest solved result.
- **Not implemented, by design, not oversight:** the one-edit seam bridge
  for a failed exact join. Correctness doesn't need it (a failed join
  reports `solved=False` with both halves' cheapest cost, never a wrong
  answer — the joined sequence is asserted to replay from `start_state` to
  `end_state` before being returned), and the acceptance test below passed
  without it. Add it only if a real session's bidirectional decode fails
  specifically because the two halves' cheapest states are one edit apart
  and neither meets exactly.
- **A real design tradeoff surfaced during implementation, and needs to be
  understood before touching this code:** `_Beam._rank`'s three ranking
  heuristics (the suffix-consistency ladder, the pattern-database capacity
  bound, the parity bound) are ALL calibrated around a FIXED target of
  solved/identity — the ladder literally checks `rel == SOLVED`. A partial
  bidirectional beam has no such fixed target (its endpoint is whatever
  the OTHER half's search finds), so applying those heuristics unmodified
  would not just be uninformative, it would actively bias the search
  toward states that happen to look close to solved rather than toward
  the true unknown meeting point. `decode_bidirectional` therefore runs
  both partial beams with `use_bounds=False` (new `_Beam`/`_rank`
  parameter): pure cost ranking, no lookahead. This is *correct* — proven
  by a `--selftest` case that fails at beam 512 and recovers at 2048 with
  no logic change, confirmed by direct beam sweep to be pressure, not a
  bug — but it means bidirectional needs MORE beam per unit of error mass
  than the guided single-pass search does. **This tempers the "roughly
  doubles the budget at the same beam" framing above**: the budget gain
  from splitting the cost is real, but part of it is spent buying back the
  ranking guidance that was given up for target-independence. Whether the
  net effect is still a win has to be measured, not assumed — that's
  exactly what the acceptance test below is for.
- **Acceptance test — RUN, and the cited example was STALE.** The
  docstring's `solve_20260721_103149` example (cost 10.17, needs beam
  64000) was measured 2026-07-22 against whatever detector/classifier was
  deployed then. Re-measured fresh 2026-07-27 with the CURRENT models
  (`checkpoints/move_detector_all28.pt` + `move_classifier_all39_jitter.pt`): that
  session's true-path cost is now **34.5**, and it does not solve
  single-pass even at **beam 128000**. The models changed (crop-regime
  fix, jitter retrain) between the citation and now, and this specific
  session's classifier errors on it changed enough to move it completely
  out of any beam's reach — this is exactly the mistake the doc's own
  "never cite a stale baseline" rule at the top warns against, caught
  only by actually re-running the cited number before building a test
  around it. **Do not reuse that citation again without re-measuring.**
  A substitute short-session example was found (rank all cached sessions
  by `gt_path_cost`, pick a NOT-SOLVED one with moderate cost) but even
  the best current candidate (`solve_20260724_134516_scramble`, cost
  19.65, 22 onsets) doesn't solve single-pass through beam 128000 either
  — the "cost ~10, solves at 64000" regime this acceptance test wanted to
  probe may simply not exist in the current data at all. The real test
  ended up being the honest-subset sweep below, which doesn't depend on
  finding one matching anecdote.

**MEASURED 2026-07-27 — REJECTED (flat), both configurations tried.**
Single meet point (n//2) on the full 40-session sweep: honest 1/6 (same
session as baseline), all-sessions 19/40 (same as baseline, but exact
reconstruction 10→11/40 — a small, real, cost-model-neutral gain from
finding equal-or-cheaper equivalent stories, not from solving anything
new). `--meet-sweep` (3 meet points) on the 6 honest sessions
specifically: still 1/6, identical. Same 9–30-insertion-unit gap as
D1/D2 explains it: splitting a 100+ cost story across two 40-60 cost
halves still leaves each half far outside the proven envelope. The
use_bounds=False ranking-guidance tradeoff (above) may also be costing
real ground, but at this error magnitude it's very unlikely to be the
binding constraint — the gap is too large for ranking quality alone to
close. Falsifiability NOT re-run (see §0 — outstanding). Commands:
`python reconstruct.py --session ../training_data/solve_*/ --bidir
--beam 4000` and (on the honest sessions) `... --bidir --meet-sweep
--beam 4000`.
**Effort:** ~1 day to implement (done), ~1 day to test and diagnose the
stale-citation dead end. **Risk realized:** not the join mechanics
(exact, selftest-verified) — the actual finding was that this session's
error magnitude, like the others, is simply too large for any decoder
lever.

### D4 (contingent — only if D1–D3 leave "search failure" sessions). Two-edit repair ladder

Extend the exact `rep1` lookahead to residuals repairable by TWO future
edits: the product set {a·b} over pairs of rep1 elements (a conjugation-
ordered superset is fine — the ladder only ever awards an undeserved
bonus, never hides one, same argument as the existing position superset
in `rep1`). Size: 25n elements → (25n)² products; at n≈100 that's ~6M
hashes (~50 MB, minutes to build, once per orientation). Gate: build it
lazily, only when the beam's best level is stuck at `RESID_CAP`.

**Only worth it if** the post-D3 failure attribution still shows truths
being evicted during long ladder-flat stretches (decode cost >
`gt_path_cost`). If failures are model-failures instead, D4 cannot help
— that's D2 territory or a model-accuracy problem.
**Effort:** 2–3 days. **Risk:** low correctness risk (ranking-only),
real memory/compute cost.

**GATE CHECKED 2026-07-27 — NOT BUILT.** The failure attribution this
gate asks for was run directly (§0, and the per-session table in the
sprint notes): all 5 remaining honest failures have `gt_path_cost`
36–119, vastly exceeding any beam's reach at ANY of the tested
configurations — these are unambiguously model failures (the raw
classifier/detector error rate is too high for the endpoint constraint to
disambiguate, per §2), not search failures where the beam merely evicted
a findable truth. A two-edit lookahead ladder ranks candidates better; it
cannot manufacture 9–30 insertion-units of missing information. Skipped
per the gate's own stated condition — this is the gate working as
designed, not a shortcut.

---

## 6. Root-cause list (context for §5; unchanged from rev 1 except item 1)

1. ~~Decoder repair budget is a hard wall~~ → being attacked directly by
   §5; the information budget (§2) says the current ~90% regime is the
   knife edge where search/calibration bits genuinely matter.
2. Detector misses are 55% of remaining raw error and cascade into
   neighbours' classification windows (`detector-window-mismatch`). D1
   converts a chunk of these from blind insertions into cheap accepts.
3. Two live outliers: one explained (saturation z=+8.2, needs data), one
   unexplained (`solve_20260726_100142_solve`, 69.6%, nothing anomalous
   measured). Still open; see Path 2d in §8.
4. Classifier adjacent-face/direction residual that data volume stopped
   moving (`classifier-error-axis-live`). D2 prices it honestly.
5. Sub-frame anchor quantization — mitigated by jitter augmentation;
   real fix is a regression head (Path 2b, §8).
6. Algorithm-prior: measured safe, low ceiling, not yet wired into
   `reconstruct.py` (Path 2a, §8).
7. `ble_truth.get_state()` live solved-check bug — eval-side only,
   backlog (`ble-truth-end-state-bug`).
8. False-accept measurement is anecdotal — §7, mandatory.

---

## 7. Mandatory regardless of path: rigorous false-accept measurement

Unchanged from rev 1, and now MORE urgent: every §5 lever makes the
decoder better at repairing stories, which is the same mechanism that
would repair a decoy into acceptance. This harness is the gate.

**Current state:** `verify_solve.py`'s falsifiability sweep is
methodologically sound (constructive decoy bound; 1/2/4/8-quarter-turn-off
claims, never-scrambled, wrong-scramble) but has run in depth on ONE
session. 0/6 decoys at n=6 bounds the true false-accept rate below ~39%
at 95% — not shippable.

**Build (3–5 days, before or alongside D1):**
1. Batch harness running the sweep across every session in
   `training_data/` (~40) and every decoy type. Orchestration around
   `verify_solve.py`'s existing `--session` path; no new decode logic.
2. Report: N decoy attempts, K accepted, 95% Clopper–Pearson upper bound.
   0/150 → ~2.4% bound; that's a quotable claim. Also report the decoy
   cost-margin distribution, not just accept/reject — §5 levers must not
   silently shrink the margin even where the verdict stays correct.
3. Re-run on EVERY change to costs, search, beam, or thresholds (all of
   D1–D4, Path 2a). A regression here is a hard blocker, not a tradeoff.

---

## 8. Other paths — status and record

### Path 1 (rev 1's primary): scan-based mid-solve checkpoints — REJECTED 2026-07-27

Record of why, so it isn't re-proposed: the segment math required each
checkpoint to yield a *known* state, which only a full 6-face scan
provides. The scan (`cv/solver/state_finder.py`) is manual (SPACE per
face, redo on failure, orange/red calibration), ~30–60s, and itself
error-prone (lock-on, orange/red under non-daylight lighting — see
CLAUDE.md). 5–7 mid-solve scans per solve multiplies scan-failure
exposure (≈ coin-flip that some checkpoint fails at 90%/scan) and
dominates solve time — it would add error and compute, not remove it.
A "confirm-predicted-state-from-partial-view" variant was considered and
also shelved: it inherits the scanner's per-face weaknesses and adds an
orientation-matching problem, for a weaker security claim. If mid-solve
anchors ever return, they must not depend on the current scanner.

### Path 2f: input-representation rework — MEASURED FLAT 2026-07-27

The classifier's *input encoding* was the one lever never pulled: the crop
fix (07-24) and the anchor-jitter retrain (07-26) both changed what the
window contained, never how it was encoded. Four 3-channel encodings that
fold temporal order into COLOUR instead of into 12 channels
(`ble/encodings_move.py`, `--encoding`) were trained against a
**freshly retrained diffstack control**, identical recipe, identical named
holdout, same sitting, and scored end-to-end on the same 10 live takes
(none of which any classifier has trained on).

| encoding | ch | held-out val | live classifier-of-found |
|---|---|---|---|
| diffstack (deployed `all39_jitter`, re-measured) | 12 | 94.7% | **86.0%** |
| diffstack (fresh control, new seed) | 12 | 95.2% | **83.7%** |
| rgbtime | 3 | 95.4% | 86.8% |
| chroma8 | 3 | 95.2% | 86.5% |
| rgbtime0 | 3 | 95.9% | 85.1% |
| chroma | 3 | 94.1% | 83.9% |

**Read the first two rows before the rest.** Two diffstack runs differing
only in seed span 83.7-86.0% live — and every new encoding lands inside
that span. Exact-McNemar (paired, per move) against the *fresh control*
makes rgbtime look like a winner (p=0.030); against the *deployed*
checkpoint nothing is significant (rgbtime p=0.64, chroma8 p=0.82,
rgbtime0 p=0.64, chroma p=0.23). The apparent +3.1 points is a control
that landed low, not an encoding that landed high. **This is the fifth
independent intervention to measure flat** (after D1, D2, D3 and the
jitter retrain), and the first one where the null result comes with its
own seed-variance envelope attached.

Two findings worth keeping even though the headline is flat:

- **3 channels tie 12.** chroma8 and rgbtime match the 12-channel input
  at a quarter of the input width, with conv1 used exactly as pretrained
  rather than inflated. If inference cost ever matters, that swap is free.
- **Held-out val does not rank live performance.** Val spans 94.1-95.9%
  and live spans 83.7-86.8% with the orderings essentially uncorrelated —
  rgbtime0 is best on val and mid-pack live; the control is second-best on
  val and worst live. Do not select a checkpoint on the val number alone;
  `metric_audit.py` on the live takes is the arbiter.

Reproduce: `sh ble/run_encoding_sweep.sh` then `sh ble/run_encoding_audit.sh`.
Nothing was promoted — `CLASSIFIER_PATH` is unchanged.

### Path 2g: optical flow — REJECTED 2026-07-27 (significantly worse)

Same harness, same holdout, same 10 live takes. Hypothesis: a turn rotates
ONE layer while the rest of the cube is rigid, so fitting the cube's global
motion and subtracting it should leave the layer's motion alone —
*where* the residual lives naming the layer, its *sign* naming the
direction. `flow` (raw Farneback, 2ch per pair) vs `flowres` (same, minus a
robust centre-weighted affine global fit) is the ablation that isolates
exactly that one step.

| encoding | ch | held-out val | live classifier-of-found |
|---|---|---|---|
| diffstack (deployed / fresh control) | 12 | 94.7 / 95.2% | **86.0 / 83.7%** |
| flowres (compensated) | 8 | 91.8% | 80.8% |
| flow (uncompensated) | 8 | 90.8% | 79.0% |
| flowwheel (residual, colour wheel, 3ch) | 3 | 89.1% | 77.6% |

**All three land significantly BELOW the diffstack seed envelope** — not
flat, worse: vs the deployed checkpoint, flowres p=2.3e-03, flow p=4.5e-05,
flowwheel p=2.2e-06 (exact McNemar, paired per move). flow and flowwheel are
significant even against the *low* end of the envelope.

**The compensation hypothesis was directionally right and insufficient:**
flowres beats flow by +1.8 points live and +1.0 on val, but p=0.295 — real
in direction, not resolvable at this sample size, and nowhere near enough to
close a 5-point deficit. The mechanism does work: the self-test shows a
synthetic global pan reduced 62.8 → 0.8. On real moves it does not isolate a
layer, because the hands are a large independent moving object that no
single affine over the crop can describe.

Two independent pieces of evidence say the same thing, and both were free:
- `flow_direction.py` (in-repo since the reorg, never previously run) gets
  only **70.4% leave-one-session-out on the BINARY CW/CCW decision with the
  face already given** — chance is 50%. Classical flow carries weak
  direction signal on this capture. Log: `flow_direction_probe.log`.
- Ranking across all nine encodings tracks how much *appearance* survives:
  rgbtime (cube visible) 86.8% > chroma8 (motion only) 86.5% > … > flowres
  (motion only, no appearance) 80.8%. Discarding the cube's appearance
  costs more than cleaning up its motion gains.

Do not re-propose flow without first fixing 2h below, and prefer RAFT over
Farneback if it is revisited (that means dropping `--anchor-jitter` so
flows can be precomputed — see the note in `encodings_move.py`).

### Path 2h: move-window collapse — REAL DEFECT, MEASURED COST ~0. CLOSED.

**Verdict first, because this was briefly written up as the best next
lever and that was wrong.** The defect below is real and reproducible. Its
effect on accuracy is not distinguishable from zero, and it is not worth
fixing. `window_collapse_audit.py` is the measurement; re-run it before
reopening this.

Controlled comparison (held-out sessions, 608 moves): within a gap bucket —
so that move difficulty is held roughly fixed and only the sub-frame anchor
phase decides whether slots collide — collapsed windows vs clean windows,
across five independently trained checkpoints:

| model | clean | collapsed | delta | p |
|---|---|---|---|---|
| deployed jitter | 96.2% | 93.8% | -2.4 | 0.264 |
| diffstack control | 96.2% | 94.6% | -1.6 | 0.438 |
| rgbtime | 95.4% | 95.4% | 0.0 | 1.000 |
| chroma8 | 95.8% | 94.9% | -0.9 | 0.698 |
| rgbtime0 | 95.0% | 96.5% | **+1.5** | 0.405 |

The sign flips across models and nothing is significant. Even taking the
deployed model's -2.4 at face value, only 41% of moves carry addressable
(non-camera) collapse, so the **optimistic ceiling is 1.0 point** — under
half the 2.3-point seed envelope, for a change that would have to alter
training and inference in lockstep plus a retrain.

**Two premises behind the original write-up were also wrong:**
- "It hits the hardest moves." It does not. Accuracy by gap is 95.7%
  (<250ms), 92.6%, 96.6%, 90.6% (>600ms) — fast moves are the *most*
  accurate here, presumably fluent algorithm execution, while slow isolated
  moves are hesitant ones. The dead channels do concentrate on fast moves;
  those moves just are not the problem.
- "It should skew errors toward direction." Direction share of errors is
  52% for collapsed vs 44% for clean windows, on 23 and 9 errors — far too
  few to mean anything. The mechanism was never demonstrated to operate.

Most likely why it costs nothing: 60% of TRAINING windows are collapsed
too, so the classifier is fitted to that distribution and the surviving
diffs evidently carry enough; `--anchor-jitter` further trains window
robustness directly.

The defect itself, for the record (do not re-derive it):

Found while debugging why flow tiles rendered pure black. **60% of the 3300
training windows contain at least one pixel-identical consecutive frame
pair; 16.5% contain two.** The diff or flow computed across such a pair is
exactly the neutral value, so those input channels carry nothing.

It is not the camera — raw capture duplicates only 4-6% of consecutive
frames (~28.5 effective fps of a nominal 30). It is `move_window()` /
`window_from_anchor` squeezing the five slots toward each other when
neighbouring moves are close, until two slots round to the same frame:

| nearest-neighbour move gap | mean dead pairs (of 4) | windows with ≥1 |
|---|---|---|
| <250ms | 1.15 | 81% |
| 250-400ms | 0.51 | 50% |
| 400-600ms | 0.27 | 27% |
| >600ms | 0.19 | 19% |

So the input representation degrades by ~29% precisely on the fastest
moves — the crowded regime §3a and the detector-miss analysis already
identify as the failure mode. Every encoding in 2f and 2g inherits this
equally, which makes it a plausible common ceiling under all of them.

If this is ever reopened (it should not be without new evidence), the fix
would be to choose window slots by distinct *content* rather than fixed
scaled offsets, and it **must change training and inference together** —
`decode.window_from_anchor` is the live path — so it needs a checkpoint-
recorded window policy the way `crop_regime` is recorded.

One caveat left open honestly: all of the above is measured on recorded
held-out sessions, where accuracy is 94-96%. Live accuracy is 86%, and a
larger error budget could in principle expose a larger effect. Scoring this
on live takes needs per-move gaps from `metric_audit.py`'s BLE matching,
which `window_collapse_audit.py` does not yet read. Given a ceiling of 1.0
point and a sign that flips across models, that was not judged worth
building.

### Path 2i: BLE-anchored training vs detector-anchored inference — AUDITED

Raised 2026-07-27: every classifier trains on windows anchored on BLE move
TIMESTAMPS, but at inference the anchor comes from the detector. Audited
with `phantom_confidence.py` (10 live takes, 646 classifier calls).

The mismatch splits into two cases with different answers.

**Matched moves — already covered, and completely.** `TOLERANCE = 2`
frames, and the measured live offset histogram is `-2:29 -1:97 0:297
+1:139 +2:15`, n=577 = every matched move. So `ANCHOR_JITTER_PMF` spans
the entire achievable offset range by construction; there is no censored
tail hiding inside the matched set. `--anchor-jitter` trains exactly this
distribution.

**Phantom detections — a real, uncovered gap.** 69 of 646 calls (10.7%)
are on windows containing no move. The classifier has 12 classes and a
softmax; it cannot say "nothing here", and it is never trained on such a
window. It emits a fairly confident name for a move that did not happen,
and the decoder consumes that as `-log p` evidence.

**But the classifier is the wrong place to fix it.** Confidence barely
separates phantoms from real moves, because it conflates "no move" with
"hard move" — phantom mean confidence 0.676 vs substitution 0.667 vs
correct 0.913:

| signal | AUC (real vs phantom) | best net at any threshold |
|---|---|---|
| classifier confidence | 0.791 | **negative everywhere** (at 0.5: cuts 20 phantoms, loses 33 real) |
| detector onset score | **0.904** | +14 at 0.40 (cuts 32, loses 18) |
| score x confidence | 0.894 | — |

Combining is *worse* than the onset score alone, so the classifier carries
no phantom information the detector lacks. And the decoder **already
consumes the onset score** for exactly this — `score_del_costs()` ramps
deletion cost with onset strength, and its docstring already cites "fp odds
concentrate at weak scores". Nothing to add there.

**Conclusion: no null-class retrain on current evidence.** The residual it
could address is the ~37 phantoms that survive a 0.40 score threshold —
strong-score phantoms carrying confident wrong softmax, which are both
expensive to delete and actively misleading. That is 5.7% of calls, against
six consecutive classifier-side interventions that moved nothing. Revisit
only if phantom *evidence quality* (not phantom count) is ever shown to be
what breaks a specific decode.

### Path 2 (parallel, still valid): incremental ML improvements

Unchanged from rev 1; none of these moves the verified rate alone
(measured: jitter retrain, p=0.008, zero sessions flipped), but they
lower per-move error, which compounds with §5:
- **2a.** Wire the algorithm-prior into `reconstruct.py` (validated
  config: `match_windows` + mined library; re-run §7 after). 1–2 days.
- **2b.** Sub-frame onset regression head on the TCN (kills the anchor-
  quantization residual at source). 3–5 days + retrain.
- **2c.** 60fps capture for sub-150ms move recall. Exploratory, 1–2 wks.
- **2d.** Root-cause the unexplained live outlier
  (`solve_20260726_100142_solve`). 1–3 days; genuinely open.
- **2e.** Record sessions in the saturated-lighting condition. Data fix.

### Path 3: capture/UX levers

- **3a.** Pacing prompt ("one move at a time") — ~1 day, do whenever;
  directly avoids the crowded-move regime that drives detector misses.
- **3b.** Optional BLE second-signal for smart-cube owners — scope
  decision, deferred; camera-only must stand alone (CLAUDE.md priority).

### Path 4 (contingent): architecture rework (end-to-end sequence model)

Trigger condition, sharpened by §2: if after §5 + Path 2 the honest rate
is still far from target, the remaining gap is per-move accuracy, and the
bar is quantified — whole-solve decoding needs roughly ≥96–97% per-move
accuracy for expected error mass to sit inside even the D3-doubled
budget. A CTC-style end-to-end model is the identified candidate
(dissolves the detector/classifier boundary and the window-anchor
mismatch). 3–6 weeks, uncertain payoff. Doing it BEFORE §5 would repeat
the trap `README.md` already flags: you couldn't tell "architecture was
the problem" from "decoder wasn't finished." After §5, it has a real
baseline to beat.

---

## 9. Sequence for the prototyping sprint

```
Days 1-2    D1 (soft-onset lattice) + start §7 harness in parallel.
Days 3-4    D2 (confusion-calibrated prior). First full re-measurement:
            40-session sweep, honest subset, falsifiability, wall-time.
Days 5-9    D3 (bidirectional). Acceptance test: the beam-64000 session
            (solve_20260721_103149) recovered at 4000/side.
Day 10      Full re-measurement + failure attribution (gt_path_cost).
            Decision point: D4 if search-failures remain; otherwise stop.
Contingent  D4 (two-edit ladder), 2-3 days, only per the gate in §5.
Then        Re-assess against §1 targets → Path 2 items and/or Path 4
            decision, now with an exhausted-decoder baseline.
```

Every measurement: fresh baseline in the same sitting (rule at top),
honest subset as the headline, falsifiability alongside true-accept.

## 10. Definition of done for the sprint (not the MVP)

- [ ] All of D1–D3 implemented behind flags, each A/B-able in one command
      against the unchanged path.
- [ ] 40-session sweep + honest subset re-measured after each lever,
      same-sitting baselines, logged (same convention as
      `reconstruct_all_jitter.log`).
- [ ] §7 harness built; decoy Clopper–Pearson bound reported before/after
      the lever set; no margin regression.
- [ ] Failure attribution table (search vs. model failure per unsolved
      session) before and after — this is the input to the D4 gate and
      the Path 4 decision.
- [ ] `README.md` + this doc updated with measured results; projections
      not left standing as results.

## 11. Open decisions that need the user, not a model

- **True-accept / false-accept ship targets** — §1's 90% / 2%-bound is a
  straw man; confirm before treating any gate as pass/fail.
- **Compute budget for decoding** — D3/D4 trade wall-time for recovery;
  what per-solve decode time is acceptable in the product (current
  single-pass times are in `decode_seconds`; a wide-beam retry already
  exists via `--retry-beam`)?
- **If the sprint falls short: Path 4 go/no-go** — 3–6 weeks, uncertain;
  §8's trigger condition makes the decision data-driven, but it's still
  a calendar/scope call.
- **Path 3b (BLE second signal)** — in or out of scope; default out.

---

Related: `README.md` (measured history), `ALGORITHM_PRIOR.md` (precedent
for "measured safe, low ceiling"), `reconstruct.py` module docstring
(cost model, measured limits, and everything §4 inventories —
**required reading before D1–D4**), `verify_solve.py` docstring
(falsifiability sweep), `decode.py`/`live_detect.py` (peak picking and
the miss-bucket diagnostic D1 builds on).
