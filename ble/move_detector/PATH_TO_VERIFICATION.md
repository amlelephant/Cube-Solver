# Path to verification — decoder-first roadmap (rev 2)

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

**Implementation status (2026-07-27, same sitting as this note)**: D1
(`--candidate-threshold`/`--del-floor`), D2 (`--blend-adj`, plus a
`--confusion` measurement mode), and D3 (`--bidir`, `--meet`,
`--meet-sweep`) are all implemented in `reconstruct.py`/`verify_solve.py`
and pass `--selftest`, including new bidirectional-specific checks. Every
new flag defaults to reproducing the EXACT prior behaviour (verified: the
full existing `--selftest` suite passes unchanged before and after each
lever, byte-for-byte). Full-session measurement (40-session sweep, honest
subset, falsifiability, the beam-64000 acceptance test) is in progress —
see the per-lever sections below for what's confirmed vs. still open, and
do not treat "implemented" as "measured to help." D3 in particular
surfaced a real design tradeoff during implementation — see its section.

---

## 0. TL;DR

- Honest generalization rate (classifier-unseen sessions, full pipeline,
  camera-only): **1/6 verified, 0/6 exact** (`reconstruct_all_jitter.log`,
  40-session sweep, 2026-07-26). The all-sessions number (19/40) is
  inflated by sessions the classifier trained on — don't quote it.
- Three consecutive significant model improvements moved move-level
  accuracy and left the verified rate flat. The binding constraint is the
  decoder's repair budget vs. the error mass a 60–150-move solve
  accumulates — see the information-budget analysis in §2, which also
  answers "can the decoder correct at 75% accuracy?" (no — for any
  decoder, not just this one).
- Rev 1's answer (mid-solve scan checkpoints) is **rejected**: the 6-face
  scan is slow, manual, and itself error-prone (orange/red, lock-on), so
  chaining 5–7 of them per solve multiplies scan-failure exposure and
  wrecks UX. Record kept in §8.
- **New primary path (§5): four decoder levers, ordered by
  cost-of-trying** — (D1) feed sub-threshold detector peaks into the
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

As measured 2026-07-26/27, deployed detector `move_detector_all28.pt` +
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
  coordinates, `h_light`/`h_full`, cached in `reconstruct_tables.npz`) —
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
- **Acceptance test:** session `solve_20260721_103149` decodes to the true
  story at cost 10.17 at beam 64000 and is LOST at 4000/16000 (docstring).
  Bidirectional at beam 4000/side should find it. **Status: not yet run**
  — needs the session's replay cache (GPU inference), which is what the
  in-progress 40-session baseline sweep is populating. Run
  `python reconstruct.py --session ../training_data/solve_20260721_103149/
  --bidir --beam 4000` once that baseline finishes and the replay cache
  exists (fast afterward — decode-only, no GPU).

**Accept/reject:** the 64000-beam session recovered at 4000/side (not yet
measured); honest subset count; decode_seconds comparable to or better
than single-pass at the wide beams it replaces, ACCOUNTING for the extra
beam the use_bounds=False tradeoff above costs. Falsifiability sweep
re-run (search changes alter which decoys get found, in both directions;
`verify_solve.py --bidir` is wired for this).
**Effort:** ~1 day to implement (done). **Risk:** the ranking-guidance
tradeoff above is the real remaining unknown — not the join mechanics,
which are exact and selftest-verified.

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
