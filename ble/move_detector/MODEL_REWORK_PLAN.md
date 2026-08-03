# Model rework plan — road to verification, rev 3 (2026-07-28)

Successor to `PATH_TO_VERIFICATION.md` rev 2. Rev 2's primary path
(decoder levers D1–D4) was executed 2026-07-27 and is **exhausted** —
all levers measured flat, the reason quantified (the 5 stubborn honest
sessions are 9–30 insertion-units outside any decoder's repair budget;
see rev 2 §0/§5 and the `decoder-sprint-exhausted` record). This
document is the follow-on: the remaining lever is **per-move accuracy of
the evidence stream**, i.e. the models, and the bar is quantified by rev
2 §2: roughly **96–97% end-to-end per-move accuracy** for a 100+ move
solve to sit inside the endpoint-constraint information budget. Current
live end-to-end is 73.2% (free solves; ~91% prescribed scrambles).

Written to be executed by a cheaper model, cold. Rules:

- **Do not propose another decoder-side tweak as the fix.** Four decoder
  interventions plus two classifier retrains measured flat on the honest
  verified rate. Re-derive why your idea escapes those six data points
  before spending a day on it.
- **Never cite a stale baseline.** Re-measure both sides of every
  comparison fresh, same sitting. This rule has caught two real errors
  already (rev 2 top note, and the D3 stale-citation trap).
- **Two seeds minimum** for any training-based claim. The measured
  same-recipe seed spread on live scoring is **~2.3 points** — one
  control run is one draw, not a baseline (`encoding-rework-flat`).
- **Named holdouts only** (the `CLASSIFIER_UNSEEN` sessions), never
  random splits (`named-holdouts-cross-env`).
- **The arbiter is `metric_audit.py` over the 10 saved live takes.**
  Trainer val accuracy does not rank live performance (measured:
  orderings uncorrelated). Never select a checkpoint on val.
- **Headline number = honest-subset verified count** (currently 1/6),
  reported with move-level accuracy alongside. All-sessions 19/40 is
  contaminated by training data; don't quote it.
- **Any change that alters what the decoder consumes must re-run the
  falsifiability sweep** (rev 2 §7). Still outstanding from the last
  sprint and now overdue — it is Phase 0 below, not optional.
- Replay caches are keyed by model names — after any model or threshold
  change, `--refresh-cache` or bump the cache key, or you will silently
  measure the old lattice (this bit once already, rev 2 D1 notes).
- Training runs: always `--workers 16` on this machine; crop regime is
  recorded/enforced in artifacts — run `cache_crops.py --check` before
  training (`crop-provenance-guards`).

---

## 1. The design brief: where the error actually lives

Measured decomposition of the live gap (all numbers 2026-07-26/27,
re-verify fresh before use):

| error source | size | reference |
|---|---|---|
| Detector misses | 55% of remaining raw error; live recall 78–84% vs ~96% recorded | rev 2 §6, `detector-window-mismatch` |
| Misses also corrupt **neighbours'** windows | 91 moves got >25%-wide windows: 90.1%→72.5% on those | `detector-window-mismatch` |
| Phantom detections | 10.7% of classifier calls; ~37 strong-score phantoms survive the score threshold and carry confident wrong softmax | `ble-vs-detector-anchors` |
| Classifier on found moves | 85.8–86% live; **six consecutive classifier-side interventions flat** on the verified rate | `jitter-retrain-decoder-flat`, `encoding-rework-flat` |

Three structural observations that the current architecture cannot fix
from inside:

1. **The detector/classifier boundary itself generates errors.** The
   window/anchor machinery (frame-quantized anchors, gap-squeezed
   windows, neighbour corruption, window collapse) exists only because
   detection and classification are separate models glued by index
   arithmetic. Every one of those defects was individually measured;
   individually each is small, but they are all artifacts of the seam.
2. **The classifier cannot say "nothing happened".** 12 classes + a
   softmax means every phantom onset gets a confident name, which the
   decoder eats as `-log p` evidence. Measured: the classifier carries
   *no* phantom information the detector's onset score lacks (AUC 0.79
   vs 0.90, combining is worse). The fix belongs in a model that has a
   background class competing per-frame, not a 13th class bolted onto
   windowed crops.
3. **The decoder is fed impoverished evidence.** It gets hard onsets +
   one softmax per onset. Rev 2's D1 proved the lattice mechanism works
   (weak candidates absorbed near-free) — it just had too little to add
   at the onset level. A per-frame posteriorgram is the full-fat version
   of the same idea, and the decode machinery to consume priced
   candidates already exists.

This is why "specialize the architecture" is the right instinct, and why
the right shape is a **joint frame-synchronous model**, not a better
window classifier. That is Path 4 from rev 2 §8, whose trigger condition
is now genuinely met, scoped below as an evolution of existing code
rather than a rewrite.

---

## 2. Verdict on optical flow and PWC-Net (the standing question)

Honest reading of the 2026-07-27 flow rejection before re-litigating it:

- **What was measured:** Farneback flow inputs (raw, global-motion
  compensated, colour-wheel) were significantly *worse* than diff
  encodings (p ≤ 2.3e-03), and the ranking across all nine encodings
  tracked **how much appearance survives**, not how clean the motion is.
  The compensation mechanism itself verifiably worked (synthetic pan
  62.8→0.8) but did not isolate the turning layer on real frames —
  hands are a large independent moving object no single affine explains.
- **What was NOT ruled out:** that Farneback specifically is too weak
  for this footage. That part of the user's critique is fair: a cube
  turn is 100–300ms of fast, blurred, small-object motion — exactly
  where classical pyramidal flow breaks and learned flow (RAFT family)
  is dramatically better.
- **PWC-Net specifically: do not use it.** It is a 2018 model and is
  dominated on every axis by the RAFT family. More importantly,
  **torchvision 0.28.0 (already pinned in requirements.txt) ships
  `torchvision.models.optical_flow.raft_small` and `raft_large` with
  pretrained weights** — a learned-flow test costs zero new
  dependencies. SEA-RAFT (2024) is the current speed/accuracy frontier
  but needs an external repo; only consider it if RAFT passes the gate
  and is too slow.
- **Two constraints on any retest, both learned the hard way:**
  1. Flow may only be tested as **added channels alongside appearance**,
     never replacing it. "Do not propose another appearance-free
     representation" is a measured conclusion, not a preference.
  2. Horizontal-flip augmentation must negate the u component
     (`flip_fixup` in `encodings_move.py` already implements this) —
     and precomputed flow conflicts with `--anchor-jitter` (see the
     note in `encodings_move.py`). With RAFT, compute flow on the fly
     on GPU or restrict jitter accordingly.

### Gate G1 — half a day, no training, decides whether flow lives

1. Run RAFT (`raft_small`, pretrained) on the same sample move-window
   frame pairs `viz_flow/` used, render side-by-side with the Farneback
   tiles (reuse the `viz_encodings.py` tile plumbing). Question: does
   RAFT visibly isolate the turning layer where Farneback didn't?
2. Re-run the `ble/flow_direction.py` probe with RAFT flow substituted
   for Farneback in `compute_features()`. Baseline to beat: **70.4%
   leave-one-session-out on binary CW/CCW with the face given** (chance
   50%) — logged in `flow_direction_probe.log`.
3. **Gate:** probe ≥ ~85% *or* unambiguous visual layer isolation →
   flow earns a slot as Stage C added-channels experiment (§4). Below
   that → flow stays dead, including PWC-Net, and the answer to "was it
   the algorithm?" is measured rather than argued.

**MEASURED 2026-07-28 — GATE FAILS, decisively.** Implemented as
`flow_direction.py --flow-backend {farneback,raft}` (`raft_small`,
torchvision 0.28's pretrained weights, lazy-loaded so the module's
zero-torch-dependency default path is untouched) plus `raft_flow_viz.py`
for the visual half. Both run on the IDENTICAL session set the original
70.4% baseline used (`training_data/solve_20260724_*/`, 820 samples,
re-verified fresh first: reproduced 70.4%/74.4% exactly, confirming the
refactor changed nothing on the Farneback path).

RAFT leave-one-session-out: **54.4%** (pooled 80/20: 54.3%) — barely
above chance (50%), 16 points below Farneback's own 70.4%, and nowhere
near the ~85% gate. A 2-session spot-check before the full run showed
the same direction even more starkly (RAFT 49.6% vs Farneback 73.5% on
that subset) — this is not a small-sample fluke that the fuller run
smoothed over.

The visual comparison (`viz_flow/raft_vs_farneback.png`, 12 sampled
moves across classes) agrees with the number and adds a mechanism: RAFT's
global-motion-compensated residual renders as fine-grained noise spread
across the WHOLE crop, not a cleaner isolation of the turning layer than
Farneback's already-diffuse blobs — if anything less localized. Plausible
cause, not chased further given the size of the gap: RAFT (`raft_small`)
is trained on natural-scene benchmarks (Sintel/KITTI) at resolutions and
motion statistics far from a 224-256px cube crop with fast blur-heavy
quarter-turns: this is exactly the domain-mismatch risk the plan's design
brief flagged as the fair alternative reading of the original rejection,
and it appears to be real, just not fixable by swapping the estimator.

**Conclusion: optical flow (Farneback OR RAFT) is closed.** Do not
re-propose flow, including PWC-Net (already out of consideration — 2018,
dominated by RAFT) or SEA-RAFT (the natural next escalation, but the
mechanism problem — appearance beats clean motion, §2f/2g of the
predecessor doc, now reinforced by RAFT actively producing WORSE input
than Farneback rather than merely equal — makes further estimator
swaps an unpromising place to spend time). Stage C is cancelled; Stage A
and Stage B stand unaffected. Reproduce: `python flow_direction.py
--sessions "training_data/solve_20260724_*/" --flow-backend raft` and
`python raft_flow_viz.py --sessions "training_data/solve_20260724_*/"`.

---

## 3. Gate G2 — oracle attribution on the honest six (~1 day, do first)

Before committing weeks to architecture, measure **which side's
perfection would actually verify the stubborn sessions.** All machinery
exists (replay cache, `ble_truth` move log, `reconstruct.py`,
`metric_audit.py` matching). On the 6 `CLASSIFIER_UNSEEN` sessions, run
the decoder with:

- **(a) Oracle classifier:** real detector onsets, but each matched
  onset's softmax replaced by one-hot BLE truth (unmatched onsets keep
  their real softmax — they're phantoms, truth has no label for them).
- **(b) Oracle detector:** onsets taken from BLE timestamps (perfect
  recall, zero phantoms), real classifier softmax on windows anchored
  there.
- **(c) Both** (sanity: should verify ~6/6; if not, the harness or the
  endpoint states are wrong — check `ble-truth-end-state-bug`, use the
  move log, never `get_state()`'s live solved-check).

Caution: BLE truth is the *supervision source*, so (a)/(b) are upper
bounds, not reachable targets — the point is the *split*, not the level.

**What each outcome means:**
- (b) verifies most sessions, (a) doesn't → the binding error is
  detector-side (misses + phantoms = insertions/deletions, the expensive
  edits). Stage A (§4) should weight the background/onset channel and
  recall; classifier capacity is secondary. *This is the expected
  outcome given the 55%-of-error and phantom numbers, but measure it.*
- (a) verifies most, (b) doesn't → substitutions dominate after all;
  Stage A should weight per-class discrimination (colour input, encoder
  capacity).
- Neither alone verifies → both error streams must fall together; the
  joint model is the only shape that plausibly does that, and expected
  gains from any single-side fix should be discounted accordingly.

Also log the per-session `gt_path_cost` split (insertion-units vs
substitution mass) under (a) and (b) — this converts "9–30 units over
budget" into a per-session attribution table that Stage A's evaluation
reuses.

**MEASURED 2026-07-28 — decisive, detector side dominates.** Implemented
as `oracle_attribution.py` (session selection computed live via
`reconstruct.classifier_unseen()` against the deployed classifier — 6
honest sessions currently, matching `decoder-sprint-exhausted`'s count
exactly once the one session lacking `frames/` is excluded). Oracle
classifier: real onsets, matched ones (via `metric_audit.gt_onset_frames`
+ `score_by_time` — TIME-based matching, not `align_sequences`, which has
no timing information) get a near-one-hot softmax on the truth label
through the same `onset_costs()` the real pipeline uses. Oracle detector:
onsets placed exactly at the true BLE-anchored frame, classified by
RE-RUNNING the real deployed classifier there (not an assumed softmax).
Sanity check `both` (oracle onsets + oracle softmax) verified 6/6 — the
harness and endpoint states are correct.

| session | real | oracle-cls | oracle-det | both |
|---|---|---|---|---|
| solve_20260721_102711 | 106.2 | 94.8 | 14.7 | 0.0 |
| solve_20260722_101225 | 73.7 | 60.7 | 18.0 | 0.0 |
| solve_20260723_105530_solve | 36.1 | 34.1 | 3.7 | 0.0 |
| solve_20260724_100120_solve | 93.6 | 82.7 | 15.1 | 0.0 |
| solve_20260726_165044_scramble | 14.8 | 12.9 | 2.0 | 0.0 |
| solve_20260726_165044_solve | 119.3 | 100.0 | 43.6 | 0.0 |

(values are `gt_path_cost`, lower = closer to verifiable.) Verified
counts: real 1/6, oracle-classifier 1/6, **oracle-detector 2/6**, both
6/6. Averaged over sessions, oracle-classifier closes **12.5%** of the
(real → both) gap; **oracle-detector closes 81.0%** — and this holds
individually on EVERY session, not just on average (oracle-detector's
`gt_path_cost` is 5–10x lower than oracle-classifier's on each one).

**Consequence for Stage A:** the binding error stream is detector-side
(misses cascading into `C_INS`-priced insertions, plus phantoms), not
classifier substitutions — confirming the design brief's expectation
from the raw-error decomposition (§1) with a real causal measurement
rather than a correlational one. Stage A's dense 13-way posteriorgram
should be built and evaluated with detector recall / background-class
separation as the PRIMARY target; per-class (which-of-12) discrimination
is real but secondary — six independent classifier-side interventions
were already flat (`jitter-retrain-decoder-flat`, `encoding-rework-flat`,
`optical-flow-rejected`) before this measurement, which is now the causal
confirmation of why. Do not spend Stage A's first iteration tuning
class-balance or input colour before the background/onset channel is
solid.

Reproduce: `python oracle_attribution.py --out results/2026-07-28/oracle_attribution_result.json`
(run from `move_detector/`; ~30–40 min on GPU, dominated by beam-16000
retries on the `real`/`oracle-classifier` conditions that don't verify).

---

## 4. Phase 2 — the architecture (Path 4, scoped as evolution)

**Core change: one temporal model emitting a per-frame 13-way
posteriorgram (12 quarter-turn classes + background) over the whole
session, replacing onset-score → peak-pick → window → ResNet softmax.**

What this dissolves, by construction: the anchor quantization (no
anchors), window squeeze/collapse and neighbour corruption (no windows),
the phantom problem (background competes in the same softmax, per
frame), the peak-picking threshold (the decoder consumes the lattice
directly), and the miss cliff (a sub-threshold move is attenuated
evidence, not an absent row).

### Stage A — minimal version (~1 week including 2-seed training)

Evolve `move_detector/model.py`, don't replace it:

- **Input:** the current stream is grayscale+diff at 96px — the detector
  literally cannot see colour, so it cannot name faces. Extend
  `prepare_data.py`'s stream to RGB + diff (4–5ch; keep 96px first,
  128px only if measured necessary). This touches the stream cache
  format — version it.
- **Model:** keep `FrameEncoder` + TCN skeleton. Widen the head from
  `Conv1d(feat, 1, 1)` to two heads: the existing onset head (unchanged,
  so every existing metric/decode path still runs during transition) and
  a new `Conv1d(feat, 13, 1)` class head. ~1.4M params today; even
  doubling encoder width stays cheap. `score_stream()`'s chunked
  inference with discarded margins carries over — extend it to return
  the (T, 13) posteriorgram.
- **Targets:** dense framewise labels from BLE timestamps — reuse the
  sigma=1 Gaussian-bump construction already in `train.py`, one bump per
  move in its class channel; frames beyond the bumps are background.
  Per-frame cross-entropy (soft targets). Background dominates ~90% of
  frames: class weighting will need measuring, but note `--pos-weight`
  was measured at sigma=1 and deliberately left OFF for the onset head
  (commit d551fe1) — measure, don't assume, for the 13-way head too.
- **Why framewise dense supervision and NOT CTC:** CTC exists to solve
  an alignment problem we do not have — BLE gives exact sub-frame
  timestamps for every move. CTC's peaky-blank behaviour would also
  reinvent the thresholding this rework removes. CTC is the fallback if
  framewise targets somehow misalign, not the default. (Rev 2 §8 said
  "CTC-style"; the *style* — frame-synchronous lattice with a blank —
  is kept, the CTC loss is not needed.)
- **Decode integration:** feed the posteriorgram to `reconstruct.py` as
  the lattice. This generalizes D1's `--candidate-threshold` path
  (already built and verified mechanically sound): candidate onsets =
  local maxima of (1 − background posterior), each carrying its own
  12-way distribution; deletion cost from the posterior itself replaces
  `score_del_costs`' calibration. Behind a flag, defaulting to the old
  path; A/B in one command, per house style. Bump the replay cache key.
- **Evaluation:** 2 seeds; `metric_audit.py` on the 10 live takes
  reporting **insertion / deletion / substitution counts**, not just
  accuracy (G2's attribution says which count has to move); honest-six
  decode with the G2 table refreshed; falsifiability sweep (Phase 0
  harness) re-run since the decoder input changed.
- **Ship gate:** live end-to-end must beat 73.2% by more than the
  2.3-point seed envelope, with the improvement concentrated in the
  error type G2 named. Verified-rate movement on the honest six is the
  headline but may lag — the §2 budget says it stays near 1/6 until
  per-move accuracy approaches the mid-90s, so *trend in gt_path_cost*
  (sessions moving from 36–119 toward the ~4–8 envelope) is the honest
  intermediate metric.

  **FIRST RESULT, MEASURED 2026-07-28 — real, seed-consistent, and the
  biggest single movement in this metric since the project started.**
  Implementation: `prepare_data.py --color` (BGR+onset_class streams,
  `detector_stream_color.npz`, independent of the deployed grayscale
  stream), `model.build_joint_model`/`score_stream_joint` (widened
  `OnsetDetector` — RGB+diff input, 13-way class head sharing the onset
  trunk), `dataset.JointSessionStream`/`JointClipDataset`
  (`build_dense_targets` — dense per-frame 13-way soft targets from the
  same Gaussian-bump construction the onset head already used;
  `WCA_FLIP_PERM` remaps the class target on flip augmentation),
  `train_joint.py` (mirrors `train.py`'s structure; loss = onset BCE +
  soft-label class CE), `joint_decode.py` (`posteriorgram_to_moves` —
  peaks on 1-background, each carrying its own renormalized 12-way
  softmax — turns the joint model's output into the EXACT `moves`-list
  format `reconstruct.costs_from_moves` already consumes, so no decoder
  code changed at all), `verify_joint.py` (decode-level measurement,
  parallel to `oracle_attribution.py`'s methodology).

  Trained 2 seeds (0.59M params, same 40-session corpus, SAME named
  holdout the deployed detector/classifier use — `solve_20260721_102711`,
  `solve_20260722_101225`, `solve_20260723_105530_solve`,
  `solve_20260724_100120_solve`; `solve_20260720_142006` excluded, no
  `frames/`). Both converged to onset F1 96.7–96.9% and at-onset class
  accuracy 95.8–97.0% on these 4 held-out sessions — already exceeding
  the deployed detector's own recorded-session F1 (~93.9%) with ONE
  model and no window/anchor machinery at all.

  Decoded through the identical `reconstruct.py` machinery (same cost
  model, same beam/retry policy) every other measurement in this project
  uses, and compared to `oracle_attribution.py`'s "real" (deployed
  detector+classifier) column on the SAME 4 sessions:

  | session | raw acc: real → seed0 → seed1 | gt_path_cost: real → seed0 → seed1 |
  |---|---|---|
  | solve_20260721_102711 | 89.5 → 90.8 → 90.8% | 106.2 → 58.7 → 62.0 |
  | solve_20260722_101225 | 90.4 → 93.3 → 97.2% | 73.7 → 42.7 → 30.2 |
  | solve_20260723_105530_solve | 91.5 → 97.2 → 97.2% | 36.1 → 20.3 → 27.8 |
  | solve_20260724_100120_solve | 91.8 → 97.8 → 97.8% | 93.6 → 36.6 → 19.0 |

  `gt_path_cost` falls on EVERY session, BOTH seeds, no exceptions —
  average reduction 49% (range 23–80% per session/seed pair). Phantom
  counts drop sharply too (e.g. session 4: 17 phantoms real → 6 → 2) —
  consistent with the hypothesis that a real competing background class
  per frame suppresses false onsets far better than a hard peak-pick
  threshold on a separate model's score curve. **Four independent prior
  interventions (D1, D2, D3, the jitter retrain) moved this number NOT
  AT ALL** (`decoder-sprint-exhausted`, `jitter-retrain-decoder-flat`) —
  this is the first thing that has actually closed the detector-side gap
  G2 identified, rather than working around it.

  **Read the honest limits before calling this a win:**
  - **Verified count did not move**: 0/4 on this exact subset for BOTH
    the real baseline and both Stage A seeds (the "1/6" honest figure
    elsewhere includes `solve_20260726_165044_scramble`, not in this
    4-session holdout). Per §2's information budget, gt_path_cost still
    has to reach the ~4–8 envelope before verification itself moves —
    halving a 106-cost gap to 59 is real progress but still far outside
    it. Track the TREND, not the verified count, as this doc's own rule
    says.
  - **This is NOT yet the ship-gate metric.** §5's gate is live
    end-to-end vs. 73.2% (10 saved live takes, `metric_audit.py`-style
    TIME-based scoring) — that needs a LIVE inference bridge (webcam
    capture → colour crop stream → joint model), which does not exist
    yet. What's measured above is the recorded-session decode-level
    number, methodologically identical to G2's own comparison and a
    fully legitimate signal, but distinct from and a prerequisite check
    before the literal ship gate.
  - Only 2 seeds, per-session spread is real (e.g. session 3: 20.3 vs
    27.8) — directionally unanimous and large relative to that spread,
    but don't quote a single seed's number as *the* result.

  **Full 40-session sweep, MEASURED same sitting.** `verify_joint.py
  --sessions ../training_data/solve_*/` (both seeds), matching the
  all-sessions/honest-subset convention this project always reports:

  | metric | deployed (baseline) | Stage A seed0 | Stage A seed1 |
  |---|---|---|---|
  | ALL verified | 19/40 | **29/40** | **29/40** |
  | ALL exact | 10/40 | 25/40 | 27/40 |
  | ALL raw → system acc | 92.3% → 92.9% | 97.9% → 98.2% | 98.2% → 98.4% |
  | ALL gt_path_cost median | — | 5.5 | (not logged, ALL not the headline) |
  | UNSEEN (4 sessions) verified | 0/4 | 0/4 | 0/4 |
  | UNSEEN gt_path_cost median | 83.65 (median of 106.2/73.7/36.1/93.6 above) | 39.7 | 29.0 |

  Both seeds land on the EXACT SAME all-sessions verified count (29/40)
  independently — a striking level of agreement, not just directional
  consistency. Exact reconstruction more than doubled (10→25–27).
  **Read this number correctly: it is inflated by training-session
  inclusion, exactly the same way the deployed baseline's own 19/40
  always was** (only 4 of 40 sessions are held out) — it is a fair
  like-for-like comparison of the SAME kind of number, not a claim about
  real-world generalization. The UNSEEN subset is where that claim lives,
  and there the verified count is unchanged (0/4 both, matching the
  deployed baseline's own 0/4 on this identical subset) — but its
  `gt_path_cost` median fell from the low-60s/70s range down to
  29.0–39.7, continuing the same halving trend the 4-session table
  showed individually. These 4 sessions are also, not coincidentally,
  the deployed classifier's OWN canonical cross-environment holdout
  (chosen specifically to span different recording environments) — the
  hardest generalization test this project has, so it not fully cracking
  yet after one architecture iteration is not a surprising result; the
  trend is the strongest ever measured toward it.

  **Live capture bridge: BUILT 2026-07-28, not yet human-tested.**
  `verify_solve.py --joint [--joint-model checkpoints/move_joint_seed0.pt]` swaps the
  live analysis step for `joint_decode.analyse_joint_live` (colour crop
  stream via `prepare_data.build_color_stream`, `score_stream_joint`,
  `posteriorgram_to_moves`) — everything else (scramble generation,
  phase1/phase2 flow, `verify_claim`, the falsifiability sweep, `--ble`
  ground truth) is untouched, since it only ever consumed the `moves`
  list format, unchanged here. `--joint --session` is refused with a
  pointer to `verify_joint.py` (offline replay isn't wired through this
  path — no reason to duplicate it). Validated end-to-end (crop → colour
  stream → model → move sequence → confidence report) by replaying a
  recorded session's raw frames through `analyse_joint_live` exactly as
  live capture would feed it, with no webcam involved — this is the
  furthest the pipeline can be exercised without a physical camera/cube
  in hand, and it ran clean. **What it has NOT been checked against**:
  a real webcam, a real cube, real lighting, or the literal 73.2%
  ship-gate number — that first live run **is** the check, and it should
  be run before trusting `--joint` beyond curiosity. The deployed
  path (`verify_solve.py` without `--joint`) is completely unaffected —
  same models, same code, byte-identical behaviour.

  **Not yet done, in priority order:** (1) an actual live human test
  session with `--joint` (mandatory before any live-metric claim); (2)
  more honest-holdout sessions or a second cross-environment holdout to
  get more than 4 data points on the claim that matters most; (3) Stage
  B (capacity) is very likely NOT needed yet given how far ALL-sessions
  numbers moved with a 0.59M-param model — re-assess only after (1) and
  (2).
  Reproduce: `python prepare_data.py --sessions ../training_data/solve_*/
  --color`, `python train_joint.py --sessions ../training_data/solve_*/
  --val-session-names solve_20260721_102711 solve_20260722_101225
  solve_20260723_105530_solve solve_20260724_100120_solve --seed 0
  --output checkpoints/move_joint_seed0.pt`, `python verify_joint.py --model
  checkpoints/move_joint_seed0.pt --sessions ../training_data/solve_*/`, `python
  verify_solve.py --joint` for a live test.

### Stage B — capacity, only if Stage A is limited by it

Diagnose first (train vs val vs live gaps), then in order of cost:
wider/deeper TCN (receptive field is only ~2s; algorithm bursts span
more); pretrained encoder trunk (ResNet-18 first blocks on the crop,
matching the classifier's proven transfer); a small temporal transformer
over frame embeddings *only* with strong regularization. **Explicitly
deprioritized: large video transformers (VideoMAE/TimeSformer-class).**
~40 sessions / ~3300 moves will not feed them, and the val-doesn't-rank-
live finding makes their val-score appeal worthless here.

### Stage C — RAFT flow channels, only if G1 passed

**CANCELLED 2026-07-28 — G1 failed decisively (54.4% vs Farneback's own
70.4% on the identical set, vs an ~85% gate; see G1's measured result
above).** This stage does not exist. Do not revive it without new
evidence that specifically addresses why RAFT produced WORSE motion
information than Farneback here, not just "a different flow model
might work" — the mechanism gap (appearance beats clean motion) has now
survived two different estimators.

---

## 5. Parallel levers (run during training downtime — they raise the ceiling the model reaches for)

- **Phase 0, before anything trains: the falsifiability batch harness**
  (rev 2 §7). Pure orchestration around `verify_solve.py`, ~40 sessions
  × decoy types, Clopper–Pearson bound. Outstanding since the last
  sprint, required by every later measurement, and the ideal first task
  for the implementing model. 0/150 → ~2.4% bound = quotable.

  **MEASURED 2026-07-28.** Implemented as `falsifiability_batch.py`
  (pure orchestration around `verify_solve.run_session`, no new decode
  logic; Clopper–Pearson via bisection on the exact binomial CDF — no
  scipy dependency in `requirements.txt`). Ran the full 40-session sweep:
  19/40 verified (matches the known aggregate), every verified session
  got a complete 6-decoy sweep (19×6=114 attempts).

  **Pooled bound is weak and would be misleading quoted alone: 38/114
  accepted → 95% upper bound 41.3%.** But this pools near-decoys the cost
  model is EXPECTED to sometimes accept (that's what "repair budget"
  means, not a defect) with the two actual frauds a real verifier has to
  reject. Broken out by type, the story is completely different:

  | decoy type | accepted/n | 95% bound |
  |---|---|---|
  | 1 move off | 16/19 | 95.6% |
  | 2 moves off | 12/19 | 81.3% |
  | 4 moves off | 8/19 | 63.2% |
  | 8 moves off | 0/19 | 14.6% |
  | never scrambled | 2/19 | 29.6% |
  | scrambled by something else | 0/19 | 14.6% |

  The near-decoy rows (1/2/4 moves off) are not a problem — falling
  inside the repair budget is what the cost model is calibrated to do,
  per `verify_solve.py`'s own falsifiability docstring. **The two
  outright-fraud rows are the actual security claim, and one of them has
  a real gap: "never scrambled" was accepted 2/19 times** (a claim that
  the cube started scrambled, when it was actually already solved,
  slipped through) — "scrambled by something else" was clean (0/19).
  No session had a wrong claim CHEAPER than the true one (the
  worse failure mode) — every accepted decoy cost strictly more, median
  +8.00, min +1.67 (matches one `C_INS`≈4.0-per-quarter-turn-of-error, as
  the cost model predicts). This is the honest baseline every future
  decoder-facing change must be re-swept against — a shrinking median
  margin, or a rising "never scrambled" accept rate, is the early warning
  that a lever is eroding security while true-accept goes up.

  Reproduce: `python falsifiability_batch.py --sessions
  "../training_data/solve_*/" --out results/2026-07-28/falsifiability_batch_result.json`
  (run from `move_detector/`; ~1 hour on GPU — dominated by beam-16000
  retries on sessions that don't verify at the default beam).
- **Pacing prompt** (rev 2 §3a, ~1 day): "one move at a time" UX
  directly thins the crowded-move regime that drives misses.
- **Data in the failure conditions** (rev 2 §2e): record sessions in
  saturated lighting; more live-style free solves generally — the
  detector's live-vs-recorded recall gap (96%→78-84%) smells like
  condition shift, and no architecture conjures signal that isn't in
  the training distribution.
- **60fps capture** (rev 2 §2c, exploratory): the posteriorgram model
  makes higher fps nearly free to exploit (no window arithmetic to
  redo) — revisit after Stage A ships.
- **Unexplained outlier** `solve_20260726_100142_solve` (rev 2 §2d):
  still open; cheap to re-examine with the Stage A model's posteriorgram
  visualized over its timeline.

## 6. Sequence

```
Day 1      Phase 0 harness + G1 (RAFT probe) + G2 (oracle attribution).
           All three independent; none trains anything.
Day 2      Decision point: G1 verdict on flow, G2 verdict on which error
           stream binds. Start Stage A (stream format + heads + targets).
Days 3–7   Stage A: train 2 seeds, posteriorgram decode integration,
           full measurement block (live takes, honest six, gt_path_cost
           attribution, falsifiability).
Then       Stage B/C per their gates. Re-assess against §7 targets.
```

## 7. Success criteria (unchanged from rev 2 §1, restated)

- Headline: honest-subset verified count > 1/6, trending toward the
  ~90% true-accept MVP target (still a straw man — confirm with user).
- Per-move live end-to-end climbing toward the 96–97% budget line, with
  gt_path_cost on the stubborn five shrinking toward the ~4–8 envelope.
- False-accept bounded by the Phase 0 harness, re-run after every
  decoder-facing change; margin regressions are hard blockers.

Related: `PATH_TO_VERIFICATION.md` (rev 2 — record of everything
rejected and why; §4's do-NOT-reinvent inventory still applies),
`reconstruct.py` docstring (required reading before touching decode),
`encodings_move.py` (flow utilities, flip_fixup), `flow_direction.py`
(the G1 probe), `phantom_confidence.py` (the phantom audit),
`window_audit.py` (the seam-error ladder).
