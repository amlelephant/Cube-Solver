# CubeArena — Final Roadmap to Launch

**Status:** adopted 2026-08-03 · **Owner:** Aiden
**Relationship to other docs:** this is now the single execution plan to launch.
`docs/ROADMAP.md` (product phases) and `SYSTEM_DESIGN.md` (platform S0–S6) feed
into it and remain the detailed reference for their tracks;
`ble/move_detector/GAMEPLAN.md` remains the model-side plan and its measured
records stand. `TODO.md`'s five items are each dispositioned in §7. Where this
doc and an older doc disagree, this doc wins.

---

## 1. The strategic reframe this plan is built on

One measurement reorganizes everything: **move-by-move verification is dead,
and coaching is alive — on the same model, at the accuracy we already have.**

- Verification from the move stream is a cliff at ~98–99% raw per-move
  accuracy (`ACCURACY_TARGET.md` §4.1), held-out verification is 0/5, and the
  decode rework's measured ceiling (~96.5–97.1% daytime) sits below the cliff.
  No scheduled work reaches it. Stop aiming the move detector at verification.
- Coaching does not sit on that cliff. Phase splits, TPS curves, pause maps,
  and algorithm timings **degrade gracefully** with word errors and **average
  out across solves**. At the measured 95.6–96.1% held-out daytime post-decode
  accuracy, per-solve analytics are already trustworthy and multi-solve trends
  are rock solid. The paid feature was mis-specified as verification; correctly
  specified, it is already within reach.

So the product splits cleanly, and each half stands on what is actually proven:

| Tier | Feature | Trust mechanism | Status |
|---|---|---|---|
| **Free** | Verified solves + weekly comps | State bracketing (6-face scans) + continuity guard + server re-verification | CV pipeline works on desktop; browser port + backend are the work |
| **Paid ("Coach")** | Move recording + solve coaching | None claimed — accuracy tier only, confidence-graded | Model meets its daytime ship gate today; coach product layer is the work |

Verification is never paywalled (growth loop, per `docs/ROADMAP.md` §1).
`VERIFIED` verdicts never come from the move decode — the anticheat stream owns
them, and any decoder change touching a verdict re-runs the decoy sweep first.

## 2. Where we are — measured, 2026-08-03

Model side (12 held-out sessions × 2 CTC seeds, post-decode word accuracy vs
BLE truth, never pooled across lighting):

| Regime | Seed 0 | Seed 1 | CTC structural ceiling |
|---|---|---|---|
| Daytime mean | **96.1%** | **95.6%** | 99.2% |
| Daytime per-session floor | 90.9% | **88.2%** (`102711`) | — |
| Evening mean | 73.3% | 73.6% | 99.1% |

Ship-gate bundle (GAMEPLAN §2), item by item:

1. Daytime held-out mean ≥95% — **met, both seeds.**
2. Daytime per-session floor ≥90% — **met seed 0; missed by 1.8 pts on seed 1**
   (one session, 88.2%). Treat as marginal, expect the evening-corpus retrain
   to move it; re-check then rather than building anything for it now.
3. Capture-time lighting prompt — **not built.** Prototype exists
   (`lighting_check.py`); needs the weak-peak-fraction check verified against
   the known takes, then wired into capture. This is the cheapest +21-point
   feature in the project.
4. Evening — not a ship blocker given (3); fix is the evening recording corpus
   (GAMEPLAN §5), which is a **human, calendar-bound** task.

Platform side: `web/` is a real Next.js shell with mock data — no camera, no
inference, no backend, no ONNX export anywhere. Everything server-side is
greenfield, planned in `SYSTEM_DESIGN.md` S0–S6.

Known structural limits, so nobody re-attacks them: decoder anchor-selection
ceiling +0.65 daytime (measured, rejected); algorithm library is user-specific
by construction; 30 fps caps elite solvers (R2-class fix known, in software);
speed drift means the corpus goes stale — recording is maintenance, not a
one-off.

## 3. The Coach — how the coaching feature actually gets built

This was the open design question; here is the answer. The coach is a layered
analytics engine over the decoded, timestamped move list. The layers are
ordered by robustness to decode errors, and each layer only consumes what the
layer below can guarantee.

### 3.1 What we already have per solve

- Timestamped move list (CTC decode + reconstruct), ~96% daytime accuracy.
- The scramble — server-issued and signed, so the start state is **known**,
  which makes replay-based analysis deterministic.
- Per-segment grading (D3 schema: `exact` / `repaired (n edits)` /
  `best-effort`) — planned in GAMEPLAN §4-D3, not yet built. **This is the
  coach's foundation and is hereby promoted from "worth doing" to
  launch-critical.**
- The mined algorithm library: identifies this solver's last-layer algorithms
  at 91–97% held-out, including on spans the decode got wrong
  (`algorithm_gate.py`, the backward peel).

### 3.2 The four layers

**L1 — Timing (no move identity needed; works even in evening light).**
TPS curve over the solve, pause map (inter-onset gaps above a threshold),
longest pause, inspection usage. Pure onset timestamps. Ships everywhere,
always.

**L2 — Phase splits (needs the move list; confidence-graded).**
Replay the decoded moves from the known scramble state and detect CFOP
milestones with pure group-theory predicates: cross complete (and on which
color), F2L slots filled 1→4, OLL done, PLL done. Deterministic, ~200 lines,
no ML. New module: `ble/move_detector/coach/phases.py`. Output: time and move
count per phase, recognition pause entering each LL stage. A segment graded
`best-effort` contributes timing only; milestones inside it are suppressed,
not guessed.

**L3 — Case and algorithm recognition (needs L2 + the user's library).**
Which OLL/PLL case occurred (from the replayed state at LL entry) and which
algorithm the user executed for it (word-match via the existing
`algorithm_gate` machinery). Per-case execution time and recognition time,
tracked across solves. The library is user-specific **by construction**
(GAMEPLAN §4-D1: it matches executed words, not permutations) — so the coach
has an explicit **enrollment phase**: it mines the user's repertoire over
their first ~15–20 solves, and alg-level insights unlock as coverage grows.
Product copy writes itself: *"your coach is learning your solutions."* This
converts the library constraint from a defect into onboarding.

**L4 — Insights (rules over L1–L3 aggregates; no ML, no LLM required).**
Rule-based comparisons against the user's own history: slowest PLL cases vs
their case average, recognition pauses concentrated on specific cases, F2L
pause clustering (the lookahead signature), move count per phase vs personal
median, week-over-week trends. Aggregation is the error-killer: at ~4% word
error a single solve's split can shift by a move; a 10-solve average cannot.
Single-solve views show graded segments honestly; trend views are the
headline. An LLM-phrased weekly summary is an optional garnish later, not a
dependency.

### 3.3 Delivery architecture — decided

**The coach runs server-side in Python. The decode stack is never ported to
JS.** The client already runs the YOLO detector for the continuity guard
(free tier); for paid solves it additionally uploads the move-window crops
(detector boxes it already computed) plus the evidence bundle. A Python worker
(Celery, per SYSTEM_DESIGN S3) imports `ble/move_detector/` unchanged, runs
CTC + decode + coach layers, and posts results async — "your analysis is
ready" is acceptable and honest UX. Costs scale only with paying users;
inference is CPU-fine at this model size. Full-frame video upload is the
fallback if crop-only proves lossy — decide by measuring decode parity on
crops vs frames during B4 (below), not by preference.

### 3.4 Rotations — launch policy

Cube rotations break face identity for everything downstream (TODO item 5).
Launch policy: **detect and degrade, don't guess.** A decode that fails its
endpoint check or grades mostly `best-effort` is flagged "limited analysis —
cube rotation suspected"; L1 timing stats still ship for that solve. Rotation
*inference* (or a lightweight rotation detector) is the first post-launch
coach upgrade, and the coach is not "done" until it lands — but it does not
block launch, because the failure mode is honest degradation, not wrong data.

## 4. The three tracks

Work runs on three parallel tracks. Track A is the critical path in calendar
time; Track B's first item is human and starts tonight; Track C rides on A's
infrastructure.

### Track A — Platform + free tier (critical path; SYSTEM_DESIGN S0–S5 verbatim)

| Step | Weeks | What | Gate |
|---|---|---|---|
| A0 | 1 | Deployment skeleton: Django+DRF, compose, Caddy one-origin, COOP/COEP, CI | `docker compose up` serves Next + `/api/health` from one domain |
| A1 | 2–5 | **Browser inference spike** — ONNX export, ORT-Web harness, color-stratified parity (orange/red over-represented), latency on ≥4 device classes, tracker decision | 9 sticker labels in-browser matching Python on a test set |
| A2 | 4–7 | Django core: models, allauth, **signed server-side scrambles**, admin review queue | Scramble issued → solve recorded → visible in admin |
| A3 | 7–9 | Evidence bundle schema, R2 upload, CPU re-verification worker importing `cv/` | Tampered bundle caught; 20 honest bundles pass |
| A4 | 8–12 | Solve flow in browser: getUserMedia, guided 6-face scan, pre-scan orange/red calibration (never gated on detection), continuity guard v1, result card | Full flow <3 min; deliberate swap DQ'd; 20 legit solves no false DQ |
| A5 | 12–15 | Weekly async competition + leaderboard, 100% re-verify for leaderboard entries | Two consecutive comps with ≥30 participants |

Phase 0 (waitlist + demo video + community seeding, `docs/ROADMAP.md`) starts
**now**, in parallel, and gates nothing technical — but if it can't find ~1k
emails, reposition before Track A's later steps.

### Track B — Move engine + Coach (the paid tier)

| Step | When | What | Gate |
|---|---|---|---|
| B1 | **Now, human, calendar-bound** | Evening recordings (10–15 sessions, ≥3 evenings, varied lamps, separate-evening holdout) **plus** ongoing daytime sessions (library coverage + speed-drift maintenance) | Corpus recorded per GAMEPLAN §5.1 |
| B2 | Now, small | Wire the lighting prompt: verify `lighting_check.py` separates the known takes (56/63% evening vs 90%+ morning), add clock-time heuristic, surface at capture | Prompt fires on the known evening takes, silent on morning takes |
| B3 | Next code work | **D3 graded per-segment output** — `exact`/`repaired`/`best-effort` schema through decode → result dict → JSON | Numbers reconcile with the algo-sweep output |
| B4 | After B3 | Coach L1+L2 (`coach/phases.py`, pause/TPS analyzer) against recorded sessions; decide crops-vs-frames upload by measuring decode parity | Phase splits match hand-checked truth on 10 sessions |
| B5 | After B1 lands | Evening retrain (2 seeds, widened aug kept), re-run `eval_lighting.py` then `algo_sweep.py`, re-check the full ship-gate bundle incl. the seed-1 floor | Evening up substantially, morning within 1 pt, floor ≥90 both seeds |
| B6 | Weeks ~8–12 | Coach L3+L4 + enrollment flow + server worker integration (rides on A3's worker plumbing) | End-to-end: paid solve → async analysis with graded insights |
| B7 | Post-launch #1 | Rotation inference or detector; then 60 fps / R2-class work as speed demands | — |

Optional, strictly time-boxed: D4 adaptive LM fusion (helps cross-day both
seeds) — attempt only if B5's retrain leaves daytime short of 97%, and judge
on cross-day holdouts only. Everything on GAMEPLAN's do-not list stays dead.

### Track C — Anticheat + trust (owns every VERIFIED verdict)

| Step | When | What |
|---|---|---|
| C1 | With A4 | Record deliberate swap-attack sessions, **including the table-edge/below-frame swap** (TODO item 1's known hole); tune trajectory-continuity and boundary-exit flags against them; add a frame-edge occlusion heuristic if flags alone miss it |
| C2 | With A4 | The never-scrambled fraud check (baseline: 2/19 slip) and the decoy-sweep discipline: any change to what emits VERIFIED re-runs the sweep |
| C3 | With A2/A3 | Human review queue in Django admin — the honest backstop for top leaderboard slots; sleight-of-hand review stays cheap because it's only ever the top slice |
| C4 | Standing | The paid tier's move stream is **corroborating evidence** for anticheat (moves visibly occur on camera), never the verdict source |

## 5. Launch definition and go/no-go

**Launch = free verified weekly comps + Coach in paid beta, same day.**
Target: **~16 weeks** (A5 complete + B6 complete), consistent with
SYSTEM_DESIGN's week numbering. "Features worth paying for" is satisfied by
the Coach at founding-member pricing (~$4/mo, per `docs/ROADMAP.md` Phase 4 —
beta pricing derisks promising too much too early).

Go/no-go checklist, all measurable:

- [ ] A4 exit: ≥95% first-try face-scan success across ≥5 webcams/rooms
- [ ] A4 exit: deliberate swap (incl. table-edge variant) DQ'd; false-DQ <1%
      on ≥20 legit solves
- [ ] A5 exit: two weekly comps with ≥30 participants
- [ ] Ship-gate bundle holds after B5: daytime mean ≥95% both seeds,
      per-session floor ≥90% both seeds, lighting prompt live
- [ ] One live session and its replay scored by the **same** scorer agree
      (GAMEPLAN §2's quote-rule; the live-scoring misattribution means the
      live and replay numbers are not yet comparable)
- [ ] Coach end-to-end on 10 internal solves: phase splits sane, no insight
      emitted from a best-effort segment, enrollment flow completes

Post-launch order (unchanged from `docs/ROADMAP.md` Phase 3–4): creator
outreach and public comps → retention gate (≥40% week-2) → live rooms only
after liquidity → rotation inference and 97% daytime as the first model
milestones.

## 6. Metrics dashboard (weekly)

The six from `docs/ROADMAP.md` §4 (face-scan success ≥95%, activation ≥40%,
week-2 retention ≥40%, shares ≥10/100, false-DQ <1%, infra <$0.02/MAU —
annotated: fixed costs dominate below ~5k MAU), plus two new paid-tier lines:

| Metric | Target | Why |
|---|---|---|
| Coach solve-analysis completion rate | ≥95% of paid solves get a full (non-degraded) analysis | The paid promise, honestly measured |
| Daytime post-decode holdout mean | ≥95%, tracking toward 97% | The model's health line; re-measured per retrain, both seeds, split by regime |

## 7. Disposition of TODO.md

1. **Anticheat finalization** → Track C (C1–C3). The table-edge swap gets its
   own recorded attack sessions and gate. The "charge for move recording"
   idea → adopted as the Coach tier (§1); the move stream corroborates but
   never issues verdicts (C4).
2. **Django backend** → Track A, A0–A3, per SYSTEM_DESIGN unchanged.
3. **Backtracker redesign** → resolved by measurement, not redesign. The
   algorithm method *was* implemented correctly — the forward arm is
   structurally inert (gate never admits, `ALGORITHM_PRIOR.md` §8a) and
   anchor *selection* has a measured +0.65 ceiling with the obvious rule
   scoring −6.4. Insertions/deletions are confirmed as the real channel (81%
   of error mass) and their fix is CTC + data, already shipped/scheduled.
   What survives: D3 graded output (B3) and optional D4. Nothing else.
4. **Move detection finalization** → the goal is now set (this was the item's
   own first ask): **not** 97%-for-verification (cliff unreachable) but the
   ship-gate bundle for the *coach* tier, which daytime already meets. The
   cross-environment gap is quantified (proximity ladder; evening −22 pts)
   and its fix is B1/B5 data, with speed drift as a standing maintenance axis.
5. **Rotations** → §3.4: detect-and-degrade at launch, inference as
   post-launch coach work B7. The product claim at launch is scoped to
   rotation-free solves, stated in the UI, with honest degradation otherwise.
