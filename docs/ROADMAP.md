# CubeArena — Roadmap to Launch

**Status:** adopted 2026-07-14 · **Owner:** Aiden · **Supersedes:** nothing (VISION.md remains the long-range product doc; this is the execution plan)

This document is written to be executable: every task has a definition of done
(DoD), and every decision states its reasoning so it doesn't get silently
relitigated. If you are picking up work, find the current phase, take the top
unchecked task, and satisfy its DoD. Do not pull tasks from later phases
forward without updating this file first.

---

## 1. The one-sentence strategy

CubeArena wins by owning **trust**: a camera-verified, cheat-proof solve is
the product; everything else (timers, leaderboards, live play) is packaging
around that proof.

Consequences that all other decisions follow from:

- **Verification is never paywalled.** It is the growth loop (shareable
  "verified solve" cards).
- **Verification runs client-side** (in the browser). Server costs stay near
  zero, which is what makes the business viable at hobby-market scale.
- **Async before live.** Weekly same-scramble competitions work with 10
  users; live matchmaking feels dead below ~1k concurrent. Build the arena
  only after there are gladiators.

---

## 2. Decision record: how do we determine moves?

### The question

To verify a solve, do we need to know every **move** the solver made, or only
the **states** the cube passed through?

### The options, honestly assessed

| # | Approach | How it works | Verdict |
|---|----------|--------------|---------|
| A | **State-only verification** (scan before + after, continuous video between) | Verify scramble state with 6-face scan → record uncut video while solving → verify solved state. Validity checked by group theory (`cv/solver/twophase/`). | ✅ **Adopt for launch.** Already ~works in `cv/solver/state_finder.py`. Cheat-proofness: a continuous, uncut camera feed bracketed by two verified states cannot be faked without actually performing the solve — **provided the solve-continuity guard (§2.1) also holds**, otherwise a cube-swap defeats it. |
| B | **BLE smart cube move stream** (`ble/cube_ble.py`) | Hardware reports every move with timestamps. Ground truth. | ✅ **Keep as secondary input**, never a requirement. ~1% of cubers own compatible hardware. Its real value: (1) premium replay/analytics feature later, (2) the *labeling engine* for option C training data (`ble/record_training.py` pipeline). |
| C | **Video move classification** (current R&D: ResNet-18 on temporal diff images, `ble/train_move_classifier.py`) | CNN classifies each turn from webcam frames. | ⚠️ **Defer. Do not block launch on this.** See analysis below. |
| D | **Mid-solve state snapshots + solver reconstruction** | Periodically re-scan state during the solve; infer moves between snapshots. | ❌ Reject. Hands occlude the cube during actual solving; you cannot get clean face scans mid-solve at speed. |
| E | **Audio/IMU novelty approaches** | Classify moves from turning sounds etc. | ❌ Reject. Unreliable, no independent value. |

### Why C (video move classification) is deferred — the honest math

- Real solves run **3–10 turns per second**. A 30 fps webcam gives 3–10 frames
  per move, with motion blur and with **hands occluding most of the cube most
  of the time**. This is a genuinely hard fine-grained temporal action
  recognition problem, not an incremental extension of the sticker classifier.
- The current temporal-diff ResNet approach is a reasonable first attempt, but
  expect a long tail: distinguishing R from R' from Rw under occlusion at
  speed is where it will plateau.
- **And the product does not need it.** Option A is sufficient for cheat-proof
  verification. Move data adds *analytics* (reconstructions, TPS graphs,
  phase splits) — a differentiating **post-launch premium feature**, not a
  launch requirement.

### 2.1 The swap attack and the solve-continuity guard

**The attack:** verify the scramble on cube A, then mid-solve swap in an
identical cube B that is already solved, and present B at the end scan. State
bracketing alone cannot detect this — it assumes the object at second 0 is the
same object at second 12.

**The countermeasure (adopted):** run cube detection continuously from the
moment the scramble scan completes until the solved-state scan completes (not
just while the timer runs — the swap window includes inspection), and enforce:

1. **Uniqueness — hard DQ:** two confidently-detected cubes in the same frame
   at any point.
2. **Presence — DQ with tolerance:** no cube detected for longer than a gap
   threshold (start at 1.0s, tune on beta data). Brief detection flicker is
   normal — hands occlude most of the cube during real solving — and is
   bridged by the tracker (`cube_detector.py`'s YOLO-interval + CSRT
   architecture is already exactly this shape). A hard "any zero-cube frame →
   DQ" rule would DQ legitimate solves and must not be implemented.
3. **Trajectory continuity — flag:** the tracked cube cannot teleport
   (bounding-box displacement above a threshold between consecutive
   detections) and exits near the frame boundary are flagged. This closes the
   hole uniqueness alone leaves open: a swap performed just below the
   camera's field of view, where cubes A and B never appear simultaneously.

Every result stores a **continuity report** (per-interval detection count,
gap durations, max displacement, flags) alongside the state scans.

**Honesty clause:** no automated guard beats a determined sleight-of-hand
artist. The backstop is human review of the recording for top leaderboard
placements (top N% of any comp), which stays cheap precisely because it is
only ever the top slice. The guard's job is to make cheating cost more than a
fake time is worth, not to be unbeatable.

**Dataset dependency:** the detector is currently trained on ~270 clean cube
images and struggles with occlusion. Continuity monitoring needs it fine-tuned
on frames of *cubes being actively solved* (hands on, motion blur). The BLE
recording sessions (`ble/training_data/solve_<timestamp>/`) already contain
exactly these frames — use them.

### Standing decision

1. **Launch on A.** State verification + continuous recording is the MVP.
2. **Keep B alive** as the data-collection engine and future premium input.
   Don't let it crowd out CV work (per CLAUDE.md priority order).
3. **C is a post-launch research track** with an explicit promotion bar:
   ≥95% per-move accuracy at ≥3 TPS on held-out real solves before any
   product surface mentions it. Until then it ships nowhere.
4. If C keeps stalling, the fallback for analytics is B-only (smart-cube
   owners get reconstructions; camera users get times and states). That is an
   acceptable end state.

---

## 3. Phased roadmap

### Phase 0 — Proof + waitlist (weeks 0–6)

Goal: 1,000+ emails and one magical demo video. This phase also gates the
whole plan: if the community won't give ~1k free signups, reposition before
building more.

- [ ] Wire the landing page CTA to a real email capture (form → hosted list,
      e.g. Buttondown/Mailerlite free tier).
      **DoD:** submitting an email from the live page lands it in a list you
      can export; confirmation state shown in-page.
- [ ] Record the 45-second demo video: scramble issued → camera verifies →
      solve on camera → "VERIFIED 12.43s" card.
      **DoD:** one take, no cuts, real lighting, under 60s.
- [ ] Seed posts in r/Cubers, SpeedSolving.com, 2–3 cubing Discords, framed
      as a builder story ("I built a camera judge that makes online comps
      cheat-proof"), not an ad.
      **DoD:** posted in all three channel types; responses triaged into a
      feedback doc.
- [ ] Instrument the landing page (privacy-light analytics: page views,
      CTA conversion).
      **DoD:** you can state signup conversion % from real data.

**Exit criteria:** ≥1,000 emails OR a documented decision to reposition.

### Phase 1 — Port verification to the browser (weeks 2–10, overlaps Phase 0)

Goal: the `cv/` pipeline runs client-side in a browser. This is the single
largest technical lift in the plan and the reason the margins work.

- [ ] Export `detect_full_cube.pt` (YOLOv8) and `sticker_cnn.pt` to ONNX.
      **DoD:** ONNX models produce outputs matching PyTorch within tolerance
      on 20 test images (script checked into `cv/export/`).
- [ ] Build a browser inference harness (ONNX Runtime Web, WASM backend with
      WebGPU when available) reproducing the detect → warp → slice →
      ensemble-classify flow of `cv/detection/cube_detector.py` +
      `cv/classification/ensemble.py`.
      **DoD:** given a webcam frame in-browser, returns 9 sticker labels with
      confidences; matches Python pipeline on a recorded test set.
- [ ] Port the orange/red per-session calibration (`ensemble.calibrate()` /
      `run_calibration()`) — pre-scan step sampling a fixed box, **not**
      dependent on detection succeeding (see CLAUDE.md: this dependency was
      the historical bug; do not reintroduce it).
      **DoD:** calibration runs before any face scan; L/R centers sampled
      from raw frame.
- [ ] Port state assembly + validation. The two-phase solver is pure Python;
      either transpile the validity check (parity/permutation/orientation
      checks only — full solving not required client-side) or compile via
      Pyodide as a stopgap.
      **DoD:** invalid states rejected in-browser with a human-readable
      reason; valid states accepted; matches `cv/solver/state_finder.py`
      verdicts on a test suite of 50 states (include the known-tricky ones).
- [ ] Scan UX: guided 6-face capture with live overlay, per-face confidence,
      instant retake.
      **DoD:** a first-time user completes a full 6-face scan unaided.

**Exit criteria — the metric that decides everything:**
**≥95% first-try face-scan success in ordinary room lighting** across ≥5
distinct webcams/rooms. Track this number weekly; it is the product.

### Phase 2 — Closed beta: the Verified Solo Time Trial (weeks 10–16)

Goal: 100–300 waitlist invitees complete verified solves weekly.

- [ ] Solve flow: issue scramble → verify scrambled state → inspection +
      timer while recording continuously → verify solved state → result.
      **DoD:** end-to-end flow completable in under 3 minutes including scans.
- [ ] Verified result card: time, scramble, date, verification mark —
      rendered as a shareable image/clip.
      **DoD:** one-click download/share; looks good pasted in Discord.
- [ ] Minimal accounts + results storage (managed auth + DB, e.g.
      Supabase/Firebase free tier; video stays client-side, only states +
      times + hashes uploaded).
      **DoD:** results persist across sessions; total infra cost <$25/mo.
- [ ] One weekly async competition: same scramble set, leaderboard, closes
      Sunday.
      **DoD:** two consecutive weekly comps complete with ≥30 participants.
- [ ] Solve-continuity guard v1 (§2.1): continuous detection from
      scramble-scan to solved-scan; uniqueness DQ, presence DQ with gap
      tolerance, trajectory flags; continuity report stored with each result.
      **DoD:** a deliberate cube swap on camera is DQ'd in testing; 20
      legitimate beta solves in ordinary lighting pass without false DQ.
- [ ] Fine-tune the detector on hands-on solving frames sampled from
      `ble/training_data/` sessions (plus any beta failure frames).
      **DoD:** detection gap rate during real solves measurably drops vs. the
      current `detect_full_cube.pt`; no regression on the static-scan test set.
- [ ] Human-review queue for flagged results and top leaderboard placements.
      **DoD:** reviewer can watch the recording + continuity report and
      approve/reject in under 2 minutes per result.
- [ ] Feedback loop: in-app report on any failed scan (auto-attaches the
      offending frame with consent) feeding a triage queue.
      **DoD:** scan-failure rate measurable per week from real users.

**Exit criteria:** ≥40% of beta users return for a second weekly comp
(retention is the go/no-go signal for public launch).

### Phase 3 — Public launch (months 4–6)

- [ ] Open registration; weekly comps public; all-time leaderboards.
- [ ] Creator outreach: early access to 10–15 cubing YouTubers (50k-sub tier
      realistically; J Perm tier if reachable) with a ready-made format:
      "compete against my subscribers, judged by camera."
      **DoD:** ≥3 creator videos live in launch month.
- [ ] Cube shop sponsorship for launch comp prizes (TheCubicle /
      SpeedCubeShop tier).
      **DoD:** one signed sponsorship, even if $200 in gift cards.
- [ ] Launch posts (Reddit/forums/Discords) + Product Hunt for the outside-
      cubing halo.

**Exit criteria:** 5,000 registered users or 8 weeks elapsed, whichever
first; then re-plan against actuals.

### Phase 4 — Post-launch (months 6–12): the arena, then money

In order, each gated on the previous:

1. **Live 1v1 duels** with Glicko ratings, seasons, spectating (only now is
   there liquidity).
2. **Monetization** (never before a habit loop exists): Pro at ~$4/mo —
   stats/history, replays, private leagues, custom comps. Sponsored-prize
   tournaments (free entry — paid-entry contests with a minor-heavy audience
   are a legal minefield; do not rake entry fees).
3. **Move-classification R&D (Option C)** resumes as capacity allows, using
   the BLE labeling pipeline to grow the dataset; promotion bar per §2.

---

## 4. Metrics dashboard (track weekly from Phase 1 onward)

| Metric | Target | Why it's the one that matters |
|---|---|---|
| First-try face-scan success | ≥95% | A failed scan is a churned user; the "magic" depends on this |
| Waitlist → beta activation | ≥40% | Measures whether the demo promise matches the product |
| Week-2 comp retention | ≥40% | The habit loop; gates public launch |
| Verified-card shares per 100 solves | ≥10 | The growth loop working (or not) |
| False-DQ rate (continuity guard, legit solves) | <1% | A wrongful DQ destroys trust faster than a cheater does; if this climbs, loosen thresholds and lean on human review |
| Infra cost per MAU | <$0.02 | The client-side architecture holding |

---

## 5. Risks and pre-agreed responses

| Risk | Signal | Response (decided now, so no panic later) |
|---|---|---|
| Scan reliability plateaus <95% | Weekly metric stalls | Stop feature work; collect failure frames; retrain detector with them (known weakness: orange/L face, 270-image training set — more data is the real fix) |
| Community shrugs (Phase 0 <1k emails) | Low signups despite reach | Reposition around the strongest reaction in feedback before building further |
| Browser port too slow on low-end devices | Inference >200ms/frame on 2018-era laptops | Ship detector-only tracking at lower cadence; classify stickers on captured stills instead of live video |
| A competitor (e.g. a smart-cube app) adds camera verification | — | Speed + community are the moat; do not respond by expanding scope, respond by shipping the weekly comp habit loop faster |
| Solo-founder burnout | Two silent weeks | Cut scope to: scan works, weekly comp runs. Everything else can pause |

---

## 6. What this plan deliberately does not do

- No mobile app before product-market fit (the web app must work on phone
  browsers, which is different and sufficient).
- No paid ads at any phase; every channel above is community/earned.
- No live play before Phase 4. No exceptions for demos.
- No monetization of verification itself, ever.
- No move-by-move claims in any marketing until Option C passes its bar.
