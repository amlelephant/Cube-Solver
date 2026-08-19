# TODO

Working list — things still to DO. Completed work is recorded in git,
`ANTICHEAT.md`, `LAUNCH_ROADMAP.md` and `server/README.md`; it is summarised
here in one line only where it changes what is left.

---

## 1. Anticheat

Built and measured: `ble/move_detector/anticheat_gate.py` (count floor above
God's number in QTM, post-timer test, three-tier verdict) and
`server/core/timing.py` (server-derived timing). 36/36 legit solves verified
with 0 false DQ, 21/21 proxy attacks caught, both seeds. Details in
`ANTICHEAT.md`.

**What the count gate does and does not cover.** It catches every attack that
arrives with *too few moves*: substitution, solve-one-face-then-swap,
solver-following, hidden solve after the timer stops, and fabricated times
(the timing attack collapses into the count attack). It does **not** catch a
cheat who makes a full solve's worth of plausible moves on camera without
solving and swaps a solved cube in at the end — that reads as ~50 moves and
clears the floor of 32.

- [ ] **The enough-moves-then-swap attack — REOPENED 2026-08-10.** The
      mechanism is right and built (`cv/detection/solved_check.py` plus a
      post-stop custody guard, both wired into `adjudicate()`), but the
      0-false-DQ number does **not** reach the live call. Details and the
      full 2×2 in `ANTICHEAT.md` §2.1a; harness
      `ble/move_detector/stop_window_check.py`.
  - The published cell — `tail_window` (1.5 s ending 0.5 s before the last
    frame) with `trajectory.npz` boxes — re-runs fresh and reproduces
    exactly: 0 false DQ / 28, 20/28 caught. The live call reads a different
    window (straddling the timer stop) with different boxes
    (`per_frame_boxes`, +12% `CROP_MARGIN`) and lands at **3/28 false DQ =
    10.7%**, against a launch requirement of <1%.
  - Each factor costs ~2 false DQ alone. The threshold has **zero false-DQ
    margin by construction** (set one step above the worst legit session),
    so any input change spends margin it does not have.
  - Retuned to zero false DQ on the live configuration the catch is
    **13/28 (46%)**, not 71–100%.
  - **Do not raise `SOLVED_MAX_REGIONS`** — that returns to the attack the
    margin the test exists to take away. Two real options: re-derive on
    what the live call feeds and accept ~46%, or move the evaluation into
    the post-stop scan window where the cube is deliberately presented
    (custody already covers that window, so it costs the attacker nothing
    they did not have). Neither is built.
  - Encouraging: the *window* move helps the catch side (23/28 vs 20/28 at
    the shipped threshold). It is the false-DQ side that breaks.
  - The idea that made it tractable: stop asking "is this the same cube"
    (unanswerable against a same-brand second cube — appearance magnitude is
    measured dead) and ask **"was the cube solved when the timer stopped"**.
    The flailer's cube is not, and that fires *before* any swap happens.
  - Deterministic, no ML, no facelet registration: count distinct solid
    colour regions. Opposite faces merge into 3 axis classes, which is free
    on the solved side (opposite faces can't both be visible) and deletes the
    orange/red problem entirely.
  - Measured on `*_solve` vs `*_scramble` tails, held out by date at three
    cut points: **0 false DQ every time**, catch 71 / 90 / 100%. Verified
    that the live path (`per_frame_boxes` + real detector) reproduces the
    offline numbers on real sessions.
  - **Residual, and it is the usual one:** in warm evening light the
    background classifies as a solid region and a scrambled cube can read as
    solved. Three geometric fixes were tried and all measured *worse*; the
    real fix is cube segmentation instead of a bounding box.
- [ ] **Record the attack anyway** — enough moves, then swap, including the
      table-edge/below-frame variant. Only 1 session exists
      (`table_edge_20260804_175814`). The proxy above covers the
      **solved-at-stop** side well (a `*_scramble` tail genuinely is "real
      moves performed, cube not solved"); it does **not** proxy the
      **custody** side at all, which has never seen a real swap. That is the
      measurement this recording buys.
- [ ] Record timer-stop-to-scan footage. The post-stop phantom rate is
      extrapolated off 1–3s session tails and the real window is tens of
      seconds. **`verify_solve.py` now records exactly this** as phase 3
      (`solve_<stamp>_scan`, saved with `--save`), so every live take from
      here on buys some — the `_scan` suffix is skipped by the coach and
      robustness harnesses so it cannot be mistaken for a solve.
  - Re-measured meanwhile (2026-08-10, `anticheat_gate.py
    calibrate-poststop --min-s 1.5`, fresh on all three checkpoints): on the
    SHIPPING models the phantom rate on move-free tails is **0.00 per 10s,
    5 held-out tails, both seeds**, where `aug44_s0` — the checkpoint
    `POST_STOP_PHANTOM_RATE = 4.0` was set from — reaches 4.41/10s and now
    exceeds its own constant. Speed augmentation appears to have removed
    post-stop phantoms entirely. **Do not tighten the rate on this**: 5
    short tails of a cuber admiring a solved cube is not the scan window,
    and a no-false-DQ constant must not be calibrated on it. Keep 4.0; the
    finding is that it now has headroom, not that it should shrink.
- [x] Speed-augmented model wired into `verify_solve.py` — `--ctc-model` now
      defaults to `checkpoints/move_ctc_spd_s0.pt`. **Ready for a live take:**
      `python verify_solve.py --ble --front blue --top yellow --ctc --save`
      The point of that take is the thing the offline numbers cannot settle:
      the speed gains were measured by *simulating* speed (dropped frames,
      blur-approximated exposure), so real fast footage with real motion blur
      is the arbiter. Solve fast and see whether the count holds.
- [x] **`adjudicate()` wired into `verify_solve.py`'s live path — DONE
      2026-08-10.** One sitting now produces both verdicts on the same
      frames: the reconstruction verdict ("is there a consistent story") and
      the gate's ("should this count"). A new **phase 3** records the
      post-timer scan, so `solved_at_stop` and `post_stop_continuity` are
      both supplied — a caller omitting them silently loses the only arm
      covering the enough-moves-then-swap attack, which is why the phase
      exists rather than being optional-by-default.
  - `--anticheat-session DIR` replays the whole gate on a recorded session
    with no camera (splits it into a timed window and its move-free tail),
    so the wiring can be checked before spending a live take and re-checked
    after any change. Not a measurement of the gate — `anticheat_gate.py
    score` is that.
  - `--no-anticheat` skips phase 3 for a quick reconstruction-only take. It
    is not a way to make a take pass: a missing scan window makes the gate
    abstain, not approve.
  - `cv/detection/continuity_guard.run_guard()` is now the single driver for
    the guard loop (live, post-hoc and server), and takes real per-frame
    timestamps — the guard's gap and cadence tests are wall-clock tests, so
    feeding them `idx / mean_fps` hides exactly the stall a cadence check
    exists to catch.
- [x] **60s cap on the post-timer scan enforced in the capture UI — DONE
      2026-08-10.** `live_detect.capture(max_seconds=...)` hard-stops the
      recording, with a countdown from 15 s out and the reason on screen.
      Past the cap the phantom allowance could hide a whole solve; the gate
      abstains rather than passing, so an uncapped UI converted every
      VERIFIED into a REVIEW rather than opening a hole.
  - Fixed alongside it: the REC indicator's elapsed-seconds readout was
    computed from a 45-frame ring buffer, so it read ~1 s for the whole of
    every take regardless of length.
- [x] Abstain band re-derived on **two seeds** (2026-08-05).
      `RETENTION_FLOOR` now holds the worse-of-both-seeds curve for
      `move_ctc_spd_s0/s1`, moving `separation_tps_limit()` from **7.11 →
      9.62 TPS** (~4.7s for a 45-move solve, vs a ~3.1s world record). Gate
      re-scored on the new model: 36/36 legit verified, 0 false DQ, 21/21
      proxies caught. Note the earlier "10+ TPS" was seed 0 alone; seed 1
      gives 9.62 and that is the number, since the gate must not false-DQ.
- [ ] **The gate now depends on which checkpoint runs.** `RETENTION_FLOOR`
      describes `move_ctc_spd_*`. Whatever verification actually deploys
      must be the model those numbers were measured on — re-measure with
      `speed_sim.py --blur` on any checkpoint change. Also run the decoy
      sweep before this widened band ever emits a live `VERIFIED`
      (LAUNCH_ROADMAP C2); it does not today because `adjudicate()` is not
      yet wired into `verify_solve.py`.
- [ ] **Camera injection** — the only threat no cube analysis can touch.
      Needs challenge-response: the server demands a specific face at a
      random instant, and the verification face order is randomised per
      solve so no recording can satisfy it. Not built, not designed beyond
      `ANTICHEAT.md` §4.
- [ ] Decide what the client is allowed to *not* upload. Anything withheld
      is something the server cannot re-check. (The rest of the
      client-vs-server question is settled: client checks are UX, server
      checks are verdicts, both run the same code; sampling is 100% for
      leaderboard-eligible and ~10% otherwise, SYSTEM_DESIGN §2.3.)

## 2. The website

Three parts, tracked separately because they fail and ship independently.
Running locally right now via `RUNNING.md` (two terminals, no Docker).

### 2A. Front end — React / Next

Done: `/api/*` proxied to Django (one origin, no CORS, same shape as
production behind Caddy); allauth-backed login/signup page matching the
design system; `RequireAuth` gating every route in the `(app)` group while
`/` and `/auth/*` stay public; real sign-out and account initial in the nav;
landing-page detector-tracking demo.

- [ ] **Connect the app pages to the API.** `/home`, `/compete`,
      `/leaderboard`, `/profile` still render `lib/mockData.ts` and make zero
      real calls — "Welcome home, Aiden" is hardcoded. Biggest frontend gap.
- [ ] **Landing page copy.** Aiden writes this — do not edit. The current
      copy is not engaging enough and the whole narrative wants a rewrite.
- [ ] Remaining auth screens: email verification landing
      (`/auth/verify-email/{key}`) and password reset
      (`/auth/reset-password/{key}`). The keys live in Django's
      `HEADLESS_FRONTEND_URLS` and **must** match what Next serves or the
      links in verification email 404.
- [ ] The solve flow itself — camera, guided 6-face scan, result card. The
      largest single piece of work left anywhere in the project.
- [ ] Error/empty/loading states once real data lands. Mock data never fails,
      so none of these exist yet.

### 2B. Back end — Django

Running: `server/` (Django + DRF + allauth headless + Postgres config), 43
tests. See `server/README.md`.

- [ ] **Run the Postgres path once.** Settings-verified (engine switch, the
      `DJANGO_DEBUG=0` guard, `check --deploy`, migration has no
      backend-specific SQL) — but this machine has neither Docker nor
      Postgres, so all 43 tests ran on SQLite. `docker compose up` is its
      first real exercise.
- [ ] **Delete `DeviceKeyAuthentication`** (`core/auth.py`). It mints a user
      for any string it is shown. Only still enabled because the capture
      tooling has no other way to authenticate yet.
- [ ] Evidence-bundle upload + re-verification worker (SYSTEM_DESIGN S3).
      **Still the load-bearing gap:** the verdict path scores a bundle the
      client *describes* rather than one the server re-analyzes, so
      `frame_count` is a self-report. Every verdict currently rests on the
      client's honesty about its own evidence.
- [ ] Endpoints the app pages will need: solve history, leaderboard, profile.
      None exist — `/api/solves/` returns only the caller's own rows.
- [ ] Decide HSTS scope before launch. `SECURE_HSTS_SECONDS` is on but
      `INCLUDE_SUBDOMAINS`/`PRELOAD` are deliberately off; browsers cache
      HSTS, so those are hard to walk back.
- [ ] `EMAIL_HOST` to a transactional relay + SPF/DKIM/DMARC before any real
      send (see 2C — the list lives in Postgres, the relay only carries it).

### 2C. Information — Postgres

**Deliberately generic for now.** Which metrics matter is still an open
product question, so the schema stays minimal rather than guessing at
columns we would then have to migrate away from.

Current tables: `WaitlistSignup` (the mailing list itself, with campaign
send-state and unsubscribe tokens), `Scramble` (server-issued, signed,
single-use), `Solve` (server-derived timing, verdict, reasons, and a
`detail` JSON blob).

- [ ] **Decide what a solve actually stores** before the schema hardens. The
      `detail` JSONField is the pressure valve — put new fields there while
      they are still speculative, and promote to real columns only once
      something queries or aggregates on them.
- [ ] User profile / stats tables. Nothing exists; the profile page is
      entirely mock.
- [ ] Competition + leaderboard tables (weekly comps, entries, standings).
- [ ] Retention policy for evidence bundles — they are the expensive thing to
      store and nobody has decided how long they live.
- [ ] Postgres in compose is configured but never run (see 2B).

## 3. Move detection

Daytime held-out sits at 95.6–96.1% post-decode, which meets the Coach ship
gate. Verification no longer depends on this number (the count gate does not
read move identity).

New Idea: I may not be an expert in how this thing actually works but I do know how it fails. It lacks any sort of anchor points and it seems that determining it by any measurement other than visual verification has not worked. I think that we should move towards a visual algorithim similar to one my partner has designed. It is basically trained to fix a lattice onto the cube and read color tiles that are visible. I think that it might be best to not reinvent the wheel since he has already done it. I also think that the animation that he designed would be a really cool addition to the visual aspect of our solve screen. 

- [ ] **Speed-aug seed 1** — running. Seed 0 gave a tie on val MER (5.4%
      greedy vs `aug44`'s 5.26%) and a large retention gain: worst held-out
      session +0.145 at 6 TPS, +0.250 at 8, +0.270 at 10, with all four
      sessions improving. One seed is not a result on this pipeline.
- [ ] Evening corpus, then retrain (LAUNCH_ROADMAP B5). This is the
      cross-environment gap (−22 pts) and the fix is data, not architecture.
      2 sessions recorded 2026-08-04; more needed.
- [ ] Wire `lighting_check.py` into capture so the user is warned *before*
      recording — the cheapest large win available (LAUNCH_ROADMAP B2).
      (`verify_solve.py` already prints `time_of_day_note()` pre-take; what
      is missing is the same warning in the product's capture UI.)
- [x] **The lighting probe feeding `adjudicate()` never worked — FIXED
      2026-08-10.** `live_anticheat.py` passed JPEG-**encoded buffers** to
      `frame_stats`, which expects an `(N,H,W,3)` array. It raised
      `AxisError` on every call, a bare `except` swallowed it, and
      `lighting_ok` was therefore `None` on every run the program has ever
      done — **the lighting abstention has never once fired live.** Two
      faults worth naming separately: the wrong pixels, and an
      `except Exception` broad enough to make a permanent failure invisible.
  - The right input is the **cropped** block the model was scored on, not
    raw camera frames: `build_reference` aggregates
    `detector_stream_color.npz`, which is already cropped, so raw frames
    compare a picture of a room against a distribution of pictures of a
    cube. Measured on `solve_20260803_100135_solve`, a 10:01 take inside
    the corpus by construction: raw frames give |z| > 3 and a REVIEW
    verdict, the cropped block gives 1.7 and passes. Same crop-regime trap
    that cost a day on the classifier in 2026-07-25, in a new place.
  - One shared implementation now: `lighting_check.assess(block, ref)`,
    called by both `live_anticheat` and `verify_solve`.
  - **Turning a dormant path on has a cost, so it was measured**
    (`python lighting_check.py --abstain-rate`): at `Z_ABSTAIN = 3.0`,
    **8 of 76** corpus sessions read `False` — 5 of 62 daytime, 3 of 14
    evening. Roughly one honest daytime take in twelve now goes to REVIEW
    on lighting alone. Two caveats pulling opposite ways: the corpus
    *defines* the reference, so a member exceeding |z|=3 says the spread is
    tight rather than the room dark; and 7 of the 8 are driven by
    `luma_std`, the spatial spread inside the crop, which is as much a
    measure of how much background the box caught as of the light. That
    makes it partly a crop-quality gate wearing a lighting label — worth
    re-deriving `Z_ABSTAIN` deliberately before launch.
- [ ] **Cheap measurement worth doing before more data:** is the detector's
      low-light failure the *same* axis as its fast-turn failure? Both are
      low-contrast temporal evidence. If so the speed augmentation may move
      evening numbers too — check on the next retrain rather than treating
      it as a separate project.

Attributed 2026-08-05: low light costs the **onset detector** 40+ points and
the classifier only ~5. One problem, not two; retraining the classifier is
not the lever.

## 4. Cube rotations

- [ ] Rotation inference, or a lightweight rotation detector. Launch policy
      is detect-and-degrade rather than guess: a decode that fails its
      endpoint check is flagged "limited analysis" and still ships L1 timing,
      and the product claim at launch is scoped to rotation-free solves,
      stated in the UI. First post-launch coach upgrade.

## 5. Launch / waitlist

Live form, honeypot, dedupe, Django-backed, tested end to end. Mailing list
is the Postgres table (2C) and `manage.py send_waitlist` mails it.

- [ ] **Landing page copy** — Aiden's, not to be edited by anyone else. The
      information on the page is not engaging enough as written.
- [ ] Send setup: `EMAIL_HOST` → transactional relay, plus SPF/DKIM/DMARC on
      the sending domain. Send to yourself with `--only` first, every time.
- [ ] The demo film. `DEMO_VIDEO_URL` is empty and the card sits in its
      "filming now" placeholder. Deferred by decision 2026-08-05: the
      product is not yet good enough to film. Revisit once the browser solve
      flow exists — the section is titled "Don't trust us? You don't have
      to" and currently offers no proof.

---

## 6. Live deployment

Built 2026-08-06: hardened compose overlay, production Caddyfile, VM
bootstrap, backup/restore, deploy script. Full runbook in `DEPLOY.md`; the
threat table there says what is defended and what is not. What remains is
the part that needs an account, a card, or a decision.

### 6A. Before the first deploy — blocking

- [ ] **Buy the domain.** Everything below waits on it: the certificate, the
      email records, `DJANGO_ALLOWED_HOSTS`, `SITE_URL`. Registrar does not
      matter much; Cloudflare and Namecheap are both fine.
- [ ] **Provision the VM.** Hetzner CPX21 (~€8/mo) is the recommendation —
      `DEPLOY.md` §1 has the comparison and why 4 GB is the floor (the Next
      build gets OOM-killed on 2 GB, and it presents as a bare exit 137).
      Take the provider's snapshot option; it covers the machine, which the
      pg_dumps do not.
- [ ] **DNS first, deploy second.** A/AAAA for apex and `www` → the VM, and
      confirm `dig +short` answers before deploying. Let's Encrypt validates
      over port 80; DNS not live means no certificate and no site.
- [ ] **Transactional email relay.** Postmark / SES / Mailgun, plus SPF,
      DKIM and DMARC on the sending domain. **This is load-bearing, not
      polish:** `ACCOUNT_EMAIL_VERIFICATION` is mandatory, so with
      `EMAIL_HOST` unset mail goes to the container log and *nobody can
      finish signing up*. Overlaps §5's send setup — same relay.
- [ ] Generate the two secrets into `.env` (`DJANGO_SECRET_KEY`,
      `POSTGRES_PASSWORD`), `chmod 600`. `deploy.sh` refuses to run if the
      mode is wrong — the file holds the key that signs scrambles.

### 6B. First deploy

- [ ] `ssh root@<ip> 'bash -s' < deploy/bootstrap.sh`, then verify SSH still
      works **from a second terminal before closing the first**. The script
      will not disable password auth without a key present, but verify
      anyway; the failure mode is a rebuilt server.
- [ ] `./deploy/deploy.sh`, then `createsuperuser`.
- [ ] **Set `ADMIN_ALLOW_CIDR` to your own address and redeploy.** Until
      this is set, `/api/admin/` is reachable from anywhere and a password
      is the only thing in front of every row. With it, Caddy answers 404 so
      the admin does not appear to exist.
- [ ] Seed nothing. `seed_demo` refuses to run with `DJANGO_DEBUG=0` — that
      is deliberate, its accounts share one printed password.

### 6C. Prove it works before telling anyone

- [ ] Walk the real auth flow on the live domain: sign up → receive the
      verification email *in a real inbox* → verify → sign in → change
      username → hit the 7-day limit. This is the path that has never run
      against a real relay.
- [ ] `./deploy/restore.sh --verify` — restore the newest dump into a
      scratch database and compare row counts against `--counts`. **Until
      this passes once, there are no backups, only dumps.**
- [ ] Check the headers land: `curl -sI https://<domain>` should show HSTS,
      `nosniff`, `X-Frame-Options: DENY`, and no `Server:`.
- [ ] Confirm nothing else is listening: `sudo ss -tlnp` should show only
      sshd and the two Docker-published ports.
- [ ] Set a calendar reminder for the monthly `--verify`.

### 6D. Soon after — highest value first

- [ ] **Cloudflare in front** (free tier), then tighten `ufw` to accept
      80/443 from Cloudflare ranges only. This is the single biggest
      remaining win: volumetric DDoS is the one attack in `DEPLOY.md`'s
      "not closed" list that a bigger VM cannot fix, and it is an afternoon.
- [ ] Off-site the backups. On the box they survive a bad migration and
      nothing else — not a failed volume, not a deleted VM.
- [ ] Uptime monitoring on `/api/health/` (UptimeRobot free tier). Nothing
      currently tells you the site is down except looking at it.
- [ ] Error reporting (Sentry free tier). A 500 in production is currently
      invisible unless you are reading logs at the time.
- [ ] Decide the CSP. Caddy sets the adjacent headers but no
      `Content-Security-Policy` — Next's inline styles and the landing
      page's inline script need a nonce or hash strategy, which is real work
      and easy to ship broken. Do it deliberately, not in a hurry.

### 6E. Deferred, with the trigger that should un-defer it

- [ ] Managed Postgres — when point-in-time recovery or failover matters
      more than the bill.
- [ ] Staging environment — when a deploy has broken production once.
- [ ] Splitting the VM — when a measured metric says so (`SYSTEM_DESIGN.md`
      §4), which is a long way past the closed beta.
- [ ] Secrets manager — when more than one person has server access.

---

## 7. Analytics (the Coach product)

The design is already written — `LAUNCH_ROADMAP.md` §3, four layers L1–L4.
This section is the executable version of **L1 plus the storage everything
above it needs**, in the order it gets built. Each entry names its
*analytic type* so it is clear what kind of thing is being computed and
what can therefore break it.

**The premise is right, with one correction.** The hard part *is* done: the
decode already emits a timestamped move list — `ctc_to_moves` produces
`{frame, time, move, conf, probs, score}` per move
(`ble/move_detector/ctc_decode.py:196`). What is missing is not computation,
it is that **nothing keeps it**: the server stores `observed_moves` as a bare
integer (`server/core/models.py:268`), and `reconstruct.py`'s decode cache
writes `frame` but drops `time` (`reconstruct.py:1991`). The correction is
that every metric below is a *difference of timestamps*, and this pipeline
has only ever been scored on move **identity** (MER / word accuracy). Onset
*timing* error has never been measured — see 7B's first item, which gates
whether the rest of 7B is a product or a plot.

### 7A. Storage — the move stream

Prerequisite for everything else. Nothing here is speculative; the data
already exists in memory at decode time and is thrown away.

- [ ] **Persist the timestamped move list per solve.**
      *Type: storage / schema.*
  - Per move: `{i, t, move, conf, graded}` — `t` in seconds relative to
    timer start, `conf` the model's own posterior, `graded` the D3 segment
    grade (`exact` / `repaired` / `best-effort`, GAMEPLAN §4-D3) once it
    exists.
  - **Split raw from derived, per 2C's rule.** The raw stream goes in a
    JSONField (`Solve.moves`) — it is never queried element-wise, only
    fetched whole for one solve's detail view. The *derived scalars* (TPS,
    pause count, pause seconds) become real indexed columns, because those
    are what trend views sort and average over. ~50 moves is 2–4 KB/solve,
    so storage cost is not a consideration at any plausible scale.
  - **This data is client-reported and must never touch a verdict.** Until
    the re-verification worker exists (2B), the move stream is whatever the
    client says it is. That is acceptable for coaching — it is the user's
    own data and there is no incentive to lie to your own coach — and
    unacceptable for the leaderboard. Keep the boundary explicit in code,
    not just in this file: analytics reads `Solve.moves`, the gate reads
    `observed_moves`.
- [x] **Carry the real capture timestamp through the decode — DONE
      2026-08-10.** *Type: data plumbing (defect).* One definition,
      `decode.move_time`, is now called by all three emitters
      (`live_detect.analyse`, `joint_decode.posteriorgram_to_moves`,
      `ctc_decode.ctc_to_moves`), and `verify_solve` passes `ftimes` on
      every arm. `frame` is kept — it is what re-decoding and the evidence
      bundle key on.
  - **Measured before fixing** (`frame_time_audit.py`, 76 sessions / 5811
    intervals, `results/2026-08-10/frame_time_audit.json`): the two clocks
    drift up to **145 ms** apart within a session, but almost entirely
    common-mode, so on the inter-onset *intervals* every L1 metric is built
    from it is **9.2 ms median / 27.7 ms p95** — under the 41 ms 30fps
    jitter floor on 74 of 76 sessions.
  - So it is a **tail fix, not a correction to the headline**: worst
    session 12.1% error on hesitation seconds and 9.9% on execution TPS,
    which is the same order as the evening error that already suppresses
    those metrics — and it would have arrived with no regime flag to
    explain it.
  - **Why it survived**, worth keeping: every offline harness
    (`onset_timing`, `execution_tps`, `metric_robustness`, `coach_report`)
    re-times onsets itself from `frames.jsonl` and never reads the emitted
    `time` field. The defect could therefore only ever bite the LIVE path,
    and no offline measurement could see it. Offline numbers are unchanged
    by the fix, by construction.
  - Still open, and the other half of this item: **persisting** the stream
    server-side (`Solve.moves`).
- [ ] **Define t=0 once, in one place.** *Type: definition.*
  - `live_anticheat.py`'s `t_start` (SPACE, IDLE→SOLVING) is the timer
    start. Moves before it are inspection/setup; moves after the stop are
    the verification scan, already counted separately as
    `observed_moves_after_stop`. Every metric below is over
    `[t_start, t_stop]` and nothing else.
  - Inspection time is a *separate* metric (7B), not part of the solve
    window — conflating them would inflate the first pause on every solve.

### 7B. The three metrics

- [x] **Onset timing accuracy — MEASURED 2026-08-06, both seeds.**
      *Type: validation / error distribution.* Built:
      `ble/move_detector/onset_timing.py`. Results:
      `results/2026-08-06/onset_timing_s0.json` / `_s1.json`
      (`move_ctc_spd_s0/s1`, 9 held-out sessions, 726 / 736 matched onsets).
      **Verdict: daytime hesitation is shippable; evening is not.**

  | | daytime s0 / s1 | evening s0 / s1 |
  |---|---|---|
  | per-onset bias (median) | +25 / +55 ms | +42 / +68 ms |
  | **interval (IOI) error, median** | **+3.4 / +2.8 ms** | +0.8 / +3.8 ms |
  | IOI jitter (MAD) | 41 / 44 ms | 48 / 47 ms |
  | frame interval (floor) | 32 ms | 29 ms |
  | pause F1 | **0.950 / 0.933** | 0.788 / 0.796 |
  | total pause seconds, true → decoded | 80.9 → 79.8 / 83.8 | 62.7 → **72.9 / 69.2** |

  - **The bias cancels, exactly as expected.** Per-onset lag is a real
    constant (+25 to +68 ms) and it is *model-dependent* — seed 0 and seed 1
    differ by 30 ms on identical footage — but the IOI error median lands
    within 4 ms of zero in all four cells. No 5% correction, no correction
    at all: differencing removes it for free. **One exception:** inspection
    time is measured from the timer (a real clock) to the first onset (a
    model-timed event), so it carries the full bias uncancelled. ~50 ms on a
    multi-second inspection — record it, don't correct it.
  - **Jitter is at the quantisation floor.** IOI MAD of 41–48 ms against a
    32 ms frame interval is ~1.3 frames; the model is not the limit, 30 fps
    is. It does not average out within a solve (differencing two independent
    onset errors *adds* their variances) but it does not need to: the
    smallest thing the pause rule adjudicates is 250 ms.
  - **Daytime totals are honest to 1–4%** (80.9 s of true pause vs 79.8 /
    83.8 s decoded). That is the thinking-vs-turning headline, and it is
    accurate enough to show a user without hedging.
  - **Evening fails on precision, not recall** (0.68 / 0.72 vs recall
    0.93 / 0.89) and inflates total pause time by 10–16%. This is exactly
    the predicted mechanism — a missed move merges two intervals into one
    fake pause — and it is the *known* lighting cliff rather than a new
    axis, so it needs no separate fix: it is the same B5 evening-corpus
    retrain. Until then, degrade honestly (below).
  - Measurement caveat worth keeping: per-onset **std** (110–178 ms) is 4×
    its MAD (27–33 ms). That gap is Levenshtein pairing occasionally
    matching moves that are not the same physical event, not a fat-tailed
    model. Read MAD/IQR here; std is the wrong summary for this population.
- [ ] **Gate hesitation output on the lighting regime.** *Type: policy.*
      Evening pause precision (0.68–0.72) is below anything worth showing.
      `lighting_check.py` already exists and is scheduled for capture-time
      warning (§3); the same signal should suppress or grade the pause map
      rather than printing a number that is 16% wrong. Timing totals
      (TPS, solve duration) stay — those survive evening intact.
- [x] **Moves per second, hesitation removed — BUILT + MEASURED
      2026-08-06, both seeds.** *Type: scalar + cluster decomposition.*
      `coach/timing.py` (the shipped metric, pure functions) and
      `execution_tps.py` (the harness). Results:
      `results/2026-08-06/execution_tps_s0.json` / `_s1.json`.

  | truth median, 6 held-out solves | daytime | evening |
  |---|---|---|
  | **execution TPS** (hesitation removed) | **4.85** | **5.29** |
  | span TPS (plain moves/seconds) | 2.38 | 2.89 |
  | share of solve spent hesitating | **53%** | **53%** |
  | decode error, execution TPS (s0/s1) | **2.7% / 1.0%** | 8.6% / 6.0% |
  | decode error, hesitation seconds | 0.8% / 4.8% | 20.8% / 13.7% |

  - **Removing hesitation roughly doubles the number** (2.4 → 4.9 TPS), and
    the split is the point: execution TPS is hands, hesitation is
    recognition, and plain TPS is the one number that improves when either
    does and tells you which at neither.
  - **Half the solve is hesitation** — 53%, identically in both regimes.
    That is the single most useful thing this metric has produced and it is
    a coaching headline on its own.
  - Daytime decode error of 1–3% clears anything a user would notice.
    Evening is again the weak cell (6–9% on TPS, 14–21% on hesitation
    seconds) for the same false-pause reason; gate it with the item above.
  - Still open, deliberately not built: the **TPS curve** over the solve
    (sliding k=5-move window — scale-free, and it stretches rather than
    collapsing to zero through a pause). The scalars are the ship-critical
    part and they are done.
- [ ] **Re-derive the 3× pause factor whenever the corpus changes.**
      *Type: calibration guard.* Execution TPS moves ~30% across a 2×–5×
      threshold sweep (5.86 → 4.08), so on its own it would be partly a
      re-parameterisation of a convention. What rescues it: at 3× it
      reproduces the **threshold-free** estimator `1/median(IOI)` to within
      2.8% daytime / 3.1% evening. That is why 3× is the default — not
      taste. `coach.timing.timing_report` returns both; if they ever
      diverge, trust the threshold-free one and re-derive the factor.
  - **Unit trap, state it in the UI.** The decoder has 12 classes — quarter
    turns only (`reconstruct.py:189`); R2 as its own class was measured and
    rejected. So a half turn decodes as two moves and every TPS here is
    **QTM**, while cubers quote HTM and every other timer will show a lower
    number. Label it, or the metric reads as broken. The count gate already
    uses QTM (God's number 26), so QTM is the consistent choice — do not
    convert one and not the other.
- [ ] **Hesitation.** *Type: distribution over inter-onset intervals.*
  - Compute IOIs `t[i+1] − t[i]`; a pause is an IOI above threshold.
  - **Threshold must be relative, with an absolute floor.** The user's
    solve speed doubled in 12 days (1.30 → 2.83 moves/s), so a hardcoded
    400 ms would silently change meaning as they improve — early solves
    would show no pauses, later ones nothing but. Use `max(3× personal
    median IOI, ~250 ms)` and store the threshold that was used on the row,
    so old solves stay interpretable.
  - Outputs: total pause time, pause count, longest pause, pause timeline
    (position within the solve), and the headline coaching number —
    **% of the solve spent thinking vs turning**.
  - **A missed move manufactures a fake pause.** A deletion merges two IOIs
    into one that looks like a hesitation. At ~4% word error that is
    roughly 1–2 phantom pauses per solve, which is the same order as the
    real ones. Mitigation: suppress pauses inside segments graded
    `best-effort` (D3), and prefer aggregates over ≥10 solves for any
    displayed claim. Single-solve pause maps ship with their grade visible.
  - Accept gate: pause map computed from decoded onsets vs the same map
    computed from BLE truth, on held-out sessions. If they disagree on
    *where* the pauses are, the metric is not shippable regardless of how
    good the totals look.
- [x] **Average move duration + turn consistency — BUILT 2026-08-06.**
      *Type: scalar + dispersion.* `coach.timing.timing_report` returns
      `mean_move_duration_s` (execution only), `median_move_duration_s`,
      `mean_move_duration_incl_pauses_s`, and `move_duration_cv`.
  - The durations are reciprocals of the TPS figures and carry no new
    information — they exist because "your average move takes 206 ms" is a
    more legible sentence than "4.9 TPS", and the UI should not be doing
    that arithmetic itself.
  - **`move_duration_cv` is the one that is genuinely new.** Coefficient of
    variation over execution intervals is turning *consistency*: two
    solvers can share an execution TPS while one turns metronomically and
    the other alternates bursts with micro-stutters. Truth median 0.68
    daytime. Decodes to 5.1% / 5.0% error (both seeds) daytime — shippable;
    13.7% / 16.2% evening — not.
- [ ] **Inspection time.** *Type: scalar.* Free once t=0 is defined —
      time between the scramble being shown and the first onset after
      `t_start`. Listed here because it is the one L1 metric that needs no
      move identity at all, so it survives evening lighting intact.

### 7C. Which metrics survive our error — re-measured 2026-08-10

> **The whole registry was re-measured on 2026-08-10** and the numbers in
> `coach/report.py` now come from that run. The held-out set has more than
> doubled since 2026-08-06 — **14 solves (9 daytime / 5 evening), up from
> 6** — because the corpus grew. Before replacing anything, the harness was
> re-run restricted to the original six sessions and reproduced the 08-06
> table *exactly*, all 25 metrics in both regimes, so the numbers moved
> because there is more data and for no other reason.
>
> **Two consequences that are decisions, not just numbers.**
>
> 1. **Nothing is suppressed in either regime any more.** Evening
>    `hesitation_seconds` went 20.8% → 8.4% and `move_duration_cv` 16.2% →
>    8.1%, both under the 15% suppression bar. The three metrics
>    `/analytics` currently demonstrates as *withheld* would now be shown.
>    That is what the measurement says, and the rule is that the
>    measurement decides — but it rests on **five** evening solves, and the
>    worst column for those same metrics is still 24.2% and 19.6%. If
>    evening hesitation should stay hidden until the evening corpus is real
>    (LAUNCH_ROADMAP B5), that is a *policy* change — lower
>    `SUPPRESS_ABOVE_PCT`, or gate evening on the worst column instead of
>    the median — not an edit to these numbers.
> 2. **Medians settled, tails appeared.** `hesitation_seconds` daytime fell
>    4.8% → 2.1% while its worst case rose 6.5% → 17.4%. That is what a
>    bigger sample does. Read the worst column.
>
> Also fixed: **`metric_robustness.py` had been broken since 2026-08-06** —
> it read `mv["awkward_face_fraction"]`, which `coach/moves.py` correctly
> stopped returning when that metric was cut, so every session raised
> `KeyError`, was caught by a blanket `except Exception ... skipped`, and
> the run printed "Nothing scored" while exiting 0. The gate-filler could
> not run at all. The rejected candidate is now computed inside the
> harness (a rejection nobody can re-check is a rejection nobody can
> revisit), and programming errors re-raise instead of being swallowed.

The original 2026-08-06 analysis, whose *conclusions* all survived
re-measurement:

`metric_robustness.py` computes 21 candidate metrics twice per session —
once from BLE truth, once from the decode the product would actually have
— and ranks them by disagreement. Both seeds, 6 held-out solves.
Results: `results/2026-08-06/metric_robustness_s0.json` / `_s1.json`.

**The headline finding is that my own taxonomy was wrong.** The plan
assumed robustness tracks *what a metric needs* (timing < count <
identity < order). It does not. Identity-needing aggregates are among the
most robust things measured (`face_share_L1` 1.3–2.7%), while the single
worst tail in the whole table is a timing-only metric. What actually
predicts robustness is **what kind of statistic it is**, and that cuts
straight across the other taxonomy:

| kind | median \|err\| s0 / s1 | worst | why |
|---|---|---|---|
| **mean** (sum/average over all moves) | **5.2% / 3.9%** | 24–30% | errors average out |
| local (adjacent-pair) | 13.2% / 7.0% | 41–83% | one insertion corrupts two pairs |
| ratio (estimate ÷ estimate) | 17.7% / 14.8% | 56–69% | two errors compound |
| **threshold** (count of events) | **27.1% / 25.1%** | 67% | small continuous error → discrete miscount |
| extreme (max/min) | 1.7–1.9% | **48–72%** | one bad move *is* the statistic |

- **Build means. Avoid counts of thresholded events.** That is the whole
  design rule, and it is why `n_pauses` (25–27% error) is the wrong way to
  express hesitation while `hesitation_seconds` (0.8–4.8% daytime) is the
  right way — same underlying signal, one thresholded and one summed.
- **`extreme` is the trap.** `longest_pause_s` shows a *median* error of
  1.7–1.9% and a worst-session error of 48–72%. Any metric of this kind
  must be judged on its worst column; the median flatters it structurally.
- **Move count is far better than word error predicts** — `n_moves` 1.9–5.3%
  daytime against ~4% word error, because misses and phantoms cancel.
- **n = 3 solves per regime.** Medians over three sessions are weak and the
  seeds disagree on several rows (`ccw_fraction` evening: 12.9% vs 4.5%).
  Treat the *kind* rollup as the result and individual rows as indicative.

**Shippable today (both seeds, daytime, ≤6% with a sane worst case):**
`solve_seconds` (0.1%) · `n_moves` · `face_share_L1` · `top_face_share` ·
`distinct_face_runs` · `span_tps` · `execution_tps` ·
`mean_move_duration_ms` · `ccw_fraction` · `hesitation_seconds` ·
`move_duration_cv`.

- [x] **The shippable set is BUILT — 2026-08-06.** `coach/` is now three
      modules plus a harness pair:

  | file | what |
  |---|---|
  | `coach/timing.py` | L1 timing: pauses, bursts, TPS, move duration, consistency |
  | `coach/moves.py` | identity aggregates: face share/entropy, R-U share, direction |
  | `coach/report.py` | assembles both, **gates each metric on its measured error** |
  | `metric_robustness.py` | measures decoded vs truth; fills the registry |
  | `coach_report.py` | runs the shipped analysis on real solves; `--inventory` |

  - **16 metrics ship**: 16 usable daytime, 13 evening (3 suppressed there).
    Full inventory with per-regime median *and worst-case* error:
    `python coach_report.py --inventory`. Results:
    `results/2026-08-06/coach_report_s0.json` / `_s1.json`.
  - **`report.py`'s `MEASURED` registry is a gate, not documentation.** A
    metric absent from it cannot be reported, and only
    `metric_robustness.py` may add to it. This is what stops the obvious
    failure of someone adding a metric by analogy ("it's just another
    average") and shipping it unmeasured.
  - **`metric_robustness.py` now measures the shipped code** — it imports
    `coach.moves.move_report` rather than reimplementing it, so the
    accuracy numbers describe what actually runs.
  - One rejection worth keeping visible: **`awkward_face_fraction` (D+B
    share) was built and cut** — 5.6% median daytime but 30.5% worst, and
    26–42% evening with a 100% worst case. Cause generalises: D+B is ~6% of
    moves, and **a share with a small numerator does not inherit the
    robustness of a mean** — there is nothing for errors to average
    against. That refines §7C's rule: means are safe *when the denominator
    is large*.
- [x] **TPS curve — BUILT + MEASURED 2026-08-10.** `coach.timing.tps_curve`
      (sliding k=5, `(k-1)/span` per window so a pause stretches a window
      rather than zeroing it), registered in `coach/report.py` as the
      registry's first `series` metric. Scored by resampling truth and
      decode onto one shared time grid — they disagree about how many
      windows there are but share a wall clock — and taking the median
      disagreement across 24 points.

  | | daytime | evening |
  |---|---|---|
  | median error (worse seed) | **6.8%** | **10.2%** |
  | worst session (worse seed) | **9.9%** | **14.0%** |

  - Ships `high` daytime and `caution` evening — the only *rate* metric
    reportable in both regimes. Its worst case is tighter than most scalars
    in the registry, which is the pointwise-mean construction working.
  - It **supersedes `slowdown_ratio`**, re-measured on the same run at
    10.5–15.9% daytime and 12.2–21.2% evening: the curve is better in every
    cell, and it is the same question asked as a sequence of means rather
    than as one ratio of two estimates.
  - Not yet rendered — `/analytics` shows scalars only, and a `series`
    metric needs a chart. The payload carries `series: true` so the client
    can branch rather than sniffing the JSON type.
- [ ] **Still unbuilt, in priority order.**
  - **Cross-solve trends.** Aggregation is the error killer and every
    number above improves with it — this is what turns a 5%-accurate
    single-solve metric into a trustworthy trend.
  - **Inspection time** (needs t=0 wired from the capture UI).
- [ ] Do **not** build, on current evidence: pause *counts* (25–27%),
      burst-size statistics (37–63%), longest-pause as a headline (48–72%
      worst), `same_face_pair_rate` (27–75%), or any single ratio of two
      estimates. Each is measured and each fails; the numbers are recorded
      in `coach/report.py`'s docstring so they are not re-litigated.

### 7D. Surfacing it

- [x] **The `/analytics` page — BUILT 2026-08-06.** Linked from the primary
      nav and the footer. Visualises the metrics on the landing page's cube
      rather than only tabulating them.

  | file | what |
  |---|---|
  | `web/lib/cube/renderer.ts` | the cube, driven by `tps` + per-face brightness |
  | `web/components/CubeViz.tsx` | React wrapper; params update without remounting |
  | `web/lib/analytics.ts` | the registry + **real decoded values**, not mock data |
  | `web/app/(app)/analytics/page.tsx` | the page |

  - **The cube runs `R U R' U'` at the solve's measured execution TPS**, and
    the toggle swaps it to overall TPS — the two rates side by side is what
    makes "half your solve is not turning" land without a paragraph. For
    face usage each face is lit in proportion to its share, floored at 0.14
    so an unused face still reads as a face.
  - **Shares the landing page's material, not its machinery.** Mesh, shader
    and palette are lifted from `public/landing.html` so it is recognisably
    the same object; the scroll choreography, detector HUD, motion pass and
    bloom chain are all dropped. One pass onto a transparent canvas, so it
    works in both themes with no second palette. A tone-map + inverse gamma
    had to be added at the end of the fragment shader — the landing page
    does that in its post chain, and without it every specular clipped.
  - **Real numbers, not mock data.** `lib/analytics.ts` carries decoded
    output copied from `results/2026-08-06/coach_report_s0.json`, with the
    measured error beside each figure. The evening solve genuinely shows
    three metrics as *withheld* — a mock dataset would have hidden exactly
    the behaviour the page exists to demonstrate.
  - Verified in-browser at 1568×708: cube animates, both visualisations
    work, suppression renders, no hydration errors.
- [x] **Replace the fixture with the API — DONE 2026-08-07.** The page now
      reads `GET /api/solves/analysis/`, which returns the signed-in
      account's own analysed solves, oldest first. The swap was mechanical
      as predicted: `lib/analytics.ts`'s types already matched
      `coach/report.py`'s payload, so `SOLVES` was deleted and `seriesFor`
      took the list as an argument.

  | file | what |
  |---|---|
  | `core/models.py` | `Solve.analysis` — the coach payload, stored verbatim |
  | `core/views.py` | `solve_analysis` + `_analysis_json` |
  | `core/management/commands/import_real_solves.py` | backfill from recordings |

  - **`analysis` is one JSONField, not columns.** The registry in
    `coach/report.py` IS the gate on what may be reported and it moves
    whenever `metric_robustness.py` re-measures; mirroring it as columns
    would make that a migration each time and put a second, silently
    diverging copy of the gate in the schema. The payload already carries
    each metric's label, unit, confidence and measured error.
  - **The gate stays server-side.** `suppressed` and per-value
    `confidence` are decided when the analysis is stored; the client never
    re-derives either. `METRICS` in `lib/analytics.ts` keeps only display
    copy (label, blurb, ordering) plus the accuracy figures for the
    footnote.
  - Analytics reads `analysis`; the verdict gate still reads
    `observed_moves`. Unchanged, and asserted in the field's docstring.
  - Solves without an analysis are omitted, not returned empty — a row of
    dashes reads as a broken metric rather than as work not yet done.
    Verified: an account with no analysed solves gets the empty state.
- [x] **Per-solve view — DONE 2026-08-07.** `/api/solves/<id>/analysis/`,
      and `solve/[id]` rebuilt against it. Profile history rows link
      through — your own only, since someone else's solve is 404 by design.
  - **The old page was entirely `lib/mockData`**, including a cross / F2L /
    OLL / PLL phase breakdown. Phase splits are not built (§7E) and those
    numbers stood behind nothing, so the page now shows the coach payload
    instead and `getSolveDetail`/`phaseBreakdown` were deleted rather than
    left for someone to re-wire.
- [x] **Free vs paid split — DONE 2026-08-07.** `Profile.is_premium`
      (hand-granted; no billing yet). Free is served its most recent
      analysed solve and nothing else; paid gets every solve, the averaging
      window and per-solve history.
  - **The gate is the response, not the UI.** `solve_analysis` slices to
    `[-1:]` for free accounts, so a client ignoring `is_premium` still
    cannot compute an average it was never sent. Per-solve is 402 with
    `upgrade_required`, distinct from the 404 for someone else's solve.
  - `is_premium` is in `/api/me/` only — `is_founder` is a badge others
    see, this is billing state, and `public_profile_json` omits it.
  - Averaging is client-side over the gated list, so the window toggle is
    instant. Windows: last 5 / 12 / all, defaulting to 12 — an all-time
    mean on an improving cuber sits above current form and moves slower the
    longer they play.
- [x] **The `/coach` page — BUILT 2026-08-07.** The pitch, what Coach
      includes, and the plan frames ($4/mo, $36/yr). Reachable by everyone —
      it is where the locked nav entry points, so gating it would leave the
      lock leading nowhere — and it swaps to an owned state for
      subscribers. Every figure on it is derived from `lib/analytics.ts`'s
      registry rather than typed in, so the copy cannot drift into an
      overstatement when a metric is added or suppressed. It quotes the
      MEDIAN metric's error, not the best: `span_seconds` is 0.1% and
      advertising that would be true of one measure and flattering of the
      other fifteen.
- [ ] **Billing.** `is_premium` is set by hand and there is no checkout, so
      the plan button says "Not on sale yet" rather than opening a flow that
      fails. *Type: delivery.*

### 7F. Pre-launch UI pass (2026-08-07)

- [x] **The nav vanished below `md`.** `hidden md:flex` with no replacement,
      so at half-width — an ordinary desktop window, not just phones — the
      only way to another page was the footer. Now a hamburger sheet, with
      the settings/sign-out icons folded in below `sm`.
- [x] **Coach entry in the nav**, with a lock for accounts without it. A
      MARKER, not a guard: the page is the upsell.
- [x] **One history table.** `/home` and a profile rendered the same rows
      through two components that had drifted on row height, date format and
      whether a row was clickable. Both now use `RecentSolves`; only the row
      cap differs. "Current Best" is a link to your PB, backed by a new
      `best_solve_id` on the profile payload — NOT computed client-side from
      the history rows, which are capped at 50, so on a bigger account the
      client's "best" would silently be the best of the most recent 50.
- [x] **The rating chart is back**, beside the heat map. `EloChart` still
      existed but nothing rendered it. There is no rating-history table, so
      `ratingSeries` walks `rating_delta` BACKWARDS from the stored current
      rating; forwards from a presumed start would drift on any adjustment
      made outside a solve. Solo trials are unrated, so the card's empty
      state is the point — it is the one surface explaining why you would
      play a match, and hiding it until you had would be backwards.
- [x] **Preset avatars** (`core/avatars.py` + `lib/avatars.tsx`). No
      uploads: image moderation is unbuilt and a public profile is exactly
      where unmoderated imagery does damage. The server stores a key from an
      allowlist and the artwork is drawn as inline SVG, so there is no file
      to serve and no upload endpoint to abuse. New `avatar_preset` column,
      deliberately NOT `avatar_path` — that one means "a file this account
      uploaded" and overloading it would make the upload pipeline's first
      query ambiguous the day it ships.
- [x] **Real flag artwork** (`components/Flag.tsx`, SVGs vendored in
      `public/flags/` from flag-icons v7.5.0, MIT). **Windows ships no
      country-flag glyphs in any system font**, so the emoji approach
      rendered as the bare letters "US" for most of the desktop audience —
      the leaderboard's flag column was two grey letters. There is no font
      or CSS workaround. `countryFlag()` survives for plain-text contexts
      only and says so.
- [x] **`useMe` is one shared store**, not one copy per caller. Six call
      sites each held their own state and fired their own `/api/me/`, so the
      NavBar and the profile disagreed about your avatar the moment you
      changed it — the picker's `setMe` updated the picker's copy alone.
      `clearMe()` on sign-out, or the next session flashes the previous
      user's name.
- [x] **Deleted `lib/mockData.ts` and `HistoryTable.tsx`**, both fully
      unused once the above landed. Mock data left lying around a launching
      product is mock data that gets wired back in.
- [ ] Multiplayer is out of scope for MVP — the Live Match tiles and the
      rating card's "Find a match" CTA point at `/compete`, which is not a
      real matchmaking flow yet. *Type: scope.*
- [ ] Wire `lighting_check.py` through to the response so `regime` is a
      measurement rather than a recorded field. The suppression logic is
      already regime-driven and correct; it is the input that is fixed.
- [ ] **Trend views over ≥10 solves.** *Type: aggregation.* Per
      LAUNCH_ROADMAP §3.2 L4 this is deliberately the headline rather than
      the single-solve view: at ~4% word error one solve's numbers move
      with a single decode edit, and an average does not. Single-solve views
      exist but show their grading honestly.
  - Partly DONE 2026-08-07: the headline now averages over a chosen window
    (last 5 / 12 / all) rather than showing one solve, for paid accounts.
    `averageOf` in `lib/analytics.ts`. Still unbuilt: a *rolling* ao5/ao12
    charted over time, as opposed to one average of the latest window.
  - **Two traps averaging hit, both fixed, both worth not re-learning.**
    (1) A metric withheld for a solve must be dropped from that metric's
    average, not counted as zero — so `hesitation_*` averages over daytime
    solves only, and correctly keeps *daytime* error bars rather than
    inheriting evening's. Regime is therefore tracked per metric, not per
    aggregate. (2) `span_seconds - hesitation_seconds` is invalid once
    averaging: evening solves contribute a span but withhold hesitation, so
    the two means cover different solve sets and their difference is a
    turning time no solve had. Measured wrong by 1.94s over all 36.
    `Aggregate.hesitationSpanSeconds` is the matched-denominator span.
  - Averaging several solves will in practice beat the per-solve error
    figures, but by how much is **not measured**, so the footnote still
    quotes the per-solve number and says so.

### 7E. Later — phases, then algorithms

Not yet; listed so the earlier work does not paint it into a corner.

- [ ] **Phase splits.** *Type: deterministic replay, not inference.* Worth
      correcting the assumption up front: phase boundaries do **not** need
      to be inferred from timing clusters. The scramble is server-issued and
      signed, so the start state is known — replay the decoded moves and
      detect cross / F2L 1-4 / OLL / PLL with group-theory predicates.
      Deterministic, ~200 lines, no ML (`coach/phases.py`, LAUNCH_ROADMAP
      §3.2 L2).
- [ ] **Pause-cluster cross-check.** *Type: unsupervised, validation only.*
      First data in (2026-08-06, `execution_tps.py` burst histogram, 103
      bursts over 6 solves): **bursts are real but they are smaller than
      algorithms.** Median burst is 5 moves, mean 7.0, with a long tail to
      33 and a visible secondary bump at 14–16. Reference lengths are F2L
      pair ~7–10 QTM, OLL ~9–11, PLL ~10–13 — so at the calibrated 3×
      threshold the clusters are *sub*-algorithm units (triggers, partial
      insertions), not whole algorithms. Raising the threshold to 5× moves
      median burst size to 10, squarely in algorithm range, but that
      threshold is not the one that makes execution TPS correct — so the
      two uses want different cuts, and pretending one cut serves both
      would corrupt whichever metric loses.
  - What that means for the plan: the burst decomposition is **not** by
    itself a phase detector, which reinforces §7D's first item — get
    boundaries from the deterministic replay, then use pauses to confirm.
    The honest test is still ahead: do the large pauses land on the
    replay-derived boundaries? That needs `coach/phases.py`.
  - Its real value is unchanged and is the **fallback for degraded
    decodes** (evening light, suspected rotation) where replay fails but
    timing survives.
- [ ] **Algorithm usage.** *Type: sequence matching against a mined,
      user-specific library.* Deferred per the user's ordering. The
      machinery already exists and scores 91–97% held-out
      (`algorithm_gate.py`); what it needs is the enrollment flow
      (LAUNCH_ROADMAP §3.2 L3).

---

## Closed

- **Backtracker redesign** — closed by measurement. The algorithm method
  *was* implemented correctly; the forward arm is structurally inert and
  anchor selection has a measured +0.65 ceiling. The original instinct was
  right (insertions/deletions, not identification, are 81% of error mass)
  and CTC + data fixed those. What survives is D3 graded output
  (LAUNCH_ROADMAP B3).
- **Verification-phase continuity** — built as the count gate's post-stop
  test, rate-based rather than a fixed constant because phantoms accumulate
  with window length. `ANTICHEAT.md` §1.4.
- **DataLoader workers / 7h training runs** — fixed 2026-08-05, 447s →
  63s/epoch. Lazy per-worker stream rebuild (pickle 3.2 GB → 7 KB) plus
  memory-mapped frame sidecars. Two non-code lessons that cost as much as the
  bug: killed runs leave orphaned workers holding GBs (`TaskStop` does not
  reap them on Windows), and `--workers 3` left 19 of 24 cores idle — worker
  count is the throughput knob.
