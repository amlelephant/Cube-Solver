# CubeArena — System Design & Implementation Roadmap

**Status:** draft 2026-07-30 · **Owner:** Aiden
**Relationship to other docs:** `docs/VISION.md` is the long-range product spec.
`docs/ROADMAP.md` is the adopted *product* execution plan (phases, DoD, metrics).
**This document is the *platform* counterpart** — how the backend, hosting, and
client-side inference actually get built, what it costs, and in what order.
Where this doc and `docs/VISION.md` §12 disagree, this doc wins: VISION assumed a
server-side GPU inference fleet, which the client-side architecture deletes.

The original design notes that seeded this document are preserved verbatim in
[Appendix A](#appendix-a--original-design-notes).

---

## 1. Where the code actually is today

Grounding, so the roadmap starts from facts rather than intentions:

| Piece | State | Notes |
|---|---|---|
| CV pipeline (`cv/`) | Working, Python, desktop | detect → warp → slice → ensemble classify → group-theory validate |
| `detect_full_cube.pt` | **3.02M params, fp16 checkpoint** | YOLOv8n, 640×640 |
| `sticker_cnn.pt` | **0.36M params** | 3 conv blocks, 32×32×3 input, 9 patches per face |
| ONNX export | **Does not exist anywhere in the repo** | Phase 1's first blocker |
| Web app (`web/`) | Next.js 16 / React 19 / Tailwind 4 | Pages scaffolded with mock data |
| Camera in browser | **No `getUserMedia` anywhere in `web/`** | Nothing of the CV pipeline is wired in yet |
| Backend | **Does not exist** | No Django project, no DB, no auth |
| Move classifier (`ble/move_classifier*.pt`) | 11.2M params / 43MB × 20 variants | R&D only — deferred per ROADMAP §2 Option C |

So: the models are real and small, the frontend shell is real, and the entire
backend plus the entire browser-inference layer are greenfield.

---

## 2. Design decisions

### 2.1 Django + DRF — confirmed, but for a stronger reason than stated

The original notes justify Django by "user accounts are built in." That's true and
worth having (`django-allauth` gives Google/Discord OAuth, which VISION §7.1.1
requires, in an afternoon). But it isn't the decisive reason. Two better ones:

1. **Server-side re-verification must run the same code as `cv/`.** Section 2.3
   explains why re-verification is non-negotiable. `cv/` is Python. A Django
   backend can import the *actual* detection and classification modules; a Node
   backend would need either a second implementation of the pipeline (which will
   drift, and a drifted verifier is worse than none) or a Python sidecar, which
   is Django-with-extra-steps.
2. **Django admin is most of the human-review queue for free.** `docs/ROADMAP.md`
   Phase 2 requires a reviewer who can "watch the recording + continuity report
   and approve/reject in under 2 minutes." A `ModelAdmin` with a custom
   changelist and two admin actions gets ~80% of that on day one.

The relational-database instinct is also right and needs no defence — solves,
ratings, matches, and competitions are exactly the shape Postgres is for.

### 2.2 "Do we have to host frontend and backend separately?" — No

This was flagged in the notes as an open problem. It isn't one. Three options,
ranked:

| Option | How | Verdict |
|---|---|---|
| **Reverse proxy, one origin** | One box. Caddy/nginx routes `/api/*` → Django (gunicorn), everything else → Next (node). One domain, one TLS cert, **no CORS**, one `docker compose up`. | ✅ **Adopt.** Delivers the "one total stack" goal without giving anything up. |
| Static export | `output: 'export'` in Next, Django serves the build via WhiteNoise | ❌ Costs you Server Components, route handlers, and `web/app/route.ts` (which reads `landing.html` off disk at request time). Solves a problem you don't have. |
| Separate hosts | Next on Vercel, Django on Fly | ⚠️ Fallback only. Adds CORS, a second deploy pipeline, and cross-origin auth-cookie friction, in exchange for nothing at this scale. |

Concretely: keep `web/` exactly as it is, add `server/` (Django + DRF), add a
`Caddyfile`, ship both from one VPS. Split them later only if the frontend's
traffic profile actually diverges from the API's — which, for a hobby-market
product, it won't.

### 2.3 The trust boundary — the one correction to the design

> "offload compute capacity to end users as opposed to paying for server compute"

This is right about **compute** and wrong about **trust**, and the distinction
decides the whole architecture.

A client that runs the model also controls the model's output. Anyone can open
devtools and `POST {verified: true, time: 4.21}`. Since `docs/ROADMAP.md` §1
states the product *is* trust ("a camera-verified, cheat-proof solve is the
product"), a purely client-side verdict is worth nothing — it would be the
CubingTime failure mode (§4.1 of VISION: top times widely believed fake) with
extra steps.

**The resolution — client-side inference is a bandwidth optimization, not a trust
mechanism:**

```
Client (WebGPU)                          Server (Django)
─────────────────────────────────────    ──────────────────────────────────
run detector + classifier live      →    (nothing trusted yet)
  · guide the user's scan (UX)
  · run the continuity guard
  · pick ~12 key frames
  · build evidence bundle             ──▶ store bundle, mark PENDING
    (6 face scans, solved-state
     frames, continuity report,
     frame hashes, timings)
                                         re-run the SAME cv/ pipeline on
                                         the uploaded key frames, CPU-only
                                              │
                                              ├─ sampled (~10%) for ordinary results
                                              └─ always for leaderboard-eligible
                                         ──▶ VERIFIED / FLAGGED
```

The client never uploads 20 seconds of 1080p video — it uploads ~12 frames and a
report. That is where the bandwidth and storage savings actually come from, and
it survives the fact that the client is hostile.

The economics still work because re-verification is sampled, runs on ~12 frames
rather than ~600, and a 3M-parameter model on CPU is cheap. Quantified in §4.4 —
it comes to single-digit dollars per month even at 100k MAU, which is why **no
GPU server is required at any scale in this plan.**

Scramble generation stays fully server-side and signed regardless (VISION §8.1,
§9.3): the client must never be able to choose its own scramble.

### 2.4 Competitive rooms — Django Channels, and not yet

The notes correctly say rooms are not a novel problem. The specific answer:
**Django Channels + Redis channel layer**, which keeps rooms inside the one stack
rather than adding a Node service. Matchmaking is a Redis sorted set keyed by
rating; the queue-pop-and-pair loop is ~100 lines.

But per `docs/ROADMAP.md` §3 Phase 4, **live play is explicitly gated behind
async weekly competitions**, for the good reason that live matchmaking feels dead
below ~1k concurrent users. Rooms are scheduled as S6 below and should not be
built earlier. Building an arena before there are gladiators is the single most
common way this kind of product dies.

---

## 3. Is WebGPU-serving every model to the client practical?

**Verdict: yes for the launch-critical models, comfortably — with four caveats,
one of which is a genuine unsolved porting problem (§3.5).** The models are small
enough that this is not a close call.

### 3.1 What actually has to run client-side

Only the state-verification path, per ROADMAP §2 Option A:

| Component | Params | Frequency | Backend |
|---|---|---|---|
| YOLOv8n cube detector | 3.02M | Continuous during solve (continuity guard) + per scan | **WebGPU** |
| Sticker CNN | 0.36M | 9 patches × 6 faces, on capture only | **WASM** (see below) |
| HSV classifier + ensemble | none (arithmetic) | with the CNN | JS |
| State validity check | none | once per scan | JS (~200 lines) |

Two things deliberately excluded:

- **The full two-phase solver stays server-side.** The client only needs to
  answer "is this state legal?" (parity, permutation, orientation) — not "solve
  it." That's a pure-JS port of a few hundred lines, and it avoids shipping
  `cv/solver/twophase/` plus its 30–60s pruning-table precompute into a browser.
- **The move classifier (43MB, 11.2M params) is not in the client bundle** and
  must not be until Option C passes its promotion bar (ROADMAP §2: ≥95% per-move
  accuracy at ≥3 TPS). If it ever ships it would be ~11MB int8 — a material
  change to the download budget, flagged now so it isn't a surprise later.

**The sticker CNN should run on WASM, not WebGPU.** A batch of nine 32×32 patches
through a 0.36M-param net is on the order of 0.1 GFLOP total. The GPU dispatch
and readback round-trip will cost more than the arithmetic. Put it on the CPU
backend and keep WebGPU for the detector.

### 3.2 Download budget

| Asset | fp32 | fp16 | int8 |
|---|--:|--:|--:|
| Detector ONNX | 12.1 MB | 6.0 MB | ~3.2 MB |
| Sticker CNN ONNX | 1.4 MB | 0.7 MB | ~0.4 MB |
| **ORT Web runtime (jsep/WebGPU build)** | **~3–4 MB brotli** | — | — |
| **First-visit total** | **~17 MB** | **~11 MB** | **~8 MB** |

Notes that matter:

- **The runtime is bigger than the models.** It's the single largest item and is
  easy to forget when budgeting.
- **Start at fp16.** The detector checkpoint is *already stored fp16*, so fp16
  export costs no additional accuracy versus what's on disk today. Reach for int8
  only if measurement demands it — see the orange/red caveat in §3.5.
- **Lazy-load everything.** None of this may touch the landing page, which is the
  Phase 0 conversion surface. Fetch on entry to the solve flow, behind a progress
  indicator.
- **Content-hash the filenames** and serve `Cache-Control: public, max-age=31536000, immutable`
  so returning users pay zero bytes. This turns model delivery from a recurring
  cost into a per-release one.

~11 MB cached-forever is comparable to a mid-size SPA. This is not a problem.

### 3.3 Compute budget

**These are engineering priors, not measurements. Replacing them with real
numbers is the entire point of stage S1 — do not quote them as results.**

YOLOv8n at 640×640 is ~8.7 GFLOPs per inference.

| Device class | Backend | Est. detector latency | Usable cadence |
|---|---|--:|---|
| Discrete GPU (RTX 3060-class) | WebGPU | ~5–8 ms | 30 fps trivially |
| Modern integrated (Iris Xe-class) | WebGPU | ~15–30 ms | 15–30 fps |
| Older integrated (UHD 620-class) | WebGPU | ~60–120 ms | 8–15 fps |
| Mid CPU, no WebGPU | WASM SIMD ×4 | ~150–400 ms | 2–5 fps |

The load that matters is **not** the 6-face scan (six discrete captures — even
400 ms is fine). It's the **continuity guard**, which per ROADMAP §2.1 runs
continuously from scramble-scan completion through solved-state scan — inspection
included, so call it 20–40 seconds of sustained inference.

The good news: the guard's own tolerance is ~1.0 s gap detection. It does not
need 30 fps. **5–8 fps is sufficient for uniqueness, presence, and trajectory
checks**, which brings even the WASM fallback tier within reach and makes the
"browser port too slow on low-end devices" risk (ROADMAP §5) much less likely to
bite than that row assumes.

### 3.4 Browser coverage

WebGPU is broadly available in mid-2026 — Chrome/Edge desktop and Android,
Safari, and Firefox on Windows all ship it — but coverage is not universal, and
enterprise/older/Linux-Firefox configurations remain a real tail.

**Therefore the WASM path is a shipped, tested fallback, not a theoretical one.**
Feature-detect `navigator.gpu`, fall back to `ort-wasm-simd-threaded`, and drop
the continuity cadence rather than refusing the solve. ROADMAP §5 already
pre-agreed this response ("ship detector-only tracking at lower cadence") — this
just confirms it's the right one and that it's cheap.

Note that WASM threading needs `crossOriginIsolated`, i.e. COOP/COEP headers.
That is a **server config task** (§S1), and COEP will break any third-party embed
that lacks CORP headers. Worth knowing before it surprises you in staging.

### 3.5 The four caveats

1. **The CSRT tracker does not exist in the browser — and this is not currently
   in ROADMAP Phase 1's task list.** `cv/detection/cube_detector.py` is a
   two-stage design: YOLO every `YOLO_INTERVAL = 12` frames, OpenCV CSRT tracking
   in between. That architecture is *why* momentary confidence dips get bridged,
   and CLAUDE.md documents the outage that happened when the tracker silently
   returned `None`. `cv2.TrackerCSRT_create` is contrib-only C++; there is no
   ORT-Web equivalent. Three ways out, in order of preference:
   - Run the detector at a flat 5–8 fps with **no** tracker, and widen the
     continuity guard's gap tolerance to match. Simplest; likely sufficient given
     §3.3's cadence finding.
   - Port a lightweight tracker to JS (IoU association + a small correlation
     step). Maybe 150 lines, no WASM blob.
   - `opencv.js` with the tracking module — a ~9 MB WASM download that would
     nearly double the bundle. Last resort.

   **Decide this in S1 by measurement, not preference**, and update
   `docs/ROADMAP.md` Phase 1 with the outcome.

2. **fp16 may move the orange/red decision boundary.** CLAUDE.md is explicit that
   orange/red is the hardest pair and that the calibration path is delicate. The
   ROADMAP Phase 1 DoD ("ONNX outputs match PyTorch within tolerance on 20 test
   images") must be tightened: **the parity test needs to be stratified by
   colour, with orange and red over-represented.** An aggregate tolerance can pass
   while the only pair that matters regresses.

3. **Sustained mobile thermals are unmeasured.** One 30-second guard run is
   almost certainly fine. Five back-to-back solves in an Ao5 session is not
   obviously fine — phones throttle. Measure in S1 on a real mid-range Android,
   not a flagship.

4. **Client-side inference does not establish trust.** Restated here because it's
   the caveat with architectural consequences: see §2.3.

### 3.6 Verdict

Practical, and the deciding factor is that **these models are unusually small**.
3.02M + 0.36M parameters is roughly two orders of magnitude below the models
people normally worry about shipping to browsers. The download is ~11 MB cached
forever; the compute fits comfortably on integrated graphics and degrades
gracefully to WASM.

The real work in S1 is not "can WebGPU do this" — it's the tracker gap (§3.5.1)
and colour-stratified numerical parity (§3.5.2).

---

## 4. Server cost model

All figures are list-price estimates as of writing, monthly, USD. Two columns
throughout: self-managed (Hetzner-class VPS) and fully-managed (AWS/Vercel/Neon
tier). The gap between them is roughly 4–6×, and at these volumes it is genuinely
a matter of how you want to spend your time.

### 4.1 Tier A — Closed beta (≤300 users, ~50 solves/day)

| Item | Choice | Self-managed | Managed |
|---|---|--:|--:|
| App + DB + Redis | 1× 4 vCPU/8 GB VPS, docker-compose | $8 | Fly + Neon + Vercel: $25 |
| Backups | daily snapshots | $2 | included |
| Object storage | Cloudflare R2, <10 GB | $1 | $1 |
| CDN / DNS | Cloudflare free | $0 | $0 |
| Email, error tracking | Resend + Sentry free tiers | $0 | $0 |
| **Total** | | **≈$11** | **≈$26** |

Comfortably inside `docs/ROADMAP.md` Phase 2's `<$25/mo` DoD on the self-managed
path, marginal on the managed one.

### 4.2 Tier B — Public launch (5k registered, ~1k MAU, ~500 solves/day)

| Item | Self-managed | Managed |
|---|--:|--:|
| App servers (2× 4 vCPU/8 GB) | $16 | $120 |
| Postgres (8–16 GB, + backups) | $15 | $60 |
| Redis | $0 (co-located) | $10 |
| Object storage (~50 GB, R2) | $3 | $3 |
| Error tracking (Sentry Team) | $26 | $26 |
| Transactional email (~50k) | $20 | $20 |
| **Total** | **≈$80** | **≈$240** |
| **Per MAU** | **$0.08** | **$0.24** |

**This misses ROADMAP's `<$0.02/MAU` target, and that's expected** — at 1k MAU
the bill is almost entirely fixed cost. The target is a scale metric, not a
launch metric; it's met from roughly 5k MAU onward without changing anything.
Worth annotating the metrics table in `docs/ROADMAP.md` §4 so it doesn't read as
a failure later.

### 4.3 Tier C — 100k MAU (VISION §12.4's "Growth" row, which claims ~$20,000/mo)

| Item | Self-managed | Managed (AWS) |
|---|--:|--:|
| App tier (4× 8 vCPU/16 GB) | $60–120 | $600 |
| Postgres primary + read replica | $30–60 | $400–700 |
| Redis (cache + matchmaking queues) | $20–40 | $100–200 |
| WebSocket / room tier (3× 4 GB, Channels) | $25–50 | $200 |
| Object storage (R2, ~1.5 TB avg, **$0 egress**) | $25 | $25 |
| Re-verification workers (CPU, sampled — §4.4) | $5 | $25 |
| Model/CDN delivery (§4.5) | $0–25 | $0–25 |
| Observability | $100 | $200 |
| **Total** | **≈$300–450** | **≈$1,600–2,000** |
| **Per MAU** | **$0.003–0.005** | **$0.016–0.020** |

**Versus VISION's ~$20,000/mo.** The discrepancy is almost entirely one line
item: VISION §12.2 specifies "AWS inf2.xlarge / g5 (TorchServe)" and §12.4 scales
that to "4–16× GPU." A `g5.xlarge` is ~$1.00/hr ≈ $730/mo on demand, so 4–16 of
them is ~$2,900–11,700/mo — which is how you get to $20k. **The client-side
architecture deletes that fleet entirely**, and §4.4 shows what little remains
runs on CPU for pocket change.

### 4.4 Why no GPU is needed, quantified

Server-side re-verification per solve: ~12 key frames × (one 8.7 GFLOP detector
pass + nine trivial CNN passes) ≈ **105 GFLOPs**. A modern server core under
ORT/oneDNN sustains roughly 50–100 GFLOPS, so call it **~1.5 core-seconds per
solve**.

At 100k MAU and ~15 solves/MAU/month = 1.5M solves/month:

| Sampling rate | Core-hours/mo | Cost @ ~$0.04/vCPU-hr (spot) |
|---|--:|--:|
| 10% (ordinary results) | 62 | **~$2.50** |
| 100% (every solve) | 625 | **~$25** |

Even verifying *every single solve* on the server costs about as much as one
lunch. The sampled-plus-leaderboard policy from §2.3 is therefore driven by
latency and queue depth, not by cost — and if it's ever simpler to just verify
everything, that's affordable. Good position to be in.

### 4.5 Model delivery — the cost the design notes correctly worried about

> "We won't have to deal with much traffic issues besides serving the model to
> the end user"

Right instinct. At 100k MAU × ~11 MB × ~1.3 fetches/mo (cache misses + releases)
≈ **1.4 TB/mo of egress**. Where you serve it from is the entire cost:

| Origin | Egress rate | Cost at 1.4 TB |
|---|---|--:|
| **Cloudflare R2 / Pages** | **$0** | **$0** |
| S3 + CloudFront | ~$0.085/GB | ~$120 |
| Vercel (beyond included) | ~$0.15/GB | ~$150+ |

**Serve models and static assets from Cloudflare R2 or Pages, never from the
Next.js origin.** Combined with immutable content-hashed filenames (§3.2), model
delivery is a rounding error rather than a line item. This one decision is worth
~$150/mo at Tier C and costs nothing to make now.

### 4.6 The two things that could actually blow the budget

Everything above is comfortable. These are not:

1. **TURN relay for live video (Phase 4+).** If spectating relays peer video,
   costs scale with viewer-hours, not users: 1,000 concurrent spectators at
   1 Mbps is ~450 GB/hour, and TURN egress runs ~$0.40–0.60/GB → **~$200/hour**.
   That single feature can exceed the entire rest of the infrastructure by an
   order of magnitude. **Mitigation, decided now:** spectators see the timer,
   states, and continuity status — *not* relayed webcam video. If video
   spectating ships, it goes through one server-side transcode to HLS, never
   mesh-relayed per viewer.
2. **Storing raw video instead of evidence bundles.** 20 s of 1080p is ~40 MB;
   a bundle is ~200 KB. At 1.5M solves/mo that's 60 TB/mo versus 300 GB/mo — a
   200× difference, and it converts storage from $25/mo into thousands. The
   evidence-bundle format (§S3) is what keeps this from happening by accident.
   Retain full video **only** for flagged and top-leaderboard results.

---

## 5. Implementation roadmap

Sequenced against `docs/ROADMAP.md`'s phases rather than replacing them. Week
numbers are that document's timeline.

### S0 — Deployment skeleton (week 1)

Prove the shape before building into it.

- `server/` Django project + DRF, `docker-compose.yml` (django, next, postgres, redis, caddy).
- Caddy path-routing: `/api/*` → gunicorn, `/*` → Next. One origin, no CORS.
- COOP/COEP headers set for `crossOriginIsolated` (needed by WASM threads, §3.4).
- CI: GitHub Actions running lint + the (currently single) test suite.

**DoD:** `docker compose up` serves the existing Next pages and a DRF
`/api/health/` from one domain, locally and on a staging VPS.

### S1 — Browser inference spike (weeks 2–5) — **do this before anything else substantial**

This is the highest-variance item and it validates the cost model. If WebGPU
inference doesn't hold up, §4's numbers change and the plan needs rework — so
find out now, not in month four.

- Export both models to ONNX (fp16 first), script checked into `cv/export/`.
- **Colour-stratified parity harness** (§3.5.2): PyTorch vs ONNX on ≥20 images
  with orange/red over-represented. Aggregate tolerance is not sufficient.
- ORT Web harness: WebGPU detector + WASM sticker CNN, reproducing detect → warp
  → slice → ensemble-classify.
- **Measure** detector latency across ≥4 device classes including one mid-range
  Android and one no-WebGPU fallback machine. Replace §3.3's estimates with real
  numbers in this document.
- **Resolve the tracker gap** (§3.5.1) by measurement; write the decision into
  `docs/ROADMAP.md` Phase 1.
- Sustained-load test: 5 consecutive 30 s guard runs on mobile, watching for
  thermal throttling.

**DoD:** a webcam frame in-browser returns 9 sticker labels with confidences,
matching the Python pipeline on a recorded test set; a measured latency table
exists; the tracker decision is recorded.

*Maps to `docs/ROADMAP.md` Phase 1, tasks 1–2.*

### S2 — Django core (weeks 4–7, overlaps S1)

- Models: `User`, `Solve`, `EvidenceBundle`, `Competition`, `CompetitionEntry`,
  `Result`. Defer `Rating`/`Match` to S6 — no live play until Phase 4.
- `django-allauth`: email + Google + Discord (VISION §7.1.1).
- **Scramble service: server-authoritative, seeded, signed, logged** (VISION
  §8.1/§9.3). WCA random-state for 3×3. The client may never generate one.
- DRF endpoints for the solo-solve flow + the weekly competition.
- Django admin tuned as the review queue: bundle preview, continuity report,
  approve/reject actions.

**DoD:** a scramble can be issued, a solve recorded against it, and both are
visible and actionable in admin.

### S3 — Evidence protocol + re-verification worker (weeks 7–9)

The trust boundary from §2.3 — the load-bearing piece.

- Freeze the evidence bundle schema: 6 face scans, solved-state frames, ~12 key
  frames, continuity report, per-frame hashes, client timings, model version.
- Upload path: direct-to-R2 presigned PUT, metadata to Django.
- Re-verification worker importing `cv/` directly, CPU-only, Celery/RQ on Redis.
- Sampling policy: 10% of ordinary results, 100% of leaderboard-eligible,
  100% of anything the continuity guard flagged.
- Client/server disagreement → `FLAGGED`, into the review queue, never silently
  dropped.

**DoD:** a tampered bundle (edited client verdict, substituted frame) is caught
by the worker and lands in review; 20 honest bundles pass clean.

### S4 — Solve flow in the browser (weeks 8–12)

- Wire `getUserMedia` + the S1 harness into `web/app/(app)/solve/`.
- Guided 6-face scan UX with live overlay, per-face confidence, instant retake.
- **Port the orange/red calibration as a pre-scan step sampling a fixed box from
  the raw frame** — *not* gated on detection succeeding. CLAUDE.md documents this
  exact dependency as the bug that made the original fix inert; do not
  reintroduce it in the browser port.
- Continuity guard v1 (ROADMAP §2.1): uniqueness DQ, presence DQ with gap
  tolerance, trajectory flags, at the cadence S1 measured.
- Pure-JS state validity check (parity/permutation/orientation).
- Verified result card, shareable image.

**DoD:** end-to-end flow in under 3 minutes including scans; a deliberate cube
swap is DQ'd; 20 legitimate solves pass without false DQ.

*Maps to `docs/ROADMAP.md` Phase 1 tasks 3–5 and Phase 2.*

### S5 — Async weekly competition (weeks 12–15)

- Same scramble set per week, entry window, Sunday close, leaderboard.
- Leaderboard-eligible results force 100% re-verification before publishing.
- Result notification email.

**DoD:** two consecutive weekly comps complete with ≥30 participants.

**This is the launch product.** Everything below is gated on ROADMAP Phase 2's
exit criterion: ≥40% week-2 retention.

### S6 — Live rooms (Phase 4, months 6+, gated)

Only after async retention is proven.

- Django Channels + Redis channel layer; room = channel group.
- Matchmaking: Redis sorted set on rating, widening bands per VISION §10.5.
- Server-authoritative timing (VISION §9.3) — client never reports elapsed time.
- Glicko-2 (`glicko2` on PyPI; do not implement it yourself).
- **No relayed video spectating** (§4.6.1).

**DoD:** 100 successful verified rated matches in internal testing.

---

## 6. Open questions

1. **Self-managed or managed hosting?** §4 prices both. Self-managed is ~4–6×
   cheaper and, at one box, genuinely not much work; managed buys back time when
   something breaks at 2am. This is a temperament call, not a technical one —
   but it should be made before S0 rather than drifted into.
2. **Is the ~11 MB first-visit download acceptable on mobile data?** It's cached
   forever after, but the first solve on a phone is a real barrier. Worth an
   explicit "download models (11 MB)" gate rather than a silent fetch.
3. **Retention policy for evidence bundles.** VISION §17.3 says 90 days free /
   1 year premium *for replays*. Bundles are tiny — keeping them permanently
   costs almost nothing and makes retroactive audits possible if a cheat method
   is discovered later. Recommend: bundles permanent, full video 30 days unless
   flagged.
4. **Does the Phase 0 landing page need any of this?** No — and it must not load
   any of it (§3.2). Confirming so S1's bundle work doesn't leak into the
   conversion surface.

---

## Appendix A — Original design notes

Preserved verbatim as the decision seed for this document.

> We should use a sql relational database because the information we are trying
> to store is deterministic. The info will be in standard formats. For a dbms we
> could couple it with django and make it one full back end system. That system
> can serve static compiled react files so combign it into one system and host it
> with one total stack. We won't have to deal with much traffic issues besides
> serving the model to the end user because of the design of using a web gpu
> system in order to offload compute capacity to end users as opposed to paying
> for server compute. The main features like user accounts and solve data can be
> stored easily with django because the features are built into the backend. For
> django we actually probably won't be able to do a static compile so we could
> have the django framework be a django rest framework and use the frontend to
> ping the backend. The only problem might be that we would have to host our
> backend and frontend seperately. This needs to be considered. We might have to
> have a little more complex programs running on our servers in order to host the
> competitive rooms. This is not a novel concept so we can just copy existing
> frameworks. We just need to spin up rooms when people get into the queue and
> attempt to match users with open rooms.

**Disposition of each point:**

| Note | Outcome |
|---|---|
| SQL relational DB | ✅ Confirmed — Postgres |
| Django as one full backend | ✅ Confirmed, §2.1 — with a stronger reason (Python re-verification shares `cv/`) |
| Serve static compiled React from Django | ⚠️ Revised, §2.2 — reverse proxy instead; static export costs Next features for no gain |
| DRF + frontend pings backend | ✅ Confirmed |
| "Would have to host frontend and backend separately" | ✅ Resolved, §2.2 — no, one origin via Caddy |
| WebGPU offloads compute to end users | ✅ Confirmed for compute (§3), ⚠️ corrected for trust (§2.3) |
| Only real traffic cost is serving the model | ✅ Confirmed and quantified, §4.5 — and it's $0 from R2 |
| Django user accounts built in | ✅ Confirmed — `django-allauth` |
| Rooms need more complex server programs | ✅ Confirmed — Django Channels, §2.4 |
| Rooms are not novel, copy existing frameworks | ✅ Confirmed — but gated to Phase 4, §2.4 |
