"""
continuity_guard.py — solve-continuity guard (ROADMAP.md §2.1).

Watches the interval from scramble-scan to solved-scan and enforces.

TWO-TIER by design (2026-07-17): the verdict is "pass" or "dq" — there is
NO human-review tier. This is a product decision: a verified solve must be
certifiable with no human ever watching the clip. So every swap-indicative
signal is a hard DQ, and only genuinely benign interior events (table
slams, brief interior occlusion) are forensic notes that do not affect the
verdict. The tradeoff is deliberate strictness: the cube must stay visible
and roughly centered through the whole solve — an appearance change across
an occlusion, a border-zone gap, or a pipeline stall all DQ, because none
can be distinguished from a swap without a human looking.

  1. UNIQUENESS  — two confidently-detected cubes in frame → hard DQ,
                   but only when SUSTAINED (>= MULTI_DQ_HITS multi-cube
                   YOLO frames inside MULTI_WINDOW_S seconds). Below that
                   bar every sub-DQ multi-cube frame is a forensic note
                   only: rare background false positives (e.g. a light-up
                   keyboard firing once every few hundred frames) must not
                   DQ a good player, and a real swap sustains multi-cube
                   frames past the DQ bar or trips the gap/appearance DQs.
                   Additionally, two boxes that ABUT each other and whose
                   union still looks like one cube (aspect sane AND the
                   boxes tile >= MERGE_COVER of the union) are merged
                   before counting — a resting cube seen corner-on often
                   yields one YOLO box per visible face, which must never
                   read as two cubes. Two real cubes held side-by-side
                   produce an elongated union; held diagonally they produce
                   a square-ish union they only half-cover — both still
                   count as two.
  2. PRESENCE    — no cube detected for longer than GAP_DQ_S seconds → DQ.
                   Brief flicker (hand occlusion, motion blur) is expected
                   and tolerated. Never DQ on a single missed frame.
                   Sub-DQ gaps (>= GAP_FLAG_S) are judged by WHERE they
                   happen AND how the cube comes back: a gap that starts
                   or ends in the border zone (EDGE_MARGIN) → hard DQ
                   (dq_edge_gap) — the border is a swap route. An interior
                   gap is benign ONLY if the cube reappears in place and
                   looking the same: a mid-frame occluder like a table
                   edge is an interior exit route (2026-07-18 — an
                   under-the-table swap exploited the old interior-note
                   leniency), so a far reappearance (dq_reappear_jump) or
                   an appearance change (dq_appearance_change) DQs even
                   mid-frame. Both GAP_FLAG_S and GAP_DQ_S are tuned to
                   the ~22 fps pipeline (2026-07-18) — at that rate 0.18s
                   is still 4 missed frames, and every extra fps narrows
                   the window a fast swap can hide in.
  3. TRAJECTORY  — a teleport under near-continuous detection
                   (gap < GAP_FLAG_S) is border-gated: touching the border
                   zone → hard DQ (dq_teleport); fully interior it is a
                   note (slamming the cube onto the table to stop the
                   timer produces violent interior jumps with no real
                   detection gap — a legit solve must never DQ for that).
                   Across a gap >= GAP_FLAG_S there is no interior
                   leniency for position: reappearing more than
                   REAPPEAR_JUMP diagonals away → hard DQ anywhere
                   (dq_reappear_jump).
                   Appearance is checked across any gap >= SIG_CHECK_S
                   regardless of position (HSV signature before vs after —
                   a palmed-in solved cube looks nothing like the scrambled
                   cube that vanished) → hard DQ (dq_appearance_change),
                   with a laxer bar (SIG_DIST_INTERIOR) for fully-interior
                   gaps so motion blur from fast center-frame moves stays
                   a note (2026-07-18);
                   combined with a border gap and a position jump it also
                   records the dq_swap_reappearance detail.
  4. CADENCE     — the guard can only certify the interval it actually
                   samples. If wall-clock time between analyzed frames
                   grows into a swap-sized window (slow model / pipeline
                   lag), a cube swap can complete entirely between two
                   frames with no presence gap ever recorded — the model
                   sees a cube before and a cube after and never a hole.
                   So frame rate is itself a security parameter here: an
                   inter-frame interval >= FRAME_STALL_DQ_S is a hard DQ
                   (dq_analysis_stall). Enforced only once a cube is being
                   tracked; model-load warmup and pre-cube idle are exempt.

Batch-testing real ble/training_data sessions (2026-08-03, all legit solves,
none deliberately swapped — see batch_test_guard.py) found two false-DQ
mechanisms, one fixed and one still open:
  * BEST-BOX SELECTION (fixed) — a static background object (a room
    fixture: a fish-tank stand, a dresser corner) was briefly MORE
    confident than the real cube and hijacked best-box selection for a
    frame, registering a false dq_teleport when tracking snapped back.
    _pick_best() now prefers the box nearest the previously-tracked
    position (TRACK_CONTINUITY_RADIUS) over raw confidence — this doesn't
    touch the uniqueness count, so it can't be gamed by a still or moving
    decoy.
  * STATIC SECONDARY BOXES (open, NOT fixed) — the same kind of fixture
    sustained past MULTI_DQ_HITS and false-DQ'd via two_cubes (6 of 10
    false-DQs in the batch, the dominant cause). A within-window "has this
    box moved" check was prototyped and reverted: on this guard's 1s
    MULTI_WINDOW_S timescale, a real second cube held still for a second
    (e.g. propped on a table edge) is not distinguishable from furniture
    that's been there for hours — the synthetic adversarial suite confirmed
    this concretely (sustained-two-cubes and diagonal-touching-cubes tests
    both got waved through). The safe version of this fix is a pre-solve
    background calibration step (snapshot the empty scene before the
    tracked interval starts, exempt only boxes seen in that snapshot) —
    same shape as state_finder.py's orange/red run_calibration() — not yet
    built; needs wiring into the actual capture flow, not just the guard.

Verdicts: "pass" | "dq" (two-tier — no human-review tier, see above).

The core state machine (ContinuityGuard) is pure — it consumes
(timestamp, boxes) pairs and never touches the camera or model — so it is
unit-testable without weights. detect_cubes()/analyze_* wire it to YOLO.

Run from inside cv/ (bare model filenames — see CLAUDE.md):
    python continuity_guard.py --video solve.mp4 [--display]
    python continuity_guard.py --frames ../ble/training_data/<session>/frames --fps 30
"""

import argparse
import json
import os

# ---------------------------------------------------------------------------
# Thresholds (ROADMAP §2.1 — start values, tune on beta data)
# ---------------------------------------------------------------------------
GAP_DQ_S        = 0.5    # presence: max tolerated detection gap, seconds.
                         #   Tightened 0.6 -> 0.5 (2026-07-18) alongside the
                         #   fps fix: normal hand occlusion measured ~0.4-0.5s
                         #   at 9.5 fps, but at ~22 fps the detector gets 2.3x
                         #   more chances to catch the cube mid-occlusion, so
                         #   legit gaps measure shorter. This is also the hard
                         #   ceiling on an under-the-table swap's round trip.
                         #   Validate on live legit solves before tightening
                         #   further.
GAP_FLAG_S      = 0.18   # any gap >= this runs the reappearance continuity
                         #   checks: DQ if it touches the border zone or the
                         #   cube comes back far away / looking different;
                         #   note if fully interior, in place, same-looking.
                         #   Tightened 0.30 -> 0.18 (2026-07-18): at ~22 fps
                         #   this is still 4 missed frames, and a side-swap
                         #   now has to complete in under 0.18s to duck the
                         #   border checks — at 9.5 fps the old 0.30 bar was
                         #   the fastest we could afford without flagging
                         #   single missed frames.
REAPPEAR_JUMP   = 1.2    # reappearance center jump after a >= GAP_FLAG_S
                         #   gap, in units of box diagonal, beyond which it
                         #   is a hard DQ — EVERYWHERE, interior included
                         #   (2026-07-18; was a note in the interior). A
                         #   table edge crossing the middle of the frame is
                         #   an interior exit route: dip the cube below it,
                         #   swap, raise the other cube elsewhere — the old
                         #   interior leniency passed that with a note. A
                         #   legit occluded cube reappears where it was; only
                         #   in-place, same-looking interior reappearances
                         #   stay benign. (Slams stay legal via the teleport
                         #   path: near-continuous detection, gap < GAP_FLAG_S.)
SIG_CHECK_S     = 0.12   # gaps >= this compare appearance signatures.
                         #   Tightened 0.20 -> 0.12 (2026-07-18): at ~22 fps
                         #   even a 3-frame blink gets an appearance check,
                         #   so a fast swap can't duck under the gap bar.
                         #   Paired with LOOSER distance bars below — check
                         #   more gaps, but demand a bigger difference to DQ.
SIG_DIST        = 0.45   # ...Bhattacharyya distance beyond this → hard DQ
                         #   when the gap touches the border zone. Loosened
                         #   0.35 -> 0.45 (2026-07-18): motion blur on real
                         #   solves lands ~0.35-0.5 and was DQing legit
                         #   attempts; a scrambled-vs-solved-face swap still
                         #   measures well above.
SIG_DIST_INTERIOR = 0.65 # interior appearance-DQ bar (loosened 0.55 -> 0.65
                         #   2026-07-18, same false-DQ reason). Interior
                         #   distances in the SIG_DIST..this band are
                         #   forensic notes. The under-the-table swap is not
                         #   handled here — that's the reappear-jump DQ and
                         #   GAP_DQ_S ceiling (see TRAJECTORY).
PRESENCE_CONF   = 0.10   # presence uses a loose floor ("a cube-ish thing is
                         #   there") — measured on real solves this cuts gap
                         #   time roughly in half vs the 0.35 scan floor.
                         #   Uniqueness still requires MULTI_CONF below.
MULTI_CONF      = 0.50   # uniqueness: second box must be at least this confident
MULTI_DQ_HITS   = 3      # uniqueness: multi-cube frames needed to hard-DQ...
MULTI_WINDOW_S  = 1.0    # ...within this rolling window. Below this bar every
                         #   sub-DQ multi-cube frame is a forensic note (kept
                         #   at 3 so a rare background FP can't DQ a good
                         #   player; a real swap sustains past it)
DUP_IOU         = 0.40   # boxes overlapping this much are one cube (YOLO dup)
ADJ_GAP_FRAC    = 0.15   # boxes whose edges are closer than this fraction of
                         #   the smaller box's diagonal are "touching"...
MERGE_ASPECT    = 1.8    # ...and merge into one cube IF their union box is no
                         #   more elongated than this (two faces of one cube);
                         #   widened 1.6 -> 1.8 (2026-08-03): batch-testing
                         #   against real ble/training_data sessions found a
                         #   single tilted cube split top/bottom-half by YOLO
                         #   unioning to 1.63 (verified visually — one cube,
                         #   both hands, no swap) and wrongly kept as two,
                         #   hard-DQing a legit solve. Two real cubes
                         #   side-by-side still union to ~2:1 → kept, with
                         #   comfortable margin either side of the new bar.
MERGE_COVER     = 0.75   # ...AND the two boxes tile at least this fraction of
                         #   their union's area (2026-07-18). A genuine face
                         #   split covers ~95%+ of its union; two real cubes
                         #   held diagonally union to a deceptively square box
                         #   (which passed the aspect gate and was wrongly
                         #   merged — "MERGED 2->1" on screen, no DQ) but
                         #   leave the union's off-diagonal corners empty,
                         #   covering only ~50-60% → kept as two cubes.
INFER_IMGSZ     = 416    # YOLO inference size for the guard (default 640).
                         #   fps is a security parameter here (see CADENCE):
                         #   416 cuts inference 62ms -> 40ms/frame (measured
                         #   2026-07-18) and a cube is a large object, so
                         #   presence detection at PRESENCE_CONF is unhurt.
TELEPORT_JUMP   = 1.8    # trajectory: center jump, in units of box diagonal...
TELEPORT_DT_S   = 0.45   # ...within this dt → teleport (DQ at border, note
                         #   interior)
EDGE_MARGIN     = 0.12   # last-seen or reappearance center within this
                         #   fraction of the border → border-grade event.
                         #   Widened 0.08 -> 0.12 (2026-07-18): fast side
                         #   swaps were staging just inside the old 8% zone.
FRAME_STALL_DQ_S = 0.20  # cadence: wall-clock time between analyzed frames
                         #   above which a swap could complete unseen between
                         #   samples → hard DQ (analysis_stall). Single cutoff,
                         #   no review tier: if the pipeline can't sample fast
                         #   enough to rule out a swap, the attempt can't be
                         #   certified. Tightened 0.30 -> 0.20 (2026-07-18)
                         #   after the fps fix: at ~22 fps intervals are
                         #   ~0.045s, so this is still a 4x hiccup allowance,
                         #   and it shrinks the blind window an attacker gets
                         #   from a stalled pipeline.
TRACK_CONTINUITY_RADIUS = 1.0  # trajectory: when picking which box is "the"
                         #   cube this frame, prefer the one nearest the
                         #   previous tracked position (within this many
                         #   diagonals) over the most confident box (added
                         #   2026-08-03: a background object briefly MORE
                         #   confident than the real cube — e.g. 0.81 vs 0.76
                         #   — hijacked best-box selection for one frame and
                         #   registered as a teleport when tracking snapped
                         #   back). Falls back to highest confidence only when
                         #   nothing is near the last position (first frame,
                         #   genuine occlusion recovery).


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a[:4]
    bx1, by1, bx2, by2 = b[:4]
    ix = max(0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    if inter == 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def _sig_dist(a, b):
    """Bhattacharyya distance between two normalized histograms.

    0.0 = identical distributions, 1.0 = disjoint. Pure math so the guard
    stays testable without cv2 — signatures are plain flat float sequences
    (see box_signature() in the wiring section).
    """
    bc = sum((x * y) ** 0.5 for x, y in zip(a, b))
    return max(0.0, 1.0 - min(1.0, bc)) ** 0.5


def _diag(b):
    return ((b[2] - b[0]) ** 2 + (b[3] - b[1]) ** 2) ** 0.5


def pick_continuity(boxes, ref_box, radius=None):
    """Pick the box nearest ref_box's center (within `radius` diagonals) over
    raw confidence; falls back to highest confidence when ref_box is None or
    nothing is near it (bootstrap / genuine occlusion recovery). Shared by
    ContinuityGuard._pick_best and mine_hard_negatives.py, which both need to
    know "which box is the cube we were already tracking" rather than
    whichever box YOLO happened to be most confident about this frame."""
    if radius is None:
        radius = TRACK_CONTINUITY_RADIUS
    if ref_box is None:
        return max(boxes, key=lambda b: b[4])
    diag = max(1.0, _diag(ref_box))
    lx1, ly1, lx2, ly2 = ref_box[:4]
    lcx, lcy = (lx1 + lx2) / 2, (ly1 + ly2) / 2

    def dist(b):
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        return ((cx - lcx) ** 2 + (cy - lcy) ** 2) ** 0.5 / diag

    nearest = min(boxes, key=dist)
    if dist(nearest) <= radius:
        return nearest
    return max(boxes, key=lambda b: b[4])


def _edge_gap(a, b):
    """Shortest distance between two boxes' edges (0 if they intersect)."""
    gx = max(a[0] - b[2], b[0] - a[2], 0)
    gy = max(a[1] - b[3], b[1] - a[3], 0)
    return (gx * gx + gy * gy) ** 0.5


def dedup_boxes(boxes):
    """Merge detections that belong to the same physical cube.

    Two failure modes would otherwise count one cube as two — a
    catastrophic false DQ:
      * YOLO duplicate: two heavily-overlapping boxes on one cube.
        Keep the most confident box of any overlapping cluster.
      * Face split: a resting cube seen corner-on gets one box per
        visible face — adjacent, barely overlapping. Merge touching
        boxes into their union, but only when the union still has the
        aspect ratio of a single cube AND the boxes actually tile it
        (MERGE_COVER); two real cubes side-by-side union to roughly 2:1
        and are deliberately NOT merged, and two cubes held diagonally
        union to a square-ish box the pair covers only ~half of — also
        NOT merged.
    """
    out = []
    for b in sorted(boxes, key=lambda b: -b[4]):
        if all(_iou(b, k) < DUP_IOU for k in out):
            out.append(b)
    merged = True
    while merged and len(out) > 1:
        merged = False
        for i in range(len(out)):
            for j in range(i + 1, len(out)):
                a, b = out[i], out[j]
                if _edge_gap(a, b) > ADJ_GAP_FRAC * min(_diag(a), _diag(b)):
                    continue
                u = (min(a[0], b[0]), min(a[1], b[1]),
                     max(a[2], b[2]), max(a[3], b[3]), max(a[4], b[4]))
                w, h = u[2] - u[0], u[3] - u[1]
                if max(w, h) > MERGE_ASPECT * max(1, min(w, h)):
                    continue
                area_a = (a[2] - a[0]) * (a[3] - a[1])
                area_b = (b[2] - b[0]) * (b[3] - b[1])
                ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
                iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
                if area_a + area_b - ix * iy < MERGE_COVER * max(1, w * h):
                    continue
                out[i] = u
                del out[j]
                merged = True
                break
            if merged:
                break
    return out


class ContinuityGuard:
    """Pure state machine. Feed update(t, boxes[, sig]) for every analyzed frame.

    t      : seconds (monotonic, e.g. frame_idx / fps)
    boxes  : list of (x1, y1, x2, y2, conf) raw detections for that frame
    sig    : optional appearance signature of the best box (flat normalized
             histogram, see box_signature()); enables the appearance-
             continuity check across detection gaps
    """

    def __init__(self, frame_w, frame_h):
        self.fw, self.fh = frame_w, frame_h
        self.events = []          # [{t, type, detail}]
        self.dq_reasons = []
        self.start_t = None
        self.last_seen_t = None   # time of last accepted detection
        self.last_update_t = None # time of last analyzed frame (any content)
        self.max_frame_gap = 0.0  # largest interval between analyzed frames
        self.last_box = None
        self.last_sig = None      # appearance signature at last detection
        self.multi_hits = []      # timestamps of multi-cube frames
        self.frames = 0
        self.detected_frames = 0
        self.max_gap = 0.0
        self._open_gap_flagged = False
        self._open_gap_dqed = False

    # -- internals ----------------------------------------------------------

    def _event(self, t, typ, **detail):
        self.events.append({"t": round(t, 3), "type": typ, **detail})

    def _dq(self, t, reason, **detail):
        if reason not in self.dq_reasons:
            self.dq_reasons.append(reason)
        self._event(t, "dq_" + reason, **detail)

    def _near_edge(self, box):
        x1, y1, x2, y2 = box[:4]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        mx, my = EDGE_MARGIN * self.fw, EDGE_MARGIN * self.fh
        return cx < mx or cy < my or cx > self.fw - mx or cy > self.fh - my

    def _pick_best(self, boxes):
        # Prefer continuity with the cube we're already tracking over raw
        # confidence — see TRACK_CONTINUITY_RADIUS. A background object that
        # is momentarily MORE confident than the real cube must not hijack
        # the tracked identity for a frame (that hijack-and-snap-back is
        # exactly what a false teleport DQ looks like).
        return pick_continuity(boxes, self.last_box)

    # -- public -------------------------------------------------------------

    def update(self, t, boxes, sig=None):
        if self.start_t is None:
            self.start_t = t
        self.frames += 1

        # 4. CADENCE ---------------------------------------------------------
        # Only meaningful once a cube is being tracked (last_seen_t set):
        # before that, slow frames carry no swap risk, and the model-load
        # warmup happens before the first update pair. A stall between two
        # analyzed frames is a window the guard never got to inspect.
        frame_dt = None if self.last_update_t is None else t - self.last_update_t
        self.last_update_t = t
        if frame_dt is not None and self.last_seen_t is not None:
            self.max_frame_gap = max(self.max_frame_gap, frame_dt)
            if frame_dt >= FRAME_STALL_DQ_S:
                self._dq(t, "analysis_stall", frame_dt_s=round(frame_dt, 2))

        boxes = dedup_boxes([b for b in boxes if b[4] >= 0.0])
        confident = [b for b in boxes if b[4] >= MULTI_CONF]
        best = self._pick_best(boxes) if boxes else None

        # 1. UNIQUENESS ------------------------------------------------------
        if len(confident) >= 2:
            self.multi_hits.append(t)
            self.multi_hits = [h for h in self.multi_hits
                               if t - h <= MULTI_WINDOW_S]
            hits = len(self.multi_hits)
            if hits >= MULTI_DQ_HITS:
                self._dq(t, "two_cubes", count=len(confident))
            else:
                # below the DQ bar: forensic note only. Rare background FPs
                # must not DQ a good player; a real swap sustains multi-cube
                # frames past the bar or trips the gap/appearance DQs.
                self._event(t, "note_multi_frame", count=len(confident))

        # 2 + 3. PRESENCE / TRAJECTORY --------------------------------------
        if boxes:
            if self.last_seen_t is not None:
                gap = t - self.last_seen_t
                self.max_gap = max(self.max_gap, gap)
                if gap > GAP_DQ_S and not self._open_gap_dqed:
                    self._dq(t, "presence_gap", gap_s=round(gap, 2))
                # border involvement decides severity: the border zone is
                # the only route a swap cube can take, so events touching
                # it are DQ/flag-grade; identical events fully in the
                # interior are benign notes (slams, fast waves).
                at_border = ((self.last_box is not None
                              and self._near_edge(self.last_box))
                             or self._near_edge(best))
                # position continuity, scaled by cube size
                jump_frac = None
                if self.last_box is not None:
                    lx1, ly1, lx2, ly2 = self.last_box[:4]
                    bx1, by1, bx2, by2 = best[:4]
                    diag = max(1.0, ((lx2 - lx1) ** 2 + (ly2 - ly1) ** 2) ** 0.5)
                    jump = (((bx1 + bx2) - (lx1 + lx2)) ** 2
                            + ((by1 + by2) - (ly1 + ly2)) ** 2) ** 0.5 / 2
                    jump_frac = jump / diag
                    # teleport: big jump with (near-)continuous detection.
                    # At the border it is the swap route → DQ; identical
                    # jump fully interior (table slam) is a benign note.
                    if gap <= TELEPORT_DT_S and jump_frac > TELEPORT_JUMP:
                        if at_border:
                            self._dq(t, "teleport", jump_frac=round(jump_frac, 2))
                        else:
                            self._event(t, "note_teleport",
                                        jump_frac=round(jump_frac, 2))
                # appearance continuity across any non-trivial gap: checked
                # everywhere — a palmed swap never touches the border. A cube
                # that reappears looking different across an occlusion is the
                # swap signature → hard DQ (no review tier). Interior gaps
                # get the laxer SIG_DIST_INTERIOR bar: fast center-frame
                # moves blur the crop into the ~0.35-0.5 range and must not
                # DQ; a real swapped-in cube still clears the interior bar.
                sig_d = None
                if (gap >= SIG_CHECK_S and sig is not None
                        and self.last_sig is not None):
                    sig_d = _sig_dist(sig, self.last_sig)
                    dq_bar = SIG_DIST if at_border else SIG_DIST_INTERIOR
                    if sig_d > dq_bar:
                        self._dq(t, "appearance_change",
                                 dist=round(sig_d, 2), gap_s=round(gap, 2))
                    elif sig_d > SIG_DIST:
                        self._event(t, "note_appearance_change",
                                    dist=round(sig_d, 2), gap_s=round(gap, 2))
                # long gap: DQ at the border; in the interior, benign ONLY
                # if the cube comes back in place — a far reappearance
                # across a gap is swap-grade everywhere (2026-07-18: a
                # table edge mid-frame is an interior exit route, so the
                # old interior-note leniency let an under-the-table swap
                # through).
                if gap >= GAP_FLAG_S:
                    jumped = (jump_frac is not None
                              and jump_frac > REAPPEAR_JUMP)
                    if jumped:
                        self._dq(t, "reappear_jump", gap_s=round(gap, 2),
                                 jump_frac=round(jump_frac, 2))
                    if at_border:
                        self._dq(t, "edge_gap", gap_s=round(gap, 2))
                        if jumped and sig_d is not None and sig_d > SIG_DIST:
                            self._dq(t, "swap_reappearance",
                                     gap_s=round(gap, 2),
                                     jump_frac=round(jump_frac, 2),
                                     sig_dist=round(sig_d, 2))
                    else:
                        self._event(t, "note_gap", gap_s=round(gap, 2))
            self.detected_frames += 1
            self.last_seen_t = t
            self.last_box = best
            if sig is not None:
                self.last_sig = sig
            self._open_gap_flagged = False
            self._open_gap_dqed = False
        else:
            # gap in progress: note an edge exit once, DQ when it exceeds max.
            # The exit itself is a forensic note; severity is decided on
            # reappearance (edge_gap DQ) or on total absence (presence_gap).
            if self.last_seen_t is not None:
                gap = t - self.last_seen_t
                self.max_gap = max(self.max_gap, gap)
                if (not self._open_gap_flagged and self.last_box is not None
                        and self._near_edge(self.last_box)):
                    self._event(t, "note_edge_exit")
                    self._open_gap_flagged = True
                if gap > GAP_DQ_S and not self._open_gap_dqed:
                    self._dq(t, "presence_gap", gap_s=round(gap, 2))
                    self._open_gap_dqed = True
            elif t - self.start_t > GAP_DQ_S:
                self._dq(t, "never_seen", waited_s=round(t - self.start_t, 2))

    def report(self):
        # Two-tier by design: pass or dq, no human-review tier. Every
        # swap-indicative signal is already a hard DQ; anything left is a
        # forensic note that does not change the verdict.
        verdict = "dq" if self.dq_reasons else "pass"
        notes = sum(1 for e in self.events if e["type"].startswith("note_"))
        return {
            "verdict": verdict,
            "dq_reasons": self.dq_reasons,
            "notes": notes,
            "frames": self.frames,
            "detected_frames": self.detected_frames,
            "detection_rate": round(self.detected_frames / self.frames, 3)
                              if self.frames else 0.0,
            "max_gap_s": round(self.max_gap, 2),
            "max_frame_gap_s": round(self.max_frame_gap, 2),
            "events": self.events,
        }


# ---------------------------------------------------------------------------
# YOLO wiring — multi-cube detection (cube_detector returns only the best box)
# ---------------------------------------------------------------------------

def detect_cubes(frame, conf_floor=None, imgsz=INFER_IMGSZ):
    """All size-sane cube detections in a frame → [(x1,y1,x2,y2,conf)].

    Runs at INFER_IMGSZ (not YOLO's 640 default): the guard's fps is a
    security parameter (CADENCE), and a cube is a large target."""
    import cube_detector as cd
    model, _ = cd._load_yolo()
    if model is None:
        raise RuntimeError("ultralytics not installed / no model")
    floor = conf_floor if conf_floor is not None else cd.YOLO_CONF
    fh, fw = frame.shape[:2]
    frame_area = fh * fw
    boxes = []
    for r in model(cd.normalize_lighting(frame), verbose=False, conf=floor,
                   imgsz=imgsz):
        if r.boxes is None:
            continue
        for b in r.boxes:
            c = float(b.conf[0])
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            frac = (x2 - x1) * (y2 - y1) / frame_area
            if cd.MIN_BOX_FRAC <= frac <= cd.MAX_BOX_FRAC:
                boxes.append((x1, y1, x2, y2, c))
    return boxes


def box_signature(frame, box):
    """Appearance signature of a box crop: flat normalized HSV hue-sat
    histogram (16x8 bins). Fed to ContinuityGuard.update() so appearance
    continuity can be checked across detection gaps."""
    import cv2
    x1, y1, x2, y2 = (int(v) for v in box[:4])
    crop = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 8],
                        [0, 180, 0, 256]).flatten()
    total = float(hist.sum())
    return (hist / total).tolist() if total > 0 else None


def _analyze(frames_iter, fw, fh, fps, stride, display=False):
    import cv2
    guard = ContinuityGuard(fw, fh)
    for idx, frame in frames_iter:
        if idx % stride:
            continue
        t = idx / fps
        boxes = detect_cubes(frame, conf_floor=PRESENCE_CONF)
        deduped = dedup_boxes(boxes)
        sig = (box_signature(frame, max(deduped, key=lambda b: b[4]))
               if deduped else None)
        guard.update(t, boxes, sig)
        if display:
            for x1, y1, x2, y2, c in boxes:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 230, 60), 2)
                cv2.putText(frame, f"{c:.2f}", (x1, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 230, 60), 1)
            cv2.imshow("continuity", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    if display:
        cv2.destroyAllWindows()
    return guard.report()


def analyze_video(path, stride=1, display=False):
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def frames():
        i = 0
        while True:
            ok, f = cap.read()
            if not ok:
                break
            yield i, f
            i += 1
    try:
        return _analyze(frames(), fw, fh, fps, stride, display)
    finally:
        cap.release()


def analyze_frames_dir(path, fps=30.0, stride=1, display=False):
    """Analyze a directory of image frames (e.g. ble/training_data sessions)."""
    import cv2
    names = sorted(n for n in os.listdir(path)
                   if n.lower().endswith((".jpg", ".jpeg", ".png")))
    if not names:
        raise RuntimeError(f"no frames in {path}")
    first = cv2.imread(os.path.join(path, names[0]))
    fh, fw = first.shape[:2]

    def frames():
        for i, n in enumerate(names):
            f = cv2.imread(os.path.join(path, n))
            if f is not None:
                yield i, f
    return _analyze(frames(), fw, fh, fps, stride, display)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Solve-continuity guard (ROADMAP §2.1)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", help="video file to analyze")
    src.add_argument("--frames", help="directory of image frames")
    ap.add_argument("--fps", type=float, default=30.0,
                    help="fps for --frames mode (default 30)")
    ap.add_argument("--stride", type=int, default=1,
                    help="analyze every Nth frame (default 1 — full rate; a "
                         "fast swap can live in just a handful of frames, so "
                         "only raise this on hardware that can't keep up)")
    ap.add_argument("--display", action="store_true")
    args = ap.parse_args()

    if args.video:
        rep = analyze_video(args.video, args.stride, args.display)
    else:
        rep = analyze_frames_dir(args.frames, args.fps, args.stride, args.display)
    print(json.dumps(rep, indent=2))
