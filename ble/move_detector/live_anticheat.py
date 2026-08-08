"""
live_anticheat.py — the WHOLE anticheat stack, live, in one window.

`cv/detection/live_guard_test.py` already let you attack the continuity
guard in real time. This adds the half that had no interactive path at all:
the move-count gate, the post-timer window test, and `adjudicate()` — so one
run ends in the same verdict the server would reach on a stored bundle.

Why this is a state machine and not just "more HUD"
---------------------------------------------------
The two halves of the anticheat run on DIFFERENT CLOCKS, and pretending
otherwise would misrepresent both.

The continuity guard is per-frame: it decides things about gaps and cube
counts as they happen, and its verdict is live. The count gate cannot be. It
needs the whole timed window at once, because `per_frame_boxes` median-
smooths the crop detections and interpolates across frames — a crop box for
frame 40 depends on detections at frame 50. Cropping each frame with only
its own box would inject the detector's frame-to-frame jitter into the
temporal diffs, which is exactly the fake global motion the model is trying
to read (see crop_utils.py's header). So the count is computed ONCE, after
the window closes, on the buffered frames.

That is also how the real product behaves — you stop the timer and then get
a verdict — so the flow here is the flow, not a testing shortcut.

    IDLE      guard live, nothing buffered.        SPACE starts the solve.
    SOLVING   timed window. Frames buffered.       SPACE stops the timer.
    SCANNING  post-timer window: present the cube  SPACE finishes the scan.
              to the camera. Buffered separately.
    VERDICT   count decoded, adjudicate() run.     R for another attempt.

What to try
-----------
  * swap in a solved cube mid-solve              -> continuity / appearance
  * make ~20 moves and stop (solver-following)   -> too_few_moves
  * stop the timer unsolved, solve, then scan    -> moves_after_timer_stop
  * make 50 real moves and THEN swap             -> the known hole; the count
                                                    gate cannot see this, so
                                                    the continuity guard is
                                                    the only thing standing.
                                                    Worth attacking directly.
  * solve very fast                              -> too_fast_to_separate
                                                    (abstain, not reject)

Run from inside ble/move_detector (bare checkpoint paths — see CLAUDE.md):
    python live_anticheat.py
    python live_anticheat.py --ctc checkpoints/move_ctc_spd_s1.pt --camera 1

Keys:
    SPACE  advance the state machine (start / stop timer / finish scan)
    R      reset to IDLE for another attempt
    S      save the full evidence bundle + verdict to JSON
    Q/ESC  quit
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# --- cross-topic bootstrap (CLAUDE.md): the guard and the cube detector
#     live in cv/detection, this file lives in ble/move_detector, and both
#     halves are needed in one process.
_HERE = Path(__file__).resolve()
_BLE_DIR = _HERE.parents[1]
_DETECTION_DIR = _HERE.parents[2] / "cv" / "detection"
for _p in (str(_BLE_DIR), str(_DETECTION_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from continuity_guard import (ContinuityGuard, box_signature,     # noqa: E402
                              dedup_boxes, detect_cubes, EDGE_MARGIN,
                              FRAME_STALL_DQ_S, GAP_DQ_S, GAP_FLAG_S,
                              MULTI_CONF, MULTI_DQ_HITS, PRESENCE_CONF)
#: `_put` is deliberately NOT imported — see `_text` for why its
#: outline-plus-fill idiom is what produced doubled glyphs.
from live_guard_test import SwapMeter, SWAP_REF          # noqa: E402
from crop_utils import load_detector                     # noqa: E402
from prepare_data import (build_color_stream,            # noqa: E402
                          per_frame_boxes)
from anticheat_gate import (MIN_OBSERVED_MOVES, POST_STOP_MAX_WINDOW_S,
                            SolveEvidence, _decode_count, _load_model,
                            adjudicate, post_stop_limit,
                            separation_tps_limit)          # noqa: E402

DEFAULT_CTC = "checkpoints/move_ctc_spd_s0.pt"

#: Frames are buffered for post-hoc analysis, so the buffer is real memory.
#: Capped at max side 640 — YOLO's native input is 640 and the crop is
#: resized to 96 afterwards, so nothing downstream can tell the difference,
#: while a 1280x720 buffer would cost ~4x as much for no gain.
BUFFER_MAX_SIDE = 640

#: Buffered frames are JPEG-encoded, and this is a FRAME RATE fix, not a
#: disk one. Raw, a 640x360 frame is 675 KB; at 30fps that is 21 MB/s and a
#: 60-second solve holds 1.24 GB of live Python objects. The allocation churn
#: and paging that causes is what made the capture rate DEGRADE over an
#: attempt — it started fine and fell apart, which is the signature of the
#: buffer rather than of any per-frame cost (detect_cubes is 27ms flat).
#:
#: Measured on real camera frames: 41 KB encoded, 16.6x smaller, 75 MB for
#: the same 60 seconds. Encoding costs 1.18 ms/frame against a ~9 ms budget.
#: Decoding (2.8 ms/frame) happens post-hoc where there is no clock.
BUFFER_JPEG_QUALITY = 85
#: ~2 minutes at 30fps. A hard stop, because the alternative to a cap is
#: discovering the limit as an OOM in the middle of an attempt.
BUFFER_MAX_FRAMES = 3600

#: The corpus was recorded at ~30fps and the model's receptive field is
#: measured in FRAMES, so capture rate is a temporal scale factor: at 15fps
#: a move spans half as many frames as anything the model trained on, which
#: reads as a much faster solve than the clock says. Speed augmentation
#: (move_ctc_spd_*) widens the tolerated band but does not remove it, so a
#: run well outside this range is measuring the mismatch, not the solve.
TRAIN_FPS = 30.0
FPS_TOLERANCE = 0.25          # +/- 25% before the count is called suspect

COL_OK    = (60, 200, 60)
COL_BAD   = (50, 50, 230)
COL_WARN  = (0, 170, 230)
COL_MUTED = (170, 170, 170)

VERDICT_COLOR = {
    "verified": COL_OK,
    "rejected": COL_BAD,
    "review":   COL_WARN,
}

STATE_HELP = {
    "IDLE":     "SPACE: start the solve",
    "SOLVING":  "SPACE: stop the timer",
    "SCANNING": "SPACE: finish the scan and adjudicate",
    "VERDICT":  "R: another attempt    S: save bundle",
}


def _shrink(frame):
    """Downscaled copy. Crop geometry is scale-invariant: the box is detected
    on this frame and the crop is resized to FRAME_SIZE either way, so the
    regime training used is preserved."""
    h, w = frame.shape[:2]
    s = BUFFER_MAX_SIDE / float(max(h, w))
    if s >= 1.0:
        return frame.copy()
    return cv2.resize(frame, (int(round(w * s)), int(round(h * s))),
                      interpolation=cv2.INTER_AREA)


def _encode(frame):
    """Shrink + JPEG. See BUFFER_JPEG_QUALITY for why this is a frame-rate fix."""
    ok, buf = cv2.imencode(".jpg", _shrink(frame),
                           [cv2.IMWRITE_JPEG_QUALITY, BUFFER_JPEG_QUALITY])
    return buf if ok else None


def _decode(buf):
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def count_moves(buffer, fps, detector, model, device, beam=0):
    """Decoded move count over a buffer of JPEG-encoded frames.

    Runs the SAME path prepare_data.py runs offline — `per_frame_boxes` to
    get median-smoothed, interpolated crop boxes, then `build_color_stream`
    — so the model sees the framing it trained on. Reusing
    `anticheat_gate._decode_count` rather than re-implementing the decode
    keeps this identical to what `anticheat_gate.py score` measures; a live
    count computed a slightly different way would not be comparable to the
    corpus numbers the thresholds came from.

    Returns (n_moves, labels, n_detected).
    """
    from dataset import JointArrayStream

    n = len(buffer)
    if n < 8:
        return 0, [], 0

    # Decoded once into a list rather than per `load` call: `per_frame_boxes`
    # and `build_color_stream` each walk the whole buffer, so a decode-on-
    # demand `load` would decode every frame twice. At 2.8 ms/frame that is
    # seconds of pure waste on a long attempt.
    frames = [_decode(b) for b in buffer]

    def load(i):
        return frames[i] if 0 <= i < n else None

    if detector is not None:
        boxes, n_detected = per_frame_boxes(detector, load, n)
    else:
        # No detector: build_color_stream's own centered-square fallback.
        # The caller has already warned about the regime mismatch this
        # implies — the count will be wrong, not merely noisier.
        from prepare_data import center_square
        boxes = np.array([center_square(frames[0].shape)] * n)
        n_detected = 0

    block = build_color_stream(load, boxes, n)
    stream = JointArrayStream(frames=block, name="live", fps=fps)
    labels = _decode_count(model, stream, device, beam)
    return len(labels), list(labels), n_detected


def preflight(ctc_path, detector):
    """Check the two things that silently invalidate a count, loudly.

    Both failures produce a number that LOOKS fine. That is what makes them
    worth a banner rather than a comment.
    """
    import torch
    ck = torch.load(ctc_path, map_location="cpu", weights_only=False)
    notes = []

    # 1. The gate's constants are checkpoint-coupled. RETENTION_FLOOR — and
    #    therefore the abstain band and every headroom number — was measured
    #    on the SPEED-AUGMENTED checkpoints. A gate calibrated on a model
    #    that is not the one running is worse than no calibration, because
    #    it looks calibrated.
    if not ck.get("speed_aug"):
        notes.append(
            f"  WARNING: {Path(ctc_path).name} was trained WITHOUT speed "
            f"augmentation, but\n           anticheat_gate.RETENTION_FLOOR "
            f"was measured on the speed-augmented\n           checkpoints "
            f"(move_ctc_spd_*). The abstain band and every headroom\n         "
            f"  number below describe a different model. Re-measure with\n   "
            f"        speed_sim.py --blur before trusting these verdicts.")

    # 2. Crop regime. The classifier-era version of this bug (a cropped model
    #    fed full frames) cost a whole day of misread training logs.
    regime = ck.get("crop_regime")
    if regime == "cropped" and detector is None:
        notes.append(
            "  WARNING: this checkpoint trained on CUBE CROPS but no cube "
            "detector loaded\n           (install ultralytics + "
            "cv/detection/detect_full_cube.pt). Frames will\n           fall "
            "back to a centered square — the move count will be measured on\n"
            "           the wrong scale and the gate's floor will not mean "
            "what it says.")
    elif regime and regime != "cropped" and detector is not None:
        notes.append(
            f"  WARNING: checkpoint crop_regime is {regime!r} but this run "
            f"crops to the cube.")
    return ck, notes


class Attempt:
    """One solve attempt: the buffers, the clocks and the resulting verdict."""

    def __init__(self, fw, fh):
        self.guard = ContinuityGuard(fw, fh)
        self.swap = SwapMeter()
        self.state = "IDLE"
        self.solve_frames: list = []
        self.scan_frames: list = []
        self.t_start = None
        self.t_stop = None
        self.t_scan_end = None
        self.verdict = None
        self.detail = {}
        self.overflow = False

    @property
    def solve_seconds(self):
        if self.t_start is None:
            return None
        end = self.t_stop if self.t_stop is not None else time.perf_counter()
        return end - self.t_start

    @property
    def scan_seconds(self):
        if self.t_stop is None:
            return None
        end = self.t_scan_end if self.t_scan_end is not None else time.perf_counter()
        return end - self.t_stop

    def buffer(self, frame):
        target = (self.solve_frames if self.state == "SOLVING"
                  else self.scan_frames if self.state == "SCANNING" else None)
        if target is None:
            return
        if len(self.solve_frames) + len(self.scan_frames) >= BUFFER_MAX_FRAMES:
            self.overflow = True
            return
        enc = _encode(frame)
        if enc is not None:
            target.append(enc)

    @property
    def buffered_mb(self):
        return sum(b.nbytes for b in self.solve_frames + self.scan_frames) / 1e6


def adjudicate_attempt(att, fps, detector, model, device, beam, lighting_ref):
    """Close the attempt: decode both windows, build evidence, adjudicate."""
    t0 = time.perf_counter()

    n_moves, labels, n_det = count_moves(
        att.solve_frames, fps, detector, model, device, beam)
    n_after, _, _ = count_moves(
        att.scan_frames, fps, detector, model, device, beam)

    # Lighting. Only `False` causes an abstention — `None` means "not
    # assessed" and is deliberately harmless, so a missing reference cannot
    # silently push every attempt into review.
    lighting_ok = None
    if lighting_ref is not None and att.solve_frames:
        try:
            from lighting_check import compare, frame_stats
            stats = frame_stats(np.asarray(att.solve_frames[::5]))
            worst = max((abs(d["z"]) for d in compare(stats, lighting_ref)),
                        default=0.0)
            lighting_ok = worst < 3.0
        except Exception:
            lighting_ok = None

    # Capture rate is a temporal scale factor on the count (see TRAIN_FPS).
    # Far enough off and the honest answer is that the count is unreadable —
    # so it goes in as `observed_moves=None`, which abstains, rather than
    # being reported as a number that would reject a legitimate solve.
    fps_off = abs(fps - TRAIN_FPS) / TRAIN_FPS
    fps_bad = fps_off > FPS_TOLERANCE

    rep = att.guard.report()
    ev = SolveEvidence(
        session=f"live_{time.strftime('%Y%m%d_%H%M%S')}",
        solve_seconds=att.solve_seconds,
        observed_moves=None if fps_bad else n_moves,
        observed_moves_after_stop=None if fps_bad else n_after,
        post_stop_seconds=att.scan_seconds,
        continuity=rep,
        # The substitution meter stays DISPLAY-ONLY here, exactly as it is in
        # live_guard_test: its threshold is not calibrated (swap_check.py's
        # baseline bounds the legit ceiling; no attack corpus bounds the
        # other side). Feeding an uncalibrated bar into a verdict is how you
        # ship a false DQ.
        swap_jump=None,
        swap_threshold=None,
        lighting_ok=lighting_ok,
    )
    verdict = adjudicate(ev)

    att.verdict = verdict
    att.detail = {
        "labels": labels,
        "moves_after_stop": n_after,
        "n_detected_crops": n_det,
        "solve_frames": len(att.solve_frames),
        "scan_frames": len(att.scan_frames),
        "analysis_seconds": round(time.perf_counter() - t0, 2),
        "capture_fps": round(fps, 1),
        # Recorded even when suppressed, so a run thrown out on capture rate
        # still shows what it WOULD have counted.
        "count_suppressed_by_fps": fps_bad,
        "raw_move_count": n_moves,
        "lighting_ok": lighting_ok,
        "swap_peak": round(att.swap.peak, 4),
        "swap_legit_ceiling": SWAP_REF,
        "buffer_overflowed": att.overflow,
    }
    return verdict


def _text(frame, s, org, color, scale, thick=1):
    """Draw HUD text with ONE putText call, at one thickness.

    THIS IS THE "STACKED TEXT" BUG, and it is not a layout problem — it is a
    property of OpenCV's Hershey fonts that is easy to trip:

        cv2.getTextSize("multi 0/3   gap 17.40s   det 0.00", ..., 0.48, 1)
            -> 193 px
        cv2.getTextSize(same string,                    ..., 0.48, 2)
            -> 207 px

    **Glyph advance depends on thickness.** So the common "draw a fat black
    outline, then the thin coloured fill on top" idiom — which
    `live_guard_test._put` does at thick+2 — renders the SAME string at two
    different widths. The two copies align at the first character and drift
    apart by a whole character by the end of the line, which reads exactly
    like the text has been stamped twice slightly offset.

    A single draw cannot drift. Contrast comes from `Panel`'s dark plate
    instead of from an outline, so nothing is lost.
    """
    cv2.putText(frame, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                thick, cv2.LINE_AA)


class Panel:
    """A block of HUD text on a translucent plate.

    Exists because the previous HUD wrote lines at hand-picked pixel offsets
    tuned for one 1280x720 camera. On any other resolution those offsets no
    longer matched the glyph height and lines landed on top of each other —
    the "stacked text" that made this unreadable.

    So: nothing here is an absolute pixel. Every size derives from the frame
    height, lines are appended rather than positioned, and the plate is sized
    to its contents after they are known. Long strings are truncated to the
    plate width, because a line that runs off the edge is what starts
    colliding with its neighbours.
    """

    def __init__(self, frame, x=None, y=None, width_frac=0.42):
        self.frame = frame
        self.fh, self.fw = frame.shape[:2]
        # Clamped so text on a 480p camera does not shrink into
        # illegibility; the plate just takes proportionally more width there.
        self.s = max(0.78, self.fh / 720.0)
        self.pad = int(11 * self.s)
        # Generous relative to glyph height on purpose: cramped leading is
        # half of why the old HUD looked like overlapping text.
        self.lh = int(24 * self.s)
        self.w = int(self.fw * width_frac)
        self.x = self.pad if x is None else x
        self.y0 = self.pad if y is None else y
        self.lines = []

    def line(self, text, color=(235, 235, 235), scale=0.52, bold=False):
        self.lines.append((text, color, scale * self.s, bold))
        return self

    def gap(self, n=1):
        for _ in range(n):
            self.lines.append((None, None, 0, False))
        return self

    def _fit(self, text, scale, bold):
        """Truncate to the plate width. cv2 has no wrapping, and an
        overflowing line is exactly what reads as overlapping text."""
        limit = self.w - 2 * self.pad
        thick = 2 if bold else 1
        if cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale,
                           thick)[0][0] <= limit:
            return text
        while text and cv2.getTextSize(text + "...", cv2.FONT_HERSHEY_SIMPLEX,
                                       scale, thick)[0][0] > limit:
            text = text[:-1]
        return text + "..."

    def draw(self):
        h = len(self.lines) * self.lh + 2 * self.pad
        y1, y2 = max(0, self.y0), min(self.fh, self.y0 + h)
        x1, x2 = max(0, self.x), min(self.fw, self.x + self.w)
        roi = self.frame[y1:y2, x1:x2]
        if roi.size:
            # Darken in place rather than compositing a full-frame overlay:
            # one whole-frame addWeighted per tick is real cost at 30fps and
            # only this rectangle needs it. 0.18 is dark enough that white
            # text is legible over ANY scene behind it, which is what lets
            # the glyph halo go away.
            cv2.addWeighted(roi, 0.18, np.zeros_like(roi), 0.82, 0, roi)

        y = self.y0 + self.pad + int(self.lh * 0.72)
        for text, color, scale, bold in self.lines:
            if text is not None:
                _text(self.frame, self._fit(text, scale, bold),
                      (self.x + self.pad, y), color, scale, 2 if bold else 1)
            y += self.lh
        return y2


def draw_live(frame, att, boxes, deduped, t, fps):
    """HUD for IDLE / SOLVING / SCANNING."""
    fh, fw = frame.shape[:2]
    rep = att.guard.report()
    s = max(0.78, fh / 720.0)

    mx, my = int(EDGE_MARGIN * fw), int(EDGE_MARGIN * fh)
    cv2.rectangle(frame, (mx, my), (fw - mx, fh - my), (90, 90, 90), 1)

    for x1, y1, x2, y2, c in boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (160, 160, 160), 1)
    confident = [b for b in deduped if b[4] >= MULTI_CONF]
    ncol = COL_BAD if len(confident) >= 2 else COL_OK
    for x1, y1, x2, y2, c in deduped:
        col = ncol if c >= MULTI_CONF else COL_WARN
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), col, 2)

    # fps is load-bearing twice over: the guard's security parameter AND the
    # count's temporal scale, so a bad rate is called out rather than buried.
    fps_bad = fps > 0 and abs(fps - TRAIN_FPS) / TRAIN_FPS > FPS_TOLERANCE
    gap = (t - att.guard.last_seen_t) if att.guard.last_seen_t is not None else t
    hits = len(att.guard.multi_hits)
    scol = {"IDLE": COL_MUTED, "SOLVING": COL_OK, "SCANNING": COL_WARN}[att.state]

    p = Panel(frame)
    if att.state == "SOLVING":
        p.line(f"SOLVING   {att.solve_seconds:5.2f}s", scol, 0.72, True)
    elif att.state == "SCANNING":
        p.line(f"SCANNING  scan {att.scan_seconds:4.1f}s", scol, 0.72, True)
    else:
        p.line("IDLE", scol, 0.72, True)
    p.line(f"{fps:.1f} fps" + ("   want ~30" if fps_bad else ""),
           COL_BAD if fps_bad else COL_MUTED, 0.48)

    p.gap()
    gcol = COL_BAD if rep["verdict"] == "dq" else COL_OK
    p.line(f"continuity  {rep['verdict'].upper()}", gcol, 0.58, True)
    for r in rep["dq_reasons"][:2]:
        p.line(f"  {r}", COL_BAD, 0.46)
    p.line(f"multi {hits}/{MULTI_DQ_HITS}   gap {gap:.2f}s   "
           f"det {rep['detection_rate']:.2f}",
           COL_BAD if (hits or gap > GAP_FLAG_S) else (215, 215, 215), 0.48)
    ref_s = f"vs {SWAP_REF:.2f}" if SWAP_REF else "uncal."
    p.line(f"swap {att.swap.peak:.3f} {ref_s} - display only", COL_MUTED, 0.44)

    p.gap()
    p.line(f"count gate   floor {MIN_OBSERVED_MOVES}", COL_MUTED, 0.58, True)
    if att.state == "IDLE":
        p.line(f"abstains above {separation_tps_limit():.2f} TPS", COL_MUTED, 0.46)
        p.line("counts when the window closes", COL_MUTED, 0.44)
    elif att.state == "SOLVING":
        p.line(f"buffering {len(att.solve_frames)} fr "
               f"({att.buffered_mb:.0f} MB)", (215, 215, 215), 0.48)
        p.line("counts when the window closes", COL_MUTED, 0.44)
    else:
        secs = att.scan_seconds or 0.0
        over = secs > POST_STOP_MAX_WINDOW_S
        p.line(f"post-stop allowance {post_stop_limit(secs)} moves",
               COL_BAD if over else (215, 215, 215), 0.48)
        p.line(f"window {secs:.0f}s / cap {POST_STOP_MAX_WINDOW_S:.0f}s",
               COL_BAD if over else COL_MUTED, 0.44)
        if over:
            p.line("past the cap - goes to review", COL_BAD, 0.44)

    if fps_bad:
        p.gap()
        p.line("count will ABSTAIN on capture rate", COL_BAD, 0.46)
    if att.overflow:
        p.gap()
        p.line("BUFFER FULL - frames dropped", COL_BAD, 0.46)
    p.draw()

    # Controls on their own plate pinned to the bottom, so they can never
    # collide with the stack above however long it grows.
    c = Panel(frame, y=fh - int(58 * s), width_frac=0.40)
    c.line(STATE_HELP[att.state], (255, 255, 255), 0.5, True)
    c.line("R reset   S save   Q quit", COL_MUTED, 0.44)
    c.draw()


def draw_verdict(frame, att):
    """HUD for VERDICT — the adjudicate() output, in full.

    Two columns: WHY on the left, the numbers behind it on the right. One
    column of twenty lines was most of what made this unreadable.
    """
    fh, fw = frame.shape[:2]
    v, d = att.verdict, att.detail
    col = VERDICT_COLOR[v["verdict"]]
    s = max(0.78, fh / 720.0)

    cv2.addWeighted(frame, 0.22, np.zeros_like(frame), 0.78, 0, frame)

    tps = v.get("tps")
    scan_s = att.scan_seconds or 0.0
    allowance = post_stop_limit(scan_s)
    after = d["moves_after_stop"]

    left = Panel(frame, width_frac=0.47)
    left.line(v["verdict"].upper(), col, 1.05, True)
    left.gap()
    if v["reject_reasons"]:
        left.line("rejected because", COL_BAD, 0.55, True)
        for r in v["reject_reasons"]:
            left.line(f"  {r}", COL_BAD, 0.45)
        left.gap()
    if v["review_reasons"]:
        left.line("sent to review because", COL_WARN, 0.55, True)
        for r in v["review_reasons"]:
            left.line(f"  {r}", COL_WARN, 0.45)
        left.gap()
    if not v["reject_reasons"] and not v["review_reasons"]:
        left.line("every test passed", COL_OK, 0.55, True)
        left.gap()
    for note in v.get("caveats", []):
        left.line("caveat", COL_WARN, 0.45, True)
        left.line(f"  {note}", COL_WARN, 0.4)
    left.draw()

    right = Panel(frame, x=int(fw * 0.51), width_frac=0.47)
    right.line("evidence", COL_MUTED, 0.55, True)
    right.line(f"moves in window  {v['observed_moves']}  "
               f"(floor {v['floor']})", (235, 235, 235), 0.5)
    if "headroom_moves" in v:
        right.line(f"headroom         {v['headroom_moves']:+d}",
                   COL_OK if v["headroom_moves"] >= 0 else COL_BAD, 0.5)
    right.line(f"solve time       {v['solve_seconds']:.2f}s"
               + (f"   {tps:.2f} TPS" if tps else ""), (235, 235, 235), 0.5)
    if tps:
        right.line(f"abstain above    {separation_tps_limit():.2f} TPS",
                   COL_MUTED, 0.45)
    if "retention_floor_at_tps" in v:
        right.line(f"retention here   {v['retention_floor_at_tps']:.3f} "
                   f"-> floor {v['implied_true_floor']}", COL_MUTED, 0.43)
    right.gap()
    right.line(f"after stop       {after}  (allow {allowance})",
               COL_BAD if after > allowance else (235, 235, 235), 0.5)
    right.line(f"post-stop window {scan_s:.1f}s", COL_MUTED, 0.45)
    right.gap()
    right.line(f"continuity       {att.guard.report()['verdict'].upper()}",
               (235, 235, 235), 0.5)
    right.line(f"swap peak        {d['swap_peak']:.3f} (display only)",
               COL_MUTED, 0.45)
    right.gap()
    right.line(f"crops detected   {d['n_detected_crops']}/{d['solve_frames']}",
               COL_MUTED if d["n_detected_crops"] else COL_BAD, 0.45)
    right.line(f"capture rate     {d['capture_fps']:.1f} fps "
               f"(trained {TRAIN_FPS:.0f})",
               COL_BAD if d["count_suppressed_by_fps"] else COL_MUTED, 0.45)
    if d["count_suppressed_by_fps"]:
        right.line(f"  suppressed; would read {d['raw_move_count']}",
                   COL_BAD, 0.43)
    right.line(f"analysis took    {d['analysis_seconds']:.1f}s", COL_MUTED, 0.43)
    right.draw()

    c = Panel(frame, y=fh - int(38 * s), width_frac=0.40)
    c.line(STATE_HELP["VERDICT"], (255, 255, 255), 0.5, True)
    c.draw()


def open_camera(index):
    """DirectShow + MJPG first: Windows webcams streaming raw YUY2 at 720p
    are often capped near 10 fps by USB bandwidth, and fps is the guard's
    security parameter (continuity_guard CADENCE)."""
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        ok, frame = cap.read()
        if ok:
            return cap, frame
        cap.release()
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise SystemExit(f"ERROR: could not open webcam {index}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit("ERROR: webcam opened but returned no frame")
    return cap, frame


def main():
    ap = argparse.ArgumentParser(
        description="Live end-to-end anticheat: continuity guard + move-count "
                    "gate + post-stop test, adjudicated.")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--ctc", default=DEFAULT_CTC,
                    help=f"CTC checkpoint (default {DEFAULT_CTC})")
    ap.add_argument("--beam", type=int, default=0,
                    help="prefix-beam width; 0 = greedy (what the corpus "
                         "numbers were measured with)")
    args = ap.parse_args()

    print("[anticheat] loading the cube detector...")
    detector = load_detector()
    if detector is None:
        print("[anticheat]   none available (ultralytics or "
              "detect_full_cube.pt missing)")

    print(f"[anticheat] loading {args.ctc} ...")
    ck, notes = preflight(args.ctc, detector)
    _, model, device = _load_model(args.ctc)
    print(f"[anticheat]   {ck.get('model_type')} on {device}, "
          f"crop_regime={ck.get('crop_regime')}, "
          f"speed_aug={ck.get('speed_aug')}, val_mer={ck.get('val_mer'):.4f}")
    for n in notes:
        print("\n" + n)
    if notes:
        print()

    try:
        from lighting_check import load_reference
        lighting_ref = load_reference()
    except Exception:
        lighting_ref = None
    if lighting_ref is None:
        print("[anticheat] no lighting reference — lighting will not be "
              "assessed (never abstains on it)")

    cap, frame = open_camera(args.camera)
    fh, fw = frame.shape[:2]
    print(f"[anticheat] camera {args.camera} @ {fw}x{fh}")

    # Warm the YOLO up BEFORE the timing baseline. The first detect_cubes
    # loads weights (seconds); inside the loop that looks like a multi-second
    # inter-frame stall and can false-DQ the cadence check.
    print("[anticheat] warming up the detector...")
    detect_cubes(frame, conf_floor=PRESENCE_CONF)

    att = Attempt(fw, fh)
    t0 = time.perf_counter()
    fps, last = 0.0, time.perf_counter()
    print(f"\n[anticheat] ready. floor={MIN_OBSERVED_MOVES} moves, "
          f"abstain above {separation_tps_limit():.2f} TPS. "
          f"SPACE to start.\n")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = time.perf_counter() - t0

            if att.state != "VERDICT":
                boxes = detect_cubes(frame, conf_floor=PRESENCE_CONF)
                deduped = dedup_boxes(boxes)
                sig = (box_signature(frame, max(deduped, key=lambda b: b[4]))
                       if deduped else None)
                att.guard.update(t, boxes, sig)
                att.swap.update(t, sig)
                att.buffer(frame)

                now = time.perf_counter()
                dt = max(1e-6, now - last)
                fps = (0.9 * fps + 0.1 / dt) if fps else 1 / dt
                last = now
                draw_live(frame, att, boxes, deduped, t, fps)
            else:
                draw_verdict(frame, att)

            cv2.imshow("anticheat — full stack, try to break it", frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q"), 27):
                break

            elif key == ord(" "):
                if att.state == "IDLE":
                    att.state = "SOLVING"
                    att.t_start = time.perf_counter()
                    print("[anticheat] solve started")
                elif att.state == "SOLVING":
                    att.t_stop = time.perf_counter()
                    att.state = "SCANNING"
                    print(f"[anticheat] timer stopped at "
                          f"{att.solve_seconds:.2f}s — now scan the cube")
                elif att.state == "SCANNING":
                    att.t_scan_end = time.perf_counter()
                    att.state = "VERDICT"
                    print("[anticheat] decoding... (this is not real-time; "
                          "the count needs the whole window)")
                    v = adjudicate_attempt(att, fps or 30.0, detector, model,
                                           device, args.beam, lighting_ref)
                    print(json.dumps(v, indent=2))
                    print(f"  detail: {att.detail}")

            elif key in (ord("r"), ord("R")):
                att = Attempt(fw, fh)
                t0 = time.perf_counter()
                print("\n[anticheat] reset — new attempt\n")

            elif key in (ord("s"), ord("S")) and att.verdict:
                name = f"anticheat_live_{time.strftime('%Y%m%d_%H%M%S')}.json"
                bundle = {
                    "checkpoint": str(args.ctc),
                    "speed_aug": ck.get("speed_aug"),
                    "crop_regime": ck.get("crop_regime"),
                    "beam": args.beam,
                    "verdict": att.verdict,
                    "detail": att.detail,
                    "continuity": att.guard.report(),
                }
                Path(name).write_text(json.dumps(bundle, indent=2))
                print(f"[anticheat] saved {name}")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
