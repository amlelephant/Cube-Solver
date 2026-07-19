"""
live_guard_test.py — interactive webcam stress-test for the continuity guard.

Puts the whole anti-swap stack on screen in real time so you can try to
break it: swap cubes fast, hide the cube, present two cubes, rest the cube
corner-on (two faces visible), wave it left-right, let the keyboard glow.

What you see:
  * thin gray boxes  — raw YOLO detections (conf >= PRESENCE_CONF)
  * thick boxes      — after dedup/merge (what the guard actually counts);
                       green = one cube, red = counted as multiple cubes,
                       a "MERGED" tag means adjacent boxes (e.g. a face
                       split) were folded into one
  * HUD              — live verdict, DQ reasons, multi-hit meter (n/flag/DQ
                       thresholds), current detection gap vs the flag/DQ
                       limits, and a ticker of the most recent guard events

The guard also fingerprints the cube's appearance (HSV histogram of the
detected box) every frame; after any flagged gap the reappearing cube is
compared against the one that vanished — coming back looking different is
a hard DQ (dq_appearance_change; interior gaps get a laxer bar so motion
blur stays a note), and looking different AND somewhere else across a
border gap is the swap fingerprint (swap_reappearance).

Verdict colors: PASS green, DQ red (two-tier — no review). A DQ is sticky
(as it would be in a real solve) — press R to reset and try again.

Run from inside cv/detection (bare model filenames — see CLAUDE.md):
    python live_guard_test.py [--camera 0]

Keys:
    R      reset the guard (new attempt; prints the finished report)
    S      save the current report to guard_report_<timestamp>.json
    Q/ESC  quit (prints the final report)
"""

import argparse
import json
import time

import cv2

from continuity_guard import (ContinuityGuard, box_signature, dedup_boxes,
                              detect_cubes, EDGE_MARGIN, FRAME_STALL_DQ_S,
                              GAP_DQ_S, GAP_FLAG_S, MULTI_CONF, MULTI_DQ_HITS,
                              PRESENCE_CONF)

VERDICT_COLOR = {          # BGR
    "pass": (60, 200, 60),
    "dq":   (50, 50, 230),
}
RAW_COLOR    = (160, 160, 160)
ONE_COLOR    = (60, 200, 60)
MULTI_COLOR  = (50, 50, 230)
WEAK_COLOR   = (0, 170, 230)   # deduped box below MULTI_CONF (presence only)
TICKER_LINES = 4


def _put(frame, text, org, color=(255, 255, 255), scale=0.55, thick=1):
    # black underlay so the HUD stays readable over any background
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, thick, cv2.LINE_AA)


def draw_hud(frame, guard, rep, boxes, deduped, t, fps):
    fh, fw = frame.shape[:2]

    # border zone: gaps/jumps touching the outside of this line are
    # DQ/flag-grade (the swap route); inside it they are benign notes
    mx, my = int(EDGE_MARGIN * fw), int(EDGE_MARGIN * fh)
    cv2.rectangle(frame, (mx, my), (fw - mx, fh - my), (90, 90, 90), 1)
    _put(frame, "border zone: gaps out here DQ", (mx + 6, fh - my - 8),
         (140, 140, 140), 0.45)

    for x1, y1, x2, y2, c in boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), RAW_COLOR, 1)

    confident = [b for b in deduped if b[4] >= MULTI_CONF]
    n_color = MULTI_COLOR if len(confident) >= 2 else ONE_COLOR
    for x1, y1, x2, y2, c in deduped:
        color = n_color if c >= MULTI_CONF else WEAK_COLOR
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        _put(frame, f"{c:.2f}", (int(x1), int(y1) - 8), color)
    if len(deduped) < len(boxes) and deduped:
        x1, y1 = int(deduped[0][0]), int(deduped[0][1])
        _put(frame, f"MERGED {len(boxes)}->{len(deduped)}",
             (x1, y1 - 28), (255, 200, 0), 0.55, 2)

    verdict = rep["verdict"]
    vcol = VERDICT_COLOR[verdict]
    cv2.rectangle(frame, (0, 0), (fw, 34), (0, 0, 0), -1)
    _put(frame, f"VERDICT: {verdict.upper()}", (10, 24), vcol, 0.8, 2)
    if rep["dq_reasons"]:
        _put(frame, " / ".join(rep["dq_reasons"]), (280, 24), vcol, 0.6, 2)
    _put(frame, f"{fps:4.1f} fps", (fw - 110, 24), (255, 255, 255))

    hits = len(guard.multi_hits)
    gap = (t - guard.last_seen_t) if guard.last_seen_t is not None else t
    y = 58
    _put(frame, f"multi-cube hits in window: {hits}  "
                f"(DQ at {MULTI_DQ_HITS})",
         (10, y), MULTI_COLOR if hits else (255, 255, 255)); y += 22
    _put(frame, f"detection gap: {gap:4.2f}s  (border-DQ at {GAP_FLAG_S:.2f}s, "
                f"presence-DQ at {GAP_DQ_S:.1f}s)",
         (10, y), MULTI_COLOR if gap > GAP_FLAG_S else (255, 255, 255)); y += 22
    _put(frame, f"notes: {rep['notes']}   detection rate: "
                f"{rep['detection_rate']:.2f}   max gap: {rep['max_gap_s']:.2f}s",
         (10, y)); y += 22
    # cadence: at healthy fps the frame interval is ~0.05s; a slow model
    # opens a swap-sized blind window between frames → DQ past the cutoff.
    frame_dt = 1.0 / fps if fps > 0 else 0.0
    stalling = frame_dt >= FRAME_STALL_DQ_S
    _put(frame, f"frame gap: {frame_dt:4.2f}s  max {rep['max_frame_gap_s']:.2f}s "
                f"(DQ {FRAME_STALL_DQ_S:.2f}s)",
         (10, y), MULTI_COLOR if stalling else (255, 255, 255)); y += 22

    for e in rep["events"][-TICKER_LINES:]:
        detail = {k: v for k, v in e.items() if k not in ("t", "type")}
        _put(frame, f"[{e['t']:7.2f}s] {e['type']} {detail if detail else ''}",
             (10, y), VERDICT_COLOR["dq"] if e["type"].startswith("dq_")
             else (200, 200, 200), 0.5); y += 20

    _put(frame, "R reset   S save report   Q quit", (10, fh - 12),
         (200, 200, 200), 0.5)


def main():
    ap = argparse.ArgumentParser(
        description="Live webcam stress-test for the continuity guard")
    ap.add_argument("--camera", type=int, default=0, help="webcam index")
    args = ap.parse_args()

    # DirectShow + MJPG: Windows webcams streaming raw YUY2 at 720p are
    # often capped near 10 fps by USB bandwidth — cap.read() then throttles
    # the whole loop no matter how fast inference is. MJPG lifts that cap;
    # fps is the guard's security parameter (see continuity_guard CADENCE).
    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        ok, frame = cap.read()
        if not ok:
            cap.release()
            cap = None
    else:
        cap = None
    if cap is None:
        # fall back to the default backend without forcing a codec
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            raise SystemExit(f"ERROR: could not open webcam {args.camera}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        ok, frame = cap.read()
        if not ok:
            raise SystemExit("ERROR: webcam opened but returned no frame")
    fh, fw = frame.shape[:2]
    print(f"[guard-test] camera {args.camera} @ {fw}x{fh}")
    print("[guard-test] first inference loads the model — one-time delay...")
    # Warm the model up BEFORE the timing baseline: the first detect_cubes
    # loads weights (~seconds). If that landed inside the loop it would look
    # like a multi-second inter-frame stall and could false-DQ on the
    # cadence check (see continuity_guard §4 CADENCE).
    detect_cubes(frame, conf_floor=PRESENCE_CONF)

    guard = ContinuityGuard(fw, fh)
    t0 = time.perf_counter()
    fps, last = 0.0, time.perf_counter()

    def finish(g):
        rep = g.report()
        print(json.dumps(rep, indent=2))
        return rep

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = time.perf_counter() - t0

            boxes = detect_cubes(frame, conf_floor=PRESENCE_CONF)
            deduped = dedup_boxes(boxes)
            sig = (box_signature(frame, max(deduped, key=lambda b: b[4]))
                   if deduped else None)
            guard.update(t, boxes, sig)
            rep = guard.report()

            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 / max(1e-6, now - last) if fps else 1 / max(1e-6, now - last)
            last = now

            draw_hud(frame, guard, rep, boxes, deduped, t, fps)
            cv2.imshow("continuity guard — try to break it", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            elif key in (ord("r"), ord("R")):
                print("\n[guard-test] === attempt finished — report: ===")
                finish(guard)
                guard = ContinuityGuard(fw, fh)
                t0 = time.perf_counter()
                print("[guard-test] guard reset — new attempt\n")
            elif key in (ord("s"), ord("S")):
                name = f"guard_report_{time.strftime('%Y%m%d_%H%M%S')}.json"
                with open(name, "w") as f:
                    json.dump(rep, f, indent=2)
                print(f"[guard-test] saved {name}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\n[guard-test] === final report: ===")
        finish(guard)


if __name__ == "__main__":
    main()
