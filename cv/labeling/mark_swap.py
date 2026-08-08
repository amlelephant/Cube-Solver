"""
mark_swap.py — mark the swap moment in a recorded attack session AFTER the
fact, with the candidate moments proposed for you.

Why this exists: pressing a key at the instant of the swap is impossible.
Every exploit in the threat model is an unsolved cube being switched for a
solved one, which takes both hands — there is no hand free for the
keyboard, and a real-time keypress would carry ~200-500ms of human reaction
lag anyway. Marking afterwards is both possible and more accurate.

Model-assisted, same shape as autolabel.py: you never hunt frame-by-frame
unless you want to. Because the swap is always scrambled -> solved, the
tracked box's colour histogram changes sharply at the switch, so the
largest appearance discontinuities are proposed as candidates and you
confirm or adjust.

NOT CIRCULAR for the trajectory model: trajectory_anomaly.py scores
KINEMATICS only (speed, acceleration, straightness) and never looks at
appearance, so using appearance to establish ground truth is independent of
what is being validated. It WOULD be circular for validating the guard's
own dq_appearance_change rule — marks produced here must not be used to
claim that rule detects swaps.

Prerequisite (one command — the proposals need the tracked box):
    cd ../detection && python dump_trajectories.py --root ../labeling/attack_sessions

Run from inside cv/labeling/:
    python mark_swap.py --session attack_sessions/table_edge_20260804_120000

Keys
----
  n / p     jump to next / previous proposed candidate
  d / a     step forward / back 1 frame
  w / q     step forward / back 10 frames
  SPACE     toggle a mark at the current frame
  s         save marks into attack.json and exit
  Esc       exit without saving
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "detection"))

from continuity_guard import box_signature, dedup_boxes, pick_continuity, _sig_dist

SMOOTH = 2        # compare signatures this many frames apart, to step over
                  #   single-frame detector jitter without blurring the swap
N_CANDIDATES = 6  # proposed swap moments


def load_tracked(session):
    """(frame_paths, times, boxes) with one tracked box per frame (None if absent)."""
    tpath = session / "trajectory.npz"
    if not tpath.is_file():
        raise SystemExit(
            f"{tpath} not found.\nRun first:  cd ../detection && "
            f"python dump_trajectories.py --root {session.parent.as_posix()}")
    d = np.load(tpath, allow_pickle=True)
    n_frames = int(d["n_frames"])

    by_frame = {}
    for fi, t, b in zip(d["frame_idx"], d["t"], d["boxes"]):
        by_frame.setdefault(int(fi), {"t": float(t), "boxes": []})["boxes"].append(tuple(b))

    entries = []
    with open(session / "frames.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    entries.sort(key=lambda r: r["ts"])
    paths = [session / "frames" / r["file"] for r in entries][:n_frames]

    boxes, times, ref = [], [], None
    for i in range(len(paths)):
        rec = by_frame.get(i)
        if rec is None:
            boxes.append(None)
            times.append(times[-1] if times else 0.0)
            continue
        dd = dedup_boxes(rec["boxes"])
        if not dd:
            boxes.append(None)
            times.append(rec["t"])
            continue
        best = pick_continuity(dd, ref)
        ref = best
        boxes.append(best)
        times.append(rec["t"])
    return paths, times, boxes


def propose(paths, boxes):
    """Frames with the largest appearance discontinuity in the tracked box."""
    sigs = [None] * len(paths)
    print(f"scanning {len(paths)} frames for appearance changes...")
    for i, (p, b) in enumerate(zip(paths, boxes)):
        if b is None:
            continue
        frame = cv2.imread(str(p))
        if frame is not None:
            sigs[i] = box_signature(frame, b)
        if i % 200 == 0:
            print(f"  {i}/{len(paths)}", end="\r")
    print(" " * 40, end="\r")

    dist = np.zeros(len(paths))
    for i in range(SMOOTH, len(paths)):
        a, b = sigs[i - SMOOTH], sigs[i]
        if a is not None and b is not None:
            dist[i] = _sig_dist(a, b)

    # non-maximum suppression so one swap yields one candidate, not a cluster
    cands, taken = [], np.zeros(len(paths), dtype=bool)
    for i in np.argsort(-dist):
        if dist[i] <= 0 or taken[i]:
            continue
        cands.append(int(i))
        taken[max(0, i - 15):i + 15] = True
        if len(cands) >= N_CANDIDATES:
            break
    return sorted(cands), dist


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", required=True)
    args = ap.parse_args()
    session = Path(args.session)

    meta_path = session / "attack.json"
    meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    paths, times, boxes = load_tracked(session)
    if not paths:
        raise SystemExit("no frames")
    cands, dist = propose(paths, boxes)
    t0 = times[0] if times else 0.0

    marks = {int(m["frame_idx"]) for m in (meta.get("marks") or [])}
    print(f"\n{len(cands)} candidates proposed: "
          + ", ".join(f"{c} (t={times[c] - t0:.1f}s)" for c in cands))
    print("n/p candidates  d/a +-1  w/q +-10  SPACE mark  s save  Esc quit\n")

    i = cands[0] if cands else 0
    ci = 0
    while True:
        frame = cv2.imread(str(paths[i]))
        if frame is None:
            frame = np.zeros((720, 1280, 3), np.uint8)
        view = frame.copy()
        if boxes[i] is not None:
            x1, y1, x2, y2 = (int(v) for v in boxes[i][:4])
            cv2.rectangle(view, (x1, y1), (x2, y2), (0, 220, 60), 2)
        view = cv2.resize(view, (960, 540))

        marked = i in marks
        cv2.rectangle(view, (0, 0), (960, 30), (0, 0, 0), -1)
        cv2.putText(view, f"frame {i}/{len(paths) - 1}  t={times[i] - t0:6.2f}s"
                          f"  change={dist[i]:.3f}",
                    (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        if marked:
            cv2.putText(view, "MARKED", (700, 21), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 220, 255), 2)
        cv2.rectangle(view, (0, 510), (960, 540), (0, 0, 0), -1)
        cv2.putText(view, f"marks: {sorted(marks)}   n/p cand  d/a +-1  "
                          f"w/q +-10  SPACE mark  s save",
                    (8, 530), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        # candidate ticks along the bottom
        for c in cands:
            x = int(960 * c / max(len(paths) - 1, 1))
            cv2.line(view, (x, 496), (x, 508), (0, 170, 255), 2)
        for mk in marks:
            x = int(960 * mk / max(len(paths) - 1, 1))
            cv2.line(view, (x, 484), (x, 508), (0, 220, 255), 2)

        cv2.imshow("mark_swap", view)
        k = cv2.waitKey(0) & 0xFF
        if k == 27:
            print("exited without saving")
            break
        elif k == ord("s"):
            meta.setdefault("attack_type", session.name.rsplit("_", 2)[0])
            meta["marks"] = [{"t": round(times[m] - t0, 3), "frame_idx": int(m)}
                             for m in sorted(marks)]
            meta["marked_by"] = "mark_swap.py"
            meta_path.write_text(json.dumps(meta, indent=2))
            print(f"saved {len(marks)} mark(s) -> {meta_path}")
            break
        elif k == ord(" "):
            marks.symmetric_difference_update({i})
        elif k == ord("d"):
            i = min(i + 1, len(paths) - 1)
        elif k == ord("a"):
            i = max(i - 1, 0)
        elif k == ord("w"):
            i = min(i + 10, len(paths) - 1)
        elif k == ord("q"):
            i = max(i - 10, 0)
        elif k == ord("n") and cands:
            ci = (ci + 1) % len(cands)
            i = cands[ci]
        elif k == ord("p") and cands:
            ci = (ci - 1) % len(cands)
            i = cands[ci]
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
