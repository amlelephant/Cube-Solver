"""
dump_trajectories.py — persist per-frame detection boxes for every recorded
session, so trajectory modelling never needs another full YOLO pass.

Why this file exists: no existing artifact carries a dense per-frame box
stream. `detector_stream_color.npz` holds 96x96 CROPS but not the boxes
they came from; `crops.json` is one box per MOVE; and prepare_data.py's
per_frame_boxes() detects only every CROP_STRIDE=10 frames (~3Hz) and
median-smooths, which is far too sparse for anticheat — a swap can complete
inside GAP_FLAG_S=0.18s (~5 frames at 30fps). The continuity guard runs YOLO
on EVERY frame, so that is what gets dumped here.

Output, one file per session: <session>/trajectory.npz
    frame_idx (N,) int32     frame index, repeated once per box in that frame
    t         (N,) float32   seconds since session start
    boxes     (N,5) float32  x1, y1, x2, y2, conf
    n_frames  ()   int32     total frames analysed (including empty ones)
    fw, fh    ()   int32     frame dimensions
    name      ()   str

Ragged by construction: a frame with no detection contributes no row, a
frame with three boxes contributes three. Frames with no rows are exactly
the presence gaps, recoverable by set-difference against range(n_frames).

Storage is trivial (~2MB across all 69 sessions) and it is written next to
the session's other derived artifacts, matching the detector_stream.npz
convention.

Run from inside cv/detection (bare model filenames — see CLAUDE.md):
    python dump_trajectories.py --root ../../ble/training_data
    python dump_trajectories.py --root ../../ble/training_data --force
"""

import argparse
import json
import os
import time

import cv2
import numpy as np

from continuity_guard import detect_cubes, PRESENCE_CONF

OUT_NAME = "trajectory.npz"


def load_manifest(session_dir):
    manifest = os.path.join(session_dir, "frames.jsonl")
    frames_dir = os.path.join(session_dir, "frames")
    if not (os.path.isfile(manifest) and os.path.isdir(frames_dir)):
        return None
    entries = []
    with open(manifest) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            path = os.path.join(frames_dir, rec["file"])
            if os.path.isfile(path):
                entries.append((rec["ts"], path))
    entries.sort(key=lambda e: e[0])
    return entries


def dump_session(session_dir, name):
    entries = load_manifest(session_dir)
    if not entries:
        return None
    first = cv2.imread(entries[0][1])
    if first is None:
        return None
    fh, fw = first.shape[:2]
    t0 = entries[0][0]

    frame_idx, ts, boxes = [], [], []
    for i, (stamp, path) in enumerate(entries):
        frame = cv2.imread(path)
        if frame is None:
            continue
        # PRESENCE_CONF, not the stricter scan floor: the guard's presence
        # check uses the loose floor, and a trajectory model wants to see
        # the weak detections too (they are what bridge occlusions).
        for b in detect_cubes(frame, conf_floor=PRESENCE_CONF):
            frame_idx.append(i)
            ts.append(stamp - t0)
            boxes.append(b[:5])

    out = os.path.join(session_dir, OUT_NAME)
    np.savez_compressed(
        out,
        frame_idx=np.asarray(frame_idx, dtype=np.int32),
        t=np.asarray(ts, dtype=np.float32),
        boxes=np.asarray(boxes, dtype=np.float32).reshape(-1, 5),
        n_frames=np.int32(len(entries)),
        fw=np.int32(fw), fh=np.int32(fh),
        name=str(name),
    )
    return len(entries), len(boxes), out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True)
    ap.add_argument("--force", action="store_true",
                    help="re-dump sessions that already have a trajectory.npz")
    args = ap.parse_args()

    sessions = sorted(
        d for d in os.listdir(args.root)
        if os.path.isfile(os.path.join(args.root, d, "frames.jsonl"))
        and os.path.isdir(os.path.join(args.root, d, "frames"))
    )
    if not sessions:
        raise SystemExit(f"no sessions with frames/ + frames.jsonl under {args.root}")

    t_start = time.time()
    n_done = n_skip = 0
    for i, name in enumerate(sessions):
        session_dir = os.path.join(args.root, name)
        if not args.force and os.path.isfile(os.path.join(session_dir, OUT_NAME)):
            print(f"[{i + 1}/{len(sessions)}] {name}: exists, skipping")
            n_skip += 1
            continue
        t0 = time.time()
        res = dump_session(session_dir, name)
        if res is None:
            print(f"[{i + 1}/{len(sessions)}] {name}: SKIP (no loadable frames)")
            continue
        n_frames, n_boxes, _ = res
        n_done += 1
        print(f"[{i + 1}/{len(sessions)}] {name}: {n_frames} frames, "
              f"{n_boxes} boxes ({time.time() - t0:.0f}s)")

    print(f"\n{n_done} dumped, {n_skip} skipped, {time.time() - t_start:.0f}s total")


if __name__ == "__main__":
    main()
