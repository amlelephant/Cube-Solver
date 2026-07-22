"""
postprocess_session.py

Aligns BLE move timestamps to webcam frame timestamps after recording.
Run this after record_training.py finishes a session.

Usage:
    python postprocess_session.py --session training_data/solve_20260708_173218
    python postprocess_session.py --session training_data/solve_20260708_173218 --dry-run

Output added to the session folder:
    labeled/
        move_0001_before.jpg     STILL  (T - 150ms)
        move_0001_mid_00.jpg     MOVING (T + 30ms)
        move_0001_mid_01.jpg     MOVING (T + 60ms)
        move_0001_mid_02.jpg     MOVING (T + 100ms)
        move_0001_after.jpg      STILL  (T + 250ms)
    moves_labeled.jsonl
    metadata.json

The raw frames/ directory is KEPT by default. Pass --discard-frames to
delete it once labeled/ is written, if you are short on disk.

This default was the other way around until 2026-07-22, and it cost the
eleven 2026-07-20 sessions. labeled/ keeps five frames per move, all
within -150ms..+250ms of the BLE timestamp — so it is 100% positives with
no inter-move negatives and no rotation footage. That is enough to train
the move CLASSIFIER, which only ever sees a centred window, but it cannot
train the onset DETECTOR, which learns mainly from what is NOT a turn.
Deleting frames/ silently converts a session into one that can never be
used for detection, and the loss is not recoverable by re-postprocessing.

Keeping frames costs ~10-20x the disk of labeled/. That is the cheaper
mistake.

Window offsets above are the MAXIMUMS, used when a move has room around
it. During fast sequences (algorithms) consecutive BLE events arrive
closer together than the window span — measured median inter-move gap in
real sessions is ~180-210ms vs. the 400ms fixed window — so a fixed
window bleeds the neighboring moves' motion into 'before'/'after' frames
that are labeled STILL. Each move's window is therefore sandwiched to at
most half the gap to its temporal neighbors (never crossing the
midpoint), with the mid offsets compressed proportionally: a fast turn's
peak motion sits proportionally closer to its BLE timestamp.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

# MAXIMUM time offsets relative to BLE event timestamp T — per-move
# windows are shrunk from these when neighboring moves are close (see
# module docstring).
WINDOWS = {
    "before": -0.150,
    "mid_00": +0.030,
    "mid_01": +0.060,
    "mid_02": +0.100,
    "after":  +0.250,
}

BASE_PRE   = -WINDOWS["before"]   # max span before T
BASE_POST  = WINDOWS["after"]     # max span after T
MIN_SPAN   = 0.033                # one frame at 30fps — never shrink below
HALF_FRAME = 0.017                # matching tolerance beyond the sandwich


def move_window(gap_prev: float | None, gap_next: float | None) -> dict:
    """
    Per-move window offsets, sandwiched to half the gap to each
    neighboring move (None = no neighbor on that side).
    """
    pre  = BASE_PRE  if gap_prev is None else \
        min(BASE_PRE,  max(MIN_SPAN, 0.5 * gap_prev))
    post = BASE_POST if gap_next is None else \
        min(BASE_POST, max(MIN_SPAN, 0.5 * gap_next))

    scale = post / BASE_POST
    return {
        "before": -pre,
        "mid_00": WINDOWS["mid_00"] * scale,
        "mid_01": WINDOWS["mid_01"] * scale,
        "mid_02": WINDOWS["mid_02"] * scale,
        "after":  post,
    }

MOTION_LABELS = {
    "before": "STILL",
    "mid_00": "MOVING",
    "mid_01": "MOVING",
    "mid_02": "MOVING",
    "after":  "STILL",
}


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def find_nearest(frame_times: list[float], frame_recs: list[dict],
                 target: float, max_gap: float = 0.5) -> dict | None:
    if not frame_times:
        return None
    lo, hi = 0, len(frame_times) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if frame_times[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    best = lo
    if lo > 0 and abs(frame_times[lo-1]-target) < abs(frame_times[lo]-target):
        best = lo - 1
    return frame_recs[best] if abs(frame_times[best]-target) <= max_gap else None


def postprocess(session_dir: Path, dry_run: bool = False,
                keep_frames: bool = True):
    # Check required files exist
    for req in ["moves.jsonl", "frames.jsonl", "config.json"]:
        if not (session_dir / req).exists():
            print(f"ERROR: {req} not found in {session_dir}")
            print("Run record_training.py first.")
            sys.exit(1)

    config     = json.loads((session_dir / "config.json").read_text())
    ble_meta   = {}
    if (session_dir / "ble_meta.json").exists():
        ble_meta = json.loads((session_dir / "ble_meta.json").read_text())

    moves      = load_jsonl(session_dir / "moves.jsonl")
    frame_recs = load_jsonl(session_dir / "frames.jsonl")
    frames_dir = session_dir / "frames"
    labeled_dir= session_dir / "labeled"

    frame_times = [r["ts"] for r in frame_recs]

    print(f"\n  Session:  {session_dir.name}")
    print(f"  Moves:    {len(moves)}")
    print(f"  Frames:   {len(frame_recs)}")
    if len(frame_times) >= 2:
        dur = frame_times[-1] - frame_times[0]
        fps = len(frame_recs) / dur if dur > 0 else 0
        print(f"  Duration: {dur:.1f}s  ({fps:.1f} fps avg)")
    print()

    if not dry_run:
        labeled_dir.mkdir(exist_ok=True)

    labeled_moves  = []
    missing        = 0
    total_frames   = 0
    still_count    = 0
    moving_count   = 0
    squeezed_count = 0

    for i, move in enumerate(moves):
        t        = move["timestamp"]
        move_num = move["move_num"]
        label    = f"move_{move_num:04d}"
        frame_paths = {}

        gap_prev = t - moves[i-1]["timestamp"] if i > 0 else None
        gap_next = moves[i+1]["timestamp"] - t if i < len(moves) - 1 else None
        offsets  = move_window(gap_prev, gap_next)
        squeezed = (-offsets["before"] < BASE_PRE - 1e-9
                    or offsets["after"] < BASE_POST - 1e-9)
        if squeezed:
            squeezed_count += 1

        # A matched frame must stay inside this move's sandwich — with
        # capture gaps the nearest frame to a target can otherwise sit in
        # a neighboring move's territory.
        lo_ts = t + offsets["before"] - HALF_FRAME
        hi_ts = t + offsets["after"]  + HALF_FRAME

        for win_name, offset in offsets.items():
            nearest = find_nearest(frame_times, frame_recs, t + offset)
            if nearest is not None and not (lo_ts <= nearest["ts"] <= hi_ts):
                nearest = None

            if nearest is None:
                frame_paths[win_name] = None
                missing += 1
                if dry_run:
                    print(f"  [{label}] {win_name:<8}  NO FRAME IN WINDOW "
                          f"(target t{offset*1000:+.0f}ms)")
                continue

            src = frames_dir / nearest["file"]
            if not src.exists():
                frame_paths[win_name] = None
                missing += 1
                continue

            rel_path = f"labeled/{label}_{win_name}.jpg"
            gap_ms = abs(nearest["ts"] - (t + offset)) * 1000

            if dry_run:
                print(f"  [{label}] {win_name:<8}  t{offset*1000:+6.0f}ms"
                      f"  gap={gap_ms:5.1f}ms"
                      f"  [{MOTION_LABELS[win_name]}]"
                      f"{'  [squeezed]' if squeezed else ''}"
                      f"  ← {nearest['file']}")
            else:
                shutil.copy2(src, labeled_dir / f"{label}_{win_name}.jpg")

            frame_paths[win_name] = rel_path
            total_frames += 1
            if MOTION_LABELS[win_name] == "STILL":
                still_count  += 1
            else:
                moving_count += 1

        labeled_moves.append({
            **move,
            "frames":         frame_paths,
            "motion_labels":  MOTION_LABELS,
            "window_offsets": {k: round(v, 4) for k, v in offsets.items()},
            "window_squeezed": squeezed,
        })

    if not dry_run:
        with open(session_dir / "moves_labeled.jsonl", "w") as f:
            for m in labeled_moves:
                f.write(json.dumps(m) + "\n")

        metadata = {
            **config,
            **ble_meta,
            "move_count":     len(labeled_moves),
            "total_frames":   total_frames,
            "still_frames":   still_count,
            "moving_frames":  moving_count,
            "missing_windows":missing,
            "squeezed_windows":squeezed_count,
            "wca_sequence":   [m["wca_notation"] for m in moves
                               if m.get("wca_notation")],
        }
        with open(session_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    print(f"  Labeled:  {len(labeled_moves)} moves  "
          f"({total_frames} frames: {still_count} STILL, {moving_count} MOVING)")
    print(f"  Squeezed: {squeezed_count} windows tightened to fit between "
          f"neighboring moves")
    if missing:
        print(f"  Missing:  {missing} windows (frame capture gap or timing)")
    else:
        print(f"  Coverage: 100%")

    if not dry_run:
        print(f"\n  → {session_dir}/moves_labeled.jsonl")
        print(f"  → {labeled_dir}/")

        if not keep_frames and frames_dir.is_dir():
            freed = sum(f.stat().st_size for f in frames_dir.glob("*")
                       if f.is_file())
            shutil.rmtree(frames_dir)
            print(f"  Deleted raw frames/ ({freed / 1e6:.1f} MB freed) — "
                  f"labeled/ frames are kept for training.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", type=str, required=True,
                        help="Session folder from record_training.py")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview alignment without writing files")
    parser.add_argument("--discard-frames", action="store_true",
                        help="Delete the raw frames/ directory after "
                             "labeled/ is written. Frames are kept by "
                             "default: without them the session can never "
                             "train the onset detector (labeled/ holds only "
                             "move-centred positives), and that is not "
                             "recoverable. Only pass this if you are sure "
                             "you will never want move_detector/ to use "
                             "this session.")
    parser.add_argument("--keep-frames", action="store_true",
                        help=argparse.SUPPRESS)  # now the default; accepted
                                                 # so old commands still run
    args = parser.parse_args()

    d = Path(args.session)
    if not d.exists():
        print(f"ERROR: {d} not found")
        sys.exit(1)

    postprocess(d, dry_run=args.dry_run, keep_frames=not args.discard_frames)